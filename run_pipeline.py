import os
import argparse
import numpy as np
import torch
import json
import yaml
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from src.acquisition.downloader import TessDownloader, TransitInjector
from src.preprocessing.detrending import LightCurvePreprocessor
from src.transit_search.search import TransitSearcher
from src.transit_search.folding import PhaseFolder
from src.feature_engineering.features import FeatureExtractor
from src.models.cnn import DualViewCNN
from src.models.classifier import PhysicsClassifier, HybridEnsemble
from src.fitting.fitter import TransitFitter
from src.validation.validator import ScientificValidator
from src.explainability.explain import ExplainabilityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exoplanet_pipeline(tic_id: str, 
                           sector: Optional[int] = None, 
                           inject_transit: bool = False,
                           inj_period: float = 3.5,
                           inj_t0: float = 1.0,
                           inj_depth: float = 0.005,
                           inj_duration: float = 0.1) -> Dict[str, Any]:
    """Execute the end-to-end exoplanet detection pipeline for a target TIC ID."""
    logger.info(f"=== Starting Exoplanet Detection Pipeline for TIC {tic_id} ===")
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = os.path.join(config["data"]["output_dir"], f"tic_{tic_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------ Module 1: Data Acquisition ------------------
    downloader = TessDownloader(cache_dir=config["data"]["cache_dir"])
    lc = downloader.download_lightcurve(tic_id, sector)
    
    if lc is None:
        logger.error(f"Failed to acquire data for TIC {tic_id}. Exiting.")
        return {}
        
    arrays = downloader.extract_arrays(lc)
    time = arrays["time"]
    flux = arrays["flux"]
    flux_err = arrays["flux_err"]
    
    # Optional Injection
    if inject_transit:
        logger.info(f"Injecting mock transit: P={inj_period} d, t0={inj_t0} d, depth={inj_depth}, dur={inj_duration} d")
        injector = TransitInjector()
        flux, _ = injector.inject_transit(time, flux, inj_period, inj_t0, inj_depth, inj_duration)

    # ------------------ Module 2: Detrending ------------------
    preprocessor = LightCurvePreprocessor(
        sigma_upper=config["detrending"]["sigma_upper"],
        sigma_lower=config["detrending"]["sigma_lower"],
        wotan_window_length=config["detrending"]["wotan_window_length"],
        wotan_method=config["detrending"]["wotan_method"]
    )
    prep_res = preprocessor.flatten(time, flux, flux_err)
    
    # Save Detrending Plot
    preprocessor.plot_detrending(
        prep_res["time"], prep_res["raw_flux"], prep_res["flux"], prep_res["trend"],
        save_path=os.path.join(output_dir, "detrending_comparison.png")
    )
    plt.close('all')

    # ------------------ Module 3: Transit Search ------------------
    searcher = TransitSearcher(
        min_period=config["transit_search"]["min_period"],
        max_period=config["transit_search"]["max_period"]
    )
    bls_res, tls_res, best_res = searcher.search_all(prep_res["time"], prep_res["flux"], prep_res["flux_err"])

    # ------------------ Module 4: Phase Folding ------------------
    folder = PhaseFolder(
        global_bins=config["folding"]["global_bins"],
        local_bins=config["folding"]["local_bins"]
    )
    folded_res = folder.get_global_and_local_arrays(
        prep_res["time"], prep_res["flux"],
        best_res["period"], best_res["t0"], best_res["duration"]
    )
    
    # Save Folding Plot
    folder.plot_folded_views(folded_res, save_path=os.path.join(output_dir, "phase_folded_views.png"))
    plt.close('all')

    # ------------------ Module 5: Feature Extraction ------------------
    extractor = FeatureExtractor()
    physics_features = extractor.extract_features(prep_res["time"], prep_res["flux"], best_res)

    # ------------------ Module 6: Hybrid Classification ------------------
    # Check and load models
    cnn_path = os.path.join(config["data"]["model_dir"], "cnn_model.pt")
    physics_path = os.path.join(config["data"]["model_dir"], "physics_model.pkl")
    
    cnn_model = None
    physics_clf = None
    
    if os.path.exists(cnn_path) and os.path.exists(physics_path):
        logger.info("Loading trained machine learning models...")
        try:
            cnn_model = DualViewCNN()
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
            
            physics_clf = PhysicsClassifier()
            physics_clf.load(physics_path)
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}. Proceeding with fallback dummy predictions.")
    else:
        logger.warning("Trained model weights not found. Running with un-trained/dummy classifiers. Run 'train_pipeline.py' to train models.")

    ensemble = HybridEnsemble(cnn_model=cnn_model, physics_model=physics_clf)
    classification = ensemble.classify_candidate(
        folded_res["global_flux"], folded_res["local_flux"], physics_features
    )
    
    # ------------------ Module 7: Parameter Fitting ------------------
    fitter = TransitFitter(bootstrap_iterations=config["fitting"]["bootstrap_iterations"])
    fit_res = fitter.fit_transit(
        prep_res["time"], prep_res["flux"],
        best_res["period"], best_res["t0"], best_res["duration"], best_res["depth"]
    )
    
    # Plot fitting results
    fig, ax = plt.subplots(figsize=(8, 4))
    local_min, local_max = folded_res["local_phase"][0], folded_res["local_phase"][-1]
    local_mask = (folded_res["raw_phase"] >= local_min) & (folded_res["raw_phase"] <= local_max)
    
    ax.plot(folded_res["raw_phase"][local_mask], folded_res["raw_flux"][local_mask], '.', color='gray', alpha=0.3, label='Raw data')
    
    # Generate folded fitted model
    fitted_folded_phase, fitted_folded_flux = folder.fold(prep_res["time"], fit_res["fitted_flux"], best_res["period"], best_res["t0"])
    fitted_local_mask = (fitted_folded_phase >= local_min) & (fitted_folded_phase <= local_max)
    
    ax.plot(fitted_folded_phase[fitted_local_mask], fitted_folded_flux[fitted_local_mask], color='red', lw=2.5, label='Physical Fit')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Transit Fitting for TIC {tic_id}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "transit_fit.png"), dpi=150, bbox_inches='tight')
    plt.close('all')

    # ------------------ Module 8: Scientific Validation ------------------
    validator = ScientificValidator()
    snr = validator.compute_snr(fit_res["depth"], physics_features["rms_noise"])
    
    odd_even_res = validator.perform_odd_even_test(
        prep_res["time"], prep_res["flux"],
        best_res["period"], best_res["t0"], best_res["duration"]
    )
    centroid_res = validator.perform_centroid_shift_test(
        lc, best_res["period"], best_res["t0"], best_res["duration"]
    )
    sig_res = validator.perform_significance_test(
        prep_res["flux"], fit_res["fitted_flux"]
    )
    reliability_score = validator.generate_reliability_score(snr, odd_even_res, centroid_res, sig_res)

    # ------------------ Module 9: Explainable AI ------------------
    explainer = ExplainabilityEngine(cnn_model=cnn_model, physics_classifier=physics_clf)
    
    # Try CNN Grad-CAM
    if cnn_model is not None:
        try:
            cam, _ = explainer.get_cnn_explanation(folded_res["global_flux"], folded_res["local_flux"], classification["class_idx"])
            explainer.plot_gradcam_overlay(
                folded_res["local_phase"], folded_res["local_flux"], cam,
                save_path=os.path.join(output_dir, "explainability_gradcam.png")
            )
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {str(e)}")
            
    # Try Physics SHAP/Perturbation explainability
    if physics_clf is not None:
        try:
            feature_keys = [
                "transit_depth", "transit_duration", "period", "epoch",
                "ingress_slope", "egress_slope", "symmetry_score", "u_v_score",
                "rms_noise", "odd_even_difference", "transit_signal_strength"
            ]
            contributions = explainer.get_physics_explanation(physics_features, feature_keys)
            explainer.plot_feature_importance(
                contributions,
                save_path=os.path.join(output_dir, "explainability_features.png")
            )
        except Exception as e:
            logger.error(f"SHAP/perturbation explainability failed: {str(e)}")
            
    plt.close('all')

    # ------------------ Packaging Report ------------------
    report = {
        "tic_id": tic_id,
        "classification": classification,
        "transit_parameters": {
            "period": float(fit_res["period"]),
            "t0": float(fit_res["t0"]),
            "depth": float(fit_res["depth"]),
            "duration": float(fit_res["duration"]),
            "rp_rs": float(fit_res["rp_rs"]),
            "a_rs": float(fit_res["a_rs"]),
            "inc": float(fit_res["inc"]),
            "impact_parameter": float(fit_res["impact_parameter"]),
            "uncertainties": {k: float(v) for k, v in fit_res["uncertainties"].items()}
        },
        "scientific_validation": {
            "snr": float(snr),
            "reliability_score": float(reliability_score),
            "odd_even_test": odd_even_res,
            "centroid_test": centroid_res,
            "significance_test": sig_res
        }
    }
    
    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Pipeline executed successfully. Outputs saved to {output_dir}")
    logger.info(f"Resulting Class: {classification['class_name']} (Confidence: {classification['confidence']:.2%})")
    logger.info(f"Reliability Score: {reliability_score:.1f}%")
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exoplanet Detection Pipeline")
    parser.add_argument("--tic", type=str, default="261108232", help="TESS Catalog ID (TIC)")
    parser.add_argument("--sector", type=int, default=None, help="TESS Sector number")
    parser.add_argument("--inject", action="store_true", help="Inject artificial transit for testing")
    parser.add_argument("--period", type=float, default=3.5, help="Injection Period (days)")
    parser.add_argument("--t0", type=float, default=1.0, help="Injection Epoch t0 (days)")
    parser.add_argument("--depth", type=float, default=0.005, help="Injection Depth (fraction)")
    parser.add_argument("--duration", type=float, default=0.1, help="Injection Duration (days)")
    
    args = parser.parse_args()
    
    run_exoplanet_pipeline(
        tic_id=args.tic,
        sector=args.sector,
        inject_transit=args.inject,
        inj_period=args.period,
        inj_t0=args.t0,
        inj_depth=args.depth,
        inj_duration=args.duration
    )
