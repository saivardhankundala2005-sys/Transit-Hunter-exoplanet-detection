import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
import yaml
from typing import Dict, Any, Tuple

from src.acquisition.downloader import TransitInjector
from src.preprocessing.detrending import LightCurvePreprocessor
from src.transit_search.search import TransitSearcher
from src.transit_search.folding import PhaseFolder
from src.feature_engineering.features import FeatureExtractor
from src.models.cnn import DualViewCNN
from src.models.classifier import PhysicsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_local_base_curve(time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate base stellar light curve with low-frequency variability and white noise."""
    # Low frequency stellar variability (rotational modulation)
    stellar_var = 1.0 + 0.003 * np.sin(2 * np.pi * time / 2.5) + 0.001 * np.cos(2 * np.pi * time / 0.7)
    # White noise
    noise = np.random.normal(0, 0.0008, len(time))
    raw_flux = stellar_var + noise
    return raw_flux, stellar_var

def create_synthetic_sample(label: int, time: np.ndarray) -> Dict[str, Any]:
    """
    Create a single synthetic light curve belonging to one of the 4 classes.
    0: Exoplanet Transit, 1: Eclipsing Binary, 2: Stellar Blend, 3: Detector Artifact
    """
    raw_flux, trend = generate_local_base_curve(time)
    flux_err = np.ones_like(time) * 0.0008
    injector = TransitInjector()
    
    # Default outputs
    injected_flux = raw_flux.copy()
    centroid_col = np.random.normal(150.0, 0.01, len(time))
    centroid_row = np.random.normal(230.0, 0.01, len(time))
    
    # Core parameters
    period = float(np.random.uniform(2.0, 6.0))
    t0 = float(np.random.uniform(0.5, 1.5))
    
    if label == 0:  # Exoplanet Transit
        depth = float(np.random.uniform(0.004, 0.012))
        duration = float(np.random.uniform(0.08, 0.18))
        injected_flux, _ = injector.inject_transit(time, raw_flux, period, t0, depth, duration)
        
    elif label == 1:  # Eclipsing Binary
        # Deep, V-shaped or primary/secondary depth differences
        depth = float(np.random.uniform(0.03, 0.08))
        duration = float(np.random.uniform(0.15, 0.25))
        # Primary eclipse
        injected_flux, _ = injector.inject_transit(time, raw_flux, period, t0, depth, duration)
        # Secondary eclipse at half-phase with different depth
        sec_t0 = t0 + 0.5 * period
        sec_depth = depth * 0.4
        injected_flux, _ = injector.inject_transit(time, injected_flux, period, sec_t0, sec_depth, duration)
        
    elif label == 2:  # Stellar Blend
        # Shallow transit + significant centroid shift during transit
        depth = float(np.random.uniform(0.0015, 0.003))
        duration = float(np.random.uniform(0.1, 0.2))
        injected_flux, _ = injector.inject_transit(time, raw_flux, period, t0, depth, duration)
        
        # Simulate centroid shift in-transit
        epoch_idx = np.round((time - t0) / period)
        in_transit = np.abs(time - (t0 + epoch_idx * period)) < (duration / 2.0)
        centroid_col[in_transit] += np.random.uniform(0.03, 0.06)
        centroid_row[in_transit] += np.random.uniform(0.03, 0.06)
        
    elif label == 3:  # Detector Artifact
        # Non-periodic sudden drops or custom dip that BLS/TLS wraps poorly
        # Let's inject a single massive dip that is not periodic
        dip_center = time[len(time) // 2]
        dip_mask = np.abs(time - dip_center) < 0.15
        injected_flux[dip_mask] *= 0.985
        
    # Packaging as a lightkurve-like mock object for validator
    class MockLC:
        def __init__(self, t, f, e, c, r):
            self.time = type('Time', (object,), {'value': t})()
            self.flux = type('Flux', (object,), {'value': f})()
            self.flux_err = type('FluxErr', (object,), {'value': e})()
            self.centroid_col = type('CentroidCol', (object,), {'value': c})()
            self.centroid_row = type('CentroidRow', (object,), {'value': r})()
    
    lc_obj = MockLC(time, injected_flux, flux_err, centroid_col, centroid_row)
    
    return {
        "lc_obj": lc_obj,
        "time": time,
        "flux": injected_flux,
        "flux_err": flux_err,
        "label": label,
        "true_period": period,
        "true_t0": t0
    }

def main():
    logger.info("Initializing synthetic pipeline training...")
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup directories
    os.makedirs(config["data"]["model_dir"], exist_ok=True)
    
    # Time array: 15 days of observation at 2-minute cadence (720 points per day)
    time = np.linspace(0.0, 15.0, 15 * 720)
    
    # Setup processors
    preprocessor = LightCurvePreprocessor(
        sigma_upper=config["detrending"]["sigma_upper"],
        sigma_lower=config["detrending"]["sigma_lower"],
        wotan_window_length=config["detrending"]["wotan_window_length"]
    )
    searcher = TransitSearcher(
        min_period=config["transit_search"]["min_period"],
        max_period=config["transit_search"]["max_period"]
    )
    folder = PhaseFolder(
        global_bins=config["folding"]["global_bins"],
        local_bins=config["folding"]["local_bins"]
    )
    extractor = FeatureExtractor()

    # Generate dataset
    # We will generate 32 samples (8 of each class) for a quick but functional training demo
    samples_per_class = 8
    num_classes = 4
    
    logger.info(f"Generating synthetic dataset: {samples_per_class * num_classes} samples...")
    
    global_curves = []
    local_curves = []
    physics_features_list = []
    labels = []
    
    for label in range(num_classes):
        for s in range(samples_per_class):
            logger.info(f"Generating sample {s+1}/{samples_per_class} for class {label}")
            sample = create_synthetic_sample(label, time)
            
            # Detrend
            prep_res = preprocessor.flatten(sample["time"], sample["flux"], sample["flux_err"])
            
            # Search
            _, _, best_res = searcher.search_all(prep_res["time"], prep_res["flux"], prep_res["flux_err"])
            
            # Fold
            folded_res = folder.get_global_and_local_arrays(
                prep_res["time"], prep_res["flux"],
                best_res["period"], best_res["t0"], best_res["duration"]
            )
            
            # Extract features
            feats = extractor.extract_features(prep_res["time"], prep_res["flux"], best_res)
            
            # Add to lists
            global_curves.append(folded_res["global_flux"])
            local_curves.append(folded_res["local_flux"])
            
            feature_keys = [
                "transit_depth", "transit_duration", "period", "epoch",
                "ingress_slope", "egress_slope", "symmetry_score", "u_v_score",
                "rms_noise", "odd_even_difference", "transit_signal_strength"
            ]
            feats_array = np.array([feats[k] for k in feature_keys])
            physics_features_list.append(feats_array)
            labels.append(label)
            
    # Convert to numpy arrays
    X_global = np.array(global_curves)
    X_local = np.array(local_curves)
    X_physics = np.array(physics_features_list)
    y = np.array(labels)
    
    logger.info(f"Dataset compiled. X_global: {X_global.shape}, X_local: {X_local.shape}, X_physics: {X_physics.shape}")
    
    # ------------------ Train CNN (PyTorch) ------------------
    logger.info("Training Dual-view CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn_model = DualViewCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=config["models"]["cnn_lr"])
    
    # Convert to PyTorch tensors
    g_tensor = torch.FloatTensor(X_global)
    l_tensor = torch.FloatTensor(X_local)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(g_tensor, l_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config["models"]["cnn_batch_size"], shuffle=True)
    
    cnn_model.train()
    epochs = config["models"]["cnn_epochs"]
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_g, batch_l, batch_y in loader:
            batch_g, batch_l, batch_y = batch_g.to(device), batch_l.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = cnn_model(batch_g, batch_l)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_g.size(0)
            
        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    # Save CNN weights
    cnn_path = os.path.join(config["data"]["model_dir"], "cnn_model.pt")
    torch.save(cnn_model.state_dict(), cnn_path)
    logger.info(f"CNN model weights saved to {cnn_path}")
    
    # ------------------ Train Physics Classifier ------------------
    logger.info("Training Physics Branch Classifier...")
    physics_clf = PhysicsClassifier(random_state=config["models"]["random_state"])
    physics_clf.train(X_physics, y)
    
    physics_path = os.path.join(config["data"]["model_dir"], "physics_model.pkl")
    physics_clf.save(physics_path)
    logger.info("Training complete. Models successfully saved.")

if __name__ == "__main__":
    main()
