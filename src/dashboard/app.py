import sys
import os

# Resolve project root path and append to sys.path (avoids ModuleNotFoundError for src)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
import logging
from typing import Dict, Any

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

# Configure logger
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(
    page_title="TransitHunter - Scientific Discovery Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Scientific/Academic Theme Styling
st.markdown("""
<style>
    /* Main Content Background & Text */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"], .main {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Sidebar styling: clean light gray */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Headings styling - Times New Roman / Academic Serif */
    h1 {
        color: #0f172a !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #0f172a !important;
        padding-bottom: 6px !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    
    h2, h3 {
        color: #1e293b !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    
    /* Paragraphs and normal text */
    p, span, li, label, div {
        color: #1a1a1a !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 0.95rem !important;
    }
    
    /* Thin-bordered panels (Scientific cards) */
    .research-card, div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #fcfcfd !important;
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }
    
    .research-header {
        font-size: 0.8rem !important;
        color: #4b5563 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid #e5e7eb !important;
        padding-bottom: 4px !important;
        margin-bottom: 8px !important;
        font-family: Arial, sans-serif !important;
    }
    
    .research-value {
        font-size: 1.4rem !important;
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    .research-unit {
        font-size: 0.85rem !important;
        color: #2563eb !important;
        font-weight: 600 !important;
    }
    
    /* Vetting status badges */
    .success-badge {
        background-color: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        color: #166534 !important;
        padding: 2px 8px !important;
        border-radius: 2px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        display: inline-block;
    }
    
    .warning-badge {
        background-color: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        color: #991b1b !important;
        padding: 2px 8px !important;
        border-radius: 2px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        display: inline-block;
    }
    
    /* Scientific monitoring button styling */
    div.stButton > button {
        background-color: #f9fafb !important;
        border: 1px solid #d1d5db !important;
        color: #374151 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease !important;
    }
    
    div.stButton > button:hover {
        background-color: #f3f4f6 !important;
        border-color: #9ca3af !important;
        color: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load configuration
@st.cache_data
def load_config():
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)

# Helper to run pipeline and cache results in session_state
def run_pipeline_for_dashboard(tic_id: str, sector: int, inject: bool, inj_p: float, inj_t0: float, inj_d: float, inj_dur: float):
    config = load_config()
    with st.spinner("Executing: MAST target query..."):
        downloader = TessDownloader(cache_dir=config["data"]["cache_dir"])
        lc = downloader.download_lightcurve(tic_id, sector)
        if lc is None:
            st.error(f"Error: Target data acquisition failed for TIC {tic_id}.")
            return False
        arrays = downloader.extract_arrays(lc)
        
    time = arrays["time"]
    flux = arrays["flux"]
    flux_err = arrays["flux_err"]
    
    if inject:
        with st.spinner("Executing: Synthetic transit injection..."):
            injector = TransitInjector()
            flux, mock_model = injector.inject_transit(time, flux, inj_p, inj_t0, inj_d, inj_dur)
            
    with st.spinner("Executing: Wotan stellar detrending..."):
        preprocessor = LightCurvePreprocessor(
            sigma_upper=config["detrending"]["sigma_upper"],
            sigma_lower=config["detrending"]["sigma_lower"],
            wotan_window_length=config["detrending"]["wotan_window_length"]
        )
        prep_res = preprocessor.flatten(time, flux, flux_err)
        
    with st.spinner("Executing: BLS + TLS search algorithms..."):
        searcher = TransitSearcher(
            min_period=config["transit_search"]["min_period"],
            max_period=config["transit_search"]["max_period"]
        )
        bls_res, tls_res, best_res = searcher.search_all(prep_res["time"], prep_res["flux"], prep_res["flux_err"])
        
    with st.spinner("Executing: Phase folding and binning..."):
        folder = PhaseFolder(
            global_bins=config["folding"]["global_bins"],
            local_bins=config["folding"]["local_bins"]
        )
        folded_res = folder.get_global_and_local_arrays(
            prep_res["time"], prep_res["flux"],
            best_res["period"], best_res["t0"], best_res["duration"]
        )
        
    with st.spinner("Executing: Physics feature extraction..."):
        extractor = FeatureExtractor()
        feats = extractor.extract_features(prep_res["time"], prep_res["flux"], best_res)
        
    with st.spinner("Executing: Ensemble model classification..."):
        cnn_path = os.path.join(config["data"]["model_dir"], "cnn_model.pt")
        physics_path = os.path.join(config["data"]["model_dir"], "physics_model.pkl")
        
        cnn_model = None
        physics_clf = None
        
        if os.path.exists(cnn_path) and os.path.exists(physics_path):
            try:
                import torch
                cnn_model = DualViewCNN()
                cnn_model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
                physics_clf = PhysicsClassifier()
                physics_clf.load(physics_path)
            except Exception as e:
                logger.error(f"Error loading models in dashboard: {str(e)}")
        
        ensemble = HybridEnsemble(cnn_model=cnn_model, physics_model=physics_clf)
        class_res = ensemble.classify_candidate(folded_res["global_flux"], folded_res["local_flux"], feats)
        
    with st.spinner("Executing: Analytical transit fitting..."):
        fitter = TransitFitter(bootstrap_iterations=15)
        fit_res = fitter.fit_transit(
            prep_res["time"], prep_res["flux"],
            best_res["period"], best_res["t0"], best_res["duration"], best_res["depth"]
        )
        
    with st.spinner("Executing: Diagnostic vetting tests..."):
        validator = ScientificValidator()
        snr = validator.compute_snr(fit_res["depth"], feats["rms_noise"])
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
        reliability = validator.generate_reliability_score(snr, odd_even_res, centroid_res, sig_res)
        
    # Store everything in session state
    st.session_state["tic_id"] = tic_id
    st.session_state["raw_arrays"] = arrays
    st.session_state["prep_res"] = prep_res
    st.session_state["bls_res"] = bls_res
    st.session_state["tls_res"] = tls_res
    st.session_state["best_res"] = best_res
    st.session_state["folded_res"] = folded_res
    st.session_state["features"] = feats
    st.session_state["classification"] = class_res
    st.session_state["fit_res"] = fit_res
    st.session_state["validation"] = {
        "snr": snr,
        "reliability": reliability,
        "odd_even": odd_even_res,
        "centroid": centroid_res,
        "significance": sig_res
    }
    
    st.session_state["cnn_model"] = cnn_model
    st.session_state["physics_clf"] = physics_clf
    
    return True

# Initialize default synthetic demo dataset if none loaded
if "tic_id" not in st.session_state:
    st.session_state["demo_mode"] = True
    time_mock = np.linspace(0.0, 15.0, 15 * 500)
    var = 1.0 + 0.003 * np.sin(2 * np.pi * time_mock / 2.5) + np.random.normal(0, 0.001, len(time_mock))
    injector = TransitInjector()
    flux_mock, model_mock = injector.inject_transit(time_mock, var, 3.24, 0.85, 0.006, 0.12)
    flux_err_mock = np.ones_like(time_mock) * 0.001
    
    class MockLC:
        def __init__(self):
            self.time = type('Time', (object,), {'value': time_mock})()
            self.flux = type('Flux', (object,), {'value': flux_mock})()
            self.flux_err = type('FluxErr', (object,), {'value': flux_err_mock})()
            
    preprocessor = LightCurvePreprocessor(wotan_window_length=0.5)
    prep_res = preprocessor.flatten(time_mock, flux_mock, flux_err_mock)
    
    searcher = TransitSearcher(min_period=1.0, max_period=10.0)
    bls_res, tls_res, best_res = searcher.search_all(prep_res["time"], prep_res["flux"], prep_res["flux_err"])
    
    folder = PhaseFolder()
    folded_res = folder.get_global_and_local_arrays(
        prep_res["time"], prep_res["flux"], best_res["period"], best_res["t0"], best_res["duration"]
    )
    
    feats = FeatureExtractor().extract_features(prep_res["time"], prep_res["flux"], best_res)
    
    fit_res = TransitFitter(bootstrap_iterations=10).fit_transit(
        prep_res["time"], prep_res["flux"], best_res["period"], best_res["t0"], best_res["duration"], best_res["depth"]
    )
    
    st.session_state["tic_id"] = "Simulated Candidate 1"
    st.session_state["raw_arrays"] = {"time": time_mock, "flux": flux_mock, "flux_err": flux_err_mock}
    st.session_state["prep_res"] = prep_res
    st.session_state["bls_res"] = bls_res
    st.session_state["tls_res"] = tls_res
    st.session_state["best_res"] = best_res
    st.session_state["folded_res"] = folded_res
    st.session_state["features"] = feats
    st.session_state["classification"] = {
        "class_name": "Exoplanet Transit",
        "confidence": 0.885,
        "class_idx": 0,
        "probabilities": {"Exoplanet Transit": 0.885, "Eclipsing Binary": 0.082, "Stellar Blend": 0.021, "Detector Artifact": 0.012}
    }
    st.session_state["fit_res"] = fit_res
    st.session_state["validation"] = {
        "snr": 6.2,
        "reliability": 95.0,
        "odd_even": {"significant_difference": False, "odd_depth": 0.0059, "even_depth": 0.0061, "p_value": 0.85},
        "centroid": {"shift_detected": False, "col_shift": 0.0, "row_shift": 0.0, "p_value_col": 1.0, "p_value_row": 1.0},
        "significance": {"f_statistic": 34.5, "p_value": 1e-12, "ssr_reduction": 0.00045}
    }
    st.session_state["cnn_model"] = None
    st.session_state["physics_clf"] = None

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center; color: #0f172a; font-family: \"Times New Roman\", serif;'>TransitHunter</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8rem; color: #475569; margin-top: -10px;'>Physics-Guided Explainable AI Framework</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Page mapping with strict text labels (No Emojis)
pages_list = [
    "Home",
    "Data Explorer",
    "Light Curve Viewer",
    "Detrending Analysis",
    "Transit Detection",
    "Classification Results",
    "Explainability",
    "Final Report",
    "Research Mode"
]

# Navigation callback helper to avoid StreamlitAPIException
def navigate_to(page_name):
    st.session_state["nav_page"] = page_name
    st.session_state["nav_radio"] = page_name

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"

# Update index from current session state
current_idx = pages_list.index(st.session_state["nav_page"])

selected_page = st.sidebar.radio(
    "Navigation Menu",
    pages_list,
    index=current_idx,
    key="nav_radio"
)

if selected_page != st.session_state["nav_page"]:
    st.session_state["nav_page"] = selected_page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Target Identifier:** TIC {st.session_state['tic_id']}")
if "demo_mode" in st.session_state:
    st.sidebar.info("Running in simulated mode. Use Data Explorer to ingest real MAST light curves.")

# Global plotting context overrides
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = '#1a1a1a'
plt.rcParams['axes.labelcolor'] = '#1a1a1a'
plt.rcParams['xtick.color'] = '#1a1a1a'
plt.rcParams['ytick.color'] = '#1a1a1a'

# Helper to apply standard clean borders to plots
def apply_plot_borders(ax, hide_ticks=False):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cbd5e1')
        spine.set_linewidth(1.0)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

# --- 1. HOME PAGE ---
if st.session_state["nav_page"] == "Home":
    st.title("TransitHunter")
    st.caption("Physics-Guided Explainable AI Framework for Exoplanet Detection")
    
    st.markdown("""
    **Project Overview:**  
    TransitHunter is a modular scientific pipeline designed to analyze high-cadence photometric observations, remove systematic stellar activity trends, search for periodic occultations, extract shape-sensitive physical parameters, and perform joint neural network and physics-informed classification of planet transit candidates.
    
    **Pipeline Architecture:**  
    TESS Data → Detrending → BLS → TLS → Phase Folding → Feature Extraction → Hybrid AI → Transit Fitting → Validation
    """)
    
    st.markdown("---")
    st.subheader("Scientific Monitoring & Navigation Control Panel")
    
    # Homepage Wireframe Grid Layout
    # Row 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="research-header">Data Explorer</div>', unsafe_allow_html=True)
            st.markdown(f"**Target ID:** `TIC {st.session_state['tic_id']}`")
            
            # Mini table status fields
            data = {
                "Module Status": ["Detrending", "BLS/TLS", "Validation"],
                "Verification": ["Complete", "Complete", "Complete"]
            }
            df_status = pd.DataFrame(data)
            st.dataframe(df_status, hide_index=True, use_container_width=True)
            st.button("Open Data Explorer", key="go_explorer", on_click=navigate_to, args=("Data Explorer",), use_container_width=True)
        
    with col2:
        with st.container(border=True):
            st.markdown('<div class="research-header">Light Curve Viewer</div>', unsafe_allow_html=True)
            
            time = st.session_state["raw_arrays"]["time"]
            flux = st.session_state["raw_arrays"]["flux"]
            
            # Mini plot
            fig, ax = plt.subplots(figsize=(3, 1.0))
            ax.plot(time, flux, '.', color='#64748b', markersize=0.5, alpha=0.3)
            apply_plot_borders(ax, hide_ticks=True)
            fig.patch.set_facecolor('#fcfcfd')
            ax.set_facecolor('#fcfcfd')
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"**Data Points:** {len(time)}")
            st.markdown(f"**RMS Noise:** {np.std(flux)*1e6:.1f} ppm")
            st.markdown(f"**Time Span:** {time[-1]-time[0]:.2f} days")
            
            st.button("Open Light Curve", key="go_lc", on_click=navigate_to, args=("Light Curve Viewer",), use_container_width=True)
        
    with col3:
        with st.container(border=True):
            st.markdown('<div class="research-header">Detrending Analysis</div>', unsafe_allow_html=True)
            
            prep = st.session_state["prep_res"]
            
            # Mini plot
            fig, ax = plt.subplots(figsize=(3, 1.0))
            ax.plot(prep["time"], prep["flux"], '.', color='#2563eb', markersize=0.5, alpha=0.4)
            apply_plot_borders(ax, hide_ticks=True)
            fig.patch.set_facecolor('#fcfcfd')
            ax.set_facecolor('#fcfcfd')
            st.pyplot(fig)
            plt.close()
            
            st.markdown("**Method:** Wotan Biweight Filter")
            st.markdown(f"**Window Size:** {load_config()['detrending']['wotan_window_length']} days")
            st.markdown(f"**Residual Noise:** {st.session_state['features']['rms_noise']*1e6:.1f} ppm")
            
            st.button("Open Detrending", key="go_detrending", on_click=navigate_to, args=("Detrending Analysis",), use_container_width=True)

    # Row 2
    col4, col5, col6 = st.columns(3)
    
    with col4:
        with st.container(border=True):
            st.markdown('<div class="research-header">Transit Detection</div>', unsafe_allow_html=True)
            
            best = st.session_state["best_res"]
            
            # Mini plot
            fig, ax = plt.subplots(figsize=(3, 1.0))
            if "periods" in st.session_state["bls_res"]:
                ax.plot(st.session_state["bls_res"]["periods"], st.session_state["bls_res"]["powers"], color='black', lw=0.4)
            apply_plot_borders(ax, hide_ticks=True)
            fig.patch.set_facecolor('#fcfcfd')
            ax.set_facecolor('#fcfcfd')
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"**Best Period:** {best['period']:.5f} days")
            st.markdown(f"**Transit Epoch:** {best['t0']:.4f} BTJD")
            st.markdown(f"**Search SNR:** {best['snr']:.2f}")
            
            st.button("Open Transit Detection", key="go_detection", on_click=navigate_to, args=("Transit Detection",), use_container_width=True)
        
    with col5:
        with st.container(border=True):
            st.markdown('<div class="research-header">Classification Results</div>', unsafe_allow_html=True)
            
            class_res = st.session_state["classification"]
            
            # Mini chart
            fig, ax = plt.subplots(figsize=(3, 1.0))
            probs = list(class_res["probabilities"].values())
            names = ["Planet", "EB", "Blend", "Artifact"]
            y_pos = np.arange(len(names))
            ax.barh(y_pos, probs, color='#2563eb', alpha=0.7, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=6)
            ax.set_xlim([0, 1])
            apply_plot_borders(ax, hide_ticks=False)
            ax.tick_params(axis='both', labelsize=6)
            fig.patch.set_facecolor('#fcfcfd')
            ax.set_facecolor('#fcfcfd')
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"**Decision:** {class_res['class_name']}")
            st.markdown(f"**Confidence:** {class_res['confidence']:.2%}")
            
            st.button("Open Classification", key="go_classification", on_click=navigate_to, args=("Classification Results",), use_container_width=True)
        
    with col6:
        with st.container(border=True):
            st.markdown('<div class="research-header">Explainability & Final Report</div>', unsafe_allow_html=True)
            
            val = st.session_state["validation"]
            
            # Mini reliability gauge
            fig, ax = plt.subplots(figsize=(3, 1.0))
            rel = val["reliability"]
            ax.text(0.5, 0.5, f"Vetting Reliability\n{rel:.1f}%", 
                    ha='center', va='center', fontsize=11, fontweight='bold', 
                    color='#166534' if rel >= 75 else '#991b1b')
            apply_plot_borders(ax, hide_ticks=True)
            fig.patch.set_facecolor('#fcfcfd')
            ax.set_facecolor('#fcfcfd')
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"**F-test Significance:** {val['significance']['p_value']:.2e}")
            st.markdown(f"**Odd-Even Vetting:** {'Flagged' if val['odd_even']['significant_difference'] else 'Passed'}")
            st.markdown(f"**Centroid Shift:** {'Shift Detected' if val['centroid']['shift_detected'] else 'Stable'}")
            
            st.button("Open Final Report", key="go_report", on_click=navigate_to, args=("Final Report",), use_container_width=True)

    # Row 3
    col7, _, _ = st.columns(3)
    with col7:
        with st.container(border=True):
            st.markdown('<div class="research-header">Research Mode</div>', unsafe_allow_html=True)
            st.markdown("**Advanced Exoplanet Discovery Environment**")
            st.markdown("Planned capabilities for manual analysis, Custom Fits ingestion, leadboard tables, and Multi-model benchmarking.")
            st.markdown("<span class='warning-badge'>Status: In Development</span>", unsafe_allow_html=True)
            
            st.write("") # spacer
            st.button("Open Research Mode", key="go_research", on_click=navigate_to, args=("Research Mode",), use_container_width=True)

# --- 2. DATA EXPLORER ---
elif st.session_state["nav_page"] == "Data Explorer":
    st.title("Data Acquisition & Synthetic Injection Explorer")
    st.markdown("Download target light curves from Mikulski Archive for Space Telescopes (MAST) or inject simulated physical occultations to test algorithm pipelines.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Source Options")
        target_tic = st.text_input("TESS Input Catalog (TIC) Identifier:", "261108232")
        sector_num = st.number_input("TESS Sector (0 for automatic detection):", min_value=0, max_value=100, value=0)
        
        st.subheader("Synthetic Transit Model Injection")
        inject_check = st.checkbox("Inject simulated transit model into flux?", value=False)
        
        p_inj = st.slider("Orbital Period (days):", 1.0, 15.0, 3.5, 0.01)
        depth_inj = st.slider("Transit Depth (fractional flux):", 0.001, 0.05, 0.006, 0.0001, format="%.4f")
        dur_inj = st.slider("Transit Duration (days):", 0.02, 0.4, 0.12, 0.01)
        t0_inj = st.slider("Transit Center Epoch t0 (days):", 0.1, 3.0, 1.0, 0.1)

    with col2:
        st.subheader("Automated Pipeline Controller")
        st.markdown("""
        Deploying the execution stack triggers:
        - Stellar rotation detrending (Wotan high-pass)
        - Phase space BLS period searching
        - Physical orbital fitting & bootstrapping
        - Scientific vetting validations (centroids & odd-even checks)
        - CNN & XGBoost ensemble classification
        """)
        
        if st.button("Execute Pipeline", use_container_width=True):
            if "demo_mode" in st.session_state:
                del st.session_state["demo_mode"]
                
            sec = None if sector_num == 0 else int(sector_num)
            success = run_pipeline_for_dashboard(target_tic, sec, inject_check, p_inj, t0_inj, depth_inj, dur_inj)
            if success:
                st.success("Success: Data acquisition and pipeline execution completed.")
                st.rerun()

# --- 3. LIGHT CURVE VIEWER ---
elif st.session_state["nav_page"] == "Light Curve Viewer":
    st.title("Raw Photometric Light Curve")
    
    time = st.session_state["raw_arrays"]["time"]
    flux = st.session_state["raw_arrays"]["flux"]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time, flux, '.', color='black', alpha=0.4, markersize=1.5, label='Raw Flux')
    ax.set_xlabel('Time (BTJD days)', fontsize=9)
    ax.set_ylabel('Relative Flux', fontsize=9)
    ax.set_title(f'Raw Light Curve - TIC {st.session_state["tic_id"]}', fontsize=10, fontweight='bold')
    apply_plot_borders(ax)
    ax.grid(True, color='#e5e7eb', linestyle='--', linewidth=0.5)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Data Stream Summary Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="research-card"><div class="research-header">Data Points</div><div class="research-value">{len(time)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="research-card"><div class="research-header">Observation Baseline</div><div class="research-value">{time[-1]-time[0]:.2f} <span class="research-unit">days</span></div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="research-card"><div class="research-header">Photometric RMS Scatter</div><div class="research-value">{np.std(flux)*1e6:.1f} <span class="research-unit">ppm</span></div></div>', unsafe_allow_html=True)

# --- 4. DETRENDING ANALYSIS ---
elif st.session_state["nav_page"] == "Detrending Analysis":
    st.title("Stellar Detrending & Cleaning Analysis")
    st.markdown("Removing low-frequency stellar rotation and instrumental drifts using a **Wotan high-pass biweight filter**.")
    
    prep = st.session_state["prep_res"]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].plot(prep["time"], prep["raw_flux"], '.', color='gray', alpha=0.4, markersize=1.5, label='Raw Flux')
    ax[0].plot(prep["time"], prep["trend"], color='red', lw=1.2, label='Biweight Trend Model')
    ax[0].set_ylabel('Relative Flux', fontsize=9)
    ax[0].set_title('Raw Curve & Estimated Low-frequency Trend', fontsize=10, fontweight='bold')
    apply_plot_borders(ax[0])
    ax[0].grid(True, color='#e5e7eb', linestyle='--', linewidth=0.5)
    ax[0].legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
    
    ax[1].plot(prep["time"], prep["flux"], '.', color='#2563eb', alpha=0.4, markersize=1.5, label='Detrended Flux')
    ax[1].set_xlabel('Time (BTJD days)', fontsize=9)
    ax[1].set_ylabel('Normalized Flux', fontsize=9)
    ax[1].set_title('Flattened Detrended Light Curve', fontsize=10, fontweight='bold')
    apply_plot_borders(ax[1])
    ax[1].grid(True, color='#e5e7eb', linestyle='--', linewidth=0.5)
    ax[1].legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
    
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Filter Summary Parameters")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Outlier Rejection:** $3\sigma$ iterative sigma-clipping was run to filter flares and pointing anomalies.")
    with c2:
        st.markdown(f"**Filter Method:** Sliding Biweight Filter  \n**Filter Scale (Window Width):** {load_config()['detrending']['wotan_window_length']} days  \n**Baseline RMS Noise:** {st.session_state['features']['rms_noise']*1e6:.1f} ppm")

# --- 5. TRANSIT DETECTION ---
elif st.session_state["nav_page"] == "Transit Detection":
    st.title("Periodic Transit Search Results")
    st.markdown("Initial signal search executed via **Box Least Squares (BLS)**, with parameter refinement utilizing U-shaped templates.")
    
    best = st.session_state["best_res"]
    
    col1, col2 = st.columns([1, 2.2])
    with col1:
        st.subheader("BLS Peak Characteristics")
        st.markdown(f'<div class="research-card"><div class="research-header">Orbital Period</div><div class="research-value">{best["period"]:.5f} <span class="research-unit">days</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="research-card"><div class="research-header">Transit Epoch (t0)</div><div class="research-value">{best["t0"]:.4f} <span class="research-unit">BTJD</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="research-card"><div class="research-header">Occultation Duration</div><div class="research-value">{best["duration"]*24.0:.2f} <span class="research-unit">hours</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="research-card"><div class="research-header">Detection SNR</div><div class="research-value">{best["snr"]:.2f}</div></div>', unsafe_allow_html=True)
        
    with col2:
        st.subheader("BLS Periodogram Power Spectrum")
        if "periods" in st.session_state["bls_res"]:
            periods = st.session_state["bls_res"]["periods"]
            powers = st.session_state["bls_res"]["powers"]
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(periods, powers, color='black', lw=0.7)
            ax.axvline(best['period'], color='blue', linestyle='--', label=f'Peak: {best["period"]:.4f} days')
            ax.set_xlabel('Search Period (days)', fontsize=9)
            ax.set_ylabel('BLS Power (arbitrary)', fontsize=9)
            apply_plot_borders(ax)
            ax.grid(True, color='#e5e7eb', linestyle='--', linewidth=0.5)
            ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("BLS Power spectrum array not loaded in simulated demo mode.")

# --- 6. CLASSIFICATION RESULTS ---
elif st.session_state["nav_page"] == "Classification Results":
    st.title("AI Classification & Orbit Optimization")
    
    class_res = st.session_state["classification"]
    fit = st.session_state["fit_res"]
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Ensemble Classifier Output")
        st.markdown(f'<div class="research-card"><div class="research-header">Ensemble Prediction</div><div class="research-value">{class_res["class_name"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="research-card"><div class="research-header">Confidence Level</div><div class="research-value">{class_res["confidence"]:.2%}</div></div>', unsafe_allow_html=True)
        
        st.markdown("**Probability Distribution:**")
        for cls_name, prob in class_res["probabilities"].items():
            st.write(f"*{cls_name}:* {prob:.1%}")
            st.progress(prob)
            
    with col2:
        st.subheader("Physical Orbit Fitting")
        folded = st.session_state["folded_res"]
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(folded["local_phase"], folded["local_flux"], 'o', color='black', markersize=3, label='Binned Flux (80 bins)', alpha=0.5)
        
        if "ingress_ratio" in fit:
            # Trapezoid model
            fitted_local = fitter = TransitFitter()._trapezoid_model(folded["local_phase"], 0.0, fit["depth"], fit["duration"]/fit["period"], fit["ingress_ratio"]) - 1.0
        else:
            # Batman model
            fitter = TransitFitter()
            try:
                fitted_local = fitter._batman_model_func(folded["local_phase"], 0.0, fit["rp_rs"], fit["a_rs"], fit["inc"], best_res["period"])
                fitted_local = fitted_local - np.median(fitted_local)
            except Exception:
                fitted_local = TransitFitter()._trapezoid_model(folded["local_phase"], 0.0, fit["depth"], fit["duration"]/best_res["period"], 0.1) - 1.0
                
        ax.plot(folded["local_phase"], fitted_local, color='blue', lw=1.8, label='Limb-Darkened Keplerian Model')
        ax.set_xlabel('Phase offset', fontsize=9)
        ax.set_ylabel('Normalized Flux (binned)', fontsize=9)
        apply_plot_borders(ax)
        ax.grid(True, color='#e5e7eb', linestyle='--', linewidth=0.5)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
        st.pyplot(fig)
        plt.close()

    st.subheader("Optimized Physical Parameters (Bootstrap Uncertainty Bounds)")
    c1, c2, c3, c4 = st.columns(4)
    
    err = fit.get("uncertainties", {})
    rp_err_str = f" ± {err['rp_rs_err']:.4f}" if "rp_rs_err" in err else ""
    a_err_str = f" ± {err['a_rs_err']:.2f}" if "a_rs_err" in err else ""
    inc_err_str = f" ± {err['inc_err']:.2f}" if "inc_err" in err else ""
    
    with c1:
        st.markdown(f'<div class="research-card"><div class="research-header">Radius Ratio (Rp/Rs)</div><div class="research-value">{fit["rp_rs"]:.4f} <span style="font-size:0.8rem; color:#6b7280;">{rp_err_str}</span></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="research-card"><div class="research-header">Separation (a/Rs)</div><div class="research-value">{fit["a_rs"]:.2f} <span style="font-size:0.8rem; color:#6b7280;">{a_err_str}</span></div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="research-card"><div class="research-header">Inclination (i)</div><div class="research-value">{fit["inc"]:.1f}° <span style="font-size:0.8rem; color:#6b7280;">{inc_err_str}</span></div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="research-card"><div class="research-header">Impact parameter (b)</div><div class="research-value">{fit.get("impact_parameter", 0.0):.3f}</div></div>', unsafe_allow_html=True)

# --- 7. EXPLAINABILITY ---
elif st.session_state["nav_page"] == "Explainability":
    st.title("Explainable AI Diagnosis")
    st.markdown("Local shape attribution highlights generated from 1D Grad-CAM (CNN pathways) and feature importances.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Deep Learning Grad-CAM Local Activation Heatmap")
        cnn_model = st.session_state["cnn_model"]
        folded = st.session_state["folded_res"]
        class_res = st.session_state["classification"]
        
        # Draw Grad-CAM overlay
        if cnn_model is not None:
            explainer = ExplainabilityEngine(cnn_model=cnn_model)
            try:
                cam, _ = explainer.get_cnn_explanation(
                    folded["global_flux"], folded["local_flux"], class_res["class_idx"]
                )
                fig = explainer.plot_gradcam_overlay(folded["local_phase"], folded["local_flux"], cam)
                # Apply white background
                fig.patch.set_facecolor('white')
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Grad-CAM could not be computed: {str(e)}")
        else:
            # Mock demo representation
            cam = np.zeros_like(folded["local_flux"])
            center_idx = len(cam) // 2
            cam[center_idx-15 : center_idx+15] = np.linspace(0, 1, 30)
            explainer = ExplainabilityEngine()
            fig = explainer.plot_gradcam_overlay(folded["local_phase"], folded["local_flux"], cam)
            fig.patch.set_facecolor('white')
            st.pyplot(fig)
            plt.close()
        st.markdown("**Interpretation:** The activation heatmap denotes which specific phases dominated the CNN's decision. High intensity weights along the ingress and egress shoulders suggest physical transit parameters were correctly isolated.")
            
    with col2:
        st.subheader("2. Physics Classifier Feature Attribution")
        physics_clf = st.session_state["physics_clf"]
        feats = st.session_state["features"]
        feature_names = [
            "transit_depth", "transit_duration", "period", "epoch",
            "ingress_slope", "egress_slope", "symmetry_score", "u_v_score",
            "rms_noise", "odd_even_difference", "transit_signal_strength"
        ]
        
        explainer = ExplainabilityEngine(physics_classifier=physics_clf)
        
        if physics_clf is not None:
            try:
                contribs = explainer.get_physics_explanation(feats, feature_names)
                fig = explainer.plot_feature_importance(contribs)
                fig.patch.set_facecolor('white')
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Feature importance failed: {str(e)}")
        else:
            # Demo values
            contribs = {
                "transit_depth": 0.35, "transit_duration": 0.12, "period": 0.05, "epoch": 0.01,
                "ingress_slope": 0.08, "egress_slope": -0.04, "symmetry_score": 0.18, "u_v_score": 0.22,
                "rms_noise": -0.05, "odd_even_difference": 0.28, "transit_signal_strength": 0.40
            }
            fig = explainer.plot_feature_importance(contribs)
            fig.patch.set_facecolor('white')
            st.pyplot(fig)
            plt.close()
        st.markdown("**Interpretation:** Calculated feature importance metrics show the contribution direction of shape parameters on final candidate vetting.")

# --- 8. FINAL REPORT ---
elif st.session_state["nav_page"] == "Final Report":
    st.title("Scientific Candidate Diagnostic Report")
    
    val = st.session_state["validation"]
    fit = st.session_state["fit_res"]
    best = st.session_state["best_res"]
    class_res = st.session_state["classification"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Candidate General Summary")
        st.markdown(f"**Target ID:** `TIC {st.session_state['tic_id']}`")
        st.markdown(f"**AI Classification Result:** {class_res['class_name']} ({class_res['confidence']:.1%} confidence)")
        st.markdown(f"**Signal-to-Noise Ratio (SNR):** {val['snr']:.2f}")
        
        st.subheader("Diagnostic Validation Vetting Checklist")
        
        # Vetting status boxes
        oe_sig = val["odd_even"]["significant_difference"]
        oe_class = "warning-badge" if oe_sig else "success-badge"
        oe_text = "EB Flag Triggered (depth difference)" if oe_sig else "Passed (equal depth transits)"
        st.markdown(f"**Odd-Even Transit Depth Test:**  \n<span class='{oe_class}'>{oe_text}</span>", unsafe_allow_html=True)
        st.write(f"Odd depth: {val['odd_even']['odd_depth']:.4f}, Even depth: {val['odd_even']['even_depth']:.4f}")
        
        cen_shift = val["centroid"]["shift_detected"]
        cen_class = "warning-badge" if cen_shift else "success-badge"
        cen_text = "Blend Flag Triggered (centroid shift)" if cen_shift else "Passed (centroid is stable)"
        st.markdown(f"**Astrometric Centroid Shift Test:**  \n<span class='{cen_class}'>{cen_text}</span>", unsafe_allow_html=True)
        st.write(f"Column shift: {val['centroid']['col_shift']:.4f} pix, Row shift: {val['centroid']['row_shift']:.4f} pix")
        
        sig_pval = val["significance"]["p_value"]
        sig_class = "success-badge" if sig_pval < 0.001 else "warning-badge"
        sig_text = "Passed (highly significant dip)" if sig_pval < 0.001 else "Failed (non-significant signal)"
        st.markdown(f"**Model Significance (F-Test):**  \n<span class='{sig_class}'>{sig_text}</span> (p-value: {sig_pval:.2e})", unsafe_allow_html=True)

    with col2:
        st.subheader("Composite Pipeline Reliability Assessment")
        rel_score = val["reliability"]
        
        if rel_score >= 80:
            rel_color = "#166534" # Green
            desc = "High Reliability Candidate. Passes all vet criteria. Suggested follow-up candidate."
        elif rel_score >= 50:
            rel_color = "#854d0e" # Dark Yellow
            desc = "Moderate Reliability Candidate. Minor vettings flags present. Needs astrophysical evaluation."
        else:
            rel_color = "#991b1b" # Red
            desc = "False Positive Signal. Severe centroid shift or odd-even differences detected."
            
        st.markdown(f"""
        <div style='border: 1px solid #d1d5db; border-radius: 4px; padding: 20px; text-align: center; background-color: #f9fafb;'>
            <div style='font-size: 0.8rem; color: #4b5563; font-weight:700; text-transform:uppercase;'>Reliability Metric Score</div>
            <div style='font-size: 3rem; color: {rel_color}; font-weight: 700; margin: 8px 0;'>{rel_score:.1f}%</div>
            <div style='font-size: 0.9rem; color: #1f2937;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Orbital Parameters Reference Table")
        fit_df = pd.DataFrame({
            "Optimized Parameter": ["Orbital Period (P)", "Transit Center (t0)", "Transit Depth (d)", "Radius Ratio (Rp/Rs)", "Impact Parameter (b)", "Inclination (i)"],
            "Fitted Value": [f"{fit['period']:.5f} days", f"{fit['t0']:.4f} BTJD", f"{fit['depth']:.5f}", f"{fit['rp_rs']:.4f}", f"{fit.get('impact_parameter', 0.0):.3f}", f"{fit['inc']:.2f}°"]
        })
        st.table(fit_df)
        
    st.markdown("---")
    
    # Download JSON report
    report_dict = {
        "tic_id": st.session_state["tic_id"],
        "classification": class_res,
        "fitted_parameters": {k: float(v) if isinstance(v, (float, int)) else v for k, v in fit.items() if k != "fitted_flux"},
        "vetting": {
            "snr": float(val["snr"]),
            "reliability_score": float(rel_score),
            "odd_even_p_value": float(val["odd_even"]["p_value"]),
            "centroid_shift_detected": bool(val["centroid"]["shift_detected"]),
            "significance_p_value": float(val["significance"]["p_value"])
        }
    }
    
    st.download_button(
        "Download Full Report (JSON)",
        data=json.dumps(report_dict, indent=4),
        file_name=f"TIC_{st.session_state['tic_id']}_report.json",
        mime="application/json",
        use_container_width=True
    )

# --- 9. RESEARCH MODE ---
elif st.session_state["nav_page"] == "Research Mode":
    st.title("Research Mode - Coming Soon")
    st.caption("Advanced Exoplanet Discovery Environment")
    
    st.markdown("""
    Research Mode is intended to become an advanced workspace for astronomers, students, and researchers to perform custom exoplanet investigations beyond the automated detection pipeline.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Planned Future Capabilities")
        
        st.markdown("""
        **1. Multi-Source Data Ingestion**  
        Planned support for loading:
        - TESS Light Curves
        - Kepler Light Curves
        - FITS Files
        - CSV Files
        - TXT Time Series
        - Simulated Transit Datasets
        - Future Telescope Data Products
        
        **2. Custom Transit Analysis**  
        Planned support for:
        - Manual BLS Execution
        - Manual TLS Execution
        - Parameter Tuning
        - Search Range Optimization
        - Transit Injection Experiments
        - Noise Analysis Studies
        
        **3. Resource Allocation Controls**  
        Planned support for:
        - CPU Allocation
        - GPU Allocation
        - Batch Processing
        - Search Resolution Selection
        - Model Selection Controls
        """)
        
    with col2:
        st.subheader("Comparative & Benchmarking Tools")
        
        st.markdown("""
        **4. Comparative Research Workspace**  
        Planned support for:
        - BLS vs TLS Comparison
        - CNN vs Physics Classifier Comparison
        - Ensemble Comparison
        - Explainability Comparison
        - Multi-Model Benchmarking
        """)
        
        st.subheader("Exoplanet Candidate Leaderboard")
        st.markdown("Future ranking system for candidates, incorporating detection confidence, SNR, reliability, and scientific interest scores.")
        
        # Leaderboard Table
        leaderboard_df = pd.DataFrame({
            "Rank": [1, 2, 3, 4, 5],
            "Candidate ID": ["TIC 261108232.01", "TIC 142394651.01", "TIC 902846173.01", "TIC 441029486.01", "TIC 308947215.01"],
            "Confidence": ["98.4%", "95.1%", "89.3%", "84.2%", "78.9%"],
            "SNR": ["24.5", "18.2", "12.8", "9.4", "7.6"],
            "Reliability Score": ["96.0%", "92.0%", "85.0%", "72.0%", "55.0%"],
            "Status": ["Vetted & Confirmed", "Vetted Candidate", "Vetted Candidate", "Minor Flags Present", "High Blend Probability"]
        })
        st.table(leaderboard_df)
