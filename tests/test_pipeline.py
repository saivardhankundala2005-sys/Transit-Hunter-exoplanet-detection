import unittest
import numpy as np
import torch

from src.preprocessing.detrending import LightCurvePreprocessor
from src.transit_search.search import TransitSearcher
from src.transit_search.folding import PhaseFolder
from src.feature_engineering.features import FeatureExtractor
from src.fitting.fitter import TransitFitter
from src.models.cnn import DualViewCNN
from src.models.classifier import PhysicsClassifier

class TestExoplanetPipeline(unittest.TestCase):

    def setUp(self):
        # Generate stable mock light curve data for testing
        self.time = np.linspace(0.0, 10.0, 1000)
        self.flux_err = np.ones_like(self.time) * 0.001
        
        # Injected simple square transit: period = 4.0, t0 = 2.0, depth = 0.01, duration = 0.2
        self.base_flux = np.ones_like(self.time)
        phase = ((self.time - 2.0) % 4.0)
        in_transit = (phase < 0.1) | (phase > 3.9)
        self.base_flux[in_transit] -= 0.01
        
        # Add slight noise
        np.random.seed(42)
        self.flux = self.base_flux + np.random.normal(0, 0.0002, len(self.time))

    def test_preprocessing(self):
        preprocessor = LightCurvePreprocessor(savgol_window=7, wotan_window_length=0.5)
        prep_res = preprocessor.flatten(self.time, self.flux, self.flux_err)
        
        self.assertIn("time", prep_res)
        self.assertIn("flux", prep_res)
        self.assertIn("smoothed_flux", prep_res)
        self.assertEqual(len(prep_res["time"]), len(prep_res["flux"]))

    def test_transit_search(self):
        searcher = TransitSearcher(min_period=1.0, max_period=5.0)
        bls_res = searcher.run_bls(self.time, self.flux, self.flux_err)
        
        # We expect a period close to 4.0
        self.assertAlmostEqual(bls_res["period"], 4.0, delta=0.2)
        self.assertTrue(bls_res["depth"] > 0.0)
        self.assertTrue(bls_res["snr"] > 0)

    def test_phase_folding(self):
        folder = PhaseFolder(global_bins=200, local_bins=80)
        folded = folder.get_global_and_local_arrays(
            self.time, self.flux, period=4.0, t0=2.0, duration=0.2
        )
        
        self.assertEqual(len(folded["global_flux"]), 200)
        self.assertEqual(len(folded["local_flux"]), 80)
        # Check shapes
        self.assertEqual(folded["global_flux"].shape, (200,))
        self.assertEqual(folded["local_flux"].shape, (80,))

    def test_feature_extraction(self):
        transit_params = {"period": 4.0, "t0": 2.0, "duration": 0.2, "depth": 0.01}
        extractor = FeatureExtractor()
        feats = extractor.extract_features(self.time, self.flux, transit_params)
        
        expected_keys = [
            "transit_depth", "transit_duration", "period", "epoch",
            "ingress_slope", "egress_slope", "symmetry_score", "u_v_score",
            "rms_noise", "odd_even_difference", "transit_signal_strength"
        ]
        for key in expected_keys:
            self.assertIn(key, feats)

    def test_cnn_model_dimensions(self):
        model = DualViewCNN(global_size=200, local_size=80, num_classes=4)
        
        # Mock inputs: batch size of 2
        g_in = torch.randn(2, 200)
        l_in = torch.randn(2, 80)
        
        outputs = model(g_in, l_in)
        self.assertEqual(outputs.shape, (2, 4))

    def test_transit_fitting(self):
        fitter = TransitFitter(bootstrap_iterations=5)
        fit_res = fitter.fit_transit(self.time, self.flux, period=4.0, t0_init=2.0, duration_init=0.2, depth_init=0.01)
        
        self.assertIn("t0", fit_res)
        self.assertIn("depth", fit_res)
        self.assertIn("duration", fit_res)
        self.assertIn("uncertainties", fit_res)

if __name__ == "__main__":
    unittest.main()
