import numpy as np
from scipy.stats import linregress
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Class to extract physics-based features from light curves."""

    def __init__(self):
        pass

    def calculate_slopes_and_shape(self, 
                                   phase: np.ndarray, 
                                   flux: np.ndarray, 
                                   duration_phase: float) -> Dict[str, float]:
        """
        Calculate ingress/egress slopes, symmetry, and U/V shape scores.
        phase and flux should be sorted and phase centered at 0.
        """
        # Ingress zone: phase in [-duration_phase, 0]
        # Egress zone: phase in [0, duration_phase]
        ingress_mask = (phase >= -duration_phase) & (phase <= -0.15 * duration_phase)
        egress_mask = (phase >= 0.15 * duration_phase) & (phase <= duration_phase)
        bottom_mask = (phase >= -0.15 * duration_phase) & (phase <= 0.15 * duration_phase)
        
        # Ingress slope
        if np.sum(ingress_mask) >= 3:
            slope_in, _, _, _, _ = linregress(phase[ingress_mask], flux[ingress_mask])
        else:
            slope_in = 0.0
            
        # Egress slope
        if np.sum(egress_mask) >= 3:
            slope_out, _, _, _, _ = linregress(phase[egress_mask], flux[egress_mask])
        else:
            slope_out = 0.0

        # Symmetry score: compare left half of transit with flipped right half
        # We sample points on both sides and check difference
        left_mask = (phase >= -duration_phase) & (phase <= 0)
        right_mask = (phase >= 0) & (phase <= duration_phase)
        
        left_flux = flux[left_mask]
        left_phase_abs = np.abs(phase[left_mask])
        
        right_flux = flux[right_mask]
        right_phase_abs = np.abs(phase[right_mask])
        
        if len(left_flux) > 5 and len(right_flux) > 5:
            # Interpolate right flux onto left phase grid to compare directly
            right_interp = np.interp(left_phase_abs, right_phase_abs, right_flux)
            symmetry_score = float(np.mean(np.abs(left_flux - right_interp)))
        else:
            symmetry_score = 0.0

        # U/V shape score: ratio of middle transit depth to maximum depth
        # For a flat-bottomed U-shape, middle is deep, so shape_score is ~1.0
        # For a V-shape, the bottom is a single point, middle average is shallower, so shape_score is < 0.7
        min_flux = np.min(flux)
        max_depth = 1.0 - min_flux if min_flux < 1.0 else 1e-5
        
        if np.sum(bottom_mask) > 0:
            mean_bottom_flux = np.mean(flux[bottom_mask])
            bottom_depth = 1.0 - mean_bottom_flux
            u_v_score = float(bottom_depth / max_depth)
        else:
            u_v_score = 0.5
            
        return {
            "ingress_slope": float(slope_in),
            "egress_slope": float(slope_out),
            "symmetry_score": symmetry_score,
            "u_v_score": u_v_score
        }

    def calculate_odd_even_difference(self, 
                                      time: np.ndarray, 
                                      flux: np.ndarray, 
                                      period: float, 
                                      t0: float, 
                                      duration: float) -> float:
        """
        Calculate the depth difference between odd and even transits.
        A high difference suggests an Eclipsing Binary.
        """
        # Calculate transit epoch index for each data point
        epoch_idx = np.round((time - t0) / period)
        
        # Mask for points in transit
        in_transit = np.abs(time - (t0 + epoch_idx * period)) < (duration / 2.0)
        
        # Out-of-transit median
        out_median = np.median(flux[~in_transit])
        
        # Group in-transit points by odd/even epoch index
        odd_mask = in_transit & (epoch_idx % 2 == 1)
        even_mask = in_transit & (epoch_idx % 2 == 0)
        
        if np.sum(odd_mask) > 3 and np.sum(even_mask) > 3:
            odd_depth = out_median - np.median(flux[odd_mask])
            even_depth = out_median - np.median(flux[even_mask])
            
            # Relative difference
            avg_depth = 0.5 * (odd_depth + even_depth)
            if avg_depth > 0:
                diff = np.abs(odd_depth - even_depth) / avg_depth
                return float(diff)
            
        return 0.0

    def extract_features(self, 
                         time: np.ndarray, 
                         flux: np.ndarray, 
                         transit_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all 11 physical features from light curve data and search results.
        """
        period = transit_params["period"]
        t0 = transit_params["t0"]
        duration = transit_params["duration"]
        depth = transit_params["depth"]
        
        # Out-of-transit RMS noise
        epoch_idx = np.round((time - t0) / period)
        in_transit = np.abs(time - (t0 + epoch_idx * period)) < (duration / 2.0)
        
        if np.sum(~in_transit) > 10:
            rms_noise = float(np.std(flux[~in_transit]))
        else:
            rms_noise = float(np.std(flux))
            
        # Signal Strength (depth to noise ratio)
        signal_strength = float(depth / rms_noise) if rms_noise > 0 else 0.0
        
        # Odd-Even difference
        odd_even_diff = self.calculate_odd_even_difference(time, flux, period, t0, duration)
        
        # Folded shape metrics
        # Phase fold
        phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
        sort_idx = np.argsort(phase)
        sorted_phase = phase[sort_idx]
        sorted_flux = flux[sort_idx]
        
        duration_phase = duration / period
        shape_metrics = self.calculate_slopes_and_shape(sorted_phase, sorted_flux, duration_phase)
        
        features = {
            "transit_depth": float(depth),
            "transit_duration": float(duration),
            "period": float(period),
            "epoch": float(t0),
            "ingress_slope": shape_metrics["ingress_slope"],
            "egress_slope": shape_metrics["egress_slope"],
            "symmetry_score": shape_metrics["symmetry_score"],
            "u_v_score": shape_metrics["u_v_score"],
            "rms_noise": rms_noise,
            "odd_even_difference": odd_even_diff,
            "transit_signal_strength": signal_strength
        }
        
        return features
