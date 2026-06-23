import numpy as np
from scipy.stats import ttest_ind, f
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ScientificValidator:
    """Class to perform scientific validation of detected exoplanet candidates."""

    def __init__(self):
        pass

    def compute_snr(self, depth: float, rms_noise: float) -> float:
        """Calculate the Signal-to-Noise Ratio (SNR)."""
        return float(depth / rms_noise) if rms_noise > 0 else 0.0

    def perform_odd_even_test(self, 
                               time: np.ndarray, 
                               flux: np.ndarray, 
                               period: float, 
                               t0: float, 
                               duration: float) -> Dict[str, Any]:
        """
        Perform a t-test to check if odd and even transits have different depths.
        A low p-value (e.g., < 0.05) indicates they are statistically different.
        """
        epoch_idx = np.round((time - t0) / period)
        in_transit = np.abs(time - (t0 + epoch_idx * period)) < (duration / 2.0)
        
        # Out-of-transit median as baseline
        out_median = np.median(flux[~in_transit])
        
        # Odd and even in-transit points
        odd_mask = in_transit & (epoch_idx % 2 == 1)
        even_mask = in_transit & (epoch_idx % 2 == 0)
        
        odd_flux = flux[odd_mask]
        even_flux = flux[even_mask]
        
        if len(odd_flux) > 3 and len(even_flux) > 3:
            # Perform two-sample independent t-test
            stat, p_val = ttest_ind(odd_flux, even_flux, equal_var=False)
            
            odd_depth = out_median - np.median(odd_flux)
            even_depth = out_median - np.median(even_flux)
            
            return {
                "stat": float(stat),
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "odd_depth": float(odd_depth),
                "even_depth": float(even_depth),
                "significant_difference": bool(p_val < 0.05) if not np.isnan(p_val) else False
            }
            
        return {
            "stat": 0.0, "p_value": 1.0, "odd_depth": 0.0, "even_depth": 0.0,
            "significant_difference": False
        }

    def perform_centroid_shift_test(self, 
                                    lc_object: Any, 
                                    period: float, 
                                    t0: float, 
                                    duration: float) -> Dict[str, Any]:
        """
        Perform centroid shift analysis during transit vs out of transit.
        Uses centroid columns from the light curve if available.
        """
        # Try to find centroid columns in lightkurve object
        # In TESS SPOC files, they are usually 'centroid_col'/'centroid_row' or 'mom_centr1'/'mom_centr2'
        col_array = None
        row_array = None
        
        for col in ['centroid_col', 'mom_centr1', 'pos_corr1']:
            if hasattr(lc_object, col):
                col_array = getattr(lc_object, col).value
                break
                
        for row in ['centroid_row', 'mom_centr2', 'pos_corr2']:
            if hasattr(lc_object, row):
                row_array = getattr(lc_object, row).value
                break

        if col_array is None or row_array is None:
            logger.warning("Centroid columns not found in light curve. Skipping centroid shift test.")
            return {"col_shift": 0.0, "row_shift": 0.0, "p_value_col": 1.0, "p_value_row": 1.0, "shift_detected": False}

        time = lc_object.time.value
        
        # Filter NaNs in centroid arrays
        valid_mask = ~np.isnan(time) & ~np.isnan(col_array) & ~np.isnan(row_array)
        time = time[valid_mask]
        col_array = col_array[valid_mask]
        row_array = row_array[valid_mask]

        if len(time) < 10:
            return {"col_shift": 0.0, "row_shift": 0.0, "p_value_col": 1.0, "p_value_row": 1.0, "shift_detected": False}

        epoch_idx = np.round((time - t0) / period)
        in_transit = np.abs(time - (t0 + epoch_idx * period)) < (duration / 2.0)
        
        col_in = col_array[in_transit]
        col_out = col_array[~in_transit]
        row_in = row_array[in_transit]
        row_out = row_array[~in_transit]
        
        if len(col_in) > 3 and len(col_out) > 3:
            # Perform t-tests
            stat_col, p_col = ttest_ind(col_in, col_out, equal_var=False)
            stat_row, p_row = ttest_ind(row_in, row_out, equal_var=False)
            
            p_col_val = float(p_col) if not np.isnan(p_col) else 1.0
            p_row_val = float(p_row) if not np.isnan(p_row) else 1.0
            
            col_shift = float(np.abs(np.mean(col_in) - np.mean(col_out)))
            row_shift = float(np.abs(np.mean(row_in) - np.mean(row_out)))
            
            # Significant shift if either col or row shifts significantly
            shift_detected = bool(p_col_val < 0.01 or p_row_val < 0.01)
            
            return {
                "col_shift": col_shift,
                "row_shift": row_shift,
                "p_value_col": p_col_val,
                "p_value_row": p_row_val,
                "shift_detected": shift_detected
            }
            
        return {"col_shift": 0.0, "row_shift": 0.0, "p_value_col": 1.0, "p_value_row": 1.0, "shift_detected": False}

    def perform_significance_test(self, flux: np.ndarray, fitted_flux: np.ndarray) -> Dict[str, Any]:
        """
        Compare transit model fit against a flat baseline (no transit).
        Calculates F-statistic and its p-value.
        """
        flat_flux = np.ones_like(flux)
        
        # Sum of squared residuals
        ssr_flat = np.sum((flux - flat_flux) ** 2)
        ssr_fit = np.sum((flux - fitted_flux) ** 2)
        
        n_points = len(flux)
        # Flat model has 0 free parameters (fixed at 1.0)
        # Transit model has 4 free parameters (t0, depth, duration, ingress/LD parameters)
        df_flat = n_points
        df_fit = n_points - 4
        
        if ssr_fit > 0 and df_fit > 0:
            # F-statistic formula
            f_stat = ((ssr_flat - ssr_fit) / (df_flat - df_fit)) / (ssr_fit / df_fit)
            p_val = 1.0 - f.cdf(f_stat, df_flat - df_fit, df_fit)
            
            return {
                "f_statistic": float(f_stat),
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "ssr_reduction": float(ssr_flat - ssr_fit)
            }
            
        return {"f_statistic": 0.0, "p_value": 1.0, "ssr_reduction": 0.0}

    def generate_reliability_score(self, 
                                   snr: float, 
                                   odd_even_res: Dict[str, Any], 
                                   centroid_res: Dict[str, Any], 
                                   sig_res: Dict[str, Any]) -> float:
        """
        Combine all checks into a single reliability percentage (0 to 100).
        """
        score = 100.0
        
        # 1. Check SNR
        if snr < 4.0:
            score -= 40.0
        elif snr < 8.0:
            score -= 20.0
            
        # 2. Check Odd-Even transit difference
        if odd_even_res["significant_difference"]:
            # If the difference is also large
            odd_even_diff = np.abs(odd_even_res["odd_depth"] - odd_even_res["even_depth"])
            if odd_even_diff > 0.0001:
                score -= 30.0
                
        # 3. Check Centroid Shift
        if centroid_res["shift_detected"]:
            score -= 35.0
            
        # 4. Check Statistical Significance
        if sig_res["p_value"] > 0.01:
            score -= 25.0
            
        return float(max(0.0, min(100.0, score)))
