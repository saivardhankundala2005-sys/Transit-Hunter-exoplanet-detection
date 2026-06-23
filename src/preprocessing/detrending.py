import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip, biweight_location
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LightCurvePreprocessor:
    """Class to preprocess and detrend TESS light curves."""

    def __init__(self, 
                 sigma_upper: float = 3.0, 
                 sigma_lower: float = 3.0, 
                 savgol_window: int = 15, 
                 savgol_polyorder: int = 2,
                 wotan_window_length: float = 0.5,
                 wotan_method: str = "biweight"):
        self.sigma_upper = sigma_upper
        self.sigma_lower = sigma_lower
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.wotan_window_length = wotan_window_length
        self.wotan_method = wotan_method

    def apply_sigma_clipping(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply sigma clipping to remove extreme outliers."""
        # Use astropy's sigma_clip
        clipped = sigma_clip(flux, maxiters=5, sigma_upper=self.sigma_upper, sigma_lower=self.sigma_lower)
        mask = ~clipped.mask
        
        return time[mask], flux[mask], flux_err[mask]

    def apply_savgol_filter(self, flux: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to smooth out high-frequency noise."""
        # Ensure window length is odd and smaller than flux size
        window = self.savgol_window
        if window % 2 == 0:
            window += 1
        if window >= len(flux):
            window = len(flux) - 1 if len(flux) % 2 != 0 else len(flux) - 2
            
        if window < 3:
            return flux
            
        return savgol_filter(flux, window_length=window, polyorder=self.savgol_polyorder)

    def _fallback_flatten(self, time: np.ndarray, flux: np.ndarray, window_length: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom sliding window biweight/median filter as a fallback if wotan is unavailable.
        window_length is in days.
        """
        logger.info(f"Running fallback flattening with sliding window of {window_length} days...")
        trend = np.zeros_like(flux)
        half_window = window_length / 2.0
        
        # Sort or assume sorted
        for i, t in enumerate(time):
            # Find indices within t - half_window and t + half_window
            in_window = (time >= t - half_window) & (time <= t + half_window)
            window_flux = flux[in_window]
            
            # If biweight fails, fallback to median
            try:
                trend[i] = biweight_location(window_flux)
            except Exception:
                trend[i] = np.median(window_flux)
                
        # Handle cases where trend is zero or invalid
        trend[trend <= 0] = 1.0
        flattened_flux = flux / trend
        return flattened_flux, trend

    def flatten(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Dict[str, np.ndarray]:
        """Flatten the light curve using wotan biweight (or custom fallback)."""
        # First sigma clip
        t_clean, f_clean, e_clean = self.apply_sigma_clipping(time, flux, flux_err)
        
        try:
            import wotan
            logger.info("Using wotan for light curve flattening...")
            flattened_flux, trend = wotan.flatten(
                t_clean, f_clean, 
                window_length=self.wotan_window_length, 
                method=self.wotan_method,
                return_trend=True
            )
        except (ImportError, Exception) as e:
            logger.warning(f"wotan failed or not installed ({str(e)}). Falling back to custom sliding window filter.")
            flattened_flux, trend = self._fallback_flatten(t_clean, f_clean, self.wotan_window_length)
            
        # Also smooth the flattened flux with Savitzky-Golay for deep learning inputs
        smoothed_flux = self.apply_savgol_filter(flattened_flux)
        
        return {
            "time": t_clean,
            "raw_flux": f_clean,
            "flux": flattened_flux,
            "smoothed_flux": smoothed_flux,
            "flux_err": e_clean,
            "trend": trend
        }

    def plot_detrending(self, 
                        time: np.ndarray, 
                        raw_flux: np.ndarray, 
                        detrended_flux: np.ndarray, 
                        trend: np.ndarray, 
                        save_path: Optional[str] = None) -> plt.Figure:
        """Create before-after diagnostic detrending plots."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Raw + Trend
        axes[0].plot(time, raw_flux, '.', color='gray', alpha=0.5, label='Raw Flux')
        axes[0].plot(time, trend, color='red', lw=1.5, label='Stellar Trend')
        axes[0].set_ylabel('Normalized Flux')
        axes[0].set_title('Stellar Trend Removal & Detrending')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Detrended
        axes[1].plot(time, detrended_flux, '.', color='blue', alpha=0.6, label='Clean Detrended Flux')
        axes[1].set_xlabel('Time (Days)')
        axes[1].set_ylabel('Detrended Flux')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Detrending plot saved to {save_path}")
            
        return fig
