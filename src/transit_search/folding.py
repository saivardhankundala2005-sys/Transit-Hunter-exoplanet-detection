import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class PhaseFolder:
    """Class to fold and bin light curves for visualization and model input."""

    def __init__(self, 
                 global_bins: int = 200, 
                 local_bins: int = 80, 
                 local_phase_width: float = 0.1):
        self.global_bins = global_bins
        self.local_bins = local_bins
        self.local_phase_width = local_phase_width

    def fold(self, time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fold the light curve around period and t0.
        Returns phase (ranging from -0.5 to 0.5) and flux.
        """
        # Phase centered around t0, ranging from 0.0 to 1.0
        phase = ((time - t0) / period) % 1.0
        
        # Shift phase to be between -0.5 and 0.5
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        
        # Sort by phase
        sort_idx = np.argsort(phase)
        return phase[sort_idx], flux[sort_idx]

    def bin_curve(self, phase: np.ndarray, flux: np.ndarray, num_bins: int, phase_min: float = -0.5, phase_max: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Bin phase-folded data into a fixed number of bins."""
        bin_edges = np.linspace(phase_min, phase_max, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        binned_flux = np.zeros(num_bins)
        
        # Digitize the phases
        bin_indices = np.digitize(phase, bin_edges) - 1
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                binned_flux[i] = np.nanmedian(flux[mask])
            else:
                # Interpolate if bin is empty
                binned_flux[i] = np.nan
                
        # Handle empty/NaN bins by linear interpolation
        nans = np.isnan(binned_flux)
        if np.all(nans):
            binned_flux = np.ones(num_bins)
        elif np.any(nans):
            x = lambda z: z.nonzero()[0]
            binned_flux[nans] = np.interp(x(nans), x(~nans), binned_flux[~nans])
            
        return bin_centers, binned_flux

    def get_global_and_local_arrays(self, 
                                    time: np.ndarray, 
                                    flux: np.ndarray, 
                                    period: float, 
                                    t0: float, 
                                    duration: float) -> Dict[str, np.ndarray]:
        """
        Generate binned global and local arrays.
        Normalizes flux by subtracting the median (baseline -> 0.0).
        """
        # Phase fold the entire light curve
        phase, folded_flux = self.fold(time, flux, period, t0)
        
        # Global view: range -0.5 to 0.5, size global_bins
        global_centers, global_binned = self.bin_curve(phase, folded_flux, self.global_bins, -0.5, 0.5)
        
        # Local view: range around transit. Width is dynamic, e.g. 3-4 times transit duration as phase
        # If duration is 0.1 days, and period is 10 days, transit duration as phase is 0.01.
        # We can set the local phase window size dynamically based on duration/period, or use a fixed width.
        # Dynamic: width = 4 * (duration / period). Let's make sure it doesn't exceed 0.5.
        transit_phase_width = duration / period
        local_half_width = min(0.25, max(0.02, 3.0 * transit_phase_width))
        
        local_centers, local_binned = self.bin_curve(
            phase, folded_flux, self.local_bins, -local_half_width, local_half_width
        )
        
        # Standard normalization for CNN: subtract median out-of-transit
        # For simplicity, subtract median of the global curve
        global_median = np.median(global_binned)
        global_normalized = global_binned - global_median
        local_normalized = local_binned - global_median
        
        return {
            "global_phase": global_centers,
            "global_flux": global_normalized,
            "local_phase": local_centers,
            "local_flux": local_normalized,
            "raw_phase": phase,
            "raw_flux": folded_flux
        }

    def plot_folded_views(self, 
                          data_dict: Dict[str, np.ndarray], 
                          save_path: Optional[str] = None) -> plt.Figure:
        """Create publication-quality plots of global and local folded curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Global view plot
        axes[0].plot(data_dict["raw_phase"], data_dict["raw_flux"], '.', color='gray', alpha=0.3, label='Raw data')
        # Re-add global median to plot binned in actual flux scale
        global_median = np.median(data_dict["raw_flux"])
        axes[0].plot(data_dict["global_phase"], data_dict["global_flux"] + global_median, '-', color='red', lw=2, label='Binned (200)')
        axes[0].set_xlabel('Phase')
        axes[0].set_ylabel('Normalized Flux')
        axes[0].set_title('Global Folded Light Curve')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Local view plot
        # Find raw points that fall inside the local view range
        local_min, local_max = data_dict["local_phase"][0], data_dict["local_phase"][-1]
        local_mask = (data_dict["raw_phase"] >= local_min) & (data_dict["raw_phase"] <= local_max)
        
        axes[1].plot(data_dict["raw_phase"][local_mask], data_dict["raw_flux"][local_mask], '.', color='gray', alpha=0.4)
        axes[1].plot(data_dict["local_phase"], data_dict["local_flux"] + global_median, 'o-', color='blue', lw=2, label='Binned (80)')
        axes[1].set_xlabel('Phase')
        axes[1].set_ylabel('Normalized Flux')
        axes[1].set_title('Zoomed-in Transit (Local View)')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Phase fold plots saved to {save_path}")
            
        return fig
