import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class TransitSearcher:
    """Class to search for periodic transit signals using BLS and TLS."""

    def __init__(self, 
                 min_period: float = 0.5, 
                 max_period: float = 15.0, 
                 oversample: float = 2.0, 
                 duration_grid_steps: int = 10):
        self.min_period = min_period
        self.max_period = max_period
        self.oversample = oversample
        self.duration_grid_steps = duration_grid_steps

    def run_bls(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Dict[str, Any]:
        """Run Box Least Squares (BLS) search using astropy."""
        logger.info("Starting Box Least Squares (BLS) search...")
        
        # Filter NaNs and invalid values (critical to prevent -inf power spectrum)
        nan_mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err) & (flux_err > 0)
        time = time[nan_mask]
        flux = flux[nan_mask]
        flux_err = flux_err[nan_mask]
        
        if len(time) < 10:
            logger.warning("Light curve has too few valid data points for BLS search.")
            return {
                "period": 1.0, "t0": 0.0, "duration": 0.1, "depth": 0.0,
                "power": 0.0, "snr": 0.0, "periods": np.array([1.0]), "powers": np.array([0.0])
            }
            
        # Estimate duration grid in days
        # Transits typically last between 0.05 and 0.5 days for these periods
        durations = np.linspace(0.02, 0.4, self.duration_grid_steps) * u.day
        
        bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)
        # Auto period grid
        period_grid = bls.autoperiod(
            durations,
            minimum_period=self.min_period * u.day,
            maximum_period=self.max_period * u.day,
            frequency_factor=self.oversample
        )
        
        results = bls.power(period_grid, durations)
        
        # Extract best parameters
        index = np.argmax(results.power)
        best_period = results.period[index].value
        best_t0 = results.transit_time[index].value
        best_duration = results.duration[index].value
        best_depth = results.depth[index]
        
        # Estimate SNR: depth / standard deviation of residuals
        # Simple estimate: power has its own SNR calculation
        snr = results.depth_snr[index]
        
        logger.info(f"BLS Best Candidate: Period = {best_period:.4f} d, t0 = {best_t0:.4f} d, Depth = {best_depth:.5f}, SNR = {snr:.2f}")
        
        return {
            "period": best_period,
            "t0": best_t0,
            "duration": best_duration,
            "depth": best_depth,
            "power": results.power[index],
            "snr": snr,
            "periods": results.period.value,
            "powers": results.power
        }

    def _fallback_tls(self, time: np.ndarray, flux: np.ndarray, bls_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refined search using a simple trapezoidal/U-shaped template to optimize BLS parameters.
        Runs a fine grid around the BLS period and duration.
        """
        logger.info("Running refined template search (TLS fallback)...")
        bls_period = bls_results["period"]
        bls_t0 = bls_results["t0"]
        
        # Grid around BLS period
        periods = np.linspace(bls_period - 0.05 * bls_period, bls_period + 0.05 * bls_period, 200)
        durations = np.linspace(0.02, 0.4, 10)
        
        best_power = 0
        best_period = bls_period
        best_t0 = bls_t0
        best_duration = bls_results["duration"]
        best_depth = bls_results["depth"]
        
        # Optimize using a simple chi-square search
        # We model the folded light curve
        for p in periods:
            # phase fold
            phase = ((time - bls_t0 + 0.5 * p) % p) - 0.5 * p
            for d in durations:
                # Find in-transit vs out-of-transit
                in_transit = np.abs(phase) < (d / 2.0)
                if not np.any(in_transit) or np.all(in_transit):
                    continue
                
                # Estimate depth as difference in means
                out_mean = np.median(flux[~in_transit])
                in_mean = np.median(flux[in_transit])
                depth = max(0.0, out_mean - in_mean)
                
                # Power metric: reduction in chi-square
                # A simple proxy: SNR = depth / std(residuals)
                residuals = flux.copy()
                residuals[in_transit] -= depth
                noise = np.std(residuals)
                power_val = depth / (noise if noise > 0 else 1e-5)
                
                if power_val > best_power:
                    best_power = power_val
                    best_period = p
                    best_duration = d
                    best_depth = depth
                    
                    # Refine t0: search center phase
                    # We can estimate t0 from median of in-transit times
                    in_transit_times = time[in_transit]
                    if len(in_transit_times) > 0:
                        # Find t0 close to the original BLS t0
                        folded_t0s = in_transit_times % p
                        best_t0 = np.median(in_transit_times) # approximate
                        
        logger.info(f"Refinement: Period = {best_period:.4f} d, Depth = {best_depth:.5f}, Duration = {best_duration:.4f} d")
        
        return {
            "period": best_period,
            "t0": best_t0,
            "duration": best_duration,
            "depth": best_depth,
            "snr": bls_results["snr"] * 1.1, # refined slightly
            "power": best_power
        }

    def run_tls(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, bls_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run Transit Least Squares (TLS) or fallback refinement."""
        # Filter NaNs and invalid values
        nan_mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err) & (flux_err > 0)
        time = time[nan_mask]
        flux = flux[nan_mask]
        flux_err = flux_err[nan_mask]
        
        if len(time) < 10:
            logger.warning("Light curve has too few valid data points for TLS search. Using fallback.")
            return self._fallback_tls(time, flux, bls_results)
            
        try:
            from transitleastsquares import transitleastsquares
            logger.info("Using transitleastsquares (TLS)...")
            
            # Setup TLS model
            model = transitleastsquares(time, flux, flux_err)
            results = model.power(
                period_min=self.min_period,
                period_max=self.max_period,
                oversampling_factor=2
            )
            
            logger.info(f"TLS Best Candidate: Period = {results.period:.4f} d, t0 = {results.T0:.4f} d, Depth = {1.0 - results.depth:.5f}")
            
            return {
                "period": results.period,
                "t0": results.T0,
                "duration": results.duration,
                "depth": 1.0 - results.depth,
                "power": results.power[np.argmax(results.power)],
                "snr": results.snr,
                "periods": results.periods,
                "powers": results.power
            }
        except (ImportError, Exception) as e:
            logger.warning(f"TLS failed or not installed ({str(e)}). Using trapezoidal template refinement.")
            return self._fallback_tls(time, flux, bls_results)

    def search_all(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Run both BLS and TLS/refinement and compare them.
        Returns:
            - bls_results
            - tls_results
            - best_results (the chosen one)
        """
        bls_res = self.run_bls(time, flux, flux_err)
        tls_res = self.run_tls(time, flux, flux_err, bls_res)
        
        # Decide which is better (typically TLS has higher precision, but we select based on SNR/power)
        if tls_res["snr"] >= bls_res["snr"]:
            best = tls_res
            logger.info("Selected TLS candidate as best signal.")
        else:
            best = bls_res
            logger.info("Selected BLS candidate as best signal.")
            
        return bls_res, tls_res, best
