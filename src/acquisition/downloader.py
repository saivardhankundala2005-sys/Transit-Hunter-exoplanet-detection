import os
import numpy as np
import lightkurve as lk
import logging
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TessDownloader:
    """Class to download and handle TESS light curves using Lightkurve."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def download_lightcurve(self, tic_id: str, sector: Optional[int] = None) -> Optional[lk.LightCurve]:
        """
        Download a light curve for a given TIC ID and sector.
        If sector is None, downloads the first available sector.
        """
        search_target = f"TIC {tic_id}"
        logger.info(f"Searching for light curves of {search_target}...")
        
        try:
            if sector is not None:
                search_result = lk.search_lightcurve(search_target, mission="TESS", sector=sector, author="SPOC")
            else:
                search_result = lk.search_lightcurve(search_target, mission="TESS", author="SPOC")
            
            if len(search_result) == 0:
                # Fallback to search without author SPOC (e.g. QLP)
                search_result = lk.search_lightcurve(search_target, mission="TESS")
                
            if len(search_result) == 0:
                logger.warning(f"No TESS light curves found for {search_target}.")
                return None
            
            # Download the first available product
            logger.info(f"Downloading light curve (found {len(search_result)} files)...")
            lc = search_result[0].download(download_dir=self.cache_dir)
            return lc
        except Exception as e:
            logger.error(f"Error downloading light curve for TIC {tic_id}: {str(e)}")
            return None

    def extract_arrays(self, lc: lk.LightCurve) -> Dict[str, np.ndarray]:
        """Extract time, flux, error, and quality arrays from a LightCurve object."""
        # Clean up NaNs from the raw LightCurve
        clean_lc = lc.remove_nans()
        
        # TESS time is typically in BTJD (BJD - 2457000)
        time = clean_lc.time.value
        
        # We prefer pdcsap_flux if available, otherwise fallback to flux
        if hasattr(clean_lc, 'pdcsap_flux'):
            flux = clean_lc.pdcsap_flux.value
            flux_err = clean_lc.pdcsap_flux_err.value
        else:
            flux = clean_lc.flux.value
            flux_err = clean_lc.flux_err.value
            
        quality = clean_lc.quality.value if hasattr(clean_lc, 'quality') else np.zeros_like(time, dtype=int)
        
        # Normalize flux to average 1.0 if not already normalized
        median_flux = np.nanmedian(flux)
        if median_flux != 0:
            flux = flux / median_flux
            flux_err = flux_err / median_flux
            
        return {
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
            "quality": quality,
            "tic_id": getattr(lc, "targetid", "Unknown")
        }


class TransitInjector:
    """Class to inject synthetic transits into TESS light curves."""

    @staticmethod
    def _trapezoidal_transit(time: np.ndarray, period: float, t0: float, 
                             depth: float, duration: float, ingress_ratio: float = 0.1) -> np.ndarray:
        """
        Analytical trapezoidal transit model fallback when batman-package is unavailable.
        duration is full width of transit (in days).
        ingress_ratio is the ratio of ingress duration to total duration.
        """
        # Phase representation between -period/2 and period/2
        phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
        half_dur = duration / 2.0
        ingress_dur = half_dur * ingress_ratio
        
        flux_model = np.ones_like(time)
        abs_phase = np.abs(phase)
        
        # Full transit depth zone
        full_transit = abs_phase <= (half_dur - ingress_dur)
        flux_model[full_transit] = 1.0 - depth
        
        # Ingress / Egress zone
        ingress_zone = (abs_phase > (half_dur - ingress_dur)) & (abs_phase < half_dur)
        # Linear interpolation
        for idx in np.where(ingress_zone)[0]:
            dist = abs_phase[idx] - (half_dur - ingress_dur)
            fraction = dist / ingress_dur
            flux_model[idx] = 1.0 - depth * (1.0 - fraction)
            
        return flux_model

    def inject_transit(self, 
                       time: np.ndarray, 
                       flux: np.ndarray, 
                       period: float, 
                       t0: float, 
                       depth: float, 
                       duration: float,
                       rp_rs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject a transit into the light curve.
        Returns:
            - injected_flux: the light curve flux with transit injected.
            - transit_model: the pure transit model (centered around 1.0).
        """
        if rp_rs is None:
            rp_rs = np.sqrt(depth)
            
        # Try to use batman for physical limb-darkened model
        try:
            import batman
            logger.info("Using batman for transit injection...")
            
            # Setup transit parameters
            params = batman.TransitParams()
            params.t0 = t0
            params.per = period
            params.rp = rp_rs # planet radius (in units of stellar radii)
            
            # Estimate a (semi-major axis in stellar radii) using Kepler's 3rd law
            # Or use a simple reasonable value. For a typical TESS exoplanet, a/Rs is ~10-20.
            # We can calculate an approximate a_rs based on duration:
            # duration_days ~ (period / pi) * sqrt( (1 + rp/rs)^2 - b^2 ) / a_rs
            # Assuming circular orbit and b=0: a_rs ~ period / (pi * duration)
            a_rs = period / (np.pi * duration)
            params.a = max(a_rs, 1.5)  # must be > 1
            
            params.inc = 90.0  # inclination in degrees
            params.ecc = 0.0   # circular orbit
            params.w = 90.0    # longitude of periastron
            params.u = [0.3, 0.2]  # quadratic limb darkening coefficients
            params.limb_dark = "quadratic"
            
            model = batman.TransitModel(params, time)
            transit_model = model.light_curve(params)
        except ImportError:
            logger.warning("batman-package not found. Falling back to analytical trapezoid model for injection.")
            transit_model = self._trapezoidal_transit(time, period, t0, depth, duration)
            
        # Inject the transit: multiply the flux by the transit model (since it's normalized to 1.0)
        injected_flux = flux * transit_model
        return injected_flux, transit_model
