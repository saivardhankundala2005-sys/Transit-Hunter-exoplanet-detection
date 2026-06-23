import numpy as np
from scipy.optimize import curve_fit
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class TransitFitter:
    """Class to fit physical transit models to light curves and estimate parameters."""

    def __init__(self, bootstrap_iterations: int = 30):
        self.bootstrap_iterations = bootstrap_iterations
        try:
            import batman
            self.has_batman = True
        except ImportError:
            self.has_batman = False
            logger.warning("batman-package not found. Fitting will use analytical trapezoidal model.")

    def _trapezoid_model(self, time: np.ndarray, t0: float, depth: float, duration: float, ingress_ratio: float) -> np.ndarray:
        """Trapezoidal transit model for curve fitting fallback."""
        half_dur = duration / 2.0
        ingress_dur = half_dur * ingress_ratio
        
        flux = np.ones_like(time)
        abs_t = np.abs(time - t0)
        
        # Bottom
        bottom = abs_t <= (half_dur - ingress_dur)
        flux[bottom] = 1.0 - depth
        
        # Ingress / Egress
        slope_zone = (abs_t > (half_dur - ingress_dur)) & (abs_t < half_dur)
        for idx in np.where(slope_zone)[0]:
            dist = abs_t[idx] - (half_dur - ingress_dur)
            fraction = dist / (ingress_dur if ingress_dur > 0 else 1e-5)
            flux[idx] = 1.0 - depth * (1.0 - fraction)
            
        return flux

    def _batman_model_func(self, time: np.ndarray, t0: float, rp_rs: float, a_rs: float, inc: float, period: float) -> np.ndarray:
        """Batman model wrapper for curve fitting."""
        import batman
        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inc
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.3, 0.2]  # Quadratic limb darkening
        params.limb_dark = "quadratic"
        
        m = batman.TransitModel(params, time)
        return m.light_curve(params)

    def fit_transit(self, 
                    time: np.ndarray, 
                    flux: np.ndarray, 
                    period: float, 
                    t0_init: float, 
                    duration_init: float, 
                    depth_init: float) -> Dict[str, Any]:
        """
        Fit a transit model to the light curve.
        Fits around the local transit region to avoid out-of-transit noise.
        """
        # Focus on local transit window: t0_init +/- 1.5 * duration_init
        window_mask = np.abs(time - t0_init) < (1.5 * duration_init)
        t_fit = time[window_mask]
        f_fit = flux[window_mask]
        
        if len(t_fit) < 10:
            logger.warning("Not enough data points in transit window to perform fitting. Using initial estimates.")
            return {
                "period": period, "t0": t0_init, "depth": depth_init, "duration": duration_init,
                "rp_rs": np.sqrt(depth_init), "a_rs": period / (np.pi * duration_init),
                "inc": 90.0, "impact_parameter": 0.0, "fitted_flux": flux,
                "uncertainties": {}
            }

        # Select model and bounds
        if self.has_batman:
            # We fit t0, rp_rs, a_rs, inc. Fix period to avoid degeneracies.
            # Initial guess:
            # Rp/Rs is sqrt(depth)
            rp_init = np.sqrt(depth_init)
            # a/Rs is P / (pi * duration)
            a_init = max(1.5, period / (np.pi * duration_init))
            p0 = [t0_init, rp_init, a_init, 90.0]
            
            # Bounds: [t0, rp_rs, a_rs, inc]
            bounds = (
                [t0_init - 0.1, 0.001, 1.1, 70.0],
                [t0_init + 0.1, 0.5, 100.0, 90.0]
            )
            
            def fit_wrapper(t, t0, rp, a, inc):
                return self._batman_model_func(t, t0, rp, a, inc, period)
        else:
            # Fit trapezoid: t0, depth, duration, ingress_ratio
            p0 = [t0_init, depth_init, duration_init, 0.1]
            bounds = (
                [t0_init - 0.1, 0.0001, 0.005, 0.01],
                [t0_init + 0.1, 0.2, 1.0, 0.99]
            )
            fit_wrapper = self._trapezoid_model

        # Perform curve fit
        try:
            popt, pcov = curve_fit(fit_wrapper, t_fit, f_fit, p0=p0, bounds=bounds, method='trf')
        except Exception as e:
            logger.error(f"Curve fitting failed: {str(e)}. Using initial estimates.")
            popt = p0
            pcov = None

        # Generate fitted flux model for entire time array
        if self.has_batman:
            fitted_flux = self._batman_model_func(time, popt[0], popt[1], popt[2], popt[3], period)
            fitted_params = {
                "t0": float(popt[0]),
                "rp_rs": float(popt[1]),
                "a_rs": float(popt[2]),
                "inc": float(popt[3]),
                "depth": float(popt[1]**2),
                # Duration formula from fitted parameters:
                "duration": float((period / np.pi) * np.arcsin(np.sqrt((1 + popt[1])**2 - (popt[2]*np.cos(np.deg2rad(popt[3])))**2) / popt[2])),
                "impact_parameter": float(popt[2] * np.cos(np.deg2rad(popt[3]))),
                "period": period
            }
        else:
            fitted_flux = self._trapezoid_model(time, popt[0], popt[1], popt[2], popt[3])
            fitted_params = {
                "t0": float(popt[0]),
                "depth": float(popt[1]),
                "duration": float(popt[2]),
                "rp_rs": float(np.sqrt(popt[1])),
                "a_rs": float(period / (np.pi * popt[2])),
                "inc": 90.0,
                "impact_parameter": 0.0,
                "ingress_ratio": float(popt[3]),
                "period": period
            }

        # Perform Bootstrap uncertainty estimation
        uncertainties = self._bootstrap_uncertainties(fit_wrapper, t_fit, f_fit, popt, bounds)
        fitted_params["uncertainties"] = uncertainties
        fitted_params["fitted_flux"] = fitted_flux

        return fitted_params

    def _bootstrap_uncertainties(self, 
                                 fit_wrapper: Any, 
                                 t_fit: np.ndarray, 
                                 f_fit: np.ndarray, 
                                 popt: List[float], 
                                 bounds: Tuple[List[float], List[float]]) -> Dict[str, float]:
        """Estimate uncertainties of parameters using bootstrap resampling of residuals."""
        logger.info(f"Estimating uncertainties with {self.bootstrap_iterations} bootstrap iterations...")
        
        # Calculate residuals
        best_fit = fit_wrapper(t_fit, *popt)
        residuals = f_fit - best_fit
        
        bootstrap_popts = []
        for _ in range(self.bootstrap_iterations):
            # Resample residuals
            resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
            f_bootstrap = best_fit + resampled_residuals
            
            try:
                p_opt_b, _ = curve_fit(fit_wrapper, t_fit, f_bootstrap, p0=popt, bounds=bounds, method='trf')
                bootstrap_popts.append(p_opt_b)
            except Exception:
                continue
                
        if len(bootstrap_popts) < 5:
            return {}

        bootstrap_popts = np.array(bootstrap_popts)
        stds = np.std(bootstrap_popts, axis=0)
        
        if self.has_batman:
            return {
                "t0_err": float(stds[0]),
                "rp_rs_err": float(stds[1]),
                "a_rs_err": float(stds[2]),
                "inc_err": float(stds[3])
            }
        else:
            return {
                "t0_err": float(stds[0]),
                "depth_err": float(stds[1]),
                "duration_err": float(stds[2]),
                "ingress_ratio_err": float(stds[3])
            }
