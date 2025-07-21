"""Simple SVI (Stochastic Volatility Inspired) surface fitting utilities."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize


# --- SVI helpers -----------------------------------------------------------

Params = Tuple[float, float, float, float, float]  # a, b, rho, m, sigma


def svi_total_variance(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """Return total variance w(k) under the raw SVI parameterisation."""
    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k: float, t: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """Return implied volatility for log-moneyness ``k`` and maturity ``t``."""
    return math.sqrt(max(0.0, svi_total_variance(k, a, b, rho, m, sigma)) / max(t, 1e-12))


def fit_svi_smile(strikes: Iterable[float], vols: Iterable[float], f: float, t: float) -> Optional[Params]:
    """Calibrate SVI parameters to a volatility smile.

    Parameters
    ----------
    strikes : array-like
        Option strikes for the smile.
    vols : array-like
        Corresponding implied volatilities.
    f : float
        Forward price used for log-moneyness.
    t : float
        Time to maturity (in years).

    Returns
    -------
    tuple or None
        SVI parameters ``(a, b, rho, m, sigma)`` or ``None`` if calibration fails.
    """
    strikes = np.asarray(list(strikes), dtype=float)
    vols = np.asarray(list(vols), dtype=float)
    if t <= 0 or len(strikes) < 3:
        return None

    k = np.log(strikes / float(f))
    w_obs = vols ** 2 * t

    def objective(params: List[float]) -> float:
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        w_model = svi_total_variance(k, a, b, rho, m, sigma)
        return float(np.mean((w_model - w_obs) ** 2))

    init = [0.1, 0.1, 0.0, 0.0, 0.1]
    bounds = [(-1.0, 1.0), (1e-5, 5.0), (-0.999, 0.999), (-5.0, 5.0), (1e-5, 5.0)]
    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    if result.success:
        return tuple(result.x)  # type: ignore
    return None

