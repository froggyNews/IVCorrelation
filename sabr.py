# Simple SABR model calibration and implied volatility surface construction
# This module uses pure Python (no external dependencies) to fit SABR
# parameters per maturity and return an implied volatility surface.

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


# ---------------------- SABR utilities ----------------------

def hagan_lognormal_vol(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Approximate SABR implied volatility using Hagan's formula."""
    if f <= 0 or k <= 0:
        raise ValueError("Forward and strike must be positive")
    if alpha <= 0 or t <= 0:
        raise ValueError("Alpha and maturity must be positive")

    if abs(f - k) < 1e-12:
        fk_beta = f ** (1 - beta)
        term1 = (
            (1 - beta) ** 2 / 24 * (alpha ** 2) / fk_beta ** 2
            + 0.25 * rho * beta * nu * alpha / fk_beta
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        )
        return (
            alpha
            / f ** (1 - beta)
            * (1 + term1 * t)
        )

    log_fk = math.log(f / k)
    fk_beta = (f * k) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk
    x_z = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    term1 = (
        (1 - beta) ** 2 / 24 * (alpha ** 2) / fk_beta ** 2
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2 - 3 * rho ** 2) * nu ** 2 / 24
    )
    return (
        alpha
        / (fk_beta * (1 + (1 - beta) ** 2 / 24 * log_fk ** 2 + (1 - beta) ** 4 / 1920 * log_fk ** 4))
        * (z / x_z)
        * (1 + term1 * t)
    )


def sabr_error(
    params: Tuple[float, float, float],
    f: float,
    strikes: Iterable[float],
    t: float,
    market_vols: Iterable[float],
    beta: float,
) -> float:
    """Return mean squared error between market vols and SABR vols."""
    alpha, rho, nu = params
    error = 0.0
    n = 0
    for k, mv in zip(strikes, market_vols):
        model_vol = hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu)
        diff = model_vol - mv
        error += diff * diff
        n += 1
    return error / max(1, n)


def fit_sabr_grid(
    f: float,
    strikes: List[float],
    t: float,
    market_vols: List[float],
    beta: float = 1.0,
    alpha_range: Tuple[float, float] = (0.01, 2.0),
    rho_range: Tuple[float, float] = (-0.99, 0.99),
    nu_range: Tuple[float, float] = (0.01, 2.0),
    steps: int = 10,
) -> Tuple[float, float, float]:
    """Very naive grid search SABR calibration.

    Parameters are scanned within given ranges using equally spaced grids.
    The search returns parameters that minimize mean squared error.
    """
    best = None
    best_err = float("inf")
    for i in range(steps):
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * i / max(1, steps - 1)
        for j in range(steps):
            rho = rho_range[0] + (rho_range[1] - rho_range[0]) * j / max(1, steps - 1)
            for k_idx in range(steps):
                nu = nu_range[0] + (nu_range[1] - nu_range[0]) * k_idx / max(1, steps - 1)
                err = sabr_error((alpha, rho, nu), f, strikes, t, market_vols, beta)
                if err < best_err:
                    best_err = err
                    best = (alpha, rho, nu)
    assert best is not None
    return best


# ---------------------- Surface construction ----------------------

def construct_implied_vol_surface(
    data: List[Tuple[str, str, float, float, float]],
    forward_lookup: Dict[Tuple[str, str], float],
    beta: float = 1.0,
) -> Dict[str, Dict[str, Dict[Tuple[float, float], float]]]:
    """Construct SABR-smoothed implied volatility surfaces.

    Parameters
    ----------
    data : list of tuples
        Each tuple is (ticker, day, strike, maturity, market_vol).
    forward_lookup : dict
        Mapping from (ticker, day) to forward price used for SABR calibration.
    beta : float, optional
        SABR beta parameter. Defaults to 1.0 (log-normal).

    Returns
    -------
    dict
        Nested dictionary keyed by ticker -> day -> (K, T) with SABR implied vol.
    """
    # Organize data
    surface: Dict[str, Dict[str, Dict[Tuple[float, float], float]]] = {}
    by_key: Dict[Tuple[str, str, float], List[Tuple[float, float]]] = {}
    for ticker, day, strike, maturity, iv in data:
        by_key.setdefault((ticker, day, maturity), []).append((strike, iv))
    for (ticker, day, maturity), pairs in by_key.items():
        if (ticker, day) not in forward_lookup:
            continue
        fwd = forward_lookup[(ticker, day)]
        strikes = [p[0] for p in pairs]
        vols = [p[1] for p in pairs]
        # fit SABR for this maturity slice
        alpha, rho, nu = fit_sabr_grid(fwd, strikes, maturity, vols, beta)
        for k, _ in pairs:
            vol = hagan_lognormal_vol(fwd, k, maturity, alpha, beta, rho, nu)
            surface.setdefault(ticker, {}).setdefault(day, {})[(k, maturity)] = vol
    return surface
