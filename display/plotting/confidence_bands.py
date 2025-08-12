# display/plotting/confidence_bands.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional
import numpy as np

# We reuse your smile fitters
from volModel.sviFit import fit_svi_slice, svi_smile_iv
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv

@dataclass
class Bands:
    x: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    level: float

# -----------------------------
# Generic nonparametric bootstrap bands
# -----------------------------
def bootstrap_bands(
    x: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], Dict],
    pred_fn: Callable[[Dict, np.ndarray], np.ndarray],
    grid: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
    random_state: Optional[int] = 42,
) -> Bands:
    """
    x, y: raw points (e.g., K or moneyness, iv)
    fit_fn: returns params dict from (x, y)
    pred_fn: (params, grid) -> yhat on grid
    grid: where to compute bands
    level: 0.68 ~ 1 sigma-ish; 0.95 for wide bands
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)

    # fit once on all data for the center line
    p0 = fit_fn(x, y)
    center = pred_fn(p0, grid)

    # bootstrap
    draws = np.empty((n_boot, grid.size), dtype=float)
    n = len(x)
    idx = np.arange(n)
    for b in range(n_boot):
        resample = rng.choice(idx, size=n, replace=True)
        xb = x[resample]
        yb = y[resample]
        try:
            pb = fit_fn(xb, yb)
            draws[b] = pred_fn(pb, grid)
        except Exception:
            draws[b] = np.nan

    # compute quantiles
    alpha = 1.0 - level
    lo = np.nanquantile(draws, alpha / 2.0, axis=0)
    hi = np.nanquantile(draws, 1.0 - alpha / 2.0, axis=0)

    return Bands(x=grid, mean=center, lo=lo, hi=hi, level=level)

# -----------------------------
# SVI helper bands (expects S,K,T)
# -----------------------------
def svi_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        out = fit_svi_slice(S, K_, T, iv_)
        return out

    def _pred(p, Kq):
        return svi_smile_iv(S, np.asarray(Kq, float), T, p)

    return bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)

# -----------------------------
# SABR helper bands (expects S,K,T)
# -----------------------------
def sabr_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    beta: float = 0.5,
    level: float = 0.68,
    n_boot: int = 200,
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        out = fit_sabr_slice(S, K_, T, iv_, beta=beta)
        return out

    def _pred(p, Kq):
        return sabr_smile_iv(S, np.asarray(Kq, float), T, p)

    return bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)
