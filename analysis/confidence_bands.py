# analysis/confidence_bands.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Dict as DictType
import numpy as np

# We reuse your smile fitters
from volModel.sviFit import fit_svi_slice, svi_smile_iv
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv
from volModel.polyFit import fit_tps_slice, tps_smile_iv

__all__ = [
    "Bands",
    "bootstrap_bands",
    "svi_confidence_bands",
    "sabr_confidence_bands",
    "tps_confidence_bands",
    "generate_term_structure_confidence_bands",
    "synthetic_etf_confidence_bands",
    "synthetic_etf_weight_bands",
    "synthetic_etf_pillar_bands",
]

@dataclass
class Bands:
    x: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    level: float


# -----------------------------
# Utilities (deterministic, no sampling)
# -----------------------------
def _silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule (robust variant)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0
    iqr = np.subtract(*np.nanpercentile(x, [75, 25]))
    sigma = np.nanstd(x)
    s = min(sigma, iqr / 1.34) if np.isfinite(iqr) and iqr > 0 else sigma
    if not np.isfinite(s) or s <= 0:
        s = (np.nanmax(x) - np.nanmin(x)) / 6.0 if x.size > 1 else 1.0
        if not np.isfinite(s) or s <= 0:
            s = 1.0
    h = 0.9 * s * x.size ** (-1 / 5)
    if not np.isfinite(h) or h <= 0:
        h = 1.0
    return float(h)


def _gaussian_weights(dx: np.ndarray, h: float) -> np.ndarray:
    """Gaussian kernel weights for distances dx with bandwidth h."""
    if h <= 0:
        h = 1.0
    # Clip to 3h for numerical stability/speed
    u = dx / h
    mask = np.abs(u) <= 3.0
    w = np.zeros_like(u, dtype=float)
    z = u[mask]
    w[mask] = np.exp(-0.5 * z * z)
    return w


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile (q in [0,1]) for 1D arrays. Deterministic, no sampling."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[m], w[m]
    if v.size == 0:
        return np.nan
    if not np.any(w):
        # fall back to unweighted quantile
        return float(np.nanquantile(v, q))
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    # Normalize cumulative weights to [0,1]
    cw /= cw[-1]
    # Find first index where cw >= q
    idx = np.searchsorted(cw, q, side="left")
    idx = min(max(idx, 0), v_sorted.size - 1)
    return float(v_sorted[idx])


# -----------------------------
# Deterministic residual-quantile bands (drop-in replacement)
# -----------------------------
def bootstrap_bands(
    x: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], Dict],
    pred_fn: Callable[[Dict, np.ndarray], np.ndarray],
    grid: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,               # kept for API compatibility; ignored
    random_state: Optional[int] = 42 # kept for API compatibility; ignored
) -> Bands:
    """
    Deterministic 'global residual quantile' bands (no bootstrap).

    Steps:
      1) Fit once -> center(grid).
      2) Residuals r = y - ŷ(x).
      3) Compute global quantiles q_lo, q_hi of r (uniform weights).
      4) Bands = center + q_lo / q_hi (constant offsets across grid).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)

    # 1) Fit once for center
    params = fit_fn(x, y)
    center = pred_fn(params, grid)

    # 2) Residuals at observed x
    yhat_x = pred_fn(params, x)
    resid = y - yhat_x
    resid = resid[np.isfinite(resid)]

    # Guard: no finite residuals -> flat bands at center
    if resid.size == 0:
        return Bands(x=grid, mean=center, lo=center.copy(), hi=center.copy(), level=level)

    # 3) Global quantiles (uniform weights)
    alpha = max(0.0, min(1.0, 1.0 - level))
    q_lo = float(np.nanquantile(resid, alpha / 2.0))
    q_hi = float(np.nanquantile(resid, 1.0 - alpha / 2.0))

    # 4) Constant offsets across grid
    lo = center + q_lo
    hi = center + q_hi

    return Bands(x=grid, mean=center, lo=lo, hi=hi, level=level)


# -----------------------------
# SVI / SABR / TPS helper bands (unchanged API, now deterministic)
# -----------------------------
def svi_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,  # ignored
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        return fit_svi_slice(S, K_, T, iv_)

    def _pred(p, Kq):
        return svi_smile_iv(S, np.asarray(Kq, float), T, p)

    return bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)


def sabr_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    beta: float = 0.5,
    level: float = 0.68,
    n_boot: int = 200,  # ignored
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        return fit_sabr_slice(S, K_, T, iv_, beta=beta)

    def _pred(p, Kq):
        return sabr_smile_iv(S, np.asarray(Kq, float), T, p)

    return bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)


def tps_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,  # ignored
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        return fit_tps_slice(S, K_, T, iv_)

    def _pred(p, Kq):
        return tps_smile_iv(S, np.asarray(Kq, float), T, p)

    return bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)


# -----------------------------
# Term structure (deterministic)
# -----------------------------
def _polynomial_fit_fn(x: np.ndarray, y: np.ndarray, degree: int = 2) -> dict:
    """Fit polynomial to term structure data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if xm.size < degree + 1:
        degree = max(1, xm.size - 1)
    if xm.size < 2:
        return {"coeffs": [np.nanmean(ym)], "degree": 0}

    try:
        coeffs = np.polyfit(xm, ym, degree)
        return {"coeffs": coeffs, "degree": degree}
    except Exception:
        try:
            coeffs = np.polyfit(xm, ym, 1)
            return {"coeffs": coeffs, "degree": 1}
        except Exception:
            return {"coeffs": [np.nanmean(ym)], "degree": 0}


def _polynomial_pred_fn(params: dict, x_grid: np.ndarray) -> np.ndarray:
    """Predict using polynomial fit."""
    return np.polyval(params["coeffs"], x_grid)


def generate_term_structure_confidence_bands(
    T: np.ndarray,
    atm_vol: np.ndarray,
    level: float = 0.68,
    n_boot: int = 100,  # ignored
    fit_degree: int = 2,
    grid_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic bands for ATM term structure via residual quantiles."""
    T = np.asarray(T, dtype=float)
    atm_vol = np.asarray(atm_vol, dtype=float)

    mask = np.isfinite(T) & np.isfinite(atm_vol)
    if np.sum(mask) < 3:
        return np.array([]), np.array([]), np.array([])

    T_clean = T[mask]
    vol_clean = atm_vol[mask]

    T_min, T_max = T_clean.min(), T_clean.max()
    T_grid = np.linspace(T_min, T_max, grid_points)

    try:
        bands = bootstrap_bands(
            x=T_clean,
            y=vol_clean,
            fit_fn=lambda x, y: _polynomial_fit_fn(x, y, degree=fit_degree),
            pred_fn=_polynomial_pred_fn,
            grid=T_grid,
            level=level,
            n_boot=n_boot,
        )
        return T_grid, bands.lo, bands.hi
    except Exception:
        return np.array([]), np.array([]), np.array([])


# -----------------------------
# Synthetic ETF (deterministic)
# -----------------------------
def _weighted_quantiles_across_components(
    component_values: np.ndarray, weights: np.ndarray, q_lo: float, q_hi: float
) -> tuple[float, float]:
    """Weighted quantiles across components at a single grid point."""
    return (
        _weighted_quantile(component_values, weights, q_lo),
        _weighted_quantile(component_values, weights, q_hi),
    )


def synthetic_etf_confidence_bands(
    surfaces: Dict[str, np.ndarray],
    weights: Dict[str, float],
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,             # ignored
    weight_uncertainty: bool = False, # ignored; kept for API compat
    surface_uncertainty: bool = False # ignored; kept for API compat
) -> Bands:
    """
    Deterministic bands from cross-sectional dispersion:
      - Baseline = weighted mean across component surfaces.
      - Bands = baseline + [weighted-quantile(components) - weighted-mean].
    """
    grid_K = np.asarray(grid_K, float)
    tickers = [t for t in surfaces.keys() if t in weights]

    if not tickers:
        return Bands(x=grid_K, mean=np.full_like(grid_K, np.nan), lo=np.nan, hi=np.nan, level=level)

    # Normalize weights
    w = np.array([max(0.0, float(weights[t])) for t in tickers], dtype=float)
    sw = w.sum()
    if sw <= 0:
        w[:] = 1.0
        sw = w.sum()
    w /= sw

    # Components matrix: shape (n_tickers, n_points)
    comps = np.vstack([np.asarray(surfaces[t], float) for t in tickers])
    # Baseline (weighted mean per grid point)
    baseline = (w[:, None] * comps).sum(axis=0)

    alpha = max(0.0, min(1.0, 1.0 - level))
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    # Weighted dispersion quantiles at each grid point
    lo = np.empty_like(baseline)
    hi = np.empty_like(baseline)
    for j in range(baseline.size):
        vals = comps[:, j]
        # Weighted quantiles across components
        ql, qh = _weighted_quantiles_across_components(vals, w, q_lo, q_hi)
        # Shift around baseline (weighted mean) for symmetric interpretation
        offset_lo = ql - np.dot(w, vals)
        offset_hi = qh - np.dot(w, vals)
        lo[j] = baseline[j] + offset_lo
        hi[j] = baseline[j] + offset_hi

    return Bands(x=grid_K, mean=baseline, lo=lo, hi=hi, level=level)


def synthetic_etf_weight_bands(
    correlation_matrix: np.ndarray,
    target_idx: int,
    peer_indices: list,
    level: float = 0.68,
    n_boot: int = 200,  # ignored
    eps_corr: float = 0.05,  # deterministic correlation perturbation bound
) -> Dict[int, Bands]:
    """
    Deterministic weight intervals from bounded correlation perturbations.

    Baseline weights: w_i = |ρ_i| / Σ_j |ρ_j|, where ρ_i = corr(target, peer_i).
    Bounds: each |ρ_k| ∈ [max(0, |ρ_k|-eps_corr), min(1, |ρ_k|+eps_corr)].
    For peer i:
      w_i^lo uses min numerator, max others in denominator.
      w_i^hi uses max numerator, min others in denominator.
    """
    n_peers = len(peer_indices)
    if n_peers == 0:
        return {}

    # Extract relevant correlations (target vs peers)
    all_indices = [target_idx] + peer_indices
    corr_sub = correlation_matrix[np.ix_(all_indices, all_indices)]
    target_corrs = np.abs(corr_sub[0, 1:].astype(float))  # nonnegative

    base_den = target_corrs.sum()
    if base_den <= 0:
        baseline_weights = np.ones(n_peers, dtype=float) / n_peers
    else:
        baseline_weights = target_corrs / base_den

    # Bounds for each correlation
    lo_r = np.maximum(0.0, target_corrs - eps_corr)
    hi_r = np.minimum(1.0, target_corrs + eps_corr)

    result: Dict[int, Bands] = {}
    alpha = max(0.0, min(1.0, 1.0 - level))
    # Single-point bands per peer
    for i, peer_idx in enumerate(peer_indices):
        # Lower bound: shrink i, grow others
        num_lo = lo_r[i]
        den_lo = num_lo + np.sum(hi_r[np.arange(n_peers) != i])
        w_lo = num_lo / den_lo if den_lo > 0 else 0.0

        # Upper bound: grow i, shrink others
        num_hi = hi_r[i]
        den_hi = num_hi + np.sum(lo_r[np.arange(n_peers) != i])
        w_hi = num_hi / den_hi if den_hi > 0 else 1.0 / n_peers

        # Ensure ordered
        w0 = baseline_weights[i]
        lo_i, hi_i = (min(w_lo, w_hi), max(w_lo, w_hi))

        result[peer_idx] = Bands(
            x=np.array([peer_idx], dtype=float),
            mean=np.array([w0], dtype=float),
            lo=np.array([lo_i], dtype=float),
            hi=np.array([hi_i], dtype=float),
            level=1.0 - alpha,  # echo requested level
        )
    return result


def synthetic_etf_pillar_bands(
    atm_data: Dict[str, np.ndarray],
    weights: Dict[str, float],
    pillar_days: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,  # ignored
) -> Bands:
    """
    Deterministic ATM pillar bands from cross-sectional dispersion across components:
      - Baseline: weighted mean ATM curve.
      - Bands: weighted-quantile across component ATM values at each pillar.
    """
    pillar_days = np.asarray(pillar_days, float)
    tickers = [t for t in atm_data.keys() if t in weights]

    if not tickers:
        return Bands(x=pillar_days, mean=np.full_like(pillar_days, np.nan),
                     lo=np.full_like(pillar_days, np.nan),
                     hi=np.full_like(pillar_days, np.nan),
                     level=level)

    w = np.array([max(0.0, float(weights[t])) for t in tickers], dtype=float)
    sw = w.sum()
    if sw <= 0:
        w[:] = 1.0
        sw = w.sum()
    w /= sw

    comps = np.vstack([np.asarray(atm_data[t], float) for t in tickers])
    baseline = (w[:, None] * comps).sum(axis=0)

    alpha = max(0.0, min(1.0, 1.0 - level))
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    lo = np.empty_like(baseline)
    hi = np.empty_like(baseline)
    for j in range(baseline.size):
        vals = comps[:, j]
        ql, qh = _weighted_quantiles_across_components(vals, w, q_lo, q_hi)
        offset_lo = ql - np.dot(w, vals)
        offset_hi = qh - np.dot(w, vals)
        lo[j] = baseline[j] + offset_lo
        hi[j] = baseline[j] + offset_hi

    return Bands(x=pillar_days, mean=baseline, lo=lo, hi=hi, level=level)
