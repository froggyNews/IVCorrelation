from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

__all__ = ["impute_col_median", "zscore_cols", "safe_var", "beta", "safe_center_scale"]

def impute_col_median(X: np.ndarray) -> np.ndarray:
    """Replace NaNs/inf in each column with that column's median.

    If the entire column is NaN/inf it is filled with 0.0 so downstream
    weighting falls back gracefully instead of emitting RuntimeWarnings.
    """
    X = np.asarray(X, float)
    if X.size == 0:
        return X.reshape(0, 0) if X.ndim == 1 else X
    med = np.nanmedian(X, axis=0, keepdims=True)
    # Columns that are all NaN -> set median to 0
    all_nan = np.isnan(med)
    if all_nan.any():
        med[:, all_nan[0]] = 0.0
    mask = ~np.isfinite(X)
    if mask.any():
        X = X.copy()
        X[mask] = np.broadcast_to(med, X.shape)[mask]
    return X

def zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z‑score each column of ``X``; returns normalized array and (mean, std).

    Handles empty or all-NaN columns robustly (std=1).
    """
    X = np.asarray(X, float)
    if X.size == 0:
        return X, np.array([]), np.array([])
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z, mu, sd

def safe_center_scale(X: np.ndarray) -> Tuple[np.ndarray, dict | None]:
    """Robust center/scale with guards against empty/all-NaN input.

    Returns (transformed, stats-or-None).
    """
    X = np.asarray(X, float)
    if X.size == 0:
        return X, None
    if np.all(~np.isfinite(X)):
        return np.zeros_like(X), None
    med = np.nanmedian(X, axis=0, keepdims=True)
    med[~np.isfinite(med)] = 0.0
    Xc = X - med
    mu = np.nanmean(Xc, axis=0, keepdims=True)
    mu[~np.isfinite(mu)] = 0.0
    Xc = Xc - mu
    var = np.nanvar(Xc, axis=0, ddof=0, keepdims=True)
    var[~np.isfinite(var) | (var <= 0)] = 1.0
    std = np.sqrt(var)
    out = Xc / std
    out[~np.isfinite(out)] = 0.0
    return out, {"median": med, "mean": mu, "std": std}

def safe_var(x: pd.Series) -> float:
    """Finite variance or ``nan`` if unavailable/degenerate."""
    v = x.var()
    return float(v) if (v is not None and np.isfinite(v) and v > 0) else float("nan")

def beta(df: pd.DataFrame, x: str, b: str) -> float:
    """Simple ``beta`` of column ``x`` versus benchmark column ``b``."""
    a = df[[x, b]].dropna()
    if len(a) < 5:
        return float("nan")
    vb = safe_var(a[b])
    return float(a[x].cov(a[b]) / vb) if np.isfinite(vb) else float("nan")
