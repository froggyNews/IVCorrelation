import numpy as np
import pandas as pd


def impute_col_median(X: np.ndarray) -> np.ndarray:
    """Replace NaNs/inf in each column with that column's median."""
    X = np.asarray(X, float).copy()
    med = np.nanmedian(X, axis=0, keepdims=True)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.broadcast_to(med, X.shape)[mask]
    return X


def zscore_cols(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z‑score each column of ``X``; returns normalized array and (mean, std)."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
    return (X - mu) / sd, mu, sd


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
