import numpy as np


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
