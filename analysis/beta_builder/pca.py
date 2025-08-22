from __future__ import annotations
import numpy as np
from .utils import impute_col_median, zscore_cols
from typing import Tuple, Optional, Iterable, List, Dict
import pandas as pd
from feature_matrices import atm_feature_matrix, surface_feature_matrix

__all__ = ["pca_market_weights", "pca_regress_weights"]

def _first_pc_weights_from_rows(Z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """PCA on ``Z Z^T`` returning non‑negative first PC weights."""
    n_feat = max(Z.shape[1] - 1, 1)
    R = (Z @ Z.T) / n_feat
    R[np.arange(R.shape[0]), np.arange(R.shape[0])] += float(ridge)
    vals, vecs = np.linalg.eigh(R)
    v1 = vecs[:, -1]
    if v1.sum() < 0:
        v1 = -v1
    w = np.clip(v1, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

def pca_market_weights(X_peers: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Market‑mode weights via first principal component across peer rows."""
    Z, _, _ = zscore_cols(impute_col_median(X_peers))
    return _first_pc_weights_from_rows(Z, ridge=ridge)

def pca_regress_weights(
    X_peers: np.ndarray,
    y_target: np.ndarray,
    k: int | None = None,
    *,
    nonneg: bool = True,
) -> np.ndarray:
    """Solve ``min_w || X^T w - y ||`` using truncated SVD."""
    Z, _, _ = zscore_cols(impute_col_median(X_peers))
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    if k is None or k <= 0 or k > len(s):
        k = len(s)
    Uk, sk, Vk = U[:, :k], s[:k], Vt[:k, :].T
    w = Uk @ ((y_target @ Vk) / np.where(sk > 1e-12, sk, 1.0))
    if nonneg:
        w = np.clip(w, 0.0, None)
    ssum = float(w.sum())
    return w / ssum if ssum > 0 else np.full_like(w, 1.0 / max(len(w), 1))

# ------------------------
# PCA convenience (kept)
# ------------------------
def pca_weights(
    get_smile_slice,
    mode: str,
    target: str,
    peers: List[str],
    asof: str,
    pillars_days: Iterable[int] = (7, 30, 60, 90),
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    k: Optional[int] = None,
) -> pd.Series:
    """
    PCA-based peer weighting:

      pca_atm_market      → PC1 on ATM pillars across peers
      pca_atm_regress     → PCA‑regress target ATM on peers
      pca_surface_market  → PC1 on surface grid across peers
      pca_surface_regress → PCA‑regress target surface grid on peers
    """
    target = (target or "").upper()
    peers = [p.upper() for p in peers]
    mode = (mode or "").lower().strip()

    if mode.startswith("pca_atm"):
        atm_df, X, _ = atm_feature_matrix(get_smile_slice, [target] + peers, asof, pillars_days)
        labels = list(atm_df.index)
        if len(labels) < 2:
            return pd.Series(dtype=float)
        if "market" in mode:
            w = pca_market_weights(X[1:, :])
        else:
            y = impute_col_median(atm_df.loc[[target]].to_numpy(float)).ravel()
            Xp = atm_df.loc[peers].to_numpy(float)
            if Xp.size == 0:
                return pd.Series(dtype=float)
            w = pca_regress_weights(Xp, y, k=k, nonneg=True)
        s = pd.Series(w, index=labels[1:]).clip(lower=0.0)
        tot = float(s.sum())
        return (s / tot if tot > 0 else s).reindex(peers).fillna(0.0)

    if mode.startswith("pca_surface"):
        grids, X, _ = surface_feature_matrix([target] + peers, asof, tenors=tenors, mny_bins=mny_bins)
        labels = list(grids.keys())
        if not labels or labels[0] != target or len(labels) < 2:
            return pd.Series(dtype=float)
        if "market" in mode:
            w = pca_market_weights(X[1:, :])
        else:
            w = pca_regress_weights(X[1:, :], X[0, :], k=k, nonneg=True)
        s = pd.Series(w, index=labels[1:]).clip(lower=0.0)
        tot = float(s.sum())
        return (s / tot if tot > 0 else s).reindex(peers).fillna(0.0)

    raise ValueError(f"unknown mode: {mode}")
