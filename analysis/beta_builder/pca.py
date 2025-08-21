import numpy as np
from .utils import impute_col_median, zscore_cols


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
