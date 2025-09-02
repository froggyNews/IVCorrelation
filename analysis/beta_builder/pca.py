from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Iterable, List, Dict
from .utils import impute_col_median, zscore_cols

__all__ = [
    "pca_market_weights",
    "pca_regress_weights",
    "pca_weights",
    "pca_surface_project",
    "pca_surface_project_per_expiry",
]

# -------------------------------------------------------------------
# Internal: PC1 "market mode" weights over peers (rows)
# -------------------------------------------------------------------
def _first_pc_weights_from_rows(Z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    PCA on (Z Z^T), return non-negative first-PC weights over rows (peers).
    Z: (n_peers, n_features), already column-standardized.
    """
    n_feat = max(Z.shape[1] - 1, 1)
    R = (Z @ Z.T) / n_feat
    # tiny ridge for numerical stability
    R[np.arange(R.shape[0]), np.arange(R.shape[0])] += float(ridge)
    vals, vecs = np.linalg.eigh(R)
    v1 = vecs[:, -1]
    # sign convention: make sum(v1) >= 0
    if v1.sum() < 0:
        v1 = -v1
    w = np.clip(v1, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


# -------------------------------------------------------------------
# ATM / Surface "market lens": convex weights via PC1 on peers
# -------------------------------------------------------------------
def pca_market_weights(X_peers: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Market-mode weights via first principal component across peer rows.
    Implements the ATM/Surface 'market lens':
      - impute -> center & scale columns
      - PC1 of (Z Z^T) on rows
      - clip to >= 0 and renormalize to the simplex
    """
    Z, _, _ = zscore_cols(impute_col_median(X_peers))
    return _first_pc_weights_from_rows(Z, ridge=ridge)


# -------------------------------------------------------------------
# Factorization (Target ≈ weighted sum of peers) with SVD pseudo-inverse
# Matches the ATM factorization rows in your table (nonneg + simplex).
# -------------------------------------------------------------------
def pca_regress_weights(
    X_peers: np.ndarray,
    y_target: np.ndarray,
    k: int | None = None,
    *,
    nonneg: bool = True,
    energy: float = 0.95,
) -> np.ndarray:
    """
    Solve min_w || (Z^T) w - y ||_2 with Z = zscore(X_peers) column-wise.

    Implements the 'Target ≈ weighted sum of peers' factorization:
        ŷ = (Z^T) w,  with  w = argmin ||(Z^T)w - y|| using truncated SVD.
    Then (optionally) enforce w >= 0 and normalize sum(w)=1.

    Shapes:
        X_peers:   (n_peers, n_features)
        y_target:  (n_features,)
    """
    # 1) Impute + z-score peers; keep μ, σ to apply SAME transform to y
    Z, mu, sig = zscore_cols(impute_col_median(X_peers))
    # standardize y with peer μ/σ (aligns with the table's "center peers" step)
    y = (y_target - mu) / np.where(sig > 1e-12, sig, 1.0)

    # 2) SVD(Z) => Z = U S V^T
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)

    # Select rank k by energy if not provided
    if k is None or k <= 0 or k > len(s):
        if len(s):
            c = np.cumsum(s**2)
            total = c[-1] if np.isfinite(c[-1]) and c[-1] > 0 else 0.0
            k = int(np.searchsorted(c / total, float(energy), side="left") + 1) if total > 0 else len(s)
        else:
            k = 1
    Uk = U[:, :k]           # (n_peers, k)
    sk = s[:k]              # (k,)
    Vk = Vt[:k, :].T        # (n_features, k)

    # 3) Least-squares weights in the standardized feature space
    sk_safe = np.where(sk > 1e-8, sk, 1e-8)
    w = Uk @ ((y @ Vk) / sk_safe)

    if nonneg:
        w = np.clip(w, 0.0, None)

    # 4) Simplex normalize (sum to 1); if all zero, fall back to uniform
    ssum = float(w.sum())
    return w / ssum if ssum > 0 else np.full_like(w, 1.0 / max(len(w), 1))


# -------------------------------------------------------------------
# NEW: Pure PCA projection/reconstruction of the target in feature space
# Matches the Surface (per-expiry smiles) PCA in your LaTeX exactly:
#   μ = mean(peer smiles), S_c = [s_i - μ]
#   SVD(S_c) -> U Σ V^T, take U_K (feature PCs)
#   ŝ_tgt = μ + U_K U_K^T (s_tgt - μ)
# Note: Here rows are features, columns are peers — i.e. stack peer smiles
# as columns in a matrix of shape (n_features, n_peers).
# -------------------------------------------------------------------
def pca_surface_project(
    peers_feature_matrix: np.ndarray,  # shape (n_features, n_peers)
    target_feature_vector: np.ndarray, # shape (n_features,)
    k: Optional[int] = None,
    *,
    energy: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Project target feature vector onto the top-K peer PCs in FEATURE space.

    Returns:
        s_hat    : reconstructed target in original feature space (n_features,)
        details  : dict with {"mu","U_k","S_k","V_k","k"} for inspection
    """
    # 1) Impute NaNs in feature dimension (per feature)
    X = impute_col_median(peers_feature_matrix.T).T  # keep columns=peers, rows=features
    # 2) Mean center features using peer mean μ (no scaling here per LaTeX)
    mu = X.mean(axis=1)  # (n_features,)
    S_c = X - mu[:, None]

    # 3) SVD on centered feature matrix S_c (features × peers) => S_c = U Σ V^T
    U, s, Vt = np.linalg.svd(S_c, full_matrices=False)  # U: (n_features, r)

    # Determine K
    if k is None or k <= 0 or k > len(s):
        if len(s):
            c = np.cumsum(s**2)
            total = c[-1] if np.isfinite(c[-1]) and c[-1] > 0 else 0.0
            k = int(np.searchsorted(c / total, float(energy), side="left") + 1) if total > 0 else len(s)
        else:
            k = 1

    U_k = U[:, :k]   # feature PCs
    # 4) Project target onto U_k and reconstruct
    y = target_feature_vector
    # If target has NaNs, impute with peer μ before projection
    if np.isnan(y).any():
        y = np.where(np.isnan(y), mu, y)
    y_c = y - mu
    s_hat = mu + U_k @ (U_k.T @ y_c)

    details = {
        "mu": mu,
        "U_k": U_k,
        "S_k": s[:k],
        "V_k": Vt[:k, :].T,
        "k": k,
    }
    return s_hat, details


# -------------------------------------------------------------------
# OPTIONAL helper: do the same per-expiry (list of smiles per T_m),
# then concatenate in canonical order. Each entry peers_feature_matrix[m]
# is (n_strikes_m, n_peers), target_feature_vector[m] is (n_strikes_m,).
# -------------------------------------------------------------------
def pca_surface_project_per_expiry(
    peers_feature_matrix_list: List[np.ndarray],
    target_feature_vector_list: List[np.ndarray],
    k: Optional[int] = None,
    *,
    energy: float = 0.95,
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """
    Apply pca_surface_project to each expiry independently and concatenate.

    Returns:
        s_hat_full : concatenated reconstruction across expiries
        details_ls : list of details dicts (one per expiry)
    """
    recons: List[np.ndarray] = []
    details_ls: List[Dict[str, np.ndarray]] = []
    for X_m, y_m in zip(peers_feature_matrix_list, target_feature_vector_list):
        s_hat_m, det_m = pca_surface_project(X_m, y_m, k=k, energy=energy)
        recons.append(s_hat_m)
        details_ls.append(det_m)
    return np.concatenate(recons, axis=0), details_ls


# -------------------------------------------------------------------
# Convenience wrapper you already had for producing weights for composites.
# NOTE:
# - Modes ending in "market" (PC1 over peers) and "regress" (factorization)
#   return **peer weights**.
# - If you want the strict LaTeX "projection" result (a reconstructed smile),
#   call pca_surface_project(...) directly instead of this wrapper.
# -------------------------------------------------------------------
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
    PCA-based peer weighting (weights over peers), aligned to your table:

      pca_atm_market
        → PC1 on ATM pillars across peers (market-mode factor)
      pca_atm_regress
        → PCA-regress target ATM on peers (Target ≈ sum_i w_i ATM_i)
      pca_surface_market
        → PC1 on flattened surface grid across peers
      pca_surface_regress
        → PCA-regress target flattened surface on peers

    For strict PCA projection/reconstruction per LaTeX, use pca_surface_project(...)
    or pca_surface_project_per_expiry(...).
    """
    from .feature_matrices import atm_feature_matrix, surface_feature_matrix

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
            # y is target ATM vector in same feature space as peers; standardized using peer μ/σ internally
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
            # Factorization weights (not the LaTeX projection); for projection use pca_surface_project directly
            w = pca_regress_weights(X[1:, :], X[0, :], k=k, nonneg=True)

        s = pd.Series(w, index=labels[1:]).clip(lower=0.0)
        tot = float(s.sum())
        return (s / tot if tot > 0 else s).reindex(peers).fillna(0.0)

    raise ValueError(f"unknown mode: {mode}")


# -----------------------------------------------------------------------------
# Unified helper: compute PCA weights from an existing feature matrix
# -----------------------------------------------------------------------------
def pca_weights_from_feature_df(
    feature_df: pd.DataFrame,
    target: str,
    peers: List[str],
    *,
    k: Optional[int] = None,
    nonneg: bool = True,
) -> pd.Series:
    """Compute PCA regression weights for target vs peers using feature_df.

    Falls back to market-mode PC1 if the regression produces a degenerate
    nonnegative solution.
    """
    if feature_df is None or feature_df.empty:
        return pd.Series(dtype=float)
    tgt = (target or "").upper()
    peers = [p.upper() for p in peers]
    present = [p for p in peers if p in feature_df.index]
    if tgt not in feature_df.index or not present:
        return pd.Series(0.0, index=peers, dtype=float)

    y = impute_col_median(feature_df.loc[[tgt]].to_numpy(float)).ravel()
    Xp = feature_df.loc[present].to_numpy(float)

    try:
        w = pca_regress_weights(Xp, y, k=k, nonneg=nonneg)
    except Exception:
        try:
            w = pca_market_weights(Xp)
        except Exception:
            return pd.Series(0.0, index=peers, dtype=float)

    s = pd.Series(w, index=present).clip(lower=0.0)
    tot = float(s.sum())
    if not np.isfinite(tot) or tot <= 0:
        try:
            w_m = pca_market_weights(Xp)
            s = pd.Series(w_m, index=present).clip(lower=0.0)
            tot = float(s.sum())
        except Exception:
            s[:] = 0.0
            tot = 0.0
    s = s / tot if tot > 0 else s
    return s.reindex(peers).fillna(0.0).astype(float)
def _ul_pca_factor_weights(df_ul: pd.DataFrame, target: str, peers: list[str],
                           k: Optional[int], energy: float) -> pd.Series:
    """
    df_ul: rows=tickers, cols=time (log returns), as built by underlying_returns_matrix()
    Implements:
      R_peer (T x N)  = peer returns with time along rows
      R_peer = U S V^T, factors F = U S (T x K), loadings L = V[:, :K] (N x K)
      beta from OLS: r_tgt ~ F
      peer scores c = L @ beta  -> clip >=0 -> simplex normalize -> weights
    """
    # Ensure target + peers exist
    ticks = [t for t in [target] + peers if t in df_ul.index]
    if len(ticks) < 2:
        return pd.Series(dtype=float)

    # Build time x peers matrix
    peer_list = [p for p in peers if p in df_ul.index]
    R_peer = df_ul.loc[peer_list].to_numpy(float).T  # shape (T, N)
    if R_peer.size == 0 or R_peer.shape[1] == 0:
        return pd.Series(dtype=float)

    # Center over time (demean columns = peers); no scaling per LaTeX
    R_peer = R_peer - np.nanmean(R_peer, axis=0, keepdims=True)
    R_peer = np.nan_to_num(R_peer, nan=0.0)

    # SVD on (T x N): R = U S V^T
    U, s, Vt = np.linalg.svd(R_peer, full_matrices=False)

    # choose K
    if k is None or k <= 0 or k > len(s):
        if len(s):
            c = np.cumsum(s**2)
            tot = c[-1] if np.isfinite(c[-1]) and c[-1] > 0 else 0.0
            k = int(np.searchsorted(c / tot, float(energy), side="left") + 1) if tot > 0 else len(s)
        else:
            k = 1

    U_k = U[:, :k]                  # (T, K)
    S_k = s[:k]                     # (K,)
    V_k = Vt[:k, :].T               # (N, K)  loadings

    # Factors F = U_k S_k (T x K)
    F = U_k * S_k  # broadcast multiply

    # Target returns (T,)
    r_tgt = df_ul.loc[target].to_numpy(float)
    r_tgt = r_tgt - np.nanmean(r_tgt)
    r_tgt = np.nan_to_num(r_tgt, nan=0.0)

    # OLS beta: minimize ||F beta - r_tgt||_2
    # Solve with stable least squares
    beta, *_ = np.linalg.lstsq(F, r_tgt, rcond=None)  # (K,)

    # Map factor exposures back to peers via loadings
    c = V_k @ beta   # (N,)
    c = np.clip(c, 0.0, None)
    ssum = float(c.sum())
    w = c / ssum if ssum > 0 else np.full_like(c, 1.0 / max(len(c), 1))

    return pd.Series(w, index=peer_list)
