# analysis/pca_builder.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional
import numpy as np
import pandas as pd

from analysis.pillars import build_atm_matrix
try:
    # your current location for surface grid builder
    from analysis.syntheticETFBuilder import build_surface_grids
except Exception:
    build_surface_grids = None  # optional

def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns of X to zero mean, unit variance."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd <= 0.0, 1.0, sd)
    return (X - mu) / sd, mu, sd

def _complete_with_col_med(X: np.ndarray) -> np.ndarray:
    """Fill NaN values with column medians."""
    Xc = X.copy()
    col_med = np.nanmedian(Xc, axis=0, keepdims=True)
    inds = np.where(~np.isfinite(Xc))
    Xc[inds] = np.take_along_axis(col_med, inds[1][None, :], axis=1).ravel()
    return Xc

def _svd(X: np.ndarray):
    """Compute SVD decomposition."""
    # X: samples x features
    return np.linalg.svd(X, full_matrices=False)  # U, s, Vt

# ---------- ATM (pillars) feature matrix ----------
def atm_feature_matrix(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = 0.05,
    tol_days: float = 7.0,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Build feature matrix from ATM volatilities across pillars.
    
    Returns:
        atm_df: DataFrame with tickers as rows, pillars as columns
        X: Standardized feature matrix (tickers x pillars)
        pillars: List of pillar column names
    """
    # rows=tickers, cols=pillars; values=ATM IVs
    atm_df, _ = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
    )
    atm_df = atm_df.reindex(index=[t.upper() for t in tickers])
    X = atm_df.to_numpy(float)
    X = _complete_with_col_med(X)
    if standardize:
        X, _, _ = _zscore_cols(X)
    return atm_df, X, list(atm_df.columns)

# ---------- Surface feature matrix ----------
def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    tenors=None,
    mny_bins=None,
    standardize: bool = True,
) -> Tuple[dict, np.ndarray, list[str], list[str]]:
    """
    Build feature matrix from full volatility surfaces.
    
    Returns:
        grids: Surface grids dict
        X: Standardized feature matrix (tickers x flattened_surface)
        ok_tickers: List of tickers with valid surface data
        feat_names: List of feature names (T{tenor}_{mny_bin})
    """
    if build_surface_grids is None:
        raise RuntimeError("build_surface_grids unavailable")
    grids = build_surface_grids(
        tickers=[t.upper() for t in tickers],
        tenors=tenors, mny_bins=mny_bins, use_atm_only=False
    )
    # flatten per-ticker grid at date -> feature vector
    feats = []
    ok_tickers = []
    feat_names = None
    for t in tickers:
        tU = t.upper()
        if tU not in grids or asof not in grids[tU]:
            continue
        df = grids[tU][asof]  # index=mny labels, cols=tenor-days
        arr = df.to_numpy(float).T.reshape(-1)   # (tenors*mny,)
        if feat_names is None:
            feat_names = [f"T{c}_{r}" for c in df.columns for r in df.index]
        feats.append(arr); ok_tickers.append(tU)
    if not feats:
        raise RuntimeError("No surface data for selected date")
    X = np.vstack(feats)
    X = _complete_with_col_med(X)
    if standardize:
        X, _, _ = _zscore_cols(X)
    return grids, X, ok_tickers, feat_names

# ---------- PCA weights (two styles) ----------
def pca_market_weights(X: np.ndarray, nonneg: bool = True) -> np.ndarray:
    """
    Compute market weights based on first principal component.
    Weights align with PC1 (market mode).
    """
    # weights across samples (tickers) aligned with PC1 (market mode)
    U, s, Vt = _svd(X)   # X = U Σ V^T
    w = U[:, 0].copy()
    # pick sign so sum positive; enforce nonneg if desired
    if w.sum() < 0: 
        w *= -1.0
    if nonneg:
        w = np.abs(w)
    w_sum = w.sum()
    return w / w_sum if w_sum > 0 else np.full_like(w, 1.0/len(w))

def pca_regress_weights(
    X_peers: np.ndarray,
    y_target: np.ndarray,
    k: Optional[int] = None,
    nonneg: bool = True,
) -> np.ndarray:
    """
    Solve regression weights using PCA: min_w || X_peers^T w - y ||
    via SVD: X = U Σ V^T,  w* = U_k Σ_k^{-1} V_k^T y
    """
    U, s, Vt = _svd(X_peers)
    if k is None or k <= 0 or k > len(s):
        k = len(s)
    Uk = U[:, :k]
    sk = s[:k]
    Vk = Vt[:k, :].T  # features x k
    # y is features vector
    w = Uk @ ( (y_target @ Vk) / sk )   # equivalent to U Σ^{-1} V^T y
    if nonneg:
        w = np.clip(w, 0.0, None)
    w_sum = w.sum()
    return w / w_sum if w_sum > 0 else np.full_like(w, 1.0/len(w))

# ---------- Public entry ----------
def pca_weights_from_atm_matrix(
    atm_df: pd.DataFrame,
    target: str,
    peers: list[str],
    clip_negative: bool = True,
    min_rows_per_col: int = 3,   # columns (pillars) need >=3 finite obs
    min_cols_per_row: int = 2,   # rows (tickers) need >=2 finite pillars
    ridge: float = 1e-6,
    verbose: bool = False,       # Enable sparsity reporting
) -> pd.Series:
    """
    PCA on tickers x pillars ATM matrix (one date).
    Returns non-negative, sum-1 weights for peers (target row dropped).
    """
    target = target.upper()
    peers = [p.upper() for p in peers]

    # 1) select rows (target + peers present in atm_df)
    rows = [r for r in [target] + peers if r in atm_df.index]
    if len(rows) < 2:
        return pd.Series(index=peers, dtype=float)  # nothing to do

    X = atm_df.loc[rows].to_numpy(dtype=float)

    # Sanity check: inspect sparsity
    finite_per_col = np.isfinite(X).sum(axis=0)
    finite_per_row = np.isfinite(X).sum(axis=1)
    
    if verbose:
        print(f"PCA ATM Matrix Sparsity Check:")
        print(f"  Shape: {X.shape}")
        print(f"  Finite per pillar (cols): {finite_per_col}")
        print(f"  Finite per ticker (rows): {finite_per_row}")
    
    # 2) drop columns (pillars) that are too sparse
    keep_cols = np.where(finite_per_col >= min_rows_per_col)[0]
    if keep_cols.size < 2:
        if verbose:
            print(f"  Warning: Only {keep_cols.size} pillars have >= {min_rows_per_col} finite values")
        return pd.Series(index=peers, dtype=float)
    X = X[:, keep_cols]
    
    if verbose:
        print(f"  Kept {len(keep_cols)} pillars after sparsity filtering")

    # 3) drop rows (tickers) that are too sparse
    keep_rows = np.where(np.isfinite(X).sum(axis=1) >= min_cols_per_row)[0]
    if keep_rows.size < 2:
        if verbose:
            print(f"  Warning: Only {keep_rows.size} tickers have >= {min_cols_per_row} finite pillars")
        return pd.Series(index=peers, dtype=float)
    rows_kept = [rows[i] for i in keep_rows]
    X = X[keep_rows, :]
    
    if verbose:
        print(f"  Kept {len(keep_rows)} tickers after sparsity filtering")
        print(f"  Final matrix shape: {X.shape}")

    # 4) impute per-column median, then standardize columns
    col_med = np.nanmedian(X, axis=0, keepdims=True)
    mask = ~np.isfinite(X)
    if mask.any():
        X = X.copy(); X[mask] = np.broadcast_to(col_med, X.shape)[mask]

    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where((~np.isfinite(sd)) | (sd < 1e-12), 1.0, sd)
    Z = (X - mu) / sd

    # 5) ticker-ticker correlation-like matrix, with ridge
    ncols = max(Z.shape[1] - 1, 1)
    R = (Z @ Z.T) / ncols
    R = R + ridge * np.eye(R.shape[0])

    # 6) first PC loadings
    vals, vecs = np.linalg.eigh(R)  # symmetric PD
    v1 = vecs[:, -1]

    # make target loading positive if target present
    if target in rows_kept:
        sign = np.sign(v1[rows_kept.index(target)]) or 1.0
    else:
        sign = 1.0
    w_raw = sign * v1

    # 7) drop target, keep peers, non-neg + normalize
    ser = pd.Series(w_raw, index=rows_kept).drop(index=[target], errors="ignore")
    ser = ser.reindex(peers).fillna(0.0)
    if clip_negative:
        ser = ser.clip(lower=0.0)
    s = ser.sum()
    return ser / s if s > 0 else ser


def pca_weights(
    get_smile_slice,
    mode: str,               # 'pca_atm_market' | 'pca_atm_regress' | 'pca_surface_market' | 'pca_surface_regress'
    target: str,
    peers: list[str],
    asof: str,
    pillars_days: Iterable[int] = (7,30,60,90,180,365),
    tenors=None,
    mny_bins=None,
    k: Optional[int] = None,
    nonneg: bool = True,
    standardize: bool = True,
) -> pd.Series:
    """
    Compute PCA-based weights for synthetic ETF construction.
    
    Parameters:
    -----------
    get_smile_slice : callable
        Function to retrieve options data
    mode : str
        PCA mode: 'pca_atm_market', 'pca_atm_regress', 'pca_surface_market', 'pca_surface_regress'
    target : str
        Target ticker symbol
    peers : list[str]
        List of peer ticker symbols
    asof : str
        Date for analysis
    pillars_days : Iterable[int]
        Days for ATM pillar analysis
    tenors : optional
        Tenor days for surface analysis
    mny_bins : optional
        Moneyness bins for surface analysis
    k : Optional[int]
        Number of principal components for regression (None = all)
    nonneg : bool
        Whether to enforce non-negative weights
    standardize : bool
        Whether to standardize features
        
    Returns:
    --------
    pd.Series
        Normalized weights for peer tickers
    """
    peers = [p.upper() for p in peers]
    target = (target or "").upper()

    if mode.startswith("pca_atm"):
        # Use robust ATM matrix approach
        atm_df, _ = build_atm_matrix(
            get_smile_slice=get_smile_slice,
            tickers=[target] + peers,
            asof=asof,
            pillars_days=pillars_days,
            atm_band=0.05,
            tol_days=7.0,
        )
        # Use the robust PCA function that handles sparse/NaN data
        return pca_weights_from_atm_matrix(atm_df, target, peers, clip_negative=nonneg)

    elif mode.startswith("pca_surface"):
        if build_surface_grids is None:
            raise RuntimeError("Surface builder unavailable")
        tickers = [target] + peers
        grids, X_all, ok_tickers, feat_names = surface_feature_matrix(
            tickers, asof, tenors=tenors, mny_bins=mny_bins, standardize=standardize
        )
        # reorder to [target]+peers subset present
        keep = [t for t in [target] + peers if t in ok_tickers]
        idxs = [ok_tickers.index(t) for t in keep]
        X_all = X_all[idxs, :]
        # relabel peers present
        target_present = keep[0] == target
        if not target_present:
            raise RuntimeError("Target surface missing for date")

        y = X_all[0, :]
        Xp = X_all[1:, :]
        peer_labels = keep[1:]
        if "market" in mode:
            w = pca_market_weights(Xp, nonneg=nonneg)
        else:
            w = pca_regress_weights(Xp, y, k=k, nonneg=nonneg)
        return pd.Series(w, index=peer_labels, dtype=float)

    else:
        raise ValueError(f"Unknown PCA mode: {mode}")

if __name__ == "__main__":
    # Test the PCA functionality
    from analysis.analysis_pipeline import get_smile_slice, available_dates, available_tickers
    
    print("Testing PCA weights functionality...")
    
    # Get some test data
    tickers = available_tickers()[:4]  # First 4 tickers
    dates = available_dates()
    if not tickers or not dates:
        print("No data available for testing")
        exit()
    
    target = tickers[0]
    peers = tickers[1:]
    asof = dates[-1]
    
    print(f"Target: {target}")
    print(f"Peers: {peers}")
    print(f"Date: {asof}")
    
    # Test ATM market weights
    try:
        weights = pca_weights(
            get_smile_slice=get_smile_slice,
            mode="pca_atm_market",
            target=target,
            peers=peers,
            asof=asof
        )
        print(f"\nPCA ATM Market weights:\n{weights}")
    except Exception as e:
        print(f"PCA ATM Market test failed: {e}")
    
    # Test ATM regression weights
    try:
        weights = pca_weights(
            get_smile_slice=get_smile_slice,
            mode="pca_atm_regress",
            target=target,
            peers=peers,
            asof=asof
        )
        print(f"\nPCA ATM Regression weights:\n{weights}")
    except Exception as e:
        print(f"PCA ATM Regression test failed: {e}")
    
    print("\nPCA testing complete!")
