# analysis/beta_builder.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, List, Union
import math
import numpy as np
import pandas as pd

# Reuse your existing, tested builders
from analysis.pillars import build_atm_matrix, load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS  # :contentReference[oaicite:2]{index=2}
try:
    from analysis.syntheticETFBuilder import build_surface_grids  # :contentReference[oaicite:3]{index=3}
except Exception:
    build_surface_grids = None  # optional

# =========================
# small numeric helpers
# =========================
def _impute_col_median(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    med = np.nanmedian(X, axis=0, keepdims=True)
    mask = ~np.isfinite(X)
    if mask.any():
        # broadcast med across rows then pick masked positions
        X[mask] = np.broadcast_to(med, X.shape)[mask]
    return X

def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
    return (X - mu) / sd, mu, sd

def _safe_var(x: pd.Series) -> float:
    v = x.var()
    return float(v) if (v is not None and np.isfinite(v) and v > 0) else float("nan")

def _beta(df: pd.DataFrame, x: str, b: str) -> float:
    a = df[[x, b]].dropna()
    if len(a) < 5:
        return float("nan")
    vb = _safe_var(a[b])
    return float(a[x].cov(a[b]) / vb) if np.isfinite(vb) else float("nan")

# =========================
# feature matrices (ATM & surface)
# =========================
def atm_feature_matrix(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = 0.05,
    tol_days: float = 7.0,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Rows=tickers, cols=pillars (days). Values=ATM IVs for one date."""
    atm_df, _ = build_atm_matrix(  # builds the per‑ticker ATM vector across pillars
        get_smile_slice=get_smile_slice,
        tickers=[t.upper() for t in tickers],
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
    )  # :contentReference[oaicite:4]{index=4}
    X = atm_df.to_numpy(dtype=float)
    X = _impute_col_median(X)
    if standardize:
        X, _, _ = _zscore_cols(X)
    return atm_df, X, list(atm_df.columns)

def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    standardize: bool = True,
) -> Tuple[Dict, np.ndarray, List[str]]:
    """Rows=tickers, cols=flattened (tenor × moneyness) grid for one date."""
    if build_surface_grids is None:
        raise RuntimeError("surface grids unavailable (syntheticETFBuilder not importable)")
    grids = build_surface_grids(tickers=[t.upper() for t in tickers],
                                tenors=tenors, mny_bins=mny_bins, use_atm_only=False)  # :contentReference[oaicite:5]{index=5}
    feats, ok = [], []
    feat_names = None
    for t in tickers:
        tU = t.upper()
        if tU not in grids or asof not in grids[tU]:
            continue
        df = grids[tU][asof]  # index=mny labels, columns=tenor (days)
        arr = df.to_numpy(float).T.reshape(-1)
        if feat_names is None:
            feat_names = [f"T{c}_{r}" for c in df.columns for r in df.index]
        feats.append(arr); ok.append(tU)
    if not feats:
        raise RuntimeError("no surface data for requested date")
    X = np.vstack(feats)
    X = _impute_col_median(X)
    if standardize:
        X, _, _ = _zscore_cols(X)
    return {t: grids[t] for t in ok}, X, feat_names or []

# =========================
# PCA weights (stable)
# =========================
def _first_pc_weights_from_rows(Z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Z: rows=samples (tickers), cols=features (pillars or surface dims).
    Do PCA on ticker×ticker matrix (Z Zᵀ) with a small ridge to avoid SVD issues.
    """
    n_feat = max(Z.shape[1] - 1, 1)
    R = (Z @ Z.T) / n_feat
    R[np.arange(R.shape[0]), np.arange(R.shape[0])] += float(ridge)
    vals, vecs = np.linalg.eigh(R)  # symmetric → numerically stable
    v1 = vecs[:, -1]                # first PC loadings across rows
    if v1.sum() < 0:                # orient consistently
        v1 = -v1
    w = np.clip(v1, 0.0, None)      # non‑neg (optional but typical for “market”)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

def pca_market_weights(X_peers: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Market‑mode weights across peer rows via first PC on ridge‑regularized R."""
    Z, _, _ = _zscore_cols(_impute_col_median(X_peers))
    return _first_pc_weights_from_rows(Z, ridge=ridge)

def pca_regress_weights(X_peers: np.ndarray, y_target: np.ndarray,
                        k: Optional[int] = None, nonneg: bool = True) -> np.ndarray:
    """
    PCA regression: min_w || Xᵀ w − y || using SVD with truncation k.
    Uses a safe SVD on well‑conditioned, imputed, standardized X.
    """
    Z, _, _ = _zscore_cols(_impute_col_median(X_peers))
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    if k is None or k <= 0 or k > len(s):
        k = len(s)
    Uk, sk, Vk = U[:, :k], s[:k], Vt[:k, :].T
    w = Uk @ ((y_target @ Vk) / np.where(sk > 1e-12, sk, 1.0))
    if nonneg:
        w = np.clip(w, 0.0, None)
    ssum = w.sum()
    return w / ssum if ssum > 0 else np.full_like(w, 1.0 / len(w))

# =========================
# Public weight builders
# =========================
def pca_weights_from_atm_matrix(
    atm_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    min_rows_per_col: int = 3,
    min_cols_per_row: int = 2,
    ridge: float = 1e-6,
) -> pd.Series:
    """
    PC1 weights from an ATM (tickers×pillars) table.
    Filters sparse cols/rows, imputes medians, z‑scores, eigens on ridge R.
    """
    target = (target or "").upper()
    peers = [p.upper() for p in peers]
    rows = [r for r in [target] + peers if r in atm_df.index]
    if len(rows) < 2:
        return pd.Series(index=peers, dtype=float)

    X = atm_df.loc[rows].to_numpy(float)
    # column filter (need enough tickers per pillar)
    keep_cols = np.where(np.isfinite(X).sum(axis=0) >= int(min_rows_per_col))[0]
    if keep_cols.size < 2:
        return pd.Series(index=peers, dtype=float)
    X = X[:, keep_cols]
    # row filter (need enough pillars per ticker)
    keep_rows = np.where(np.isfinite(X).sum(axis=1) >= int(min_cols_per_row))[0]
    if keep_rows.size < 2:
        return pd.Series(index=peers, dtype=float)
    kept_names = [rows[i] for i in keep_rows]
    Z, _, _ = _zscore_cols(_impute_col_median(X[keep_rows, :]))
    w_all = _first_pc_weights_from_rows(Z, ridge=ridge)
    ser = pd.Series(w_all, index=kept_names).drop(index=[target], errors="ignore")
    ser = ser.reindex(peers).fillna(0.0)
    if clip_negative:
        ser = ser.clip(lower=0.0)
    s = ser.sum()
    return ser / s if s > 0 else ser

def pca_weights(
    get_smile_slice,
    mode: str,               # 'pca_atm_market' | 'pca_atm_regress' | 'pca_surface_market' | 'pca_surface_regress'
    target: str,
    peers: List[str],
    asof: str,
    pillars_days: Iterable[int] = (7,30,60,90,180,365),
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    k: Optional[int] = None,
) -> pd.Series:
    peers = [p.upper() for p in peers]
    target = (target or "").upper()

    if mode.startswith("pca_atm"):
        atm_df, _, _ = atm_feature_matrix(get_smile_slice, [target] + peers, asof, pillars_days)
        if "market" in mode:
            return pca_weights_from_atm_matrix(atm_df, target, peers)
        # regression: build y from target row
        if target not in atm_df.index:
            raise RuntimeError("target missing in ATM matrix")
        y = _impute_col_median(atm_df.loc[[target]].to_numpy(float)).ravel()
        Xp = atm_df.loc[peers].to_numpy(float)
        w = pca_regress_weights(Xp, y, k=k, nonneg=True)
        return pd.Series(w / w.sum(), index=peers, dtype=float)

    if mode.startswith("pca_surface"):
        grids, X, _ = surface_feature_matrix([target] + peers, asof, tenors=tenors, mny_bins=mny_bins)
        # reorder rows to [target, peers...] present in X already
        labels_present = [target] + [p for p in peers if p in grids]
        idxs = list(range(len(labels_present)))  # already in order constructed
        X = X[idxs, :]
        if "market" in mode:
            w = pca_market_weights(X[1:, :])
        else:
            w = pca_regress_weights(X[1:, :], X[0, :], k=k, nonneg=True)
        return pd.Series(w / w.sum(), index=labels_present[1:], dtype=float)

    raise ValueError(f"unknown mode: {mode}")

# =========================
# Simple beta builders (UL, IV‑ATM, surface)
# =========================
def _spot_log_returns(conn_fn) -> pd.DataFrame:
    conn = conn_fn()
    df = pd.read_sql_query("SELECT asof_date, ticker, spot FROM options_quotes", conn)
    if df.empty:
        return df
    px = df.groupby(["asof_date", "ticker"])["spot"].median().unstack("ticker").sort_index()
    ret = (px / px.shift(1)).applymap(lambda x: math.log(x) if pd.notna(x) and x > 0 else np.nan)
    return ret

def ul_betas(benchmark: str, conn_fn) -> pd.Series:
    ret = _spot_log_returns(conn_fn)
    if ret is None or ret.empty or benchmark not in ret.columns:
        return pd.Series(dtype=float)
    betas = {}
    for t in ret.columns:
        if t == benchmark:
            continue
        betas[t] = _beta(ret.rename(columns={t: "x", benchmark: "b"}), "x", "b")
    return pd.Series(betas, name="ul_beta")

def iv_atm_betas(benchmark: str, pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS) -> Dict[int, pd.Series]:
    atm = load_atm()
    if atm.empty:
        return {}
    piv = nearest_pillars(atm, pillars_days=pillar_days)
    out: Dict[int, pd.Series] = {}
    for d in sorted(set(piv["pillar_days"])):
        sub = piv[piv["pillar_days"] == d]
        wide = sub.pivot_table(index="asof_date", columns="ticker", values="iv", aggfunc="mean").sort_index()
        # de‑mean per day to reduce common level
        wide = wide.sub(wide.mean(axis=1), axis=0)
        if benchmark not in wide.columns:
            continue
        betas = {}
        for t in wide.columns:
            if t == benchmark:
                continue
            betas[t] = _beta(wide.rename(columns={t: "x", benchmark: "b"}), "x", "b")
        out[int(d)] = pd.Series(betas, name=f"iv_atm_beta_{int(d)}d")
    return out

def surface_betas(benchmark: str,
                  tenors: Iterable[int] = (7,30,60,90,180,365),
                  mny_bins: Iterable[Tuple[float,float]] = ((0.8,0.9),(0.95,1.05),(1.1,1.25)),
                  conn_fn=None) -> pd.Series:
    # cheap scalar: average grid IV per (date,ticker), then beta vs benchmark
    if conn_fn is None:
        from data.db_utils import get_conn as conn_fn  # late import
    conn = conn_fn()
    df = pd.read_sql_query("SELECT asof_date, ticker, ttm_years, moneyness, iv FROM options_quotes", conn)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.dropna(subset=["iv", "ttm_years", "moneyness"]).copy()
    df["ttm_days"] = df["ttm_years"] * 365.25
    tarr = pd.Series(list(tenors))
    df["tenor_bin"] = df["ttm_days"].apply(lambda d: tarr.iloc[(tarr - d).abs().argmin()])
    labels = [f"{lo:.2f}-{hi:.2f}" for (lo, hi) in mny_bins]
    edges = [mny_bins[0][0]] + [hi for (_, hi) in mny_bins]
    df["mny_bin"] = pd.cut(df["moneyness"], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=["mny_bin"])
    cell = df.groupby(["asof_date", "ticker", "tenor_bin", "mny_bin"])["iv"].mean().reset_index()
    grid = cell.pivot_table(index=["asof_date", "ticker"], columns=["tenor_bin", "mny_bin"], values="iv")
    level = grid.mean(axis=1).rename("iv_surface_level").reset_index()
    wide = level.pivot(index="asof_date", columns="ticker", values="iv_surface_level").sort_index()
    if benchmark not in wide.columns:
        return pd.Series(dtype=float)
    betas = {}
    for t in wide.columns:
        if t == benchmark:
            continue
        betas[t] = _beta(wide.rename(columns={t: "x", benchmark: "b"}), "x", "b")
    return pd.Series(betas, name="iv_surface_beta")
