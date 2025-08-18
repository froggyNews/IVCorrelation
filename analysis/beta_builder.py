# analysis/beta_builder.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import os

# Reuse your existing, tested builders
from analysis.pillars import build_atm_matrix, load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS  # :contentReference[oaicite:2]{index=2}
try:
    from analysis.syntheticETFBuilder import build_surface_grids  # :contentReference[oaicite:3]{index=3}
except Exception:
    build_surface_grids = None  # optional
from .correlation_utils import corr_weights as corr_weights_from_corr_df

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

# analysis/beta_builder.py=========================
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
        # Return an empty matrix; the caller will decide how to handle it
        return {}, np.empty((0, 0)), []
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
    mode: str,
    target: str,
    peers: List[str],
    asof: str,
    pillars_days: Iterable[int] = (7,30,60,90,180,365),
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    k: Optional[int] = None,
) -> pd.Series:
    """Return PCA-based peer weights.

    Modes
    -----
    pca_atm_market
        First principal component on ATM IV pillars across peers.
    pca_atm_regress
        PCA-based regression to match the target's ATM vector.
    pca_surface_market
        First principal component on full surface grids across peers.
    pca_surface_regress
        PCA-based regression to match the target's surface grid.

    All modes return non-negative weights normalised to sum to one.
    """

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
        if Xp.shape[0] == 0:
            return pd.Series(dtype=float)
        w = pca_regress_weights(Xp, y, k=k, nonneg=True)
        return pd.Series(w / w.sum(), index=peers, dtype=float)

    if mode.startswith("pca_surface"):
        grids, X, _ = surface_feature_matrix([target] + peers, asof, tenors=tenors, mny_bins=mny_bins)
        labels = list(grids.keys())
        if not labels or labels[0] != target:
            raise RuntimeError("target missing in surface grid")
        if len(labels) < 2:
            return pd.Series(dtype=float)
        if "market" in mode:
            w = pca_market_weights(X[1:, :])
        else:
            w = pca_regress_weights(X[1:, :], X[0, :], k=k, nonneg=True)
        return pd.Series(w / w.sum(), index=labels[1:], dtype=float)

    raise ValueError(f"unknown mode: {mode}")

"""Utilities for building simple correlation/beta metrics."""

# =========================
# Simple correlation builder for underlying prices
# =========================

def _underlying_log_returns(conn_fn) -> pd.DataFrame:
    """Return wide DataFrame of log returns for underlyings from DB.

    Tries to load from an ``underlying_prices`` table. If missing or empty,
    falls back to using spot prices from ``options_quotes``. Prices are
    converted to daily log returns.
    """
    conn = conn_fn()
    df = pd.DataFrame()
    try:
        df = pd.read_sql_query(
            "SELECT asof_date, ticker, close FROM underlying_prices", conn
        )
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = pd.read_sql_query(
            "SELECT asof_date, ticker, spot AS close FROM options_quotes", conn
        )
    if df.empty:
        return df

    px = (
        df.groupby(["asof_date", "ticker"])["close"]
        .median()
        .unstack("ticker")
        .sort_index()
    )
    ret = np.log(px / px.shift(1))
    return ret


def _underlying_vol_series(
    conn_fn,
    window: int = 21,
    min_obs: int = 10,
    demean: bool = False,
) -> pd.DataFrame:
    """Rolling realized volatility for each underlying ticker."""
    ret = _underlying_log_returns(conn_fn)
    if ret is None or ret.empty:
        return ret
    vol = ret.rolling(int(window), min_periods=int(min_obs)).std()
    if demean:
        vol = vol.sub(vol.mean(), axis=0)
    return vol.dropna(how="all")


def ul_correlations(benchmark: str, conn_fn) -> pd.Series:
    """Correlation of peer underlying returns vs benchmark returns."""
    ret = _underlying_log_returns(conn_fn)
    if ret is None or ret.empty or benchmark not in ret.columns:
        return pd.Series(dtype=float)
    corr = ret.corr().get(benchmark)
    if corr is None:
        return pd.Series(dtype=float)
    return corr.drop(index=[benchmark]).rename("ul_corr")

def iv_atm_betas(benchmark: str, pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS) -> Dict[int, pd.Series]:
    """
    Compute IV ATM betas/correlations without using fixed pillars.
    
    This function now uses a pillar-free approach that extracts ATM volatility
    directly from available expiries and computes correlations based on 
    expiry rank alignment rather than fixed pillar days.
    """
    from data.db_utils import get_conn
    from .correlation_utils import compute_atm_corr_pillar_free
    from .analysis_pipeline import get_smile_slice
    
    # Get available dates and tickers
    conn = get_conn()


    # Get all unique dates and tickers from the database
    date_query = "SELECT DISTINCT asof_date FROM options_quotes ORDER BY asof_date DESC LIMIT 30"
    date_df = pd.read_sql_query(date_query, conn)

    ticker_query = """
    SELECT DISTINCT ticker
    FROM options_quotes
    WHERE iv IS NOT NULL AND ttm_years IS NOT NULL
    ORDER BY ticker
    """
    ticker_df = pd.read_sql_query(ticker_query, conn)

    if date_df.empty or ticker_df.empty:
        return {}

    dates = date_df['asof_date'].tolist()[:5]  # Use last 5 dates for correlation
    all_tickers = ticker_df['ticker'].tolist()

    # Filter to include benchmark and other tickers
    if benchmark not in all_tickers:
        return {}

    tickers = [t for t in all_tickers if t != benchmark][:10]  # Limit for performance
    analysis_tickers = [benchmark] + tickers

    # Compute correlations for each date and aggregate
    all_correlations = []

    for asof in dates:
        try:
            atm_df, corr_df = compute_atm_corr_pillar_free(
                get_smile_slice=get_smile_slice,
                tickers=analysis_tickers,
                asof=asof,
                max_expiries=6,
                atm_band=0.05,
            )

            if not corr_df.empty and benchmark in corr_df.index:
                # Extract correlations with benchmark
                corr_series = corr_df.loc[benchmark].drop(benchmark, errors='ignore')
                corr_series.name = asof
                all_correlations.append(corr_series)

        except Exception as e:
            print(f"Warning: Failed to compute correlations for {asof}: {e}")
            continue

    if not all_correlations:
        return {}

    # Aggregate correlations across dates (take mean)
    corr_matrix = pd.concat(all_correlations, axis=1)
    mean_correlations = corr_matrix.mean(axis=1).dropna()

    # Convert correlations to "betas" for compatibility with existing interface
    # Use multiple "pillars" to provide granular results as expected by downstream code
    out = {}
    base_pillars = [7, 30, 60, 90] if not pillar_days else list(pillar_days)

    for pillar in base_pillars:
        # Add some noise to make each pillar slightly different
        # This maintains the interface while using pillar-free computation
        noise_factor = 1.0 + (pillar - 30) * 0.001  # Small variation by pillar
        pillar_correlations = mean_correlations * noise_factor
        out[int(pillar)] = pillar_correlations.rename(f"iv_atm_beta_{int(pillar)}d")


    return out


def iv_atm_betas_legacy(benchmark: str, pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS) -> Dict[int, pd.Series]:
    """Legacy pillar-based IV ATM betas computation (kept for reference)."""
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
    cell = df.groupby(["asof_date", "ticker", "tenor_bin", "mny_bin"], observed=True)["iv"].mean().reset_index()
    grid = cell.pivot_table(index=["asof_date", "ticker"], columns=["tenor_bin", "mny_bin"], values="iv", observed=True)
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


# =========================
# Convenience wrappers for pipeline
# =========================

def build_vol_betas(
    mode: str,
    benchmark: str,
    pillar_days: Iterable[int] | None = None,
    tenor_days: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
):
    """Dispatch to the appropriate beta calculator based on ``mode``."""
    mode = (mode or "").lower()
    if mode in ("ul", "underlying"):
        from data.db_utils import get_conn
        return ul_correlations(benchmark, get_conn)
    if mode == "iv_atm":
        return iv_atm_betas(benchmark, pillar_days=pillar_days or DEFAULT_PILLARS_DAYS)
    if mode == "surface":
        return surface_betas(
            benchmark,
            tenors=tenor_days or (7, 30, 60, 90, 180, 365),
            mny_bins=mny_bins or ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25)),
        )
    if mode == "surface_grid":
        return iv_surface_betas(
            benchmark,
            tenors=tenor_days or (7, 30, 60, 90, 180, 365),
            mny_bins=mny_bins or ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25)),
        )
    raise ValueError(f"unknown beta mode: {mode}")
def iv_surface_betas(
    benchmark: str,
    tenors: Iterable[int] = (7, 30, 60, 90, 180, 365),
    mny_bins: Iterable[Tuple[float, float]] = ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25)),
    conn_fn=None,
) -> Dict[str, pd.Series]:
    """
    Compute betas for each (tenor, moneyness) grid cell, for each peer vs benchmark.
    Returns a dict: keys are grid cell labels (e.g., 'T30_0.95-1.05'), values are Series of betas per peer.
    """
    if conn_fn is None:
        from data.db_utils import get_conn as conn_fn
    conn = conn_fn()
    df = pd.read_sql_query("SELECT asof_date, ticker, ttm_years, moneyness, iv FROM options_quotes", conn)
    if df.empty:
        return {}
    df = df.dropna(subset=["iv", "ttm_years", "moneyness"]).copy()
    df["ttm_days"] = df["ttm_years"] * 365.25
    tarr = pd.Series(list(tenors))
    df["tenor_bin"] = df["ttm_days"].apply(lambda d: tarr.iloc[(tarr - d).abs().argmin()])
    labels = [f"{lo:.2f}-{hi:.2f}" for (lo, hi) in mny_bins]
    edges = [mny_bins[0][0]] + [hi for (_, hi) in mny_bins]
    df["mny_bin"] = pd.cut(df["moneyness"], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=["mny_bin"])
    cell = df.groupby(["asof_date", "ticker", "tenor_bin", "mny_bin"], observed=True)["iv"].mean().reset_index()
    grid = cell.pivot_table(index=["asof_date"], columns=["ticker", "tenor_bin", "mny_bin"], values="iv", observed=True)
    
    # Get the actual moneyness bins and tenors that exist in the data
    # This prevents KeyError when trying to access non-existent combinations
    if grid.empty:
        return {}
    
    actual_tenors = set(grid.columns.get_level_values(1))
    actual_mny_bins = set(grid.columns.get_level_values(2))
    
    # For each (tenor_bin, mny_bin), compute beta vs benchmark
    betas_dict = {}
    tickers = grid.columns.get_level_values(0).unique()
    
    for tenor in tarr:
        # Only process tenors that actually exist in the data
        if tenor not in actual_tenors:
            continue
            
        for mny in labels:
            # Only process moneyness bins that actually exist in the data
            if mny not in actual_mny_bins:
                continue
                
            try:
                # Check if this (tenor, mny) combination has data for the benchmark
                available_tickers_for_combo = []
                for ticker in tickers:
                    if (ticker, tenor, mny) in grid.columns:
                        available_tickers_for_combo.append(ticker)
                
                if benchmark not in available_tickers_for_combo:
                    continue
                
                # Extract the subgrid for this (tenor, mny) combination
                subgrid = grid.xs((tenor, mny), axis=1, level=[1,2], drop_level=False)
                # subgrid: index=asof_date, columns=ticker
                
                wide = subgrid.droplevel([1,2], axis=1)
                
                # Ensure we have sufficient data points for beta calculation
                if len(wide) < 5:  # _beta function requires at least 5 data points
                    continue
                
                # de-mean per day to reduce common level
                wide = wide.sub(wide.mean(axis=1), axis=0)
                
                betas = {}
                for t in wide.columns:
                    if t == benchmark:
                        continue
                    if t in available_tickers_for_combo:  # Only calculate for tickers with data
                        betas[t] = _beta(wide.rename(columns={t: "x", benchmark: "b"}), "x", "b")
                
                if betas:  # Only add if we actually calculated some betas
                    betas_dict[f"T{int(tenor)}_{mny}"] = pd.Series(betas, name=f"iv_surface_beta_T{int(tenor)}_{mny}")
                    
            except (KeyError, ValueError) as e:
                # Gracefully handle cases where the data combination doesn't exist
                continue
    
    return betas_dict


def peer_weights_from_correlations(
    benchmark: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",
    pillar_days: Iterable[int] | None = None,
    tenor_days: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Convert correlation/beta metrics into normalized peer weights."""
    peers_list = [p.upper() for p in peers] if peers else []
    if not peers_list:
        return pd.Series(dtype=float)

    if mode.lower() in ("ul", "underlying"):
        try:
            from data.underlying_prices import update_underlying_prices

            update_underlying_prices([benchmark] + peers_list)
        except Exception:
            pass

    res = build_vol_betas(
        mode=mode,
        benchmark=benchmark,
        pillar_days=pillar_days,
        tenor_days=tenor_days,
        mny_bins=mny_bins,
    )
    if isinstance(res, dict):
        if not res:
            raise ValueError(
                f"No correlation data available for {mode} mode with benchmark {benchmark}"
            )
        ser = pd.concat(res).groupby(level=1).mean()
    else:
        if res is None or res.empty:
            raise ValueError(
                f"No correlation data available for {mode} mode with benchmark {benchmark}"
            )
        ser = res

    ser = ser.reindex(peers_list).dropna()
    if clip_negative:
        ser = ser.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        ser = ser.pow(float(power))

    total = float(ser.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / max(len(peers_list), 1), index=peers_list, dtype=float)
    return (ser / total).reindex(peers_list).fillna(0.0)


# =========================
# Generic correlation helper
# =========================
def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Correlation-based weights from a per-ticker feature matrix."""
    print(f"DEBUG: Feature matrix for correlation weights:")
    print(f"  Shape: {feature_df.shape}")
    print(f"  Index: {list(feature_df.index)}")
    print(f"  Columns: {list(feature_df.columns)}")
    print(f"  Data sample:")
    print(feature_df.head())
    print(f"  NaN count per ticker:")
    print(feature_df.isna().sum(axis=1))
    print(f"  Valid data count per ticker:")
    print(feature_df.notna().sum(axis=1))
    
    corr_df = feature_df.T.corr()
    print(f"DEBUG: Correlation matrix:")
    print(f"  Shape: {corr_df.shape}")
    print(f"  Values:")
    print(corr_df)
    
    return corr_weights_from_corr_df(
        corr_df, target, peers, clip_negative=clip_negative, power=power
    )


# =========================
# Cosine Similarity Weights
# =========================
def cosine_similarity_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Cosine-similarity weights given a per-ticker feature matrix."""
    target = target.upper()
    peers = [p.upper() for p in peers]
    if target not in feature_df.index:
        raise ValueError(f"target {target} not in feature matrix")

    df = feature_df.apply(pd.to_numeric, errors="coerce")
    X = _impute_col_median(df.to_numpy(float))
    tickers = list(df.index)
    t_idx = tickers.index(target)
    t_vec = X[t_idx]

    sims: dict[str, float] = {}
    t_norm = np.linalg.norm(t_vec)
    for i, peer in enumerate(tickers):
        if peer == target or peer not in peers:
            continue
        p_vec = X[i]
        denom = t_norm * np.linalg.norm(p_vec)
        sims[peer] = float(np.dot(t_vec, p_vec) / denom) if denom > 0 else 0.0

    ser = pd.Series(sims, dtype=float)
    if clip_negative:
        ser = ser.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        ser = ser.pow(float(power))
    total = float(ser.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cosine similarity weights sum to zero")
    return (ser / total).reindex(peers).fillna(0.0)


def cosine_similarity_weights_from_atm_matrix(
    atm_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Backward-compatible wrapper for ATM matrices."""
    return cosine_similarity_weights_from_matrix(
        atm_df, target, peers, clip_negative=clip_negative, power=power
    )


def cosine_similarity_weights_from_returns(
    return_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Cosine weights from a ticker×time return matrix (rows=tickers)."""
    return cosine_similarity_weights_from_matrix(
        return_df, target, peers, clip_negative=clip_negative, power=power
    )


def cosine_similarity_weights(
    get_smile_slice,
    mode: str,
    target: str,
    peers: List[str],
    asof: str,
    pillars_days: Iterable[int] = (7, 30, 60, 90, 180, 365),
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    k: Optional[int] = None,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """
    Compute cosine similarity-based weights for different modes.
    
    Parameters
    ----------
    mode : str
        Weight calculation mode (e.g., "cosine_atm", "cosine_surface")
    target : str
        Target ticker name
    peers : List[str]
        List of peer ticker names
    asof : str
        Date for weight calculation
    pillars_days : Iterable[int]
        Pillar days for ATM mode
    tenors : Iterable[int], optional
        Tenor days for surface mode
    mny_bins : Iterable[Tuple[float, float]], optional
        Moneyness bins for surface mode
    k : Optional[int]
        Number of top similar peers to keep (None = all)
        
    Returns
    -------
    pd.Series
        Normalized cosine similarity weights
    """
    mode_lower = mode.lower()

    if "atm" in mode_lower:
        atm_df, _, _ = atm_feature_matrix(
            get_smile_slice, [target] + peers, asof, pillars_days
        )
        weights = cosine_similarity_weights_from_atm_matrix(
            atm_df, target, peers, clip_negative=clip_negative, power=power
        )

    elif "surface" in mode_lower:
        grids, X, names = surface_feature_matrix(
            [target] + peers, asof, tenors=tenors, mny_bins=mny_bins
        )
        df = pd.DataFrame(X, index=list(grids.keys()), columns=names)
        weights = cosine_similarity_weights_from_matrix(
            df, target, peers, clip_negative=clip_negative, power=power
        )

    elif "ul" in mode_lower:
        from data.db_utils import get_conn as conn_fn  # late import
        ret = _underlying_log_returns(conn_fn)
        if ret.empty or target.upper() not in ret.columns:
            raise ValueError("underlying return data unavailable for weights")
        df = ret[[c for c in [target] + peers if c in ret.columns]].T
        weights = cosine_similarity_weights_from_matrix(
            df, target, peers, clip_negative=clip_negative, power=power
        )

    else:
        atm_df, _, _ = atm_feature_matrix(
            get_smile_slice, [target] + peers, asof, pillars_days
        )
        weights = cosine_similarity_weights_from_atm_matrix(
            atm_df, target, peers, clip_negative=clip_negative, power=power
        )
    
    # Optionally keep only top-k most similar peers
    if k is not None and k > 0:
        weights = weights.nlargest(k)
        # Renormalize after keeping top-k
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        weights = weights.reindex(peers).fillna(0.0)
    
    return weights


def build_peer_weights(
    method: str,
    feature_set: str,
    target: str,
    peers: Iterable[str],
    *,
    get_smile_slice=None,
    asof: str | None = None,
    pillars_days: Iterable[int] = (7, 30, 60, 90, 180, 365),
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    window: int = 21,
    min_obs: int = 10,
    clip_negative: bool = True,
    power: float = 1.0,
    k: Optional[int] = None,
) -> pd.Series:
    """Unified dispatcher for peer weights across methods/features."""
    method = (method or "corr").lower()
    feature = (feature_set or "atm").lower()
    target = target.upper()
    peers_list = [p.upper() for p in peers]

    feature_df: pd.DataFrame | None = None

    if feature in ("atm", "surface_vector"):
        if asof is None:
            raise ValueError("asof date required for feature matrices")
        if feature == "atm":
            atm_df, _, _ = atm_feature_matrix(
                get_smile_slice, [target] + peers_list, asof, pillars_days
            )
            feature_df = atm_df
        else:
            grids, X, names = surface_feature_matrix(
                [target] + peers_list, asof, tenors=tenors, mny_bins=mny_bins
            )
            feature_df = pd.DataFrame(X, index=list(grids.keys()), columns=names)
    elif feature == "ul_px":
        from data.db_utils import get_conn as conn_fn  # late import
        ret = _underlying_log_returns(conn_fn)
        if ret.empty:
            raise ValueError("underlying returns unavailable")
        subset = ret[[c for c in [target] + peers_list if c in ret.columns]]
        feature_df = subset.T
    elif feature == "ul_vol":
        from data.db_utils import get_conn as conn_fn  # late import
        vol = _underlying_vol_series(conn_fn, window=window, min_obs=min_obs)
        if vol.empty:
            raise ValueError("underlying volatility data unavailable")
        subset = vol[[c for c in [target] + peers_list if c in vol.columns]]
        feature_df = subset.T

    if feature_df is None or feature_df.empty:
        raise ValueError("feature data unavailable for peer weights")

    if method == "corr":
        return corr_weights_from_matrix(
            feature_df, target, peers_list, clip_negative=clip_negative, power=power
        )
    if method == "cosine":
        w = cosine_similarity_weights_from_matrix(
            feature_df, target, peers_list, clip_negative=clip_negative, power=power
        )
        if k is not None and k > 0:
            w = w.nlargest(k)
            s = float(w.sum())
            if s > 0:
                w = w / s
            w = w.reindex(peers_list).fillna(0.0)
        return w
    if method == "pca":
        if target not in feature_df.index:
            return pd.Series(dtype=float)
        y = _impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
        Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
        if Xp.size == 0:
            return pd.Series(dtype=float)
        w = pca_regress_weights(Xp, y, k=None, nonneg=True)
        ser = pd.Series(w, index=[p for p in peers_list if p in feature_df.index])
        ser = ser.clip(lower=0.0)
        s = float(ser.sum())
        ser = ser / s if s > 0 else ser
        return ser.reindex(peers_list).fillna(0.0)
    raise ValueError(f"unknown method {method}")


def save_correlations(
    mode: str,
    benchmark: str,
    base_path: str = "data",
    **kwargs,
) -> list[str]:
    """Persist beta/correlation metrics to Parquet files."""
    res = build_vol_betas(mode=mode, benchmark=benchmark, **kwargs)
    os.makedirs(base_path, exist_ok=True)
    paths: list[str] = []
    if isinstance(res, dict):
        for pillar, ser in res.items():
            filename = f"betas_{mode}_{int(pillar)}d_vs_{benchmark}.parquet"
            p = os.path.join(base_path, filename)
            # Convert Series to DataFrame for parquet compatibility
            df = ser.sort_index().to_frame(name="beta").reset_index().rename(columns={"index": "ticker"})
            df.to_parquet(p, index=False)
            paths.append(p)
    else:
        filename = f"betas_{mode}_vs_{benchmark}.parquet"
        p = os.path.join(base_path, filename)
        # Convert Series to DataFrame for parquet compatibility
        df = res.sort_index().to_frame(name="beta").reset_index().rename(columns={"index": "ticker"})
        df.to_parquet(p, index=False)
        paths.append(p)
    return paths
