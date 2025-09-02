"""Unified weight engine and feature builders."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple, Union, List
import logging

import numpy as np
import pandas as pd

from analysis.beta_builder.utils import impute_col_median as _impute_col_median, zscore_cols as _zscore_cols
from .correlation import (
    corr_weights_from_matrix,
    corr_weights as _corr_weights_from_corr_df,
    compute_atm_corr as _compute_atm_corr,
    compute_atm_corr_pillar_free as _compute_atm_corr_pf,
)
from .cosine import cosine_similarity_weights_from_matrix
from .pca import pca_regress_weights, pca_market_weights, pca_weights_from_feature_df
# add:
from .pca import pca_surface_project

from .open_interest import open_interest_weights
from .equal import equal_weights

from analysis.pillars import build_atm_matrix, DEFAULT_PILLARS_DAYS

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

class WeightMethod(Enum):
    CORRELATION = "corr"
    PCA = "pca"
    COSINE = "cosine"
    EQUAL = "equal"
    OPEN_INTEREST = "oi"

class FeatureSet(Enum):
    ATM = "iv_atm"
    ATM_RANKS = "iv_atm_ranks"
    SURFACE = "surface"
    SURFACE_VECTOR = "surface_grid"
    UNDERLYING_PX = "ul"

@dataclass
class WeightConfig:
    method: WeightMethod
    feature_set: FeatureSet
    pillars_days: Tuple[int, ...] = tuple(DEFAULT_PILLARS_DAYS)
    tenors: Tuple[int, ...] = (7, 30, 60, 90, 180, 365)
    mny_bins: Tuple[Tuple[float, float], ...] = ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25))
    clip_negative: bool = True
    power: float = 1.0
    asof: Optional[str] = None
    atm_band: float = 0.08
    atm_tol_days: float = 10.0
    max_expiries: int = 6
    pca_project_surface: bool = False
    pca_energy: float = 0.95
    pca_k: Optional[int] = None

    # NEW: Underlying (UL) PCA factor-path controls
    pca_ul_via_factors: bool = False   # if True, use PCs of peer returns as factors
    pca_ul_energy: float = 0.95        # variance threshold for UL PCA
    pca_ul_k: Optional[int] = None     # fixed K for UL (overrides energy)

    @classmethod
    def from_legacy_mode(cls, mode: str, **kwargs) -> "WeightConfig":
        mode = (mode or "").lower().strip()
        method_map = {
            "corr": WeightMethod.CORRELATION,
            "correlation": WeightMethod.CORRELATION,
            "pca": WeightMethod.PCA,
            "cosine": WeightMethod.COSINE,
            "equal": WeightMethod.EQUAL,
            "oi": WeightMethod.OPEN_INTEREST,
            "open_interest": WeightMethod.OPEN_INTEREST,
            "iv": WeightMethod.CORRELATION,
        }
        feature_map = {
            "atm": FeatureSet.ATM,
            "iv_atm": FeatureSet.ATM,
            "iv_atm_ranks": FeatureSet.ATM_RANKS,
            "atm_ranks": FeatureSet.ATM_RANKS,
            "atm_ranked": FeatureSet.ATM_RANKS,
            "surface": FeatureSet.SURFACE,
            "surface_full": FeatureSet.SURFACE,
            "surface_vector": FeatureSet.SURFACE_VECTOR,
            "surface_grid": FeatureSet.SURFACE_VECTOR,
            "underlying": FeatureSet.UNDERLYING_PX,
            "underlying_px": FeatureSet.UNDERLYING_PX,
            "ul": FeatureSet.UNDERLYING_PX,
            "ul_px": FeatureSet.UNDERLYING_PX,
        }
        if mode in feature_map:
            method_str, feature_str = "corr", mode
        elif "_" in mode:
            method_str, feature_str = mode.split("_", 1)
        else:
            method_str, feature_str = mode, "iv_atm"
        method = method_map.get(method_str, WeightMethod.CORRELATION)
        feature_set = feature_map.get(feature_str, FeatureSet.ATM)

        # Default: for PCA on UL, prefer factor-path unless explicitly overridden
        if (
            method == WeightMethod.PCA
            and feature_set == FeatureSet.UNDERLYING_PX
            and "pca_ul_via_factors" not in kwargs
        ):
            kwargs["pca_ul_via_factors"] = True

        return cls(method=method, feature_set=feature_set, **kwargs)

def atm_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    *,
    atm_band: float = 0.08,
    tol_days: float = 10.0,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from analysis.analysis_pipeline import get_smile_slice
    atm_df, _ = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=[t.upper() for t in tickers],
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
    )
    X = _impute_col_median(atm_df.to_numpy(dtype=float))
    if standardize:
        X, _, _ = _zscore_cols(X)
    return atm_df, X, list(atm_df.columns)

def _atm_rank_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = 6,
    *,
    atm_band: float = 0.05,
) -> tuple[pd.DataFrame, list[int]]:
    from analysis.analysis_pipeline import get_smile_slice
    from .correlation import compute_atm_corr_pillar_free
    tickers = [t.upper() for t in tickers]
    atm_df, _ = compute_atm_corr_pillar_free(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        max_expiries=max_expiries,
        atm_band=atm_band,
    )
    return atm_df.reindex(tickers), list(atm_df.columns)

def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    *,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    standardize: bool = True,
    min_coverage: float = 0.5,        # choose tenor/mny bins present in at least this fraction of tickers
    strict_common_date: bool = False, # if True, require a single common date across all tickers
    mny_range: Tuple[float, float] | None = None,  # optional numeric moneyness filter (inclusive)
) -> Tuple[Dict[str, Dict[pd.Timestamp, pd.DataFrame]], np.ndarray, List[str]]:
    # Use defaults if None were provided (some callers pass None explicitly)
    from analysis.compositeETFBuilder import (
        build_surface_grids,
        DEFAULT_TENORS as _DEF_TENORS,
        DEFAULT_MNY_BINS as _DEF_MNY_BINS,
    )
    req = [t.upper() for t in tickers]
    grids = build_surface_grids(
        tickers=req,
        tenors=(list(tenors) if tenors is not None else _DEF_TENORS),
        mny_bins=(mny_bins if mny_bins is not None else _DEF_MNY_BINS),
        use_atm_only=False,
    )
    feats: list[np.ndarray] = []
    ok: list[str] = []
    feat_names: list[str] | None = None
    asof_ts = pd.to_datetime(asof) if asof is not None else None

    # First pass: gather available columns (tenors) and index (mny bins) across tickers
    cols_list: list[list] = []
    idx_list: list[list] = []
    keys_by_ticker: dict[str, pd.Timestamp] = {}

    # Resolve date keys
    if strict_common_date:
        # Find common dates across all available tickers and pick the latest
        date_sets: list[set[pd.Timestamp]] = []
        for t in req:
            if t in grids and grids[t]:
                date_sets.append(set(grids[t].keys()))
        common_dates = set.intersection(*date_sets) if date_sets else set()
        if not common_dates:
            return {}, np.empty((0, 0)), []
        chosen_date = max(common_dates)
        for t in req:
            if t in grids and chosen_date in grids[t]:
                keys_by_ticker[t] = chosen_date
    
    for t in req:
        if t not in grids or not grids[t]:
            continue
        if t in keys_by_ticker:
            key = keys_by_ticker[t]
        else:
            key = None
            if asof_ts is not None:
                norm_map = {pd.to_datetime(k).normalize(): k for k in grids[t].keys()}
                key = norm_map.get(asof_ts.normalize())
            if key is None:
                key = max(grids[t].keys())
        df = grids[t][key]
        if df is None or df.empty:
            continue
        keys_by_ticker[t] = key
        cols_list.append(list(df.columns))
        idx_list.append(list(df.index))

    if not keys_by_ticker:
        return {}, np.empty((0, 0)), []

    # Determine canonical axes dynamically to maximize alignment:
    # choose any tenor/mny bin that appears in at least ceil(min_coverage * n_tick) tickers
    n_ok = len(keys_by_ticker)
    min_count = max(1, int(np.ceil(float(min_coverage) * n_ok)))
    from collections import Counter
    c_cols = Counter([c for cols in cols_list for c in cols])
    c_idx = Counter([i for idx in idx_list for i in idx])
    canon_cols = [c for c, cnt in c_cols.items() if cnt >= min_count]
    canon_idx = [i for i, cnt in c_idx.items() if cnt >= min_count]

    # If too aggressive (empty), fall back to strict intersection
    if not canon_cols or not canon_idx:
        canon_cols = list(set.intersection(*[set(cols) for cols in cols_list])) if cols_list else []
        canon_idx = list(set.intersection(*[set(ix) for ix in idx_list])) if idx_list else []
        if not canon_cols or not canon_idx:
            return {}, np.empty((0, 0)), []

    # Sort for stability
    try:
        canon_cols = sorted(canon_cols)
    except Exception:
        pass
    try:
        canon_idx = sorted(canon_idx)
    except Exception:
        pass

    # Second pass: align to canonical axes and stack features
    for t, key in keys_by_ticker.items():
        df = grids[t][key]
        df_aligned = df.reindex(index=canon_idx, columns=canon_cols)
        # Optional moneyness range restriction
        if mny_range is not None and len(df_aligned.index) > 0:
            lo, hi = float(mny_range[0]), float(mny_range[1])
            # Convert index labels to numeric midpoints
            idx_vals: list[float] = []
            for lab in df_aligned.index:
                try:
                    s = str(lab)
                    if "-" in s:
                        a, b = s.split("-", 1)
                        idx_vals.append((float(a) + float(b)) / 2.0)
                    else:
                        idx_vals.append(float(s))
                except Exception:
                    idx_vals.append(np.nan)
            idx_vals_arr = np.asarray(idx_vals, dtype=float)
            mask = np.isfinite(idx_vals_arr) & (idx_vals_arr >= lo) & (idx_vals_arr <= hi)
            if np.any(mask):
                df_aligned = df_aligned.iloc[np.where(mask)[0]]
        arr = df_aligned.to_numpy(float).T.reshape(-1)
        if feat_names is None:
            feat_names = [f"T{c}_{r}" for c in df_aligned.columns for r in df_aligned.index]
        feats.append(arr)
        ok.append(t)

    if not feats:
        return {}, np.empty((0, 0)), []
    X = _impute_col_median(np.vstack(feats))
    if standardize:
        X, _, _ = _zscore_cols(X)
    return {t: grids[t] for t in ok}, X, feat_names or []

def underlying_returns_matrix(tickers: Iterable[str]) -> pd.DataFrame:
    from data.db_utils import get_conn
    try:
        from data.data_pipeline import ensure_underlying_price_data
        _ = ensure_underlying_price_data({t.upper() for t in tickers})
    except Exception:
        pass
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT asof_date, ticker, close FROM underlying_prices", conn)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        df = pd.read_sql_query("SELECT asof_date, ticker, spot AS close FROM options_quotes", conn)
    if df.empty:
        return pd.DataFrame()
    px = (
        df.groupby(["asof_date", "ticker"])["close"]
        .median()
        .unstack("ticker")
        .sort_index()
    )
    ret = np.log(px / px.shift(1)).dropna(how="all")
    return ret.T
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


class UnifiedWeightComputer:
    def _choose_asof(self, target: str, peers: list[str], config: WeightConfig) -> Optional[str]:
        if config.asof:
            return config.asof
        if config.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            from data.db_utils import get_conn
            tickers = [target] + peers
            conn = get_conn()
            placeholders = ','.join('?' * len(tickers))
            q = (
                "SELECT asof_date, COUNT(DISTINCT ticker) AS n "
                "FROM options_quotes WHERE ticker IN (" + placeholders + ") "
                "GROUP BY asof_date "
                "HAVING SUM(CASE WHEN ticker = ? THEN 1 ELSE 0 END) > 0 "
                "ORDER BY n DESC, asof_date DESC LIMIT 1"
            )
            params = [t.upper() for t in tickers] + [target.upper()]
            df = pd.read_sql_query(q, conn, params=params)
            if not df.empty:
                return pd.to_datetime(df["asof_date"].iloc[0]).strftime("%Y-%m-%d")
            try:
                from analysis.analysis_pipeline import get_most_recent_date_global, available_dates
                d = get_most_recent_date_global()
                if d:
                    return d
                dates = available_dates(ticker=target, most_recent_only=True)
                if dates:
                    return dates[0]
            except Exception:
                return None
        return None

    def compute_weights(self, target: str, peers: Iterable[str], config: WeightConfig) -> pd.Series:
        target = (target or "").upper()
        peers_list = [p.upper() for p in peers]
        if not peers_list:
            return pd.Series(dtype=float)

        if config.method == WeightMethod.EQUAL:
            return equal_weights(peers_list)
        if config.method == WeightMethod.OPEN_INTEREST:
            return open_interest_weights(peers_list, config.asof)

        asof = None
        if config.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            asof = self._choose_asof(target, peers_list, config)
            if asof is None:
                raise ValueError("no surface/ATM date available to build features")

        feature_df = self._build_feature_matrix(target, peers_list, asof, config)
        if feature_df is None or feature_df.empty:
            if config.feature_set != FeatureSet.UNDERLYING_PX:
                ul_cfg = WeightConfig(
                    method=config.method,
                    feature_set=FeatureSet.UNDERLYING_PX,
                    clip_negative=config.clip_negative,
                    power=config.power,
                )
                feature_df = self._build_feature_matrix(target, peers_list, None, ul_cfg)
                if feature_df is None or feature_df.empty:
                    return equal_weights(peers_list)
                config = ul_cfg
            else:
                return equal_weights(peers_list)

        try:
            if config.method == WeightMethod.CORRELATION:
                # Mirror PCA's per-feature branches for clarity/robustness
                if config.feature_set == FeatureSet.UNDERLYING_PX:
                    # UL: correlation over returns matrix
                    return corr_weights_from_matrix(
                        feature_df, target, peers_list,
                        clip_negative=config.clip_negative, power=config.power
                    )

                if config.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS):
                    # ATM: use dedicated helpers that apply lenient coverage rules
                    tickers = [target] + peers_list
                    from analysis.analysis_pipeline import get_smile_slice as _get_smile_slice
                    if config.feature_set == FeatureSet.ATM:
                        atm_df, corr_df = _compute_atm_corr(
                            get_smile_slice=_get_smile_slice,
                            tickers=tickers,
                            asof=asof,
                            pillars_days=config.pillars_days,
                            atm_band=config.atm_band,
                            tol_days=config.atm_tol_days,
                        )
                    else:
                        atm_df, corr_df = _compute_atm_corr_pf(
                            get_smile_slice=_get_smile_slice,
                            tickers=tickers,
                            asof=asof,
                            max_expiries=config.max_expiries,
                            atm_band=config.atm_band,
                        )
                    return _corr_weights_from_corr_df(
                        corr_df, target, peers_list,
                        clip_negative=config.clip_negative, power=config.power
                    )

                if config.feature_set in (FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
                    # Surface: flatten aligned grids and correlate across features
                    return corr_weights_from_matrix(
                        feature_df, target, peers_list,
                        clip_negative=config.clip_negative, power=config.power
                    )

                # Fallback
                return corr_weights_from_matrix(
                    feature_df, target, peers_list,
                    clip_negative=config.clip_negative, power=config.power
                )
            if config.method == WeightMethod.COSINE:
                return cosine_similarity_weights_from_matrix(feature_df, target, peers_list,
                    clip_negative=config.clip_negative, power=config.power)
            if config.method == WeightMethod.PCA:
                # Build matrix once
                feature_df = feature_df  # already built above

                # --- ATM: unchanged (factorization weights) ---
                if config.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS):
                    return pca_weights_from_feature_df(feature_df, target, peers_list, k=config.pca_k, nonneg=True)

                # --- SURFACE / SURFACE_VECTOR: LaTeX projection path (opt-in) ---
                if config.feature_set in (FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR) and config.pca_project_surface:
                    # Rebuild SURFACE features WITHOUT standardization; we need raw space for μ-centering
                    grids, X_raw, names = surface_feature_matrix(
                        [target] + peers_list,
                        asof,
                        tenors=config.tenors,
                        mny_bins=config.mny_bins,
                        standardize=False,   # IMPORTANT: raw feature space
                    )
                    if X_raw.size == 0 or X_raw.shape[0] < 2:
                        raise ValueError("No peer data available for PCA projection")

                    # Arrange as: features × peers (columns); target feature vector y
                    # Our surface_feature_matrix returns rows=tickers, cols=features
                    # So transpose peers to (features, n_peers), and take target row as (features,)
                    import numpy as _np
                    tick_order = list(grids.keys())
                    if tick_order[0] != target:
                        # safety: enforce target-first ordering
                        idx_map = {t: i for i, t in enumerate(tick_order)}
                        order = [target] + [p for p in peers_list if p in idx_map]
                        X_raw = X_raw[_np.array([idx_map[t] for t in order], dtype=int), :]
                        tick_order = order

                    y_vec = X_raw[0, :].astype(float)
                    Xp_raw = X_raw[1:, :].astype(float)  # peers rows
                    if Xp_raw.shape[0] == 0:
                        raise ValueError("No peer data available for PCA projection")

                    # features x peers
                    Xp_cols = Xp_raw.T
                    # Strict LaTeX projection in feature space
                    y_hat, _details = pca_surface_project(
                        peers_feature_matrix=Xp_cols,
                        target_feature_vector=y_vec,
                        k=config.pca_k,
                        energy=config.pca_energy,
                    )

                    # Convert reconstructed target y_hat into peer weights using your SVD-LS + nonneg + simplex
                    # (standardize peers and y_hat consistently inside pca_regress_weights)
                    w = pca_regress_weights(Xp_raw, y_hat, k=config.pca_k, nonneg=True, energy=config.pca_energy)

                    import pandas as pd
                    idx = [p for p in peers_list if p in tick_order[1:]]
                    ser = pd.Series(w, index=idx).clip(lower=0.0)
                    ssum = float(ser.sum())
                    if not np.isfinite(ssum) or ssum <= 0:
                        # Final fallback to market-mode PC1
                        try:
                            w_m = pca_market_weights(Xp_raw)
                            ser = pd.Series(w_m, index=idx).clip(lower=0.0)
                            ssum = float(ser.sum())
                        except Exception:
                            ser = pd.Series(0.0, index=idx)
                            ssum = 0.0
                    ser = ser / ssum if ssum > 0 else ser
                    return ser.reindex(peers_list).fillna(0.0)

                # --- SURFACE default: keep your old factorization weights ---
                return pca_weights_from_feature_df(feature_df, target, peers_list, k=config.pca_k, nonneg=True)
            
            # --- UNDERLYING_PX: LaTeX factor path (opt-in) ---
            if config.feature_set == FeatureSet.UNDERLYING_PX and config.pca_ul_via_factors:
                # feature_df here is rows=tickers, cols=time from underlying_returns_matrix()
                ser = _ul_pca_factor_weights(
                    feature_df, target, peers_list,
                    k=config.pca_ul_k, energy=config.pca_ul_energy
                )
                if ser.empty:
                    return equal_weights(peers_list)
                # finalize, keep original peer order
                return ser.reindex(peers_list).fillna(0.0)

            # --- UNDERLYING_PX default: keep your existing factorization weights ---
            if config.feature_set == FeatureSet.UNDERLYING_PX:
                return pca_weights_from_feature_df(feature_df, target, peers_list, k=config.pca_k, nonneg=True)
                raise ValueError(f"Unsupported method: {config.method}")
        except Exception:
            # Prefer market-mode fallback over equal if peers present
            try:
                Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
                if Xp.size > 0:
                    w_m = pca_market_weights(Xp)
                    import pandas as pd
                    ser = pd.Series(w_m, index=[p for p in peers_list if p in feature_df.index]).clip(lower=0.0)
                    ssum = float(ser.sum())
                    ser = ser / ssum if ssum > 0 else ser
                    return ser.reindex(peers_list).fillna(0.0)
            except Exception:
                pass
            return equal_weights(peers_list)

    def _build_feature_matrix(self, target: str, peers_list: list[str], asof: Optional[str], config: WeightConfig):
        tickers = [target] + peers_list
        if config.feature_set == FeatureSet.ATM:
            atm_df, _, _ = atm_feature_matrix(
                tickers, asof, config.pillars_days, atm_band=config.atm_band, tol_days=config.atm_tol_days
            )
            return atm_df
        if config.feature_set == FeatureSet.ATM_RANKS:
            atm_df, _ = _atm_rank_feature_matrix(tickers, asof, max_expiries=config.max_expiries, atm_band=config.atm_band)
            return atm_df
        if config.feature_set in (FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            grids, X, names = surface_feature_matrix(tickers, asof, tenors=config.tenors, mny_bins=config.mny_bins)
            return pd.DataFrame(X, index=list(grids.keys()), columns=names)
        if config.feature_set == FeatureSet.UNDERLYING_PX:
            df = underlying_returns_matrix(tickers)
            if df.shape[1] < 2:
                return None
            return df
        return None

_weight_computer = UnifiedWeightComputer()

def compute_unified_weights(target: str, peers: Iterable[str], mode: Union[str, WeightConfig], **kwargs) -> pd.Series:
    if isinstance(mode, str):
        cfg = WeightConfig.from_legacy_mode(mode, **kwargs)
    else:
        cfg = mode
    
    # Try cached computation first
    try:
        from analysis.model_params_logger import compute_or_load
        
        # Create cache payload
        payload = {
            "target": target.upper(),
            "peers": sorted([p.upper() for p in peers]),
            "method": cfg.method.value,
            "feature_set": cfg.feature_set.value,
            "pillars_days": cfg.pillars_days,
            "tenors": cfg.tenors,
            "mny_bins": cfg.mny_bins,
            "clip_negative": cfg.clip_negative,
            "power": cfg.power,
            "asof": cfg.asof,
            "atm_band": cfg.atm_band,
            "atm_tol_days": cfg.atm_tol_days,
            "max_expiries": cfg.max_expiries,
            "pca_project_surface": cfg.pca_project_surface,
            "pca_energy": cfg.pca_energy,
            "pca_k": cfg.pca_k,
            "pca_ul_via_factors": cfg.pca_ul_via_factors,
            "pca_ul_energy": cfg.pca_ul_energy,
            "pca_ul_k": cfg.pca_ul_k,
        }
        
        def _builder():
            return _weight_computer.compute_weights(target, peers, cfg)
        
        return compute_or_load("weights", payload, _builder, ttl_sec=1800)  # 30min cache
    except Exception:
        # Fallback to direct computation if caching fails
        return _weight_computer.compute_weights(target, peers, cfg)

def compute_peer_weights_unified(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_iv_atm",
    asof: str | None = None,
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tenor_days: Iterable[int] = (7, 30, 60, 90, 180, 365),
    mny_bins: Tuple[Tuple[float, float], ...] = ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25)),
    max_expiries: int = 6,
):
    return compute_unified_weights(
        target=target,
        peers=peers,
        mode=weight_mode,
        asof=asof,
        pillars_days=tuple(pillar_days),
        tenors=tuple(tenor_days),
        mny_bins=tuple(mny_bins),
        max_expiries=max_expiries,
    )

def normalize(w: pd.Series, peers: Iterable[str]) -> pd.Series | None:
    if w is None or w.empty:
        return None
    w = w.dropna().astype(float)
    w = w[w.index.isin(peers)]
    if w.empty or not np.isfinite(w.to_numpy(dtype=float)).any():
        return None
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        return None
    return (w / s).reindex(peers).fillna(0.0).astype(float)

