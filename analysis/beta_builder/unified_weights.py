"""Unified weight engine and feature builders."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple, Union, List
import logging

import numpy as np
import pandas as pd

from .utils import impute_col_median as _impute_col_median, zscore_cols as _zscore_cols
from .correlation import corr_weights_from_matrix
from .cosine import cosine_similarity_weights_from_matrix
from .pca import pca_regress_weights
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
    from analysis.beta_builder.correlation import compute_atm_corr_pillar_free
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
) -> Tuple[Dict[str, Dict[pd.Timestamp, pd.DataFrame]], np.ndarray, List[str]]:
    from analysis.syntheticETFBuilder import build_surface_grids
    req = [t.upper() for t in tickers]
    grids = build_surface_grids(
        tickers=req,
        tenors=tenors,
        mny_bins=mny_bins,
        use_atm_only=False,
    )
    feats: list[np.ndarray] = []
    ok: list[str] = []
    feat_names: list[str] | None = None
    for t in req:
        if t not in grids or asof not in grids[t]:
            continue
        df = grids[t][asof]
        arr = df.to_numpy(float).T.reshape(-1)
        if feat_names is None:
            feat_names = [f"T{c}_{r}" for c in df.columns for r in df.index]
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
                return corr_weights_from_matrix(feature_df, target, peers_list,
                    clip_negative=config.clip_negative, power=config.power)
            if config.method == WeightMethod.COSINE:
                return cosine_similarity_weights_from_matrix(feature_df, target, peers_list,
                    clip_negative=config.clip_negative, power=config.power)
            if config.method == WeightMethod.PCA:
                y = _impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
                Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
                if Xp.size == 0:
                    raise ValueError("No peer data available for PCA weighting")
                w = pca_regress_weights(Xp, y, k=None, nonneg=True)
                import pandas as pd
                ser = pd.Series(w, index=[p for p in peers_list if p in feature_df.index]).clip(lower=0.0)
                ssum = float(ser.sum())
                ser = ser / ssum if ssum > 0 else ser
                return ser.reindex(peers_list).fillna(0.0)
            raise ValueError(f"Unsupported method: {config.method}")
        except Exception:
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
