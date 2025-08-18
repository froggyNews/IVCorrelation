"""
Unified Weight Computation System

Single, consistent interface for computing portfolio weights
across methods (correlation, PCA, cosine, equal, open-interest)
and features (ATM, surface, underlying returns).

- Options-based features (ATM, surface*) are single as-of snapshots.
- Underlying returns (UL) use time-series panels (no as-of required).
- Open-interest is a method (no feature matrix).

This module also centralizes *feature building* helpers so other modules
can reuse them without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple, Union, List
import logging

import numpy as np
import pandas as pd

# Delayed imports to avoid circular dependencies
# from analysis.analysis_pipeline import get_smile_slice, available_dates
from analysis.pillars import build_atm_matrix, DEFAULT_PILLARS_DAYS
# from analysis.syntheticETFBuilder import build_surface_grids
# from analysis.correlation_utils import compute_atm_corr_pillar_free

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class WeightMethod(Enum):
    CORRELATION = "corr"
    PCA = "pca"
    COSINE = "cosine"
    EQUAL = "equal"
    OPEN_INTEREST = "oi"   # implemented via _open_interest_weights

class FeatureSet(Enum):
    ATM = "iv_atm"                    # options ATM features
    SURFACE = "surface"               # options surface (flattened grid)
    SURFACE_VECTOR = "surface_grid"   # alias of SURFACE
    UNDERLYING_PX = "ul"              # underlying price returns (time-series)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
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

    @classmethod
    def from_legacy_mode(cls, mode: str, **kwargs) -> "WeightConfig":
        """
        Accepts 'corr_iv_atm', 'pca_surface_grid', 'ul', 'equal', 'oi', etc.
        """
        mode = (mode or "").lower().strip()

        method_map = {
            "corr": WeightMethod.CORRELATION,
            "correlation": WeightMethod.CORRELATION,
            "pca": WeightMethod.PCA,
            "cosine": WeightMethod.COSINE,
            "equal": WeightMethod.EQUAL,
            "oi": WeightMethod.OPEN_INTEREST,
            "open_interest": WeightMethod.OPEN_INTEREST,
            "iv": WeightMethod.CORRELATION,  # legacy
        }
        feature_map = {
            "atm": FeatureSet.ATM,
            "iv_atm": FeatureSet.ATM,
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

# -----------------------------------------------------------------------------
# Centralized feature builders (reusable; no circular imports)
# -----------------------------------------------------------------------------
def _impute_col_median(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    med = np.nanmedian(X, axis=0, keepdims=True)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.broadcast_to(med, X.shape)[mask]
    return X

def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
    return (X - mu) / sd, mu, sd


def atm_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    *,
    atm_band: float = 0.08,
    tol_days: float = 10.0,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Rows=tickers, cols=pillars (days). Values=ATM IVs for a single as-of date."""
    from analysis.analysis_pipeline import get_smile_slice  # Delayed import
    
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


def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    *,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    standardize: bool = True,
) -> Tuple[Dict[str, Dict[pd.Timestamp, pd.DataFrame]], np.ndarray, List[str]]:
    """Rows=tickers, cols=flattened (tenor × moneyness) grid for a single as-of date."""
    from analysis.syntheticETFBuilder import build_surface_grids  # Delayed import
    
    grids = build_surface_grids(
        tickers=[t.upper() for t in tickers],
        tenors=tenors,
        mny_bins=mny_bins,
        use_atm_only=False,
    )
    feats: list[np.ndarray] = []
    ok: list[str] = []
    feat_names: list[str] | None = None

    for t in [t.upper() for t in tickers]:
        if t not in grids or asof not in grids[t]:
            continue
        df = grids[t][asof]  # index=mny labels, columns=tenor (days)
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
    """Ticker×time matrix of log returns (rows=tickers). Pairwise-ready (keeps partial rows)."""
    from data.db_utils import get_conn
    
    # Auto-update underlying price data if needed
    tickers_set = set(str(t).upper() for t in tickers)
    logger.info(f"Checking underlying price data for {len(tickers_set)} tickers...")
    
    try:
        from data.data_pipeline import ensure_underlying_price_data
        if not ensure_underlying_price_data(tickers_set):
            logger.warning("Failed to ensure underlying price data is available")
    except Exception as e:
        logger.warning(f"Could not auto-update underlying prices: {e}")
    
    conn = get_conn()
    # prefer dedicated table; fallback to options_quotes spot
    try:
        df = pd.read_sql_query(
            "SELECT asof_date, ticker, close FROM underlying_prices", conn
        )
        logger.debug(f"Loaded {len(df)} underlying price records from dedicated table")
    except Exception:
        df = pd.DataFrame()
        logger.debug("No underlying_prices table, trying fallback")
        
    if df.empty:
        df = pd.read_sql_query(
            "SELECT asof_date, ticker, spot AS close FROM options_quotes", conn
        )
        logger.debug(f"Loaded {len(df)} price records from options_quotes fallback")
    if df.empty:
        return pd.DataFrame()

    px = (
        df.groupby(["asof_date", "ticker"])["close"]
        .median()
        .unstack("ticker")
        .sort_index()
    )
    ret = np.log(px / px.shift(1))
    # drop only all-NaN rows; keep partial rows for pairwise stats
    ret = ret.dropna(how="all")
    # return in rows=tickers layout
    return ret.T


# -----------------------------------------------------------------------------
# Public similarity/weight utilities (reused by beta_builder)
# -----------------------------------------------------------------------------
def cosine_similarity_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    target = target.upper()
    peers = [p.upper() for p in peers]
    if target not in feature_df.index:
        raise ValueError(f"target {target} not in feature matrix")

    df = feature_df.apply(pd.to_numeric, errors="coerce")
    X = _impute_col_median(df.to_numpy(float))
    tickers = list(df.index)
    t_idx = tickers.index(target)
    t_vec = X[t_idx]
    t_norm = float(np.linalg.norm(t_vec))

    sims: dict[str, float] = {}
    for i, peer in enumerate(tickers):
        if peer == target or peer not in peers:
            continue
        p_vec = X[i]
        denom = t_norm * float(np.linalg.norm(p_vec))
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


def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    target = target.upper()
    peers = [p.upper() for p in peers]
    corr_df = feature_df.T.corr()
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0]
    s = s.apply(pd.to_numeric, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("correlation weights sum to zero")
    return (s / total).reindex(peers).fillna(0.0)


def pca_regress_weights(
    X_peers: np.ndarray, y_target: np.ndarray, k: Optional[int] = None, *, nonneg: bool = True
) -> np.ndarray:
    """min_w || Xᵀ w − y || via SVD truncation."""
    Z, _, _ = _zscore_cols(_impute_col_median(X_peers))
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    if k is None or k <= 0 or k > len(s):
        k = len(s)
    Uk, sk, Vk = U[:, :k], s[:k], Vt[:k, :].T
    w = Uk @ ((y_target @ Vk) / np.where(sk > 1e-12, sk, 1.0))
    if nonneg:
        w = np.clip(w, 0.0, None)
    ssum = float(w.sum())
    return w / ssum if ssum > 0 else np.full_like(w, 1.0 / max(len(w), 1))


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
class UnifiedWeightComputer:
    """Unified weight computation engine with centralized feature building."""

    def _choose_asof(self, target: str, peers: list[str], config: WeightConfig) -> Optional[str]:
        """Robust as-of for options features; UL ignores."""
        if config.asof:
            return config.asof
        if config.feature_set in (FeatureSet.ATM, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            from data.db_utils import get_conn
            tickers = [target] + peers
            conn = get_conn()
            q = (
                "SELECT asof_date, COUNT(DISTINCT ticker) AS n FROM options_quotes "
                f"WHERE ticker IN ({','.join('?' * len(tickers))}) GROUP BY asof_date"
            )
            df = pd.read_sql_query(q, conn, params=[t.upper() for t in tickers])
            if not df.empty:
                df["asof_date"] = pd.to_datetime(df["asof_date"])
                df = df.sort_values(["n", "asof_date"])  # ascending, tie -> latest
                best_row = df.iloc[-1]
                best_date = pd.Timestamp(best_row["asof_date"])
                logger.debug(
                    "_choose_asof picked %s with %d/%d ticker coverage",
                    best_date,
                    int(best_row["n"]),
                    len(tickers),
                )
                return best_date.strftime("%Y-%m-%d")
            from analysis.analysis_pipeline import get_most_recent_date_global, available_dates
            d = get_most_recent_date_global()
            if d:
                return d
            dates = available_dates(ticker=target, most_recent_only=True)
            if dates:
                return dates[0]
        return None  # UL or if nothing found

    def _log_option_counts(self, tickers: list[str], asof: Optional[str], atm_df: Optional[pd.DataFrame]) -> None:
        if asof is None:
            return
        from data.db_utils import get_conn

        conn = get_conn()
        q = (
            "SELECT ticker, COUNT(*) AS n FROM options_quotes WHERE asof_date = ? "
            f"AND ticker IN ({','.join('?' * len(tickers))}) GROUP BY ticker"
        )
        df = pd.read_sql_query(q, conn, params=[asof] + [t.upper() for t in tickers])
        counts = {row["ticker"].upper(): int(row["n"]) for _, row in df.iterrows()}
        for t in tickers:
            total = counts.get(t.upper(), 0)
            atm_rows = 0
            if atm_df is not None and t.upper() in atm_df.index:
                atm_rows = int(atm_df.loc[t.upper()].count())
            logger.debug("asof %s %s: quotes=%d, atm_rows=%d", asof, t.upper(), total, atm_rows)

    # ---- public API ----
    def compute_weights(
        self,
        target: str,
        peers: Iterable[str],
        config: WeightConfig,
    ) -> pd.Series:
        target = (target or "").upper()
        peers_list = [p.upper() for p in peers]
        if not peers_list:
            return pd.Series(dtype=float)

        # Equal weights
        if config.method == WeightMethod.EQUAL:
            w = 1.0 / len(peers_list)
            return pd.Series(w, index=peers_list, dtype=float)

        # Open interest method
        if config.method == WeightMethod.OPEN_INTEREST:
            return self._open_interest_weights(peers_list, config.asof)

        # Pick as-of only for options features
        asof = None
        if config.feature_set in (FeatureSet.ATM, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            asof = self._choose_asof(target, peers_list, config)
            if asof is None:
                raise ValueError("no surface/ATM date available to build features")

        # Build features
        feature_df = self._build_feature_matrix(target, peers_list, asof, config)
        if feature_df is None or feature_df.empty:
            if config.feature_set != FeatureSet.UNDERLYING_PX:
                logger.warning(
                    "Feature matrix for %s empty; falling back to underlying returns",
                    config.feature_set,
                )
                ul_cfg = WeightConfig(
                    method=config.method,
                    feature_set=FeatureSet.UNDERLYING_PX,
                    clip_negative=config.clip_negative,
                    power=config.power,
                )
                feature_df = self._build_feature_matrix(target, peers_list, None, ul_cfg)
                if feature_df is None or feature_df.empty:
                    logger.warning("Underlying returns also unavailable; using equal weights")
                    return self._fallback_equal(peers_list)
                config = ul_cfg
            else:
                logger.warning("Underlying return features empty; using equal weights")
                return self._fallback_equal(peers_list)

        # Dispatch by method
        try:
            if config.method == WeightMethod.CORRELATION:
                return corr_weights_from_matrix(
                    feature_df, target, peers_list,
                    clip_negative=config.clip_negative,
                    power=config.power,
                )
            if config.method == WeightMethod.COSINE:
                return cosine_similarity_weights_from_matrix(
                    feature_df, target, peers_list,
                    clip_negative=config.clip_negative,
                    power=config.power,
                )
            if config.method == WeightMethod.PCA:
                # PCA regression of target on peers
                y = _impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
                Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
                if Xp.size == 0:
                    raise ValueError("No peer data available for PCA weighting")
                w = pca_regress_weights(Xp, y, k=None, nonneg=True)
                ser = pd.Series(w, index=[p for p in peers_list if p in feature_df.index]).clip(lower=0.0)
                ssum = float(ser.sum())
                ser = ser / ssum if ssum > 0 else ser
                return ser.reindex(peers_list).fillna(0.0)
            raise ValueError(f"Unsupported method: {config.method}")
        except Exception as e:
            logger.warning("Weight computation failed (%s); using equal weights", e)
            return self._fallback_equal(peers_list)

    # ---- feature assembly ----
    def _build_feature_matrix(
        self,
        target: str,
        peers_list: list[str],
        asof: Optional[str],
        config: WeightConfig,
    ) -> Optional[pd.DataFrame]:
        tickers = [target] + peers_list
        if config.feature_set == FeatureSet.ATM:
            atm_df, _, _ = atm_feature_matrix(
                tickers,
                asof,
                config.pillars_days,
                atm_band=config.atm_band,
                tol_days=config.atm_tol_days,
            )
            self._log_option_counts(tickers, asof, atm_df)
            return atm_df
        if config.feature_set in (FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            grids, X, names = surface_feature_matrix(
                tickers,
                asof,
                tenors=config.tenors,
                mny_bins=config.mny_bins,
            )
            self._log_option_counts(tickers, asof, None)
            return pd.DataFrame(X, index=list(grids.keys()), columns=names)
        if config.feature_set == FeatureSet.UNDERLYING_PX:
            df = underlying_returns_matrix(tickers)
            # Need at least two time rows (pairwise) to compute correlation
            if df.shape[1] < 2:
                return None
            return df
        return None

    # ---- OI method ----
    def _open_interest_weights(self, peers_list: list[str], asof: Optional[str]) -> pd.Series:
        from data.db_utils import get_conn
        if not peers_list:
            return pd.Series(dtype=float)
        conn = get_conn()
        # choose date if missing
        if asof is None:
            row = conn.execute(
                "SELECT MAX(asof_date) FROM options_quotes WHERE ticker IN ({})".format(
                    ",".join("?" * len(peers_list))
                ),
                peers_list,
            ).fetchone()
            asof = row[0] if row and row[0] else None
            if asof is None:
                return self._fallback_equal(peers_list)

        q = (
            "SELECT ticker, SUM(open_interest) AS oi FROM options_quotes "
            "WHERE asof_date = ? AND ticker IN ({}) GROUP BY ticker"
        ).format(",".join("?" * len(peers_list)))
        df = pd.read_sql_query(q, conn, params=[asof] + peers_list)
        if df.empty:
            return self._fallback_equal(peers_list)
        s = pd.Series(df["oi"].values, index=df["ticker"].str.upper())
        total = float(s.sum())
        if not np.isfinite(total) or total <= 0:
            return self._fallback_equal(peers_list)
        return (s / total).reindex(peers_list).fillna(0.0)

    @staticmethod
    def _fallback_equal(peers_list: list[str]) -> pd.Series:
        if not peers_list:
            return pd.Series(dtype=float)
        w = 1.0 / len(peers_list)
        return pd.Series(w, index=peers_list, dtype=float)


# Global instance
_weight_computer = UnifiedWeightComputer()

# -----------------------------------------------------------------------------
# Top-level API
# -----------------------------------------------------------------------------
def compute_unified_weights(
    target: str,
    peers: Iterable[str],
    mode: Union[str, WeightConfig],
    **kwargs,
) -> pd.Series:
    """
    Args
    ----
    target : str
    peers  : Iterable[str]
    mode   : weight mode string (e.g., "corr_iv_atm", "pca_surface_grid", "ul", "equal", "oi")
             or WeightConfig
    **kwargs : forwarded to WeightConfig.from_legacy_mode
    """
    if isinstance(mode, str):
        logger.info("Converting legacy mode: %r -> %r", mode, mode)
        cfg = WeightConfig.from_legacy_mode(mode, **kwargs)
        logger.info("Final mapping: method=%s, feature_set=%s", cfg.method, cfg.feature_set)
    else:
        cfg = mode
    return _weight_computer.compute_weights(target, peers, cfg)


# Legacy compatibility wrapper
def compute_peer_weights_unified(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_iv_atm",
    asof: str | None = None,
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tenor_days: Iterable[int] = (7, 30, 60, 90, 180, 365),
    mny_bins: Tuple[Tuple[float, float], ...] = ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25)),
) -> pd.Series:
    return compute_unified_weights(
        target=target,
        peers=peers,
        mode=weight_mode,
        asof=asof,
        pillars_days=tuple(pillar_days),
        tenors=tuple(tenor_days),
        mny_bins=tuple(mny_bins),
    )
