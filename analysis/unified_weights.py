"""
Unified Weight Computation System

This module provides a single, consistent interface for computing portfolio weights
across all methods (correlation, PCA, cosine, equal) and all features (ATM, surface, underlying).

Replaces the fragmented weight computation across multiple modules.
Missing data now raises explicit exceptions instead of silently returning
equal-weight fallbacks so callers can handle data issues upstream.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from analysis.analysis_pipeline import get_smile_slice, available_dates


class WeightMethod(Enum):
    """Supported weighting methods."""
    CORRELATION = "corr"
    PCA = "pca"
    COSINE = "cosine"
    EQUAL = "equal"
    OPEN_INTEREST = "oi"


class FeatureSet(Enum):
    """Supported feature sets for weight computation."""
    ATM = "atm"                    # ATM volatility across pillars
    SURFACE = "surface"            # Full volatility surface
    SURFACE_VECTOR = "surface_vector"  # Flattened surface grid
    UNDERLYING_PX = "underlying_px"    # Underlying price returns
    UNDERLYING_VOL = "underlying_vol"  # Underlying realized volatility


@dataclass
class WeightConfig:
    """Configuration for weight computation."""
    method: WeightMethod
    feature_set: FeatureSet
    pillars_days: Tuple[int, ...] = (7, 30, 60, 90)
    tenors: Tuple[int, ...] = (7, 30, 60, 90, 180, 365)
    mny_bins: Tuple[Tuple[float, float], ...] = ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25))
    clip_negative: bool = True
    power: float = 1.0
    asof: Optional[str] = None
    
    @classmethod
    def from_legacy_mode(cls, mode: str, **kwargs) -> "WeightConfig":
        """Create config from legacy weight_mode strings."""
        mode = mode.lower().strip()
        
        # Handle compound modes like "corr_iv_atm", "pca_surface", etc.
        # Map method strings
        method_map = {
            "corr": WeightMethod.CORRELATION,
            "correlation": WeightMethod.CORRELATION,
            "pca": WeightMethod.PCA,
            "cosine": WeightMethod.COSINE,
            "equal": WeightMethod.EQUAL,
            "oi": WeightMethod.OPEN_INTEREST,
            "open_interest": WeightMethod.OPEN_INTEREST,
            "iv": WeightMethod.CORRELATION,  # Legacy fallback
        }

        # Map feature strings
        feature_map = {
            "atm": FeatureSet.ATM,
            "iv_atm": FeatureSet.ATM,
            "surface": FeatureSet.SURFACE,
            "surface_vector": FeatureSet.SURFACE_VECTOR,
            "surface_grid": FeatureSet.SURFACE_VECTOR,
            "underlying": FeatureSet.UNDERLYING_PX,
            "underlying_px": FeatureSet.UNDERLYING_PX,
            "ul": FeatureSet.UNDERLYING_PX,
            "ul_px": FeatureSet.UNDERLYING_PX,
            "underlying_vol": FeatureSet.UNDERLYING_VOL,
            "ul_vol": FeatureSet.UNDERLYING_VOL,
        }

        # Allow specifying feature set directly (e.g., "ul", "ul_vol")
        if mode in feature_map:
            method_str = "corr"
            feature_str = mode
        elif "_" in mode:
            parts = mode.split("_")
            method_str = parts[0]
            feature_str = "_".join(parts[1:])
        else:
            method_str = mode
            feature_str = "atm"  # default

        method = method_map.get(method_str, WeightMethod.CORRELATION)
        feature_set = feature_map.get(feature_str, FeatureSet.ATM)

        return cls(method=method, feature_set=feature_set, **kwargs)


class UnifiedWeightComputer:
    """Unified weight computation engine."""
    
    def __init__(self):
        self._feature_cache: Dict[str, pd.DataFrame] = {}
    
    def _choose_asof(self, target: str, peers: list[str], config: WeightConfig) -> str:
        """
        Choose a robust asof date for the requested feature set:
        - SURFACE/SURFACE_VECTOR: pick the most recent date that exists for ≥1 peer
          and ideally for several (mode of available dates).
        - Others: keep existing behavior (most-recent date for target).
        """
        from analysis.syntheticETFBuilder import build_surface_grids
        from analysis.analysis_pipeline import available_dates, get_most_recent_date_global

        # If caller supplied one, honor it
        if config.asof:
            return config.asof

        if config.feature_set in (FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            # Build minimal grids for all tickers and look at date coverage
            tickers = [target] + peers
            grids = build_surface_grids(tickers=tickers)  # fast, uses DB
            # collect all candidate dates and pick the most common (mode), then latest
            counts = {}
            for t, date_map in grids.items():
                for d in date_map.keys():
                    counts[d] = counts.get(d, 0) + 1
            if counts:
                best = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[-1][0]
                return pd.Timestamp(best).strftime("%Y-%m-%d")

            # Fallback: global most recent in DB
            d = get_most_recent_date_global()
            if d:
                return d

        # Default path (ATM/UL, etc.): keep prior logic
        dates = available_dates(ticker=target, most_recent_only=True)
        if dates:
            return dates[0]
        # last ditch: any global date
        from analysis.analysis_pipeline import get_most_recent_date_global
        d = get_most_recent_date_global()
        if d:
            return d
        raise ValueError(f"No available data to select asof for {target}/{peers}")

    def compute_weights(
        self,
        target: str,
        peers: Iterable[str],
        config: WeightConfig,
    ) -> pd.Series:
        """
        Compute portfolio weights using specified method and feature set.
        
        Returns:
            pd.Series with peer weights (normalized to sum to 1.0)
        """
        target = target.upper()
        peers_list = [p.upper() for p in peers]
        
        if not peers_list:
            return pd.Series(dtype=float)
        
        # Handle equal weights (no feature computation needed)
        if config.method == WeightMethod.EQUAL:
            weight = 1.0 / len(peers_list)
            return pd.Series(weight, index=peers_list, dtype=float)

        if config.method == WeightMethod.OPEN_INTEREST:
            return self._open_interest_weights(peers_list, config.asof)
        
        # Get as-of date (robust)
        asof = self._choose_asof(target, peers_list, config)
        
        # Build feature matrix
        feature_df = self._build_feature_matrix(
            target, peers_list, asof, config
        )

        if feature_df is None or feature_df.empty:
            raise ValueError(
                "Feature matrix is empty; cannot compute portfolio weights"
            )

        # Compute weights using specified method
        return self._compute_weights_from_features(
            feature_df, target, peers_list, config
        )
    
    def _build_feature_matrix(
        self,
        target: str,
        peers_list: list[str],
        asof: str,
        config: WeightConfig,
    ) -> Optional[pd.DataFrame]:
        """Build feature matrix for weight computation."""
        tickers = [target] + peers_list
        
        if config.feature_set == FeatureSet.ATM:
            return self._build_atm_features(tickers, asof, config)
        elif config.feature_set in (FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            return self._build_surface_features(tickers, asof, config)
        elif config.feature_set == FeatureSet.UNDERLYING_PX:
            return self._build_underlying_px_features(tickers)
        elif config.feature_set == FeatureSet.UNDERLYING_VOL:
            return self._build_underlying_vol_features(tickers)
        else:
            raise ValueError(f"Unsupported feature set: {config.feature_set}")
    
    def _build_atm_features(
        self, tickers: list[str], asof: str, config: WeightConfig
    ) -> Optional[pd.DataFrame]:
        """Build ATM volatility feature matrix using pillar-free approach."""
        from analysis.correlation_utils import compute_atm_corr_pillar_free
        
        atm_df, _ = compute_atm_corr_pillar_free(
            get_smile_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=6,
            atm_band=0.05,
        )
        return atm_df
    
    def _build_surface_features(
        self, tickers: list[str], asof: str, config: WeightConfig
    ) -> Optional[pd.DataFrame]:
        """Build surface grid feature matrix."""
        from analysis.beta_builder import surface_feature_matrix
        
        grids, X, names = surface_feature_matrix(
            tickers, asof, tenors=config.tenors, mny_bins=config.mny_bins
        )
        return pd.DataFrame(X, index=list(grids.keys()), columns=names)
    
    def _build_underlying_px_features(self, tickers: list[str]) -> Optional[pd.DataFrame]:
        """Build underlying price return features."""
        from analysis.beta_builder import _underlying_log_returns
        from data.db_utils import get_conn

        ret = _underlying_log_returns(get_conn)
        if ret.empty:
            return None

        subset = ret[[c for c in tickers if c in ret.columns]]
        subset = subset.dropna(how="all")
        if subset.shape[0] < 2:
            return None
        return subset.T if not subset.empty else None

    def _build_underlying_vol_features(self, tickers: list[str]) -> Optional[pd.DataFrame]:
        """Build underlying volatility features."""
        from analysis.beta_builder import _underlying_vol_series
        from data.db_utils import get_conn

        vol = _underlying_vol_series(get_conn, window=21, min_obs=10)
        if vol.empty:
            return None

        subset = vol[[c for c in tickers if c in vol.columns]]
        subset = subset.dropna(how="all")
        if subset.shape[0] < 2:
            return None
        return subset.T if not subset.empty else None

    def _open_interest_weights(self, peers_list: list[str], asof: Optional[str]) -> pd.Series:
        """Compute weights proportional to total open interest for each peer."""
        from data.db_utils import get_conn

        if not peers_list:
            return pd.Series(dtype=float)
        conn = get_conn()
        if asof is None:
            # Fallback to most recent date available for each ticker
            # but for simplicity we'll use MAX(asof_date) overall
            asof_row = conn.execute(
                "SELECT MAX(asof_date) FROM options_quotes WHERE ticker IN ({})".format(
                    ",".join("?" * len(peers_list))
                ),
                peers_list,
            ).fetchone()
            asof = asof_row[0] if asof_row and asof_row[0] else None
            if asof is None:
                return self._fallback_weights(peers_list)

        query = (
            "SELECT ticker, SUM(open_interest) AS oi FROM options_quotes "
            "WHERE asof_date = ? AND ticker IN ({}) GROUP BY ticker"
        ).format(",".join("?" * len(peers_list)))
        df = pd.read_sql_query(query, conn, params=[asof] + peers_list)
        if df.empty:
            return self._fallback_weights(peers_list)
        series = pd.Series(df["oi"].values, index=df["ticker"].str.upper())
        total = series.sum()
        if total <= 0:
            return self._fallback_weights(peers_list)
        weights = series / total
        # Ensure all peers present; missing ones get zero weight
        return weights.reindex([p.upper() for p in peers_list]).fillna(0.0)
    
    def _compute_weights_from_features(
        self,
        feature_df: pd.DataFrame,
        target: str,
        peers_list: list[str],
        config: WeightConfig,
    ) -> pd.Series:
        """Compute weights from feature matrix using specified method."""
        if target not in feature_df.index:
            raise ValueError(f"Target {target} missing from feature matrix")

        if not any(p in feature_df.index for p in peers_list):
            raise ValueError("No peer data available in feature matrix")

        if config.method == WeightMethod.CORRELATION:
            return self._correlation_weights(feature_df, target, peers_list, config)
        elif config.method == WeightMethod.PCA:
            return self._pca_weights(feature_df, target, peers_list, config)
        elif config.method == WeightMethod.COSINE:
            return self._cosine_weights(feature_df, target, peers_list, config)
        else:
            raise ValueError(f"Unsupported method: {config.method}")
    
    def _correlation_weights(
        self, feature_df: pd.DataFrame, target: str, peers_list: list[str], config: WeightConfig
    ) -> pd.Series:
        """Compute correlation-based weights without silent fallback."""
        import numpy as np

        target = target.upper()
        peers_list = [p.upper() for p in peers_list]

        corr_df = feature_df.T.corr()
        s = corr_df.reindex(index=peers_list, columns=[target]).iloc[:, 0]
        s = s.apply(pd.to_numeric, errors="coerce")
        if config.clip_negative:
            s = s.clip(lower=0.0)
        if config.power is not None and float(config.power) != 1.0:
            s = s.pow(float(config.power))

        total = float(s.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("correlation weights sum to zero")

        return (s / total).reindex(peers_list).fillna(0.0)
    
    def _pca_weights(
        self, feature_df: pd.DataFrame, target: str, peers_list: list[str], config: WeightConfig
    ) -> pd.Series:
        """Compute PCA-based weights."""
        from analysis.beta_builder import pca_regress_weights, _impute_col_median
        import numpy as np
        
        if target not in feature_df.index:
            raise ValueError(f"Target {target} missing from feature matrix")

        y = _impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
        Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)

        if Xp.size == 0:
            raise ValueError("No peer data available for PCA weighting")

        w = pca_regress_weights(Xp, y, k=None, nonneg=True)
        ser = pd.Series(w, index=[p for p in peers_list if p in feature_df.index])
        ser = ser.clip(lower=0.0)
        s = float(ser.sum())
        ser = ser / s if s > 0 else ser
        return ser.reindex(peers_list).fillna(0.0)
    
    def _cosine_weights(
        self, feature_df: pd.DataFrame, target: str, peers_list: list[str], config: WeightConfig
    ) -> pd.Series:
        """Compute cosine similarity weights."""
        from analysis.beta_builder import cosine_similarity_weights_from_matrix

        return cosine_similarity_weights_from_matrix(
            feature_df, target, peers_list,
            clip_negative=config.clip_negative,
            power=config.power
        )


# Global instance
_weight_computer = UnifiedWeightComputer()


def compute_unified_weights(
    target: str,
    peers: Iterable[str],
    mode: Union[str, WeightConfig],
    **kwargs
) -> pd.Series:
    """
    Main API for computing portfolio weights.
    
    Args:
        target: Target asset
        peers: Peer assets  
        mode: Weight mode string (e.g., "corr_atm", "pca_surface") or WeightConfig
        **kwargs: Additional config parameters
    
    Returns:
        pd.Series with normalized weights
    """
    if isinstance(mode, str):
        config = WeightConfig.from_legacy_mode(mode, **kwargs)
    else:
        config = mode
    
    return _weight_computer.compute_weights(target, peers, config)


# Legacy compatibility aliases
def compute_peer_weights_unified(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_atm",
    asof: str | None = None,
    pillar_days: Iterable[int] = (7, 30, 60, 90),
    tenor_days: Iterable[int] = (7, 30, 60, 90, 180, 365),
    mny_bins: Tuple[Tuple[float, float], ...] = ((0.8, 0.9), (0.95, 1.05), (1.1, 1.25)),
) -> pd.Series:
    """Legacy-compatible API wrapper."""
    return compute_unified_weights(
        target=target,
        peers=peers,
        mode=weight_mode,
        asof=asof,
        pillars_days=pillar_days,
        tenors=tenor_days,
        mny_bins=mny_bins,
    )
