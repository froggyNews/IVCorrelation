"""
High-level Synthetic ETF construction utilities.

Leverages existing primitives:
- build_surface_grids / combine_surfaces (analysis.syntheticETFBuilder)
- peer_weights_from_correlations / pca_weights (analysis.correlation_builder / pca_builder)
- relative_value_atm_report_corrweighted (analysis.analysis_pipeline)
- sample_smile_curve / fit_smile_for (analysis.analysis_pipeline)

Provides:
- SyntheticETFBuilder: orchestrates weights, surfaces, synthetic surface, ATM RV.
- Convenience functions for command-line or programmatic usage.

Design Goals:
- Stateless outputs (pure DataFrames) + optional lightweight caching.
- Clean separation between "weighting strategy" and "surface assembly".
- Extensible (add factor models / custom weights later).

NOTE: For call/put separated surfaces you would extend build_surface_grids
      or add a new function. This initial implementation operates on the
      combined (current default) surface grids.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Dict, Optional, Tuple, Literal
import pandas as pd
import numpy as np
import os
import json
import time

from analysis.syntheticETFBuilder import (
    build_surface_grids,
    combine_surfaces,
    DEFAULT_TENORS,
    DEFAULT_MNY_BINS,
)
from analysis.analysis_pipeline import (
    relative_value_atm_report_corrweighted,
    available_dates,
    sample_smile_curve,
    get_smile_slice,
)
from analysis.beta_builder import peer_weights_from_correlations
from analysis.beta_builder import pca_weights, pca_weights_from_atm_matrix
from analysis.beta_builder import cosine_similarity_weights_from_atm_matrix
from analysis.analysis_pipeline import ingest_and_process, available_tickers
from analysis.analysis_pipeline import get_most_recent_date_global
from analysis.pillars import compute_atm_by_expiry, load_atm, build_atm_matrix

WeightMode = Literal["corr", "pca", "cosine", "equal", "custom"]


@dataclass
class SyntheticETFConfig:
    target: str
    peers: Iterable[str]
    pillar_days: Tuple[int, ...] = (7, 30, 60, 90)
    tenors: Tuple[int, ...] = DEFAULT_TENORS
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS
    tolerance_days: float = 7.0
    lookback: int = 60
    weight_mode: WeightMode = "corr"
    weight_power: float = 1.0
    clip_negative: bool = True
    use_atm_only_surface: bool = False
    cache_dir: Optional[str] = "data/cache_synth_etf"
    # If True we require surfaces for EVERY peer date to include a date in synthetic output
    # DISABLED: Always False to prevent date filtering
    strict_date_intersection: bool = False

    def ensure_cache(self):
        if self.cache_dir and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class SyntheticETFArtifacts:
    weights: pd.Series
    surfaces: Dict[str, Dict[str, pd.DataFrame]]
    synthetic_surfaces: Dict[str, pd.DataFrame]
    rv_metrics: pd.DataFrame
    meta: Dict[str, str] = field(default_factory=dict)


class SyntheticETFBuilder:
    def __init__(self, config: SyntheticETFConfig):
        self.cfg = config
        self.cfg.ensure_cache()
        self._weights: Optional[pd.Series] = None
        self._surfaces: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        self._synthetic_surfaces: Optional[Dict[str, pd.DataFrame]] = None
        self._rv: Optional[pd.DataFrame] = None

    # ----------------------
    # Weight Computation
    # ----------------------
    def compute_weights(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        def _custom() -> pd.Series:
            if not custom_weights:
                raise ValueError("custom weight_mode selected but no custom_weights supplied")
            w = pd.Series(custom_weights, dtype=float)
            return w / w.sum()

        def _equal() -> pd.Series:
            peers_list = list(self.cfg.peers)
            return pd.Series(
                1.0 / len(peers_list),
                index=peers_list,
                dtype=float,
                name="weight",
            )

        def _pca() -> pd.Series:
            # Get the most recent date for PCA computation
            dates = available_dates(ticker=self.cfg.target, most_recent_only=True)
            if not dates:
                raise ValueError(f"No dates available for target {self.cfg.target}")
            asof = dates[0]
            
            w = pca_weights(
                get_smile_slice=get_smile_slice,
                mode="pca_atm_market",  # Use the market-based PCA mode
                target=self.cfg.target,
                peers=self.cfg.peers,
                asof=asof,
                pillars_days=self.cfg.pillar_days,
            )
            if w.empty:
                raise ValueError("PCA weight computation returned empty series")
            return w

        def _cosine() -> pd.Series:
            w = cosine_weights_from_atm_matrix(
                target=self.cfg.target,
                peers=self.cfg.peers,
                pillar_days=self.cfg.pillar_days,
                tolerance_days=self.cfg.tolerance_days,
            )
            if w.empty:
                raise ValueError("Cosine similarity weight computation returned empty series")
            return w

        def _corr() -> pd.Series:
            w = peer_weights_from_correlations(
                benchmark=self.cfg.target,
                peers=self.cfg.peers,
                mode="iv_atm",
                pillar_days=self.cfg.pillar_days,
                tenor_days=self.cfg.tenors,
                mny_bins=self.cfg.mny_bins,
                clip_negative=self.cfg.clip_negative,
                power=self.cfg.weight_power,
            )
            if w.empty:
                raise ValueError("Correlation weights came back empty (insufficient data?)")
            return w

        dispatch = {
            "custom": _custom,
            "equal": _equal,
            "pca": _pca,
            "cosine": _cosine,
            "corr": _corr,
        }

        func = dispatch.get(self.cfg.weight_mode, _corr)
        w = func()
        self._weights = w
        return w

    # ----------------------
    # Surface Construction
    # ----------------------
    def build_surfaces(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        tickers = list({self.cfg.target, *self.cfg.peers})
        surfaces = build_surface_grids(
            tickers=tickers,
            tenors=self.cfg.tenors,
            mny_bins=self.cfg.mny_bins,
            use_atm_only=self.cfg.use_atm_only_surface,
        )
        self._surfaces = surfaces
        return surfaces

    def build_synthetic_surfaces(self) -> Dict[str, pd.DataFrame]:
        if self._weights is None:
            raise RuntimeError("Weights not computed yet. Call compute_weights() first.")
        if self._surfaces is None:
            raise RuntimeError("Surfaces not built yet. Call build_surfaces() first.")

        peer_surfaces = {
            p: self._surfaces[p]
            for p in self.cfg.peers
            if p in self._surfaces
        }
        synthetic = combine_surfaces(peer_surfaces, self._weights.to_dict())

        # Optionally restrict dates to intersection across all peer surfaces
        if self.cfg.strict_date_intersection:
            date_sets = [set(dates_dict.keys()) for dates_dict in peer_surfaces.values()]
            if date_sets:
                common = set.intersection(*date_sets)
                synthetic = {d: grid for d, grid in synthetic.items() if d in common}

        self._synthetic_surfaces = synthetic
        return synthetic

    # ----------------------
    # Relative Value (ATM)
    # ----------------------
    def compute_relative_value(self) -> pd.DataFrame:
        rv, w_used = relative_value_atm_report_corrweighted(
            target=self.cfg.target,
            peers=self.cfg.peers,
            mode="iv_atm",
            pillar_days=self.cfg.pillar_days,
            lookback=self.cfg.lookback,
            tolerance_days=self.cfg.tolerance_days,
        )
        self._rv = rv
        # w_used may differ from earlier weighting if internal weighting logic diverges;
        # we keep original in self._weights but attach w_used to metadata if needed.
        return rv

    # ----------------------
    # Merged Artifacts
    # ----------------------
    def build_all(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> SyntheticETFArtifacts:
        start = time.time()
        w = self.compute_weights(custom_weights=custom_weights)
        surfaces = self.build_surfaces()
        synth = self.build_synthetic_surfaces()
        rv = self.compute_relative_value()

        meta = {
            "target": self.cfg.target,
            "peers": ",".join(self.cfg.peers),
            "weight_mode": self.cfg.weight_mode,
            "lookback": str(self.cfg.lookback),
            "pillar_days": ",".join(map(str, self.cfg.pillar_days)),
            "tenors": ",".join(map(str, self.cfg.tenors)),
            "mny_bins": ";".join(f"{a}:{b}" for a, b in self.cfg.mny_bins),
            "tolerance_days": str(self.cfg.tolerance_days),
            "build_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "elapsed_sec": f"{time.time()-start:.2f}",
        }
        return SyntheticETFArtifacts(
            weights=w,
            surfaces=surfaces,
            synthetic_surfaces=synth,
            rv_metrics=rv,
            meta=meta,
        )

    # ----------------------
    # Export Helpers
    # ----------------------
    def export(self, artifacts: SyntheticETFArtifacts, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        # weights
        artifacts.weights.to_csv(os.path.join(out_dir, "weights.csv"), header=True)

        # meta
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(artifacts.meta, f, indent=2)

        # rv metrics
        artifacts.rv_metrics.to_csv(os.path.join(out_dir, "rv_metrics.csv"), index=False)

        # Surfaces: one folder per ticker
        surf_root = os.path.join(out_dir, "surfaces")
        os.makedirs(surf_root, exist_ok=True)
        for ticker, date_map in artifacts.surfaces.items():
            t_dir = os.path.join(surf_root, ticker)
            os.makedirs(t_dir, exist_ok=True)
            for d, df in date_map.items():
                df.to_csv(os.path.join(t_dir, f"{d}.csv"))

        # Synthetic surfaces
        syn_dir = os.path.join(out_dir, "synthetic")
        os.makedirs(syn_dir, exist_ok=True)
        for d, df in artifacts.synthetic_surfaces.items():
            df.to_csv(os.path.join(syn_dir, f"{d}.csv"))

    # ----------------------
    # Convenience Queries
    # ----------------------
    def latest_surface_pair(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """Return (target_surface, synthetic_surface, date) for most recent common date."""
        if self._surfaces is None or self._synthetic_surfaces is None:
            return None, None, None
        target_surfs = self._surfaces.get(self.cfg.target, {})
        if not target_surfs:
            return None, None, None
        dates_target = set(target_surfs.keys())
        dates_syn = set(self._synthetic_surfaces.keys())
        common = sorted(dates_target.intersection(dates_syn))
        if not common:
            return None, None, None
        d = common[-1]
        return target_surfs[d], self._synthetic_surfaces[d], d

    def sample_smile(self, T_days: float, model: str = "svi") -> pd.DataFrame:
        """Convenience wrapper for a smile at nearest expiry for latest date."""
        date_latest = get_most_recent_date_global()
        if date_latest is None:
            return pd.DataFrame()
        return sample_smile_curve(
            ticker=self.cfg.target,
            asof_date=date_latest,
            T_target_years=T_days / 365.25,
            model=model,
        )


# ----------------------
# Helper functions for weight computation
# ----------------------
def cosine_weights_from_atm_matrix(
    target: str,
    peers: Iterable[str],
    pillar_days: Tuple[int, ...] = (7, 30, 60, 90),
    tolerance_days: float = 7.0,
) -> pd.Series:
    """
    Compute cosine similarity weights using ATM vol matrix.
    
    This function builds an ATM matrix from the database and computes
    cosine similarity weights between target and peers.
    """
    from data.db_utils import get_conn
    from analysis.pillars import build_atm_matrix
    from analysis.analysis_pipeline import get_smile_slice
    
    # Get latest date for target
    dates = available_dates(ticker=target, most_recent_only=True)
    if not dates:
        peers_list = list(peers) if peers else []
        return pd.Series(1.0 / max(len(peers_list), 1), index=peers_list, dtype=float)
    
    asof = dates[0]
    tickers = [target] + list(peers)
    
    # Build ATM matrix
    atm_df, _ = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillar_days,
        tol_days=tolerance_days,
    )
    
    if atm_df.empty:
        peers_list = list(peers) if peers else []
        return pd.Series(1.0 / max(len(peers_list), 1), index=peers_list, dtype=float)
    
    # Use the cosine similarity function from beta_builder
    return cosine_similarity_weights_from_atm_matrix(
        atm_df=atm_df,
        target=target,
        peers=list(peers),
        clip_negative=True,
    )


# ----------------------
# Stand-alone convenience function
# ----------------------
def build_synthetic_etf(
    target: str,
    peers: Iterable[str],
    weight_mode: WeightMode = "corr",
    custom_weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> SyntheticETFArtifacts:
    cfg = SyntheticETFConfig(
        target=target,
        peers=tuple(peers),
        weight_mode=weight_mode,
        **kwargs,
    )
    builder = SyntheticETFBuilder(cfg)
    return builder.build_all(custom_weights=custom_weights)


__all__ = [
    "SyntheticETFConfig",
    "SyntheticETFBuilder",
    "SyntheticETFArtifacts",
    "build_synthetic_etf",
]