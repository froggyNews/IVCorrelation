"""
High-level composite ETF construction utilities.

Leverages existing primitives:
- build_surface_grids / combine_surfaces (analysis.compositeETFBuilder)
- UnifiedWeightComputer for peer weighting (analysis.unified_weights)
- sample_smile_curve / fit_smile_for (analysis.analysis_pipeline)

Provides:
- compositeETFBuilder: orchestrates weights, surfaces, composite surface, ATM RV.
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

from analysis.compositeETFBuilder import (
    build_surface_grids,
    combine_surfaces,
    DEFAULT_TENORS,
    DEFAULT_MNY_BINS,
)
from analysis.analysis_pipeline import (
    sample_smile_curve,
    get_most_recent_date_global,
    get_smile_slice,
)
from analysis.beta_builder.unified_weights import UnifiedWeightComputer, WeightConfig, FeatureSet, WeightMethod
from analysis.pillars import compute_atm_by_expiry
from analysis.compositeETFBuilder import build_composite_iv_by_rank

WeightMode = Literal["corr", "pca", "cosine", "equal", "custom"]


@dataclass
class compositeETFConfig:
    target: str
    peers: Iterable[str]
    max_expiries: int = 6
    tenors: Tuple[int, ...] = DEFAULT_TENORS
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS
    tolerance_days: float = 7.0
    lookback: int = 60
    weight_mode: WeightMode = "corr"
    weight_power: float = 1.0
    clip_negative: bool = True
    use_atm_only_surface: bool = False
    cache_dir: Optional[str] = "data/cache_synth_etf"
    # If True we require surfaces for EVERY peer date to include a date in composite output
    # DISABLED: Always False to prevent date filtering
    strict_date_intersection: bool = False

    def ensure_cache(self):
        if self.cache_dir and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class compositeETFArtifacts:
    weights: pd.Series
    surfaces: Dict[str, Dict[str, pd.DataFrame]]
    composite_surfaces: Dict[str, pd.DataFrame]
    rv_metrics: pd.DataFrame
    meta: Dict[str, str] = field(default_factory=dict)


class compositeETFBuilder:
    def __init__(
        self,
        config: compositeETFConfig,
        weight_computer: Optional[UnifiedWeightComputer] = None,
    ):
        self.cfg = config
        self.cfg.ensure_cache()
        self._weights: Optional[pd.Series] = None
        self._surfaces: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        self._composite_surfaces: Optional[Dict[str, pd.DataFrame]] = None
        self._rv: Optional[pd.DataFrame] = None
        self._weight_computer = weight_computer or UnifiedWeightComputer()

    # ----------------------
    # Weight Computation
    # ----------------------
    def compute_weights(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Compute portfolio weights using :class:`UnifiedWeightComputer`."""

        if self.cfg.weight_mode == "custom":
            if not custom_weights:
                raise ValueError(
                    "custom weight_mode selected but no custom_weights supplied"
                )
            w = pd.Series(custom_weights, dtype=float)
            return w / w.sum()

        cfg = WeightConfig.from_legacy_mode(
            self.cfg.weight_mode,
            tenors=self.cfg.tenors,
            mny_bins=self.cfg.mny_bins,
            clip_negative=self.cfg.clip_negative,
            power=self.cfg.weight_power,
            max_expiries=getattr(self.cfg, "max_expiries", 6),
        )

        w = self._weight_computer.compute_weights(
            target=self.cfg.target,
            peers=self.cfg.peers,
            config=cfg,
        )

        if w.empty:
            raise ValueError(
                f"{self.cfg.weight_mode} weight computation returned empty series"
            )

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

    def build_composite_surfaces(self) -> Dict[str, pd.DataFrame]:
        if self._weights is None:
            raise RuntimeError("Weights not computed yet. Call compute_weights() first.")
        if self._surfaces is None:
            raise RuntimeError("Surfaces not built yet. Call build_surfaces() first.")

        peer_surfaces = {
            p: self._surfaces[p]
            for p in self.cfg.peers
            if p in self._surfaces
        }
        composite = combine_surfaces(peer_surfaces, self._weights.to_dict())

        # Optionally restrict dates to intersection across all peer surfaces
        if self.cfg.strict_date_intersection:
            date_sets = [set(dates_dict.keys()) for dates_dict in peer_surfaces.values()]
            if date_sets:
                common = set.intersection(*date_sets)
                composite = {d: grid for d, grid in composite.items() if d in common}

        self._composite_surfaces = composite
        return composite

    # ----------------------
    # Relative Value (ATM)
    # ----------------------
    def compute_relative_value(self) -> pd.DataFrame:
        if self._weights is None:
            raise RuntimeError("Weights not computed yet.")

        uw = UnifiedWeightComputer()
        asof = uw._choose_asof(self.cfg.target, list(self.cfg.peers),
                               WeightConfig(method=WeightMethod.CORRELATION,
                                            feature_set=FeatureSet.ATM))
        if not asof:
            raise RuntimeError("No as-of date available for RV.")

        syn = build_composite_iv_by_rank(self._weights.to_dict(), asof=asof,
                                         max_expiries=self.cfg.max_expiries)
        if syn.empty:
            self._rv = syn
            return syn

        df = get_smile_slice(self.cfg.target, asof, T_target_years=None)
        tgt_curve = compute_atm_by_expiry(df)[["T", "atm_vol"]].dropna().sort_values("T")
        tgt_curve = tgt_curve.reset_index(drop=True).rename(columns={"atm_vol": "target_iv"})
        tgt_curve["rank"] = tgt_curve.index

        rv = tgt_curve.merge(syn, on="rank", how="inner")
        rv["spread"] = rv["target_iv"] - rv["synth_iv"]
        rv["asof_date"] = pd.to_datetime(asof)
        rv = rv[["asof_date", "rank", "T", "target_iv", "synth_iv", "spread"]]
        self._rv = rv
        return rv

    # ----------------------
    # Merged Artifacts
    # ----------------------
    def build_all(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> compositeETFArtifacts:
        start = time.time()
        w = self.compute_weights(custom_weights=custom_weights)
        surfaces = self.build_surfaces()
        synth = self.build_composite_surfaces()
        rv = self.compute_relative_value()

        meta = {
            "target": self.cfg.target,
            "peers": ",".join(self.cfg.peers),
            "weight_mode": self.cfg.weight_mode,
            "lookback": str(self.cfg.lookback),
            "tenors": ",".join(map(str, self.cfg.tenors)),
            "mny_bins": ";".join(f"{a}:{b}" for a, b in self.cfg.mny_bins),
            "tolerance_days": str(self.cfg.tolerance_days),
            "max_expiries": str(self.cfg.max_expiries),
            "build_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "elapsed_sec": f"{time.time()-start:.2f}",
        }
        return compositeETFArtifacts(
            weights=w,
            surfaces=surfaces,
            composite_surfaces=synth,
            rv_metrics=rv,
            meta=meta,
        )

    # ----------------------
    # Export Helpers
    # ----------------------
    def export(self, artifacts: compositeETFArtifacts, out_dir: str) -> None:
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

        # composite surfaces
        syn_dir = os.path.join(out_dir, "composite")
        os.makedirs(syn_dir, exist_ok=True)
        for d, df in artifacts.composite_surfaces.items():
            df.to_csv(os.path.join(syn_dir, f"{d}.csv"))

    # ----------------------
    # Convenience Queries
    # ----------------------
    def latest_surface_pair(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """Return (target_surface, composite_surface, date) for most recent common date."""
        if self._surfaces is None or self._composite_surfaces is None:
            return None, None, None
        target_surfs = self._surfaces.get(self.cfg.target, {})
        if not target_surfs:
            return None, None, None
        dates_target = set(target_surfs.keys())
        dates_syn = set(self._composite_surfaces.keys())
        common = sorted(dates_target.intersection(dates_syn))
        if not common:
            return None, None, None
        d = common[-1]
        return target_surfs[d], self._composite_surfaces[d], d

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
# Stand-alone convenience function
# ----------------------
def build_composite_etf(
    target: str,
    peers: Iterable[str],
    weight_mode: WeightMode = "corr",
    custom_weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> compositeETFArtifacts:
    cfg = compositeETFConfig(
        target=target,
        peers=tuple(peers),
        weight_mode=weight_mode,
        **kwargs,
    )
    builder = compositeETFBuilder(cfg)
    return builder.build_all(custom_weights=custom_weights)


__all__ = [
    "compositeETFConfig",
    "compositeETFBuilder",
    "compositeETFArtifacts",
    "build_composite_etf",
]