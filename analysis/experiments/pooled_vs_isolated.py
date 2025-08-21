"""Experiment comparing pooled vs isolated modeling.

This module leverages existing modeling and evaluation utilities
from the analysis package.  Individual surfaces are built using
:func:`analysis.syntheticETFBuilder.build_surface_grids` and
combined via :func:`analysis.syntheticETFBuilder.combine_surfaces`.
The resulting surfaces are evaluated with
:func:`analysis.correlation_utils.compute_atm_corr`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple, Dict
import pandas as pd

from analysis.syntheticETFBuilder import build_surface_grids, combine_surfaces
from analysis.correlation_utils import compute_atm_corr


def run_experiment(
    tickers: Iterable[str],
    weights: Mapping[str, float],
    asof: str,
    pillars_days: Sequence[int] = (7, 30, 60, 90),
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Run the pooled-vs-isolated experiment.

    Parameters
    ----------
    tickers:
        Collection of tickers to model.
    weights:
        Mapping of ticker -> weight used when pooling surfaces.
    asof:
        As-of date for evaluation.
    pillars_days:
        Pillar days used for ATM correlation evaluation.

    Returns
    -------
    pooled_surface:
        Synthetic surface created by pooling peer surfaces with ``weights``.
    atm_df:
        ATM pillar matrix used for evaluation.
    corr_df:
        Correlation matrix across tickers for the specified pillars.
    """

    # --- Modeling: build individual surfaces with existing utilities ---
    surfaces = build_surface_grids(tickers=tickers)

    peer_surfaces = {t: surfaces.get(t, {}) for t in weights.keys()}
    pooled_surface = combine_surfaces(peer_surfaces, weights)

    # --- Evaluation: compute ATM correlations using existing helper ---
    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=lambda ticker, asof, T_target_years=None: peer_surfaces.get(ticker, {}).get(asof),
        tickers=list(weights.keys()),
        asof=asof,
        pillars_days=pillars_days,
    )

    return pooled_surface, atm_df, corr_df
