from __future__ import annotations
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd

from .unified_weights import (
    atm_feature_matrix as uw_atm_feature_matrix,
    surface_feature_matrix as uw_surface_feature_matrix,
)


def atm_feature_matrix(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = 0.08,
    tol_days: float = 10.0,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Wrapper around unified weights ATM builder."""
    return uw_atm_feature_matrix(
        tickers=[t.upper() for t in tickers],
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        standardize=standardize,
    )


def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    standardize: bool = True,
) -> Tuple[Dict[str, Dict[pd.Timestamp, pd.DataFrame]], np.ndarray, List[str]]:
    """Wrapper around unified weights surface builder."""
    return uw_surface_feature_matrix(
        tickers=[t.upper() for t in tickers],
        asof=asof,
        tenors=tenors,
        mny_bins=mny_bins,
        standardize=standardize,
    )
