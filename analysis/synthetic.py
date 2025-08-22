from __future__ import annotations
from typing import Mapping, Iterable, Union

import pandas as pd

from .pillars import DEFAULT_PILLARS_DAYS
from .syntheticETFBuilder import build_synthetic_iv as build_synthetic_iv_pillars


def build_synthetic_iv_series(
    weights: Mapping[str, float],
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = 7.0,
) -> pd.DataFrame:
    """Create a weighted ATM pillar IV time series."""
    w = {k.upper(): float(v) for k, v in weights.items()}
    return build_synthetic_iv_pillars(w, pillar_days=pillar_days, tolerance_days=tolerance_days)
