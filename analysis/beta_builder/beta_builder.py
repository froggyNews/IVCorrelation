# analysis/beta_builder.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

# Methods (each file keeps its own specifics)
from .correlation import corr_weights_from_matrix
from .cosine import cosine_similarity_weights_from_matrix
from .pca import pca_regress_weights
from .open_interest import open_interest_weights

# Feature builders
from .feature_matrices import atm_feature_matrix, surface_feature_matrix
from .unified_weights import underlying_returns_matrix
from .utils import impute_col_median

__all__ = [
    "build_peer_weights",
]

def _normalize(series: pd.Series) -> pd.Series:
    s = series.clip(lower=0.0).fillna(0.0)
    tot = float(s.sum())
    return s / tot if tot > 0 else s

def build_peer_weights(
    method: str,
    feature_set: str,
    target: str,
    peers: Iterable[str],
    *,
    get_smile_slice=None,
    asof: str | None = None,
    pillars_days: Iterable[int] = (7, 14, 21, 28, 42, 56, 70, 84, 98, 112, 140, 182, 252, 365),
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    clip_negative: bool = True,
    power: float = 1.0,
    k: Optional[int] = None,   # optional top-k prune for cosine
) -> pd.Series:
    """
    4×4 dispatcher: methods × data types
      methods:      {"corr","cosine","pca","oi"}
      feature_set:  {"atm","surface","ul_px","oi"}
    """
    method = (method or "corr").lower().strip()
    feature = (feature_set or "atm").lower().strip()
    target = target.upper()
    peers_list = [p.upper() for p in peers]

    # OI: method takes precedence; no feature matrix
    if feature == "oi" or method == "oi":
        if asof is None:
            raise ValueError("asof is required for open-interest weights")
        return open_interest_weights(peers_list, asof)

    # Build feature matrix
    feature_df: pd.DataFrame | None = None
    if feature == "atm":
        if asof is None or get_smile_slice is None:
            raise ValueError("get_smile_slice and asof are required for ATM features")
        atm_df, _, _ = atm_feature_matrix(get_smile_slice, [target] + peers_list, asof, pillars_days)
        feature_df = atm_df
    elif feature == "surface":
        if asof is None:
            raise ValueError("asof is required for surface features")
        grids, X, names = surface_feature_matrix([target] + peers_list, asof, tenors=tenors, mny_bins=mny_bins)
        feature_df = pd.DataFrame(X, index=list(grids.keys()), columns=names)
    elif feature == "ul_px":
        ret_rows = underlying_returns_matrix([target] + peers_list)
        feature_df = ret_rows if ret_rows is not None and not ret_rows.empty else None
    else:
        raise ValueError(f"unknown feature_set {feature_set}")

    if feature_df is None or feature_df.empty:
        raise ValueError("feature data unavailable for peer weights")

    # Apply method
    if method == "corr":
        return corr_weights_from_matrix(feature_df, target, peers_list, clip_negative=clip_negative, power=power)

    if method == "cosine":
        w = cosine_similarity_weights_from_matrix(feature_df, target, peers_list, clip_negative=clip_negative, power=power)
        if k is not None and k > 0:
            w = w.nlargest(k)
            w = _normalize(w).reindex(peers_list).fillna(0.0)
        return w

    if method == "pca":
        if target not in feature_df.index:
            return pd.Series(dtype=float)
        y = impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
        Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
        if Xp.size == 0:
            return pd.Series(dtype=float)
        w = pca_regress_weights(Xp, y, k=None, nonneg=True)
        s = pd.Series(w, index=[p for p in peers_list if p in feature_df.index]).clip(lower=0.0)
        return _normalize(s).reindex(peers_list).fillna(0.0)

    raise ValueError(f"unknown method {method}")
