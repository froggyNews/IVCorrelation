from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import impute_col_median
from typing import Iterable, Tuple, List
__all__ = ["cosine_similarity_weights_from_matrix"]

def cosine_similarity_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Compute cosine‑similarity weights from a ticker×feature matrix."""
    target = target.upper()
    peers = [p.upper() for p in peers]
    if target not in feature_df.index:
        raise ValueError(f"target {target} not in feature matrix")
    df = feature_df.apply(pd.to_numeric, errors="coerce")
    X = impute_col_median(df.to_numpy(float))
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


# ------------------------
# Small convenience wrapper for cosine “modes”
# ------------------------
def cosine_similarity_weights(
    get_smile_slice,
    mode: str,
    target: str,
    peers: Iterable[str],
    *,
    asof: str | None = None,
    **kwargs,
) -> pd.Series:
    """
    Convenience helper that lets you call things like:
      cosine_similarity_weights(..., mode="cosine_atm", ...)
      cosine_similarity_weights(..., mode="cosine_surface", ...)
      cosine_similarity_weights(..., mode="cosine_ul", ...)  # alias for ul_px
    """
    if not mode.startswith("cosine_"):
        raise ValueError("mode must start with 'cosine_'")
    feature = mode[len("cosine_") :]
    if feature == "ul":
        feature = "ul_px"
    from .beta_builder import build_peer_weights
    return build_peer_weights(
        "cosine",
        feature,
        target,
        peers,
        get_smile_slice=get_smile_slice,
        asof=asof,
        **kwargs,
    )
