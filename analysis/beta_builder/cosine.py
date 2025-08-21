import numpy as np
import pandas as pd
from .utils import impute_col_median


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
