import numpy as np
import pandas as pd


def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Compute correlation‑based weights from a ticker×feature matrix."""
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
