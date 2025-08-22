from __future__ import annotations
import pandas as pd

__all__ = ["equal_weights"]

def equal_weights(peers_list: list[str]) -> pd.Series:
    """Return equal weights for the provided peers."""
    peers = [p.upper() for p in peers_list]
    if not peers:
        return pd.Series(dtype=float)
    w = 1.0 / len(peers)
    return pd.Series(w, index=peers, dtype=float)
