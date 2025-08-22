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

def_corr_execution(
        # New unified-style helpers for corr paths (ATM/Surface matrix)
            # Corr-based quick paths
    if feature in ("iv_atm", "surface", "ul") and method == "corr":
        return peer_weights_from_correlations(
            benchmark=target,
            peers=peers,
            mode=feature if feature != "ul" else "ul",
            pillar_days=pillar_days,
            tenor_days=tenor_days,
            mny_bins=mny_bins,
            clip_negative=True,
            power=1.0,
        )

    if feature == "atm" and method == "corr":
        if asof is None:
            dates = available_dates(ticker=target, most_recent_only=True)
            asof = dates[0] if dates else None
        if asof is None:
            return pd.Series(dtype=float)
        atm_df, corr_df = compute_atm_corr_pillar_free(
            get_smile_slice=get_smile_slice,
            tickers=[target] + peers,
            asof=asof,
            max_expiries=6,
            atm_band=0.05,
        )
        return corr_weights(corr_df, target, peers)
    if feature == "surface" and method == "corr":
        if asof is None:
            dates = available_dates(ticker=target, most_recent_only=True)
            asof = dates[0] if dates else None
        if asof is None:
            return pd.Series(dtype=float)
        corr_df = compute_atm_corr_pillar_free(
            get_smile_slice=get_smile_slice,
            tickers=[target] + list(peers),
            asof=asof,
            max_expiries=6,
            atm_band=0.05,
        )[1]
        return corr_weights_from_matrix(corr_df, target, peers)
    if feature == "ul" and method == "corr":
        if asof is None:
            dates = available_dates(ticker=target, most_recent_only=True)
            asof = dates[0] if dates else None
        if asof is None:
            return pd.Series(dtype=float)
        corr_df = compute_atm_corr_pillar_free(
            get_smile_slice=get_smile_slice,
            tickers=[target] + list(peers),
            asof=asof,
            max_expiries=6,
            atm_band=0.05,
        )[1]
        return corr_weights_from_matrix(corr_df, target, peers)
)