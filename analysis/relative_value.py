from __future__ import annotations
from typing import Iterable, Tuple

import pandas as pd

from .pillars import load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS
from .beta_builder.beta_builder import peer_weights_from_correlations
from .compositeIndexBuilder import DEFAULT_TENORS, DEFAULT_MNY_BINS, build_composite_iv_series


def _fetch_target_atm(
    target: str,
    pillar_days: Iterable[int],
    tolerance_days: float = 7.0,
) -> pd.DataFrame:
    atm = load_atm()
    atm = atm[atm["ticker"] == target].copy()
    if atm.empty:
        return pd.DataFrame(columns=["asof_date", "pillar_days", "iv"])
    piv = nearest_pillars(atm, pillars_days=list(pillar_days), tolerance_days=tolerance_days)
    out = piv.groupby(["asof_date", "pillar_days"])["iv"].mean().rename("iv").reset_index()
    return out[["asof_date", "pillar_days", "iv"]]


def _rv_metrics_join(target_iv: pd.DataFrame, composite_iv: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    tgt = target_iv.rename(columns={"iv": "iv_target"})
    syn = composite_iv.rename(columns={"iv": "iv_composite"})
    df = pd.merge(tgt, syn, on=["asof_date", "pillar_days"], how="inner").sort_values(["pillar_days", "asof_date"])
    if df.empty:
        return df

    def per_pillar(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["spread"] = g["iv_target"] - g["iv_composite"]
        roll = max(5, int(lookback // 5))
        m = g["spread"].rolling(lookback, min_periods=roll).mean()
        s = g["spread"].rolling(lookback, min_periods=roll).std(ddof=1)
        g["z"] = (g["spread"] - m) / s
        # percentile rank of latest value within window
        def _pct_rank(x: pd.Series) -> float:
            return x.rank(pct=True).iloc[-1]
        g["pct_rank"] = g["spread"].rolling(lookback, min_periods=roll).apply(_pct_rank, raw=False)
        return g

    return df.groupby("pillar_days", group_keys=False).apply(per_pillar, include_groups=False)


def relative_value_atm_report_corrweighted(
    target: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    lookback: int = 60,
    tolerance_days: float = 7.0,
    weight_power: float = 1.0,
    clip_negative: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute relative value using correlation-based peer weights."""
    w = peer_weights_from_correlations(
        benchmark=target,
        peers=peers,
        mode=mode,
        pillar_days=pillar_days,
        tenor_days=DEFAULT_TENORS,
        mny_bins=DEFAULT_MNY_BINS,
        clip_negative=clip_negative,
        power=weight_power,
    )
    if w.empty:
        empty_cols = ["asof_date", "pillar_days", "iv_target", "iv_composite", "spread", "z", "pct_rank"]
        return pd.DataFrame(columns=empty_cols), w

    composite = build_composite_iv_series(weights=w.to_dict(), pillar_days=pillar_days, tolerance_days=tolerance_days)
    tgt = _fetch_target_atm(target, pillar_days=pillar_days, tolerance_days=tolerance_days)
    rv = _rv_metrics_join(tgt, composite, lookback=lookback)
    return rv, w


def latest_relative_snapshot_corrweighted(
    target: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",
    pillar_days: Iterable[int] = (7, 30, 60, 90),
    lookback: int = 60,
    tolerance_days: float = 7.0,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper returning the latest date per pillar with RV metrics."""
    rv, w = relative_value_atm_report_corrweighted(
        target=target,
        peers=peers,
        mode=mode,
        pillar_days=pillar_days,
        lookback=lookback,
        tolerance_days=tolerance_days,
        **kwargs,
    )
    if rv.empty:
        return rv, w

    if "pillar_days" not in rv.columns:
        rv = rv.copy()
        rv["pillar_days"] = 0
    if "asof_date" not in rv.columns:
        rv = rv.copy()
        rv["asof_date"] = pd.Timestamp("1970-01-01")

    out_rows = []
    for pdays, g in rv.groupby("pillar_days"):
        out_rows.append(g.sort_values("asof_date").iloc[-1])
    out = pd.DataFrame(out_rows).reset_index(drop=True)
    out = out.sort_values("pillar_days").reset_index(drop=True)
    return out, w
