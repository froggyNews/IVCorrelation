import pandas as pd
import numpy as np
from typing import List, Dict, Iterable

"""Tools to detect implied-volatility events and measure spillovers.

This module loads a daily IV dataset, flags events where a ticker's ATM IV
moves by a configurable percentage threshold and then measures how those shocks
propagate to its peers.
"""


def load_iv_data(path: str, use_raw: bool = False) -> pd.DataFrame:
    """Load IV data from a Parquet file.

    Parameters
    ----------
    path: str
        Location of the ``iv_daily`` Parquet file.
    use_raw: bool
        If ``True`` use the ``atm_iv_raw`` column, otherwise use
        ``atm_iv_synth``.
    """
    df = pd.read_parquet(path)
    col = "atm_iv_raw" if use_raw else "atm_iv_synth"
    df = df[["date", "ticker", col]].rename(columns={col: "atm_iv"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def detect_events(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Flag dates where a ticker's IV changes by ``threshold`` or more.

    Returns a DataFrame with columns ``ticker``, ``date``, ``rel_change`` and
    ``sign`` (1 or -1).
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["rel_change"] = df.groupby("ticker")["atm_iv"].pct_change()
    events = df.loc[df["rel_change"].abs() >= threshold,
                    ["ticker", "date", "rel_change"]].copy()
    events["sign"] = np.sign(events["rel_change"]).astype(int)
    return events.reset_index(drop=True)


def select_peers(df: pd.DataFrame, lookback: int = 60, top_k: int = 3) -> Dict[str, List[str]]:
    """Identify top-K peers for each ticker using rolling correlation of ΔIV."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["dIV"] = df.groupby("ticker")["atm_iv"].pct_change()
    piv = df.pivot(index="date", columns="ticker", values="dIV")
    peers: Dict[str, List[str]] = {}
    for t in piv.columns:
        corr = piv.rolling(lookback).corr(piv[t]).iloc[-1]
        corr = corr.drop(index=t).dropna().sort_values(ascending=False).head(top_k)
        peers[t] = list(corr.index)
    return peers


def compute_responses(df: pd.DataFrame,
                      events: pd.DataFrame,
                      peers: Dict[str, List[str]],
                      horizons: Iterable[int] = (1, 3, 5)) -> pd.DataFrame:
    """Compute peer responses for each event over given horizons.

    Response for peer j at horizon h is the percentage change in j's IV from
    t0-1 to t0h.
    """
    panel = df.set_index(["date", "ticker"]).sort_index()
    dates = panel.index.get_level_values(0).unique()
    rows = []
    for _, e in events.iterrows():
        t0 = e["date"]
        i = e["ticker"]
        idx0 = dates.searchsorted(t0)
        if idx0 == 0:
            continue  # need t-1
        t_minus1 = dates[idx0 - 1]
        for j in peers.get(i, []):
            if (t_minus1, j) not in panel.index:
                continue
            base = panel.loc[(t_minus1, j), "atm_iv"]
            for h in horizons:
                idx_h = idx0 + h - 1  # t0h is idx0 + h - 1
                if idx_h >= len(dates):
                    continue
                d_h = dates[idx_h]
                if (d_h, j) not in panel.index:
                    continue
                resp = panel.loc[(d_h, j), "atm_iv"]
                pct = (resp - base) / base
                rows.append({
                    "ticker": i,
                    "peer": j,
                    "t0": t0,
                    "h": int(h),
                    "trigger_pct": e["rel_change"],
                    "peer_pct": pct,
                    "sign": e["sign"],
                })
    return pd.DataFrame(rows)


def summarise(responses: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Summarise peer responses across events."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        hr = (g["peer_pct"].abs() >= threshold).mean()
        sc = (np.sign(g["peer_pct"]) == g["sign"]).mean()
        med_resp = g["peer_pct"].median()
        med_elast = (g["peer_pct"] / g["trigger_pct"]).median()
        return pd.Series({
            "hit_rate": hr,
            "sign_concord": sc,
            "median_resp": med_resp,
            "median_elasticity": med_elast,
            "n": len(g),
        })
    return responses.groupby(["ticker", "peer", "h"]).apply(_agg).reset_index()


def persist_events(events: pd.DataFrame, path: str) -> None:
    """Write event table to Parquet."""
    events.to_parquet(path)


def persist_summary(summary: pd.DataFrame, path: str) -> None:
    """Write summary metrics to Parquet."""
    summary.to_parquet(path)


def run_spillover(
    iv_path: str,
    *,
    tickers: Iterable[str] | None = None,
    threshold: float = 0.10,
    lookback: int = 60,
    top_k: int = 3,
    horizons: Iterable[int] = (1, 3, 5),
    use_raw: bool = False,
    events_path: str = "spill_events.parquet",
    summary_path: str = "spill_summary.parquet",
) -> Dict[str, pd.DataFrame]:
    """High level helper that runs the full spillover analysis.

    Returns a dictionary with keys ``events`` and ``summary``.
    """
    df = load_iv_data(iv_path, use_raw=use_raw)
    if tickers is not None:
        tickers = [t.upper() for t in tickers]
        df = df[df["ticker"].str.upper().isin(tickers)]
    events = detect_events(df, threshold=threshold)
    peers = select_peers(df, lookback=lookback, top_k=top_k)
    responses = compute_responses(df, events, peers, horizons=horizons)
    summary = summarise(responses, threshold=threshold)
    persist_events(events, events_path)
    persist_summary(summary, summary_path)
    return {"events": events, "responses": responses, "summary": summary}
