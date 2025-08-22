#!/usr/bin/env python3
"""Precompute cache entries for IVCorrelation.

Given a ticker group name, compute all standard analysis artifacts and store
them in the ``calc_cache`` SQLite table.  This allows warming the cache ahead
of GUI usage with a single command.

Run as::

    python scripts/warm_cache.py "Tech Giants vs SPY"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

# Make project imports work when executed as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model_params_logger import compute_or_load
from analysis.analysis_pipeline import prepare_smile_data, get_smile_slice
from display.plotting.correlation_detail_plot import _corr_by_expiry_rank
from analysis.spillover.vol_spillover import run_spillover, load_iv_data
from data import load_ticker_group
from data.db_utils import get_conn, get_most_recent_date


def _warm_smile(task: Dict[str, Any]) -> None:
    ticker = task["ticker"].upper()
    asof = task["asof"]
    T_days = float(task.get("T_days", 30))
    model = task.get("model", "svi")
    ci = float(task.get("ci", 68.0))
    overlay_synth = bool(task.get("overlay_synth", False))
    peers = task.get("peers")
    weights = task.get("weights")
    overlay_peers = bool(task.get("overlay_peers", False))
    max_expiries = int(task.get("max_expiries", 6))

    payload = {
        "ticker": ticker,
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "model": model,
        "params": weights,
        "T_days": T_days,
    }

    def _builder() -> Any:
        return prepare_smile_data(
            target=ticker,
            asof=asof,
            T_days=T_days,
            model=model,
            ci=ci,
            overlay_synth=overlay_synth,
            peers=peers,
            weights=weights,
            overlay_peers=overlay_peers,
            max_expiries=max_expiries,
        )

    compute_or_load("smile", payload, _builder)


def _warm_corr(task: Dict[str, Any]) -> None:
    tickers = [t.upper() for t in task["tickers"]]
    asof = task["asof"]
    max_expiries = int(task.get("max_expiries", 6))
    atm_band = float(task.get("atm_band", 0.05))

    payload = {
        "tickers": sorted(tickers),
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "max_expiries": max_expiries,
        "atm_band": atm_band,
    }

    def _builder() -> Any:
        return _corr_by_expiry_rank(
            get_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=max_expiries,
            atm_band=atm_band,
        )

    compute_or_load("corr", payload, _builder)


def _warm_spill(task: Dict[str, Any]) -> None:
    tickers = task.get("tickers")
    threshold = float(task.get("threshold", 0.10))
    lookback = int(task.get("lookback", 60))
    top_k = int(task.get("top_k", 3))
    horizons = task.get("horizons", (1, 3, 5))
    path = task.get("path", "data/iv_data.parquet")
    df = load_iv_data(path)

    payload = {
        "tickers": sorted([t.upper() for t in tickers]) if tickers else None,
        "threshold": threshold,
        "lookback": lookback,
        "top_k": top_k,
        "horizons": tuple(horizons),
        "asof": df["date"].max().floor("min").isoformat() if not df.empty else None,
    }

    def _builder() -> Any:
        return run_spillover(
            df,
            tickers=tickers,
            threshold=threshold,
            lookback=lookback,
            top_k=top_k,
            horizons=horizons,
        )

    compute_or_load("spill", payload, _builder)


def main() -> None:
    p = argparse.ArgumentParser(description="Warm calc_cache entries for a ticker group")
    p.add_argument("group", help="Name of ticker group preset")
    args = p.parse_args()

    group = load_ticker_group(args.group)
    if not group:
        print(f"Ticker group '{args.group}' not found")
        return

    tickers: List[str] = [group["target_ticker"]] + list(group["peer_tickers"])
    conn = get_conn()
    try:
        global_asof = get_most_recent_date(conn)
        for t in tickers:
            asof_t = get_most_recent_date(conn, t) or global_asof
            if asof_t:
                _warm_smile({"ticker": t, "asof": asof_t})
        if global_asof:
            _warm_corr({"tickers": tickers, "asof": global_asof})
        _warm_spill({"tickers": tickers})
    finally:
        conn.close()


if __name__ == "__main__":
    main()
