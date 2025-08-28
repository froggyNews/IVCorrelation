#!/usr/bin/env python3
"""Precompute cache entries for IVCorrelation.

Given a ticker group name, compute all standard analysis artifacts and store
them in the database cache (calculation_cache table). This allows warming the cache ahead
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

from analysis.model_params_logger import (
    compute_or_load, get_data_snapshot, detect_data_changes, 
    invalidate_cache_for_tickers, cache_stats
)
from analysis.analysis_pipeline import prepare_smile_data, prepare_term_data, get_smile_slice
from display.plotting.relative_weight_plot import _relative_weight_by_expiry_rank
from analysis.spillover.vol_spillover import run_spillover, load_iv_data
from data import load_ticker_group
from data.db_utils import get_conn, get_most_recent_date
from data.data_downloader import save_for_tickers
from analysis.analysis_composite_etf import compositeETFBuilder, compositeETFConfig


def _warm_smile(task: Dict[str, Any]) -> None:
    from analysis.analysis_pipeline import get_smile_slice
    
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

    # Check if ticker has any data, download if missing
    try:
        df_check = get_smile_slice(ticker, asof, max_expiries=max_expiries)
        if df_check is None or df_check.empty:
            print(f"📥 No data available for {ticker} on {asof} - downloading fresh data...")
            rows_added = save_for_tickers([ticker], max_expiries=8)
            if rows_added > 0:
                print(f"✓ Downloaded {rows_added} rows for {ticker}")
                # Update asof to today's date since we downloaded fresh data
                from datetime import datetime, timezone
                asof = datetime.now(timezone.utc).date().isoformat()
                print(f"ℹ️  Updated asof date to {asof} for fresh data")
                # Re-check after download
                df_check = get_smile_slice(ticker, asof, max_expiries=max_expiries)
                if df_check is None or df_check.empty:
                    print(f"⚠️  Still no data for {ticker} after download - skipping")
                    return
            else:
                print(f"❌ Failed to download data for {ticker} - skipping")
                return
    except Exception as e:
        print(f"❌ Error with data for {ticker}: {e} - skipping")
        return

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

    try:
        compute_or_load("smile", payload, _builder)
        print(f"✓ Warmed smile cache for {ticker}")
    except Exception as e:
        print(f"❌ Failed to warm smile cache for {ticker}: {e}")
        return

    # Also warm the multi-model and confidence bands cache
    try:
        from volModel.multi_model_cache import fit_all_models_with_bands_cached
        
        # Get raw data for this ticker/date
        df = get_smile_slice(ticker, asof, T_target_years=T_days/365.25)
        if df is not None and not df.empty:
            S = float(df["S"].median())
            K = df["K"].to_numpy(float)
            IV = df["sigma"].to_numpy(float)
            T = float(df["T"].median())
            
            # Warm multi-model cache with confidence bands
            fit_all_models_with_bands_cached(S, K, T, IV, ci_level=ci/100.0 if ci > 1 else ci)
    except Exception:
        pass  # Ignore errors in cache warming


def _warm_term(task: Dict[str, Any]) -> None:
    """Warm term structure data cache."""
    from analysis.analysis_pipeline import get_smile_slice
    
    ticker = task["ticker"].upper()
    asof = task["asof"]
    ci = float(task.get("ci", 68.0))
    overlay_synth = bool(task.get("overlay_synth", True))
    peers = task.get("peers", [])
    weights = task.get("weights")
    atm_band = float(task.get("atm_band", 0.05))
    max_expiries = int(task.get("max_expiries", 6))
    weight_mode = task.get("weight_mode", "corr_iv_atm")

    # Check if ticker has any data, download if missing
    try:
        df_check = get_smile_slice(ticker, asof, max_expiries=max_expiries)
        if df_check is None or df_check.empty:
            print(f"📥 No data available for {ticker} on {asof} - downloading fresh data...")
            rows_added = save_for_tickers([ticker], max_expiries=8)
            if rows_added > 0:
                print(f"✓ Downloaded {rows_added} rows for {ticker}")
                # Update asof to today's date since we downloaded fresh data
                from datetime import datetime, timezone
                asof = datetime.now(timezone.utc).date().isoformat()
                print(f"ℹ️  Updated asof date to {asof} for fresh data")
                # Re-check after download
                df_check = get_smile_slice(ticker, asof, max_expiries=max_expiries)
                if df_check is None or df_check.empty:
                    print(f"⚠️  Still no data for {ticker} after download - skipping term structure")
                    return
            else:
                print(f"❌ Failed to download data for {ticker} - skipping term structure")
                return
    except Exception as e:
        print(f"❌ Error with data for {ticker}: {e} - skipping term structure")
        return

    payload = {
        "target": ticker,
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "ci": ci,
        "overlay_synth": overlay_synth,
        "peers": sorted(peers) if peers else [],
        "weights": weights,
        "atm_band": atm_band,
        "max_expiries": max_expiries,
        "weight_mode": weight_mode,
    }

    def _builder() -> Any:
        return prepare_term_data(
            target=ticker,
            asof=asof,
            ci=ci,
            overlay_synth=overlay_synth,
            peers=peers,
            weights=weights,
            atm_band=atm_band,
            max_expiries=max_expiries,
        )

    try:
        result = compute_or_load("term", payload, _builder)
        print(f"✓ Warmed term cache for {ticker}")

        # Also warm the term structure fitting cache
        try:
            from volModel.term_structure_cache import get_cached_term_structure_data

            atm_curve = result.get("atm_curve")
            if atm_curve is not None and not atm_curve.empty:
                # Warm term structure fit cache
                get_cached_term_structure_data(atm_curve, fit_points=100, use_cache=True)
                print(f"✓ Warmed term structure fit cache for {ticker}")
        except Exception:
            pass  # Ignore errors in term fit cache warming

    except Exception as e:
        print(f"❌ Failed to warm term cache for {ticker}: {e}")
        return


def _warm_synth(task: Dict[str, Any]) -> None:
    """Warm composite ETF surface cache for multiple weight modes."""
    target = task["target"].upper()
    peers = [p.upper() for p in task.get("peers", [])]
    weight_modes = task.get("weight_modes", ["corr"])
    max_expiries = int(task.get("max_expiries", 6))

    for mode in weight_modes:
        cfg = compositeETFConfig(
            target=target,
            peers=tuple(peers),
            max_expiries=max_expiries,
            weight_mode=mode,
        )

        def _builder() -> Any:
            b = compositeETFBuilder(cfg)
            return b.build_all()

        payload = {
            "target": target,
            "peers": tuple(peers),
            "weight_mode": mode,
            "max_expiries": max_expiries,
        }
        try:
            compute_or_load("composite_etf", payload, _builder)
            print(f"✓ Warmed composite ETF cache for {target} ({mode})")
        except Exception as e:
            print(f"❌ Failed to warm composite ETF for {target} ({mode}): {e}")


def _warm_relative_weight(task: Dict[str, Any]) -> None:
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
        return _relative_weight_by_expiry_rank(
            get_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=max_expiries,
            atm_band=atm_band,
        )

    compute_or_load("relative_weight", payload, _builder)


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
    p.add_argument("--force", action="store_true", help="Force refresh all cache entries even if no new data")
    p.add_argument("--check-only", action="store_true", help="Only check for data changes, don't warm cache")
    args = p.parse_args()

    # Get baseline data snapshot before loading new data
    baseline_snapshot = get_data_snapshot()
    print(f"📊 Current data state: {len(baseline_snapshot['tickers'])} tickers, latest date: {baseline_snapshot['global_latest_date']}")

    group = load_ticker_group(args.group)
    if not group:
        print(f"❌ Ticker group '{args.group}' not found")
        
        # Show available groups
        from data.ticker_groups import list_ticker_groups
        available_groups = list_ticker_groups()
        if available_groups:
            print("\n📋 Available ticker groups:")
            for g in available_groups:
                print(f"  • '{g['group_name']}': {g['target_ticker']} vs {g['peer_tickers']}")
        else:
            print("📋 No ticker groups found. Create some first using the GUI or data/ticker_groups.py")
        return

    tickers: List[str] = [group["target_ticker"]] + list(group["peer_tickers"])
    
    # Check for data changes
    changes = detect_data_changes(baseline_snapshot)
    
    print(f"\n🔍 Data change detection:")
    print(f"  • New tickers: {changes['new_tickers'] or 'None'}")
    print(f"  • New expiries: {changes['new_expiries'] or 'None'}")  
    print(f"  • Updated tickers: {changes['updated_tickers'] or 'None'}")
    print(f"  • Global date changed: {changes['global_date_changed']}")
    
    if args.check_only:
        print("✅ Check complete (--check-only mode)")
        return
    
    # Determine what needs refresh
    refresh_tickers = set()
    if args.force:
        refresh_tickers = set(tickers)
        print(f"\n🔄 Force mode: refreshing all {len(tickers)} tickers")
    else:
        # Auto-detect what needs refresh
        affected = set(changes['new_tickers'] + changes['new_expiries'] + changes['updated_tickers'])
        refresh_tickers = affected & set(tickers)  # Only refresh tickers in our group
        
        if refresh_tickers:
            print(f"\n🔄 Auto-refresh needed for: {sorted(refresh_tickers)}")
            
            # Invalidate cache for affected tickers
            invalidated = invalidate_cache_for_tickers(refresh_tickers)
            print(f"🗑️  Invalidated {invalidated} cache entries")
        else:
            print(f"\n✅ No refresh needed - all data is current")
            
            # Still show cache stats
            stats = cache_stats()
            print(f"📊 Cache stats: {stats['entries']} entries, kinds: {stats['kinds']}")
            return
    conn = get_conn()
    try:
        global_asof = get_most_recent_date(conn)
        
        # Warm data for affected tickers only
        for t in refresh_tickers:
            asof_t = get_most_recent_date(conn, t) or global_asof
            if asof_t:
                print(f"Warming smile data for {t}...")
                _warm_smile({"ticker": t, "asof": asof_t})
                print(f"Warming term data for {t}...")
                _warm_term({
                    "ticker": t,
                    "asof": asof_t,
                    "peers": list(group["peer_tickers"]),
                    "overlay_synth": True,
                    "ci": 68.0,
                })
                if t.upper() == group["target_ticker"].upper():
                    print("Warming composite ETF surfaces...")
                    _warm_synth({
                        "target": t,
                        "peers": list(group["peer_tickers"]),
                        "weight_modes": ["corr", "pca", "cosine", "equal"],
                        "max_expiries": 6,
                    })
        
        # Refresh relative-weight matrix if any tickers were affected or forced
        if refresh_tickers and global_asof:
            print(f"Warming relative-weight data...")
            _warm_relative_weight({"tickers": tickers, "asof": global_asof})
            
        # Only warm spillover if forced (spillover has its own error handling)
        if args.force:
            print(f"Warming spillover data...")
            try:
                _warm_spill({"tickers": tickers})
            except Exception as e:
                print(f"⚠️  Spillover warming failed: {e}")
        
        # Final cache stats
        stats = cache_stats()
        print(f"\n📊 Final cache stats: {stats['entries']} entries, kinds: {stats['kinds']}")
        print(f"✅ Cache warming complete!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
