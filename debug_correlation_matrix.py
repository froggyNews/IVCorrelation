#!/usr/bin/env python3
"""
Debug script for investigating Relative Weight Matrix issues.

This script demonstrates how to use the new debugging features added to
PlotManager to investigate why the correlation matrix shows empty data.

Usage:
    python debug_correlation_matrix.py SPY QQQ,IWM 2025-08-26
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from display.gui.gui_plot_manager import PlotManager


def debug_correlation_matrix(target: str, peers_str: str, asof: str):
    """Debug correlation matrix data availability."""
    
    peers = [p.strip().upper() for p in peers_str.split(",") if p.strip()]
    target = target.upper()
    
    print(f"🔍 Debugging Relative Weight Matrix for:")
    print(f"   Target: {target}")
    print(f"   Peers: {peers}")
    print(f"   Date: {asof}")
    
    # Create PlotManager instance
    pm = PlotManager()
    
    # Step 1: Check raw data availability
    print("\n📊 Step 1: Raw Data Availability")
    pm.debug_data_availability(target, peers, asof, max_expiries=6)
    
    # Step 2: Test bounded slicer
    print("\n🔧 Step 2: Testing Bounded Slicer")
    for ticker in [target] + peers:
        try:
            df_module = pm.get_smile_slice(ticker, asof)  # This is now bounded
            print(f"{ticker}: Bounded slicer returns {len(df_module)} rows")
        except Exception as e:
            print(f"{ticker}: Bounded slicer ERROR - {e}")
    
    # Step 3: Test pillar computation
    print("\n🎯 Step 3: Target Pillars Computation")
    try:
        pillars = pm._get_target_pillars(target, asof, max_expiries=6, peers=peers)
        print(f"Computed pillars: {pillars}")
    except Exception as e:
        print(f"Pillar computation ERROR: {e}")
    
    # Step 4: Refresh connections (useful after ingestion)
    print("\n🔄 Step 4: Refreshing Data Connections")
    pm.refresh_data_connections()
    
    print("\n✅ Debug complete. Check the output above for issues.")
    print("\nTroubleshooting tips:")
    print("- If all tickers show 0 rows: Check date format and DB timestamps")
    print("- If some tickers missing: Check ticker names and data availability")
    print("- If expiry counts vary widely: May need expiry alignment")
    print("- If pillar computation fails: Check get_smile_slice function")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python debug_correlation_matrix.py TARGET PEERS DATE")
        print("Example: python debug_correlation_matrix.py SPY QQQ,IWM 2025-08-26")
        sys.exit(1)
    
    target, peers_str, asof = sys.argv[1:]
    debug_correlation_matrix(target, peers_str, asof)
