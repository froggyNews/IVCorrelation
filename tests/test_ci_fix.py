#!/usr/bin/env python3
"""
Test script to verify that confidence intervals are now enabled by default.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.analysis_pipeline import get_smile_slice, available_dates
from analysis.pillars import compute_atm_by_expiry
from display.plotting.smile_plot import fit_and_plot_smile
from display.plotting.term_plot import plot_atm_term_structure

def test_confidence_intervals():
    """Test that confidence intervals are enabled by default."""
    
    # Get some data
    dates = available_dates(ticker="SPY", most_recent_only=True)
    if not dates:
        print("No dates available")
        return
    
    asof = dates[0]
    print(f"Using date: {asof}")
    
    # Test 1: ATM term structure with CI
    print("\n1. Testing ATM term structure confidence intervals...")
    df = get_smile_slice("SPY", asof, T_target_years=None)
    
    if df is not None and not df.empty:
        # This should now generate CI by default (n_boot=100)
        atm_curve = compute_atm_by_expiry(df, atm_band=0.05)
        print(f"   ATM curve shape: {atm_curve.shape}")
        print(f"   ATM curve columns: {list(atm_curve.columns)}")
        
        # Check if CI columns exist
        ci_cols = [col for col in atm_curve.columns if 'atm_lo' in col or 'atm_hi' in col]
        if ci_cols:
            print(f"   ✓ Confidence interval columns found: {ci_cols}")
            # Check if they have non-NaN values
            has_ci_data = any(atm_curve[col].notna().any() for col in ci_cols)
            print(f"   ✓ CI data present: {has_ci_data}")
        else:
            print("   ✗ No confidence interval columns found")
            
        # Test plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_atm_term_structure(ax, atm_curve, show_ci=True)
        ax.set_title(f"SPY ATM Term Structure ({asof})")
        plt.savefig("test_term_ci.png", dpi=100, bbox_inches='tight')
        plt.close()
        print("   ✓ ATM term plot saved as test_term_ci.png")
    
    # Test 2: Smile plot with CI
    print("\n2. Testing smile plot confidence intervals...")
    if df is not None and not df.empty:
        # Get data for a specific expiry
        expiry_groups = df.groupby('T')
        if len(expiry_groups) > 0:
            T_val, grp = next(iter(expiry_groups))
            
            # Extract data
            S = grp['S'].iloc[0]
            K = grp['K'].values
            iv = grp['sigma'].values
            
            print(f"   Testing expiry T={T_val:.4f}, S={S:.2f}, {len(K)} options")
            
            # Plot with CI (should be enabled by default now)
            fig, ax = plt.subplots(figsize=(8, 6))
            result = fit_and_plot_smile(
                ax, S, K, T_val, iv, 
                model="svi",
                # ci_level should default to 0.68 now
            )
            ax.set_title(f"SPY Smile (T={T_val:.4f}, {asof})")
            plt.savefig("test_smile_ci.png", dpi=100, bbox_inches='tight')
            plt.close()
            print("   ✓ Smile plot saved as test_smile_ci.png")
            print(f"   Model fit RMSE: {result.get('rmse', 'N/A')}")

if __name__ == "__main__":
    test_confidence_intervals()
