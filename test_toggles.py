#!/usr/bin/env python3
"""Test the updated toggle functionality for all model types."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import matplotlib.pyplot as plt
import numpy as np
from analysis.analysis_pipeline import get_smile_slice, available_dates
from display.plotting.smile_plot import fit_and_plot_smile

def test_toggles_all_models():
    """Test that toggles work for all model types."""
    
    # Get some data
    dates = available_dates(ticker="SPY", most_recent_only=True)
    if not dates:
        print("No dates available")
        return
    
    asof = dates[0]
    print(f"Using date: {asof}")
    
    df = get_smile_slice("SPY", asof, T_target_years=None)
    if df is None or df.empty:
        print("No smile data available")
        return
    
    # Get data for the first expiry
    expiry_groups = df.groupby('T')
    T_val, grp = next(iter(expiry_groups))
    
    S = grp['S'].iloc[0]
    K = grp['K'].values
    iv = grp['sigma'].values
    
    print(f"Testing with expiry T={T_val:.4f}, S={S:.2f}, {len(K)} options")
    
    # Test all model types with toggles
    models = ["svi", "sabr", "tps"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        print(f"\nTesting {model.upper()} model with toggles...")
        
        try:
            result = fit_and_plot_smile(
                ax, S, K, T_val, iv,
                model=model,
                ci_level=0.68
            )
            
            series_map = result.get("series_map")
            if series_map:
                print(f"  ✓ {model.upper()} toggles enabled, series: {list(series_map.keys())}")
            else:
                print(f"  ✗ {model.upper()} toggles not created")
                
            print(f"  Model fit RMSE: {result.get('rmse', 'N/A')}")
            
            ax.set_title(f"{model.upper()} Model (T={T_val:.4f})")
            
        except Exception as e:
            print(f"  ✗ Error with {model.upper()}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model.upper()} Model (Failed)")
    
    plt.tight_layout()
    plt.savefig("test_toggles_all_models.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Test plot saved as test_toggles_all_models.png")

if __name__ == "__main__":
    test_toggles_all_models()
