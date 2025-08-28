#!/usr/bin/env python3
"""Test the TPS implementation in smile plotting."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from display.plotting.smile_plot import fit_and_plot_smile

def test_tps_implementation():
    """Test TPS model in smile plotting."""
    
    # Create composite smile data
    S = 100.0
    K = np.array([85, 90, 95, 100, 105, 110, 115])
    T = 0.25
    # Typical smile shape (higher vol at extremes)
    moneyness = K / S
    iv = 0.20 + 0.05 * (moneyness - 1.0)**2
    
    print(f"Testing TPS smile fit with {len(K)} points")
    print(f"Strikes: {K}")
    print(f"IVs: {iv}")
    
    # Test TPS model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ["svi", "sabr", "tps"]
    for i, model in enumerate(models):
        ax = axes[i]
        
        try:
            result = fit_and_plot_smile(
                ax, S, K, T, iv,
                model=model,
                ci_level=0.68,  # Enable CI
                n_boot=50,      # Reduced for testing speed
                show_points=True
            )
            
            ax.set_title(f"{model.upper()} Model")
            ax.grid(True, alpha=0.3)
            
            print(f"{model.upper()} RMSE: {result.get('rmse', 'N/A')}")
            
        except Exception as e:
            print(f"Error with {model} model: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model.upper()} Model (Failed)")
    
    plt.tight_layout()
    plt.savefig("test_tps_smile.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Test plot saved as test_tps_smile.png")

if __name__ == "__main__":
    test_tps_implementation()
