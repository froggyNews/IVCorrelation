#!/usr/bin/env python
"""
Demonstration script for the new PolyFit module.

Shows both simple polynomial and Thin Plate Spline (TPS) fitting methods
on synthetic implied volatility smile data.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volModel.polyFit import fit_poly

def create_synthetic_smile(noise_level=0.01):
    """Create synthetic implied volatility smile data."""
    # Log-moneyness points
    k = np.linspace(-0.4, 0.4, 15)
    
    # Typical volatility smile: U-shape with slight skew
    # Higher vol for deep OTM, skew toward puts
    iv_base = 0.20 + 0.12 * k**2 + 0.03 * k
    
    # Add some realistic noise
    np.random.seed(42)  # For reproducible results
    noise = noise_level * np.random.randn(len(k))
    iv = iv_base + noise
    
    return k, iv

def demonstrate_fitting():
    """Demonstrate both fitting methods."""
    print("=== PolyFit Module Demonstration ===\n")
    
    # Create synthetic data
    k, iv = create_synthetic_smile()
    
    print(f"Generated {len(k)} implied volatility points")
    print(f"Log-moneyness range: [{k.min():.2f}, {k.max():.2f}]")
    print(f"IV range: [{iv.min():.3f}, {iv.max():.3f}]\n")
    
    # Fit using simple polynomial method
    print("--- Simple Polynomial Fit ---")
    result_simple = fit_poly(k, iv, method="simple")
    print(f"Model: {result_simple['model']}")
    print(f"ATM Vol: {result_simple['atm_vol']:.4f}")
    print(f"Skew: {result_simple['skew']:.4f}")
    print(f"Curvature: {result_simple['curv']:.4f}")
    print(f"RMSE: {result_simple['rmse']:.6f}\n")
    
    # Fit using TPS method  
    print("--- Thin Plate Spline (TPS) Fit ---")
    result_tps = fit_poly(k, iv, method="tps")
    print(f"Model: {result_tps['model']}")
    print(f"ATM Vol: {result_tps['atm_vol']:.4f}")
    print(f"Skew: {result_tps['skew']:.4f}")
    print(f"Curvature: {result_tps['curv']:.4f}")
    print(f"RMSE: {result_tps['rmse']:.6f}")
    
    # Show interpolation capability if TPS worked
    if result_tps['model'] == 'tps' and 'interpolator' in result_tps:
        print("\n--- TPS Interpolation Test ---")
        test_points = np.array([-0.05, 0.0, 0.05])
        interpolated = result_tps['interpolator'](test_points)
        
        for k_test, iv_pred in zip(test_points, interpolated):
            print(f"k={k_test:+.2f} -> IV={iv_pred:.4f}")
    
    # Create visualization if matplotlib available
    try:
        create_comparison_plot(k, iv, result_simple, result_tps)
        print(f"\nVisualization saved as 'polyfit_comparison.png'")
    except Exception as e:
        print(f"\nVisualization not available: {e}")

def create_comparison_plot(k, iv, result_simple, result_tps):
    """Create comparison plot of fitting methods."""
    # Fine grid for smooth curves
    k_fine = np.linspace(k.min(), k.max(), 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Simple polynomial fit
    ax1.scatter(k, iv, alpha=0.7, color='blue', label='Data', zorder=3)
    
    # Simple polynomial prediction
    a, b, c = result_simple['atm_vol'], result_simple['skew'], result_simple['curv']/2
    iv_simple = a + b*k_fine + c*k_fine**2
    ax1.plot(k_fine, iv_simple, 'r-', linewidth=2, label=f'Simple Poly (RMSE: {result_simple["rmse"]:.4f})')
    
    ax1.set_xlabel('Log-moneyness')
    ax1.set_ylabel('Implied Volatility')
    ax1.set_title('Simple Polynomial Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: TPS fit comparison
    ax2.scatter(k, iv, alpha=0.7, color='blue', label='Data', zorder=3)
    ax2.plot(k_fine, iv_simple, 'r--', alpha=0.7, label=f'Simple Poly (RMSE: {result_simple["rmse"]:.4f})')
    
    if result_tps['model'] == 'tps' and 'interpolator' in result_tps:
        iv_tps = result_tps['interpolator'](k_fine)
        ax2.plot(k_fine, iv_tps, 'g-', linewidth=2, label=f'TPS (RMSE: {result_tps["rmse"]:.4f})')
    else:
        ax2.text(0.05, 0.95, 'TPS fallback to\nsimple polynomial', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('Log-moneyness')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('TPS vs Simple Polynomial Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polyfit_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    demonstrate_fitting()