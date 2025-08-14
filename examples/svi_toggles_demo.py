#!/usr/bin/env python3
"""
Demo script showing the SVI smile plot with improved clickable legend toggles.

This script demonstrates how legend entries can be clicked to toggle visibility 
of SVI components in the existing smile layout.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from display.plotting.smile_plot import fit_and_plot_smile


def demo_svi_legend_toggles():
    """Demonstrate improved legend-based toggle functionality."""
    print("SVI Legend Toggle Controls Demo")
    print("=" * 45)
    
    # Generate realistic smile data
    S = 100.0
    moneyness = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    K = moneyness * S
    # Typical smile shape - higher vol at extremes
    base_vol = 0.20
    iv = base_vol + 0.08 * (moneyness - 1.0) ** 2
    
    # Add realistic noise
    np.random.seed(42)  # For reproducible results
    iv += 0.01 * np.random.randn(len(moneyness))
    T = 0.25  # 3 months
    
    print(f"Data: {len(K)} strikes, Spot=${S:.0f}, T={T:.2f}y")
    
    # Test SVI model with improved legend toggles
    print("\n🎯 SVI Model with Interactive Legend Toggles:")
    fig, ax = plt.subplots(figsize=(12, 8))
    result = fit_and_plot_smile(
        ax, S=S, K=K, T=T, iv=iv, 
        model='svi', 
        enable_svi_toggles=True,
        ci_level=0.68,
        show_points=True
    )
    
    if result.get('series_map'):
        series_keys = list(result['series_map'].keys())
        print(f"   ✅ Interactive components: {series_keys}")
        
        # Get legend info for debugging
        legend = ax.get_legend()
        if legend:
            legend_labels = [text.get_text() for text in legend.get_texts()]
            print(f"   📋 Legend entries: {legend_labels}")
        
        print(f"   📈 Model RMSE: {result.get('rmse', 0):.4f}")
    else:
        print("   ❌ No toggle controls available")
    
    plt.title('SVI Smile Plot with Interactive Legend\n'
              'Click legend entries or use keyboard shortcuts to toggle visibility',
              fontsize=12, pad=20)
    
    # Add comprehensive usage instructions
    instruction_text = (
        "INTERACTIVE CONTROLS:\n"
        "🖱️  Click any legend entry to toggle on/off\n"
        "⌨️  Keyboard: 'o'=Points, 'f'=Fit, 'c'=CI\n"
        "👁️  Toggled items become faded in legend"
    )
    
    ax.text(0.02, 0.02, instruction_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return result


def demo_comparison():
    """Compare SVI vs SABR - only SVI should have toggles."""
    print("\n" + "=" * 45)
    print("📊 Comparison: SVI vs SABR Toggle Behavior")
    print("=" * 45)
    
    # Use same data
    S = 100.0
    moneyness = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    K = moneyness * S
    base_vol = 0.20
    iv = base_vol + 0.08 * (moneyness - 1.0) ** 2
    np.random.seed(42)
    iv += 0.01 * np.random.randn(len(moneyness))
    T = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # SVI with toggles
    print("🟢 SVI Model (Should have toggles):")
    result1 = fit_and_plot_smile(
        ax1, S=S, K=K, T=T, iv=iv, 
        model='svi', 
        enable_svi_toggles=True,
        ci_level=0.68
    )
    ax1.set_title('SVI Model - Interactive Legend')
    
    if result1.get('series_map'):
        print(f"   ✅ Toggles available: {list(result1['series_map'].keys())}")
    else:
        print("   ❌ No toggles (unexpected)")
    
    # SABR without toggles  
    print("🔴 SABR Model (Should NOT have toggles):")
    result2 = fit_and_plot_smile(
        ax2, S=S, K=K, T=T, iv=iv, 
        model='sabr', 
        enable_svi_toggles=True,  # Requested but should be ignored
        ci_level=0.68,
        beta=0.5
    )
    ax2.set_title('SABR Model - Standard Legend')
    
    if result2.get('series_map'):
        print(f"   ❌ Toggles found (unexpected): {list(result2['series_map'].keys())}")
    else:
        print("   ✅ Correctly no toggles for SABR")
    
    plt.tight_layout()
    plt.show()
    
    return result1, result2


def main():
    """Run the demonstration."""
    print("🎯 SVI Smile Plot Interactive Legend Demo")
    print("=" * 50)
    
    try:
        result = demo_svi_legend_toggles()
        
        print("\n" + "="*50)
        print("📋 SUMMARY:")
        print(f"✅ Legend toggles implemented for SVI components")
        print(f"✅ Keyboard shortcuts: o=Points, f=Fit, c=CI")
        print(f"✅ Visual feedback: faded legend entries when toggled off")
        print(f"✅ Only works with SVI model (as requested)")
        
        # Optional comparison demo
        response = input("\n🤔 Show SVI vs SABR comparison? (y/n): ").strip().lower()
        if response.startswith('y'):
            demo_comparison()
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
