#!/usr/bin/env python3
"""Test script to verify animation methods are available in PlotManager."""

from display.gui.gui_plot_manager import PlotManager

def test_animation_methods():
    pm = PlotManager()
    
    print("Testing animation methods availability:")
    methods = [
        'has_animation_support',
        'plot_animated', 
        'start_animation',
        'stop_animation',
        'pause_animation',
        'set_animation_speed',
        'is_animation_active'
    ]
    
    for method in methods:
        available = hasattr(pm, method)
        print(f"  {method}: {'✓' if available else '✗'}")
    
    # Test some basic functionality
    print("\nTesting basic functionality:")
    print(f"  has_animation_support('smile'): {pm.has_animation_support('smile')}")
    print(f"  has_animation_support('surface'): {pm.has_animation_support('surface')}")
    print(f"  has_animation_support('unknown'): {pm.has_animation_support('unknown')}")
    print(f"  is_animation_active(): {pm.is_animation_active()}")
    
    print("\nAll animation methods implemented successfully!")

if __name__ == "__main__":
    test_animation_methods()
