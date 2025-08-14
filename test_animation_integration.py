#!/usr/bin/env python3
"""
Test the animation integration without requiring a full GUI.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from display.gui.gui_plot_manager import PlotManager

def test_animation_integration():
    """Test that PlotManager animation features work."""
    print("Testing PlotManager animation integration...")
    
    # Create plot manager
    mgr = PlotManager()
    
    # Test animation support detection
    assert mgr.has_animation_support("Smile (K/S vs IV)")
    assert mgr.has_animation_support("Synthetic Surface (Smile)")
    assert not mgr.has_animation_support("Term (ATM vs T)")
    print("✓ Animation support detection works")
    
    # Test animation state methods
    assert not mgr.is_animation_active()
    assert mgr.get_animation_speed() == 120  # Default speed
    print("✓ Animation state methods work")
    
    # Test animation control methods
    assert not mgr.start_animation()  # Should return False when no animation
    assert not mgr.pause_animation()  # Should return False when no animation
    mgr.stop_animation()  # Should not raise error
    print("✓ Animation control methods work")
    
    # Test speed setting
    mgr.set_animation_speed(500)
    assert mgr.get_animation_speed() == 500
    
    # Test bounds
    mgr.set_animation_speed(10)  # Too low
    assert mgr.get_animation_speed() >= 50
    
    mgr.set_animation_speed(5000)  # Too high
    assert mgr.get_animation_speed() <= 2000
    print("✓ Animation speed control works")
    
    # Test animated plotting with mock data
    fig, ax = plt.subplots()
    
    settings = {
        "plot_type": "Smile (K/S vs IV)",
        "target": "MOCK_TICKER",
        "asof": "2023-01-01",
        "T_days": 30,
        "max_expiries": 6
    }
    
    # This should return False since MOCK_TICKER doesn't exist
    result = mgr.plot_animated(ax, settings)
    assert not result
    print("✓ Animated plotting handles missing data gracefully")
    
    # Test unsupported plot type
    settings["plot_type"] = "Term (ATM vs T)"
    result = mgr.plot_animated(ax, settings)
    assert not result
    print("✓ Animated plotting rejects unsupported plot types")
    
    plt.close(fig)
    print("✓ All animation integration tests passed!")

if __name__ == "__main__":
    test_animation_integration()