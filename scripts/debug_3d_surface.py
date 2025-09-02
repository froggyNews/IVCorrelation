#!/usr/bin/env python3
"""
Debug script to isolate the 3D surface plotting issue.
"""
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_matplotlib_backend():
    """Test matplotlib backend and figure creation."""
    print("Testing matplotlib backend...")
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Backend module: {type(matplotlib.backend_bases)}")
    
    # Test basic figure creation
    try:
        fig = plt.figure(figsize=(8, 6))
        print(f"Basic figure created: {type(fig)}")
        print(f"Figure dpi: {fig.dpi}")
        print(f"Figure canvas: {fig.canvas}")
        plt.close(fig)
        print("✅ Basic figure creation works")
    except Exception as e:
        print(f"❌ Basic figure creation failed: {e}")
        return False
    
    # Test 3D figure creation
    try:
        fig3d = plt.figure(figsize=(8, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        print(f"3D figure created: {type(fig3d)}")
        print(f"3D figure dpi: {fig3d.dpi}")
        print(f"3D axes: {type(ax3d)}")
        plt.close(fig3d)
        print("✅ 3D figure creation works")
    except Exception as e:
        print(f"❌ 3D figure creation failed: {e}")
        return False
    
    return True

def test_surface_viewer():
    """Test surface viewer functions."""
    print("\nTesting surface viewer...")
    
    try:
        from display.plotting.surface_viewer import show_surface_3d, show_surface_heatmap
        
        # Create dummy data
        import pandas as pd
        import numpy as np
        
        # Create a simple surface grid
        moneyness = np.linspace(0.8, 1.2, 5)
        tenors = np.array([30, 60, 90])
        iv_data = np.random.uniform(0.2, 0.5, (len(moneyness), len(tenors)))
        
        # Create DataFrame
        df = pd.DataFrame(iv_data, index=moneyness, columns=tenors)
        print(f"Test surface data shape: {df.shape}")
        
        # Test 3D surface
        try:
            fig3d = show_surface_3d(df, "Test 3D Surface")
            print(f"3D surface created: {type(fig3d)}")
            if fig3d:
                print(f"3D figure dpi: {fig3d.dpi}")
                plt.close(fig3d)
                print("✅ 3D surface creation works")
            else:
                print("❌ 3D surface returned None")
                return False
        except Exception as e:
            print(f"❌ 3D surface creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test heatmap
        try:
            fig2d = show_surface_heatmap(df, "Test Heatmap")
            print(f"Heatmap created: {type(fig2d)}")
            if fig2d:
                print(f"Heatmap figure dpi: {fig2d.dpi}")
                plt.close(fig2d)
                print("✅ Heatmap creation works")
            else:
                print("❌ Heatmap returned None")
                return False
        except Exception as e:
            print(f"❌ Heatmap creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Could not import surface viewer: {e}")
        return False

def test_vol_structure_plots():
    """Test the new vol structure plotting functions."""
    print("\nTesting vol structure plots...")
    
    try:
        from display.plotting.vol_structure_plots import plot_3d_vol_surface
        
        # Test with a simple case
        print("Testing plot_3d_vol_surface...")
        
        # This should fail gracefully if no data, but not crash
        try:
            fig = plot_3d_vol_surface("IONQ", "2024-08-08", mode="3d")
            if fig:
                print(f"3D vol surface created: {type(fig)}")
                print(f"Figure dpi: {fig.dpi}")
                plt.close(fig)
                print("✅ 3D vol surface creation works")
            else:
                print("3D vol surface returned None (expected if no data)")
                print("✅ 3D vol surface function works (returns None when no data)")
        except Exception as e:
            print(f"❌ 3D vol surface creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Could not import vol structure plots: {e}")
        return False

def main():
    """Run all debug tests."""
    print("=" * 60)
    print("Debugging 3D Surface Plotting Issues")
    print("=" * 60)
    
    success = True
    
    # Test matplotlib backend
    success &= test_matplotlib_backend()
    
    # Test surface viewer
    success &= test_surface_viewer()
    
    # Test vol structure plots
    success &= test_vol_structure_plots()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Matplotlib and plotting functions work correctly.")
        print("\nThe issue might be in the GUI integration or data availability.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()
