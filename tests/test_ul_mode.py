#!/usr/bin/env python3
"""Test script to debug ul weight mode conversion."""

from analysis.unified_weights import WeightConfig

def test_ul_mode_conversion():
    """Test the from_legacy_mode conversion for 'ul'."""
    print("Testing 'ul' mode conversion...")
    
    config = WeightConfig.from_legacy_mode("ul")
    print(f"Method: {config.method}")
    print(f"Feature set: {config.feature_set}")
    print(f"Expected: CORRELATION method with UNDERLYING_PX features")
    
    # Test through the actual synthetic ETF interface
    print("\nTesting through synthetic ETF interface...")
    from analysis.analysis_synthetic_etf import SyntheticETFBuilder, SyntheticETFConfig
    
    cfg = SyntheticETFConfig(
        target="SPY",
        peers=("QQQ", "IWM"),
        weight_mode="ul"
    )
    
    builder = SyntheticETFBuilder(cfg)
    try:
        weights = builder.compute_weights()
        print(f"✓ Weights computed successfully: {weights}")
    except Exception as e:
        print(f"✗ Weight computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ul_mode_conversion()
