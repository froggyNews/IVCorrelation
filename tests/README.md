# Configuration Tests for IVCorrelation

This directory contains unit tests for experimenting with configuration values that influence computations without using the GUI.

## Overview

The tests exercise the same underlying functions the GUI calls, allowing you to experiment with different configuration settings and understand their impact on surface building and analysis.

## Test Files

### `test_surfaces.py`
Main test file containing core configuration tests:

- `test_use_atm_only_filters_rows`: Tests ATM-only filtering
- `test_tenor_bins_configuration`: Tests different tenor bin configurations
- `test_moneyness_bins_configuration`: Tests different moneyness bin configurations 
- `test_max_expiries_limits_data`: Tests expiry limiting functionality
- `test_build_surfaces_with_config`: Tests the high-level build_surfaces function
- `test_configuration_isolation`: Tests that different configs produce predictable results

### `test_config_edge_cases.py`
Additional tests covering edge cases and multi-ticker scenarios:

- `test_multiple_tickers_configuration`: Tests consistency across multiple tickers
- `test_empty_configuration_results`: Tests graceful handling of empty results
- `test_synthetic_surface_with_config`: Tests synthetic ETF construction
- `test_combine_surfaces_functionality`: Tests surface combination logic
- `test_configuration_caching_behavior`: Tests caching behavior with different configs

## Running the Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_surfaces.py -v

# Run with output to see detailed results
python -m pytest tests/test_surfaces.py -v -s

# Run a specific test
python -m pytest tests/test_surfaces.py::test_use_atm_only_filters_rows -v
```

## Key Configuration Parameters Tested

### PipelineConfig Parameters

- **`use_atm_only`**: Boolean to filter only ATM options (should reduce data points)
- **`tenors`**: Tuple of tenor bins (e.g., (7, 30, 60, 90, 180, 365))
- **`mny_bins`**: Tuple of moneyness bins (e.g., ((0.80, 0.90), (0.95, 1.05), (1.10, 1.25)))
- **`max_expiries`**: Optional limit on number of expiries (reduces data when set)
- **`pillar_days`**: Tuple of pillar days for ATM time series
- **`cache_dir`**: Directory for disk cache

## Test Data

Tests use lightweight in-memory SQLite databases with deterministic sample data to ensure:
- Fast execution
- Reproducible results  
- No external dependencies
- Clean isolation between tests

## Expected Relationships

The tests validate these expected behaviors:

1. **ATM-only filtering** → Fewer or equal data points
2. **Fewer tenor bins** → Fewer columns in result grids
3. **Fewer moneyness bins** → Fewer rows in result grids
4. **Lower max_expiries** → Fewer data points overall
5. **More restrictive configs** → Fewer total data points than permissive configs

## Using These Tests for Configuration Experiments

You can modify the test configurations to experiment with different settings:

```python
# Example: Test custom tenor configuration
cfg_custom = PipelineConfig(
    tenors=(15, 45, 120),  # Custom tenor days
    use_atm_only=True,
    max_expiries=2
)
```

The tests provide a framework for understanding how configuration changes affect the analysis pipeline without needing to run the full GUI application.