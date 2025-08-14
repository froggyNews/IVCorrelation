# IVCorrelation - Implied Volatility Correlation Analysis Tool

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Quick Setup (30 seconds)
- Install Python dependencies: `pip install -r requirements.txt` -- takes 30 seconds. NEVER CANCEL.
- Install pytest for testing: `pip install pytest` -- takes 10 seconds.

### Testing (3 seconds) 
- Run all tests: `python -m pytest tests/ -v` -- takes 3 seconds. NEVER CANCEL.
- Test specific file: `python -m pytest tests/test_surfaces.py -v`
- Tests cover configuration changes, synthetic ETF construction, surface building, and edge cases

### Running the Application

#### CLI Analysis (1-2 seconds per run)
- Main demo script: `python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ IWM --no-show`
- View help: `python scripts/scripts_synthetic_etf_demo.py --help`
- Caching demo: `python examples/caching_improvements_demo.py` -- takes 1 second. NEVER CANCEL.

#### GUI Application (Limited in CI environments)
- Start GUI: `python display/gui/browser.py` 
- **LIMITATION**: Requires tkinter which is not available in many CI environments
- Use `--no-show` flag in scripts to bypass GUI requirements

### Data and Network Requirements

#### Existing Data (Recommended for development)
- Uses SQLite database: `data/iv_data.db` with 35+ tickers (SPY, QQQ, AAPL, etc.)
- Check available data: 
  ```python
  from analysis.analysis_pipeline import available_tickers, available_dates
  print(available_tickers())  # Shows ~35 tickers
  ```

#### Data Ingestion (Network dependent)
- Download fresh data: `--ingest --tickers SPY QQQ` flag in scripts
- **LIMITATION**: Requires internet access to Yahoo Finance (often blocked in CI)
- Data ingestion fails quickly (~1 second) if network blocked - this is expected
- Use existing data for development when network access is limited

## Validation Scenarios

### Essential Validation Steps
ALWAYS test these scenarios after making changes:

1. **Run test suite**: `python -m pytest tests/ -v` (3 seconds)
2. **Test basic analysis**: `python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ --no-show` (1.5 seconds)
3. **Test configuration changes**: `python -m pytest tests/test_surfaces.py::test_configuration_isolation -v`
4. **Validate caching**: `python examples/caching_improvements_demo.py` (1 second)

### Manual Test Scenarios
After making changes to analysis pipeline:
1. Test working weight modes: `--weight-mode corr`, `--weight-mode equal` 
2. **KNOWN ISSUES**: `--weight-mode pca` and `--weight-mode cosine` have bugs (see limitations below)
3. Test configuration isolation with different parameters
4. Verify cache invalidation works with config changes
5. Check synthetic ETF construction produces valid weights and metrics

## Key File Locations

### Core Analysis Engine
- `analysis/analysis_pipeline.py` - Main orchestration, caching, configuration
- `analysis/syntheticETFBuilder.py` - Synthetic ETF surface construction  
- `analysis/surface_builder.py` - IV surface building
- `analysis/correlation_builder.py` - Correlation calculations

### Data Management
- `data/data_pipeline.py` - Data enrichment and processing
- `data/iv_data.db` - SQLite database with options data
- `data/data_downloader.py` - Yahoo Finance integration

### Configuration and Testing
- `tests/test_surfaces.py` - Core configuration tests
- `tests/test_config_edge_cases.py` - Edge case testing
- `analysis/analysis_pipeline.py:PipelineConfig` - Main configuration class

### Entry Points
- `scripts/scripts_synthetic_etf_demo.py` - Main CLI demo
- `display/gui/browser.py` - GUI application (tkinter required)
- `examples/caching_improvements_demo.py` - Caching demonstration

## Timing Expectations and Commands

**CRITICAL**: All operations are very fast (1-3 seconds). Set timeouts appropriately.

- **Tests**: `python -m pytest tests/ -v` -- 3 seconds. NEVER CANCEL. Use timeout 30+ seconds.
- **Analysis**: `python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ --no-show` -- 1.5 seconds. NEVER CANCEL. Use timeout 30+ seconds.
- **Caching demo**: `python examples/caching_improvements_demo.py` -- 1 second. NEVER CANCEL. Use timeout 30+ seconds.
- **Dependency install**: `pip install -r requirements.txt` -- 30 seconds. NEVER CANCEL. Use timeout 60+ seconds.
- **Data ingestion** (when network available): May take 30-60 seconds per ticker. NEVER CANCEL. Use timeout 120+ seconds.

## Common Development Tasks

### Testing Configuration Changes
```bash
# Test ATM-only filtering
python -m pytest tests/test_surfaces.py::test_use_atm_only_filters_rows -v

# Test tenor configuration
python -m pytest tests/test_surfaces.py::test_tenor_bins_configuration -v

# Test cache invalidation
python -m pytest tests/test_config_edge_cases.py::test_configuration_caching_behavior -v
```

### Analysis Pipeline Development
```bash
# Test with different peer combinations
python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ IWM GOOGL --no-show

# Test working weight modes (corr and equal work, pca/cosine have bugs)
python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ --weight-mode equal --no-show

# Export results for inspection
python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ --no-show --export-dir /tmp/results
```

### Cache Management
```python
from analysis.analysis_pipeline import clear_all_caches, cleanup_disk_cache, PipelineConfig
clear_all_caches()  # Clear in-memory caches
cfg = PipelineConfig()
cleanup_disk_cache(cfg)  # Remove old disk cache files
```

## Project Structure Overview

```
IVCorrelation/
├── analysis/           # Core analytics engine (main development area)
├── data/               # Data management and SQLite database  
├── display/            # GUI (tkinter) and plotting
├── tests/              # Unit tests (fast, comprehensive)
├── scripts/            # CLI entry points
├── examples/           # Demonstrations and workflows
├── volModel/           # Volatility models (SVI, SABR)
└── requirements.txt    # Dependencies
```

## Important Notes

### Network and Environment Limitations
- **Data ingestion requires internet**: Yahoo Finance access often blocked in CI
- **GUI requires tkinter**: Not available in many CI environments
- **Use existing data**: `data/iv_data.db` has sufficient data for development
- **CLI workflows preferred**: More reliable than GUI in restricted environments

### Performance Characteristics  
- **Very fast execution**: All operations complete in 1-3 seconds
- **Smart caching**: Configuration-aware with 41%+ file size reduction using Parquet
- **Lightweight tests**: 18 tests run in 3 seconds total
- **No build process**: Pure Python project, no compilation required

### Configuration-Driven Analysis
- **PipelineConfig class**: Controls tenor bins, moneyness bins, ATM filtering, cache settings
- **Cache invalidation**: Automatic when configuration changes
- **Weight modes**: `corr` (correlation) and `equal` work reliably
- **Flexible surface construction**: ATM-only vs full surface, expiry limits

## Known Limitations

### Weight Mode Issues
- **PCA mode (`--weight-mode pca`)**: Has parameter mismatch bug in `pca_weights_from_atm_matrix()`
- **Cosine mode (`--weight-mode cosine`)**: Has import error for `analysis.data_pipeline`
- **Working modes**: Use `corr` (correlation) or `equal` weighting

### Environment Constraints  
- **GUI requires tkinter**: Not available in CI environments - use `--no-show` flag
- **Network dependency**: Data ingestion requires Yahoo Finance access (often blocked)
- **Database dependency**: Application requires existing `data/iv_data.db` for analysis

Always run the test suite before committing changes. Always test with existing data when network access is limited. Always use `--no-show` flag in scripts when GUI display is not available.