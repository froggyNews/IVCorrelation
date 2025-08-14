# Examples Directory

This directory contains demonstration scripts and examples for IVCorrelation features.

## Caching Improvements Demo

The `caching_improvements_demo.py` script demonstrates the enhanced caching system implemented to address performance and efficiency concerns.

### Features Demonstrated:

1. **Parquet vs CSV Performance**: Shows file size reduction and I/O performance benefits
2. **Smart Configuration-Based Caching**: Demonstrates automatic cache invalidation based on configuration changes
3. **Cache Management Utilities**: Shows inspection and cleanup tools

### Running the Demo:

```bash
python examples/caching_improvements_demo.py
```

### Key Benefits Shown:

- **41%+ file size reduction** with Parquet format
- **Smart cache invalidation** prevents unnecessary recomputation
- **Configuration-aware caching** automatically handles settings changes
- **Backwards compatibility** with CSV format when needed

This demo answers the original questions from issue #47:
- "Should this be cached in CSVs or would using parquet be more effective?" → **Parquet is more effective**
- "Is there a full run done all over again, or are we able to work with cached info?" → **Smart caching avoids full recomputation**