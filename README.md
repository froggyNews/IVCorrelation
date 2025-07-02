# IVCorrelation

This project provides a skeleton framework for constructing synthetic ETF volatility surfaces.

The code base is organized into several modules:

- **data_collection** – routines to download or scrape option chain data.
- **preprocessing** – tools for building individual volatility surfaces.
- **aggregation** – utilities for combining ETF surfaces into theme-level surfaces.
- **smoothing** – helpers for surface smoothing and option pricing.
- **validation** – backtesting and validation scripts.

Run the pipeline using the package entry point:

```bash
python -m volsurf
```

Each module currently contains placeholder functions that should be filled in with concrete implementations.
