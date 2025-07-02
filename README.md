# IVCorrelation

This repository contains utilities for working with implied volatility data.

## Option Chain Utilities

The `option_data.py` module provides a helper function to download option
chains using [yfinance](https://github.com/ranaroussi/yfinance). Example
usage:

```python
from option_data import fetch_historical_option_chain

chains = fetch_historical_option_chain(
    ticker="AAPL",
    start_date="2023-01-01",
    end_date="2023-01-05",
)
```

Daily option chains are saved to `data/raw_option_chains/{ticker}_{date}.csv`.

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

