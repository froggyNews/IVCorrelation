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
