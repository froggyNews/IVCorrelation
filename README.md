# IVCorrelation

This repository configures peer tickers and data collection parameters for
implied volatility analysis.

## Configuration

Edit `config.yaml` to customize:

- `peer_tickers` organized by theme, such as the default `quantum` tickers
  (`QUBT`, `IONQ`, `RGTI`).
- `D`: number of trading days of history to collect.
- `min_maturity_days` and `max_maturity_days`: range for option maturities.
- `output_dir`: directory where data should be written.

The provided configuration is an example and can be adjusted as needed.
