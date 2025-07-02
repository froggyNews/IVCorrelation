# IVCorrelation

This repository provides a Python script to compute weighted implied volatility statistics across multiple days and tickers.

The script `compute_volatility.py` expects a CSV file containing the following columns:

- `K` – strike level
- `T` – maturity
- `sigma` – individual implied volatility observation
- `corr_weight` – correlation weight for the observation
- `liq_weight` – liquidity weight for the observation

The script groups observations by `(K, T)` and calculates:

- **weighted mean** of `sigma`
- **weighted standard deviation**
- **low** and **high** bands: mean minus/plus the standard deviation

## Usage

```bash
python compute_volatility.py your_data.csv -o output.csv
```

If the `-o` option is omitted, results are printed to stdout.
