# IVCorrelation

This repository contains utilities for smoothing implied volatility surfaces using thin-plate splines. Use `fit_spline.py` to fit continuous volatility surfaces from scattered data.
=======

This repository contains a simple option pricing example.

## Scripts

- `pricing.py` &ndash; example pricing utility providing conservative, baseline, 
  and aggressive prices calculated using Black--Scholes or Bachelier models 
  depending on the interest rate.

Run the script directly to see sample output:

```bash
python pricing.py

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

Utilities for processing implied volatility surfaces for correlation analysis.

## Surface Projection

`surface_projection.py` provides `project_surfaces` which projects a list of
smoothed IV surfaces onto a common `(K, T)` grid. The function determines a
global strike range `[K_min, K_max]` and maturity range `[Tmin, Tmax]` across
all assets and interpolates each surface onto an `n x n` grid. Missing values
are filled using nearest-neighbor interpolation with a final conservative
fallback to zero.
