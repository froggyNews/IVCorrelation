# IVCorrelation

This repository provides a small helper for combining normalized option
volatility surfaces across multiple tickers into a synthetic ETF surface.

The main entry point is `combine_surfaces` located in `src/synthetic_etf.py`.
It implements the formula

```
σ_ETF(K, T) = sum_i[ρ_i * ω_i(K,T) * σ_i(K,T)] / sum_i[ρ_i * ω_i(K,T)]
```

Where:

- `σ_i(K,T)` is the implied volatility surface for ticker `i` at strike `K`
  and maturity `T`.
- `ρ_i` is the weight assigned to each ticker.
- `ω_i(K,T)` is an optional weight grid for each ticker.

The function returns a `(K, T)` grid for each trading day present in the
input data.
