# IVCorrelation

This repository contains helper functions for calculating option-related weights.

## Correlation weights `ρ_i`
The function `correlation_weights` computes a normalized weight for each stock
based on its empirical return correlation with a theme proxy series. By
default, absolute correlations are used so weights remain non‐negative and sum
to one.

## Liquidity weights `ω_i(K, T)`
The function `liquidity_weights` produces strike‑ and maturity‑level liquidity
weights using smoothed option volume. Volume is smoothed across neighbouring
strikes before being normalized within each maturity.

See `weights.py` for implementation details.

Utilities for processing implied volatility surfaces for correlation analysis.

## Surface Projection

`surface_projection.py` provides `project_surfaces` which projects a list of
smoothed IV surfaces onto a common `(K, T)` grid. The function determines a
global strike range `[K_min, K_max]` and maturity range `[Tmin, Tmax]` across
all assets and interpolates each surface onto an `n x n` grid. Missing values
are filled using nearest-neighbor interpolation with a final conservative
fallback to zero.
