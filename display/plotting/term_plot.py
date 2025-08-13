# display/plotting/term_plot.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from display.plotting.confidence_bands import bootstrap_bands, Bands


def _polynomial_fit_fn(x: np.ndarray, y: np.ndarray, degree: int = 2) -> dict:
    """Fit polynomial to term structure data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Remove NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < degree + 1:
        # Not enough points for the requested degree
        degree = max(1, np.sum(mask) - 1)
    
    if np.sum(mask) < 2:
        # Fallback to constant
        return {"coeffs": [np.nanmean(y)], "degree": 0}
    
    try:
        coeffs = np.polyfit(x[mask], y[mask], degree)
        return {"coeffs": coeffs, "degree": degree}
    except Exception:
        # Fallback to linear
        try:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            return {"coeffs": coeffs, "degree": 1}
        except Exception:
            # Ultimate fallback
            return {"coeffs": [np.nanmean(y)], "degree": 0}


def _polynomial_pred_fn(params: dict, x_grid: np.ndarray) -> np.ndarray:
    """Predict using polynomial fit."""
    return np.polyval(params["coeffs"], x_grid)


def generate_term_structure_confidence_bands(
    T: np.ndarray,
    atm_vol: np.ndarray,
    level: float = 0.68,
    n_boot: int = 100,
    fit_degree: int = 2,
    grid_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate confidence bands for ATM term structure using bootstrap.
    
    Parameters:
    -----------
    T : np.ndarray
        Time to expiry values (years)
    atm_vol : np.ndarray
        ATM volatility values
    level : float
        Confidence level (0.68 for ~1 sigma)
    n_boot : int
        Number of bootstrap samples
    fit_degree : int
        Polynomial degree for fitting
    grid_points : int
        Number of points in interpolation grid
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (grid_T, lower_bound, upper_bound)
    """
    T = np.asarray(T, dtype=float)
    atm_vol = np.asarray(atm_vol, dtype=float)
    
    # Remove NaN values
    mask = np.isfinite(T) & np.isfinite(atm_vol)
    if np.sum(mask) < 3:
        # Not enough data for meaningful confidence bands
        return np.array([]), np.array([]), np.array([])
    
    T_clean = T[mask]
    vol_clean = atm_vol[mask]
    
    # Create grid for interpolation
    T_min, T_max = T_clean.min(), T_clean.max()
    T_grid = np.linspace(T_min, T_max, grid_points)
    
    # Generate bootstrap confidence bands
    try:
        bands = bootstrap_bands(
            x=T_clean,
            y=vol_clean,
            fit_fn=lambda x, y: _polynomial_fit_fn(x, y, degree=fit_degree),
            pred_fn=_polynomial_pred_fn,
            grid=T_grid,
            level=level,
            n_boot=n_boot,
        )
        return T_grid, bands.lo, bands.hi
    except Exception:
        # Fallback: no confidence bands
        return np.array([]), np.array([]), np.array([])


def plot_atm_term_structure(
    ax: plt.Axes,
    atm_df: pd.DataFrame,
    x_units: str = "years",   # "years" or "days"
    connect: bool = True,
    smooth: bool = False,
    show_ci: bool = False,    # draw CI bars if present
    ci_level: float = 0.68,   # confidence level for bands
    generate_ci: bool = True, # automatically generate CI if not present
    n_boot: int = 100,        # bootstrap samples for CI generation
) -> None:
    if atm_df is None or atm_df.empty:
        ax.text(0.5, 0.5, "No ATM data", ha="center", va="center")
        return

    x = atm_df["T"].to_numpy()
    y = atm_df["atm_vol"].to_numpy()

    if x_units == "days":
        x_plot = x * 365.25
        x_label = "Time to Expiry (days)"
    else:
        x_plot = x
        x_label = "Time to Expiry (years)"

    # Handle confidence intervals
    ci_rendered = False
    if show_ci:
        # First try to use pre-computed confidence intervals
        if {"atm_lo","atm_hi"}.issubset(atm_df.columns):
            y_lo = atm_df["atm_lo"].to_numpy(float)
            y_hi = atm_df["atm_hi"].to_numpy(float)
            if np.isfinite(y_lo).any() and np.isfinite(y_hi).any():
                yerr = np.vstack([np.clip(y - y_lo, 0, None), np.clip(y_hi - y, 0, None)])
                ax.errorbar(x_plot, y, yerr=yerr, fmt="o", ms=4.5, capsize=3, alpha=0.9, label="ATM (fit) ± CI")
                ci_rendered = True
        
        # If no pre-computed CI and generate_ci is True, generate them
        if not ci_rendered and generate_ci and len(x) >= 3:
            try:
                T_grid, ci_lo, ci_hi = generate_term_structure_confidence_bands(
                    T=x, atm_vol=y, level=ci_level, n_boot=n_boot
                )
                
                if len(T_grid) > 0:
                    # Convert to plot units
                    if x_units == "days":
                        T_grid_plot = T_grid * 365.25
                    else:
                        T_grid_plot = T_grid
                    
                    # Plot confidence band as filled area
                    ax.fill_between(T_grid_plot, ci_lo, ci_hi, alpha=0.2, label=f"CI ({ci_level:.0%})")
                    ax.scatter(x_plot, y, s=30, alpha=0.9, label="ATM (fit)")
                    ci_rendered = True
            except Exception as e:
                print(f"Failed to generate confidence bands: {e}")
    
    # Fallback: simple scatter plot
    if not ci_rendered:
        ax.scatter(x_plot, y, s=30, alpha=0.9, label="ATM (fit)")

    if connect and len(x_plot) > 1:
        order = np.argsort(x_plot)
        ax.plot(x_plot[order], y[order], alpha=0.5, linewidth=1.2)

    if smooth and len(x_plot) >= 4:
        order = np.argsort(x_plot)
        xp = x_plot[order]; yp = y[order]
        try:
            coeff = np.polyfit(xp, yp, 2)
            grid = np.linspace(xp.min(), xp.max(), 200)
            fit = np.polyval(coeff, grid)
            ax.plot(grid, fit, linestyle="--", alpha=0.6, label="Quadratic fit")
        except Exception:
            pass

    ax.set_xlabel(x_label)
    ax.set_ylabel("Implied Vol (ATM)")
    ax.legend(loc="best", fontsize=8)
