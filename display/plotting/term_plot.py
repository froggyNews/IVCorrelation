# display/plotting/term_plot.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict

from analysis.confidence_bands import Bands
from volModel.termFit import fit_term_structure, term_structure_iv


def plot_atm_term_structure(
    ax: plt.Axes,
    atm_df: pd.DataFrame,
    x_units: str = "years",   # "years" or "days"
    fit: bool = True,
    show_ci: bool = False,    # draw CI bars if present
    degree: int = 2,
) -> None:
    if atm_df is None or atm_df.empty:
        ax.text(0.5, 0.5, "No ATM data", ha="center", va="center")
        return

    x = atm_df["T"].to_numpy(float)
    y = atm_df["atm_vol"].to_numpy(float)

    if x_units == "days":
        x_plot = x * 365.25
        x_label = "Time to Expiry (days)"
    else:
        x_plot = x
        x_label = "Time to Expiry (years)"

    if show_ci and {"atm_lo","atm_hi"}.issubset(atm_df.columns):
        y_lo = atm_df["atm_lo"].to_numpy(float)
        y_hi = atm_df["atm_hi"].to_numpy(float)
        if np.isfinite(y_lo).any() and np.isfinite(y_hi).any():
            yerr = np.vstack([np.clip(y - y_lo, 0, None), np.clip(y_hi - y, 0, None)])
            ax.errorbar(x_plot, y, yerr=yerr, fmt="o", ms=4.5, capsize=3, alpha=0.9, label="ATM (fit) ± CI")
        else:
            ax.scatter(x_plot, y, s=30, alpha=0.9, label="ATM (fit)")
    else:
        ax.scatter(x_plot, y, s=30, alpha=0.9, label="ATM (fit)")

    if fit and len(x) > degree:
        try:
            params = fit_term_structure(x, y, degree=degree)
            grid = np.linspace(x.min(), x.max(), 200)
            fit_y = term_structure_iv(grid, params)
            grid_plot = grid * 365.25 if x_units == "days" else grid
            ax.plot(grid_plot, fit_y, linestyle="--", alpha=0.6, label="Term fit")
        except Exception:
            pass

    ax.set_xlabel(x_label)
    ax.set_ylabel("Implied Vol (ATM)")
    ax.legend(loc="best", fontsize=8)

def plot_synthetic_etf_term_structure(
    ax: plt.Axes,
    bands: Bands,
    *,
    label: str = "Synthetic ATM",
    line_kwargs: Optional[Dict] = None,
) -> Bands:
    """Plot synthetic ETF ATM term structure using pre-computed bands."""

    ax.fill_between(bands.x, bands.lo, bands.hi, alpha=0.20, label=f"CI ({int(bands.level*100)}%)")

    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 1.8)
    ax.plot(bands.x, bands.mean, label=label, **line_kwargs)

    ax.set_xlabel("Pillar (days)")
    ax.set_ylabel("Implied Vol (ATM)")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc="best", fontsize=8)

    return bands
