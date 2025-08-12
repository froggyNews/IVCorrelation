# display/plotting/term_plot.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_atm_term_structure(
    ax: plt.Axes,
    atm_df: pd.DataFrame,
    x_units: str = "years",   # "years" or "days"
    connect: bool = True,
    smooth: bool = False,
    show_ci: bool = False,    # draw CI bars if present
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
