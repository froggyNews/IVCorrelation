# display/plotting/smile_plot.py
from __future__ import annotations
from typing import Literal, Optional, Tuple
import numpy as np

import matplotlib.pyplot as plt

from volModel.sviFit import fit_svi_slice, svi_smile_iv
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv
from .confidence_bands import svi_confidence_bands, sabr_confidence_bands, Bands

ModelName = Literal["svi", "sabr"]

def fit_and_plot_smile(
    ax: plt.Axes,
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    model: ModelName = "svi",
    moneyness_grid: Tuple[float, float, int] = (0.7, 1.3, 101),
    ci_level: float = 0.68,
    show_points: bool = True,
    beta: float = 0.5,
) -> dict:
    """
    Scatter observed points + fitted curve + shaded CI.
    Returns a dict with params and basic stats.
    """
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    mlo, mhi, n = moneyness_grid
    m_grid = np.linspace(mlo, mhi, int(n))
    K_grid = m_grid * float(S)

    if show_points:
        ax.scatter(K / S, iv, s=20, alpha=0.7, label="Observed")

    if model == "svi":
        params = fit_svi_slice(S, K, T, iv)
        y_fit = svi_smile_iv(S, K_grid, T, params)
        bands: Optional[Bands] = None
        if ci_level and ci_level > 0:
            bands = svi_confidence_bands(S, K, T, iv, K_grid, level=ci_level, n_boot=200)
    else:
        params = fit_sabr_slice(S, K, T, iv, beta=beta)
        y_fit = sabr_smile_iv(S, K_grid, T, params)
        bands = None
        if ci_level and ci_level > 0:
            bands = sabr_confidence_bands(S, K, T, iv, K_grid, beta=beta, level=ci_level, n_boot=200)

    # line
    ax.plot(K_grid / S, y_fit, lw=2, label=f"{model.upper()} fit")

    # bands
    if bands is not None:
        ax.fill_between(bands.x / S, bands.lo, bands.hi, alpha=0.20, label=f"{int(ci_level*100)}% CI")
        ax.plot(bands.x / S, bands.mean, lw=1, alpha=0.6, linestyle="--")

    ax.axvline(1.0, color="grey", lw=1, ls="--")
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Implied Vol")
    ax.legend(loc="best", fontsize=8)

    # simple fit quality
    rmse = float(params.get("rmse", np.nan))
    return {"params": params, "rmse": rmse, "T": T, "S": float(S)}
