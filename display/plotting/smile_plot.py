# display/plotting/smile_plot.py
from __future__ import annotations
from typing import Literal, Optional, Tuple, Dict
import numpy as np

import matplotlib.pyplot as plt

from volModel.sviFit import fit_svi_slice, svi_smile_iv
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv
from .confidence_bands import svi_confidence_bands, sabr_confidence_bands, Bands
from src.viz.anim_utils import (
    animate_smile_over_time,
    add_checkboxes,
    add_keyboard_toggles,
    add_legend_toggles,
    apply_profile_visibility,
)

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
    label: Optional[str] = None,
    line_kwargs: Optional[Dict] = None,
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
    lw_default = 2 if show_points else 1.5
    line_kwargs = line_kwargs.copy() if line_kwargs else {}
    line_kwargs.setdefault("lw", lw_default)
    ax.plot(K_grid / S, y_fit, label=label or f"{model.upper()} fit", **line_kwargs)

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


def main():
    """Demonstration of the smile animation utilities."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Smile animation demo")
    parser.add_argument("--save", type=str, default=None, help="Output path for MP4/GIF")
    parser.add_argument(
        "--profile", type=str, default=None, help="JSON dict for initial visibility"
    )
    args = parser.parse_args()

    # generate synthetic data
    T, N_k = 40, 31
    k = np.linspace(0.7, 1.3, N_k)
    base = 0.2 + 0.1 * (k - 1.0) ** 2
    iv_syn_tk = base + 0.02 * np.sin(np.linspace(0, 4 * np.pi, T))[:, None]
    iv_raw_tk = iv_syn_tk + 0.01 * np.random.randn(T, N_k)
    ci_lo_tk = iv_syn_tk - 0.015
    ci_hi_tk = iv_syn_tk + 0.015
    dates = [f"t{i}" for i in range(T)]

    fig, ani, series_map = animate_smile_over_time(
        k,
        iv_syn_tk,
        dates,
        iv_raw_tk=iv_raw_tk,
        ci_lo_tk=ci_lo_tk,
        ci_hi_tk=ci_hi_tk,
    )

    add_checkboxes(fig, series_map)
    add_keyboard_toggles(fig, series_map, keymap={"r": "Raw", "s": "Synthetic", "c": "CI"})
    ax = fig.axes[0]
    add_legend_toggles(ax, series_map)

    if args.profile:
        profile = json.loads(args.profile)
        apply_profile_visibility(series_map, profile)

    if args.save:
        ani.save(args.save, writer="ffmpeg", dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
