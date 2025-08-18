# display/plotting/smile_plot.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt

from volModel.sviFit import fit_svi_slice, svi_smile_iv
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv
from .confidence_bands import (
    svi_confidence_bands,
    sabr_confidence_bands,
    synthetic_etf_confidence_bands,
    Bands,
)
from display.plotting.anim_utils import add_keyboard_toggles, add_legend_toggles

ModelName = Literal["svi", "sabr"]


# ------------------
# helpers
# ------------------
def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return a.ravel() if a.ndim > 1 else a


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    mask = None
    for a in arrs:
        a = np.asarray(a, float)
        m = np.isfinite(a)
        mask = m if mask is None else (mask & m)
    return mask if mask is not None else np.array([], dtype=bool)


# ------------------
# main
# ------------------
def fit_and_plot_smile(
    ax: plt.Axes,
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    *,
    model: ModelName = "svi",
    moneyness_grid: Tuple[float, float, int] = (0.8, 1.2, 121),
    ci_level: float = 0.0,          # 0 disables CI
    n_boot: int = 200,              # used only if ci_level > 0
    show_points: bool = True,
    beta: float = 0.5,              # SABR beta
    label: Optional[str] = None,
    line_kwargs: Optional[Dict] = None,
    enable_svi_toggles: bool = False,   # legend/keyboard toggles (SVI only)
    use_checkboxes: bool = False,       # keep False by default; legend is primary
) -> Dict:
    """
    Plot observed points, model fit, and optional CI on moneyness (K/S).
    Returns dict: {params, rmse, T, S, series_map or None}
    """

    # ---- sanitize
    S = float(S)
    K = _as_1d(K)
    iv = _as_1d(iv)

    m = _finite_mask(K, iv)
    if not np.any(m):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": S, "series_map": None}

    K = K[m]
    iv = iv[m]

    # ---- grid in strike space via moneyness
    mlo, mhi, n = moneyness_grid
    m_grid = np.linspace(float(mlo), float(mhi), int(n))
    K_grid = m_grid * S

    # ---- artists map for legend toggles
    series_map: Dict[str, list] = {}

    # ---- observed points
    if show_points:
        pts = ax.scatter(K / S, iv, s=20, alpha=0.85, label="Observed")
        if enable_svi_toggles and model == "svi":
            series_map["Observed Points"] = [pts]

    # ---- fit + optional CI
    bands: Optional[Bands] = None
    if model == "svi":
        params = fit_svi_slice(S, K, T, iv)
        y_fit = svi_smile_iv(S, K_grid, T, params)
        if ci_level and ci_level > 0:
            bands = svi_confidence_bands(S, K, T, iv, K_grid, level=float(ci_level), n_boot=int(n_boot))
    else:
        params = fit_sabr_slice(S, K, T, iv, beta=beta)
        y_fit = sabr_smile_iv(S, K_grid, T, params)
        if ci_level and ci_level > 0:
            bands = sabr_confidence_bands(S, K, T, iv, K_grid, beta=beta, level=float(ci_level), n_boot=int(n_boot))

    # ---- fit line
    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 2 if show_points else 1.6)
    fit_lbl = label or (f"{model.upper()} fit")
    fit_line = ax.plot(K_grid / S, y_fit, label=fit_lbl)
    if enable_svi_toggles and model == "svi":
        series_map["SVI Fit"] = list(fit_line)

    # ---- confidence bands
    if bands is not None:
        ci_fill = ax.fill_between(bands.x / S, bands.lo, bands.hi, alpha=0.20, label=f"{int(ci_level*100)}% CI")
        ci_mean = ax.plot(bands.x / S, bands.mean, lw=1, alpha=0.6, linestyle="--")
        if enable_svi_toggles and model == "svi":
            series_map["SVI Confidence Interval"] = [ci_fill, *ci_mean]

    # ---- ATM marker (not part of toggles / legend)
    ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.85, label="_nolegend_")

    # ---- axes / legend
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Implied Vol")
    if not ax.get_legend():
        ax.legend(loc="best", fontsize=8)

    # ---- legend-first toggle system (primary), keyboard helpers
    if enable_svi_toggles and model == "svi" and series_map:
        add_legend_toggles(ax, series_map)  # your improved legend system
        add_keyboard_toggles(ax.figure, series_map, keymap={
            "o": "Observed Points",
            "f": "SVI Fit",
            "c": "SVI Confidence Interval",
        })
        # checkboxes are optional; keep off unless explicitly asked
        if use_checkboxes:
            from display.plotting.anim_utils import add_checkboxes
            add_checkboxes(ax.figure, series_map)

    # ---- fit quality
    rmse = float(params.get("rmse", np.nan)) if isinstance(params, dict) else np.nan

    return {
        "params": params,
        "rmse": rmse,
        "T": float(T),
        "S": float(S),
        "series_map": series_map if (enable_svi_toggles and model == "svi") else None,
    }


def plot_synthetic_etf_smile(
    ax: plt.Axes,
    surfaces: Dict[str, np.ndarray],
    weights: Dict[str, float],
    grid: np.ndarray,
    *,
    level: float = 0.68,
    n_boot: int = 200,
    label: Optional[str] = "Synthetic ETF",
    line_kwargs: Optional[Dict] = None,
) -> Bands:
    """Plot synthetic ETF smile with confidence bands.

    Parameters
    ----------
    ax : plt.Axes
        Axis to render on.
    surfaces : dict
        ``{ticker -> iv_array}`` aligned to ``grid``.
    weights : dict
        ``{ticker -> weight}`` for the synthetic combination.
    grid : np.ndarray
        Strike or moneyness grid.
    level : float, optional
        Confidence level for the bands.
    n_boot : int, optional
        Number of bootstrap samples.
    label : str, optional
        Legend label for the mean line.
    line_kwargs : dict, optional
        Extra kwargs for the mean line.

    Returns
    -------
    Bands
        Bootstrap bands for the synthetic smile.
    """
    bands = synthetic_etf_confidence_bands(
        surfaces=surfaces,
        weights=weights,
        grid_K=np.asarray(grid, float),
        level=level,
        n_boot=n_boot,
    )

    ax.fill_between(bands.x, bands.lo, bands.hi, alpha=0.20, label=f"CI ({int(level*100)}%)")

    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 1.8)
    ax.plot(bands.x, bands.mean, label=label, **line_kwargs)

    ax.set_xlabel("Strike / Moneyness")
    ax.set_ylabel("Implied Vol")
    if not ax.get_legend():
        ax.legend(loc="best", fontsize=8)

    return bands
