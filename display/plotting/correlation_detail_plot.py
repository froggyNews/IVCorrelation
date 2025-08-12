# display/plotting/correlation_detail_plot.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.pillars import build_atm_matrix, DEFAULT_PILLARS_DAYS
# surface construction

# optional: fallback weights from persisted correlations (if you keep that path)
try:
    from analysis.correlation_builder import peer_weights_from_correlations
    _HAS_PERSISTED_WEIGHTS = True
except Exception:
    _HAS_PERSISTED_WEIGHTS = False




# ---------------------------------------------------------------
# Corr → weights
# ---------------------------------------------------------------
def corr_weights_from_matrix(
    corr_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Convert correlations with `target` into normalized positive weights on `peers`."""
    target = target.upper()
    peers = [p.upper() for p in peers]
    # Pull the target column; drop target if present in peers
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        # fallback equal
        return pd.Series(1.0 / max(len(peers), 1), index=peers, dtype=float)
    return (s / total).fillna(0.0)

# ---------------------------------------------------------------
# Correlation: compute and plot (optionally show weights)
# ---------------------------------------------------------------
def compute_and_plot_correlation(
    ax: plt.Axes,
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    *,
    target: Optional[str] = None,
    peers: Optional[Iterable[str]] = None,
    pillars_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    atm_band: float = 0.05,
    tol_days: float = 7.0,
    min_pillars: int = 3,
    corr_method: str = "pearson",    # or "spearman" | "kendall"
    demean_rows: bool = False,
    show_values: bool = True,
    clip_negative: bool = True,
    power: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Build ATM-by-pillar matrix and correlation matrix; draw heatmap.
    If `target` and `peers` are provided, also compute and display corr-based weights.
    Returns: (atm_df, corr_df, weights or None)
    """
    tickers = [t.upper() for t in tickers]
    atm_df, corr_df = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        min_pillars=min_pillars,
        corr_method=corr_method,
        demean_rows=demean_rows,
    )

    weights = None
    if target and peers:
        weights = corr_weights_from_matrix(
            corr_df=corr_df,
            target=target,
            peers=list(peers),
            clip_negative=clip_negative,
            power=power,
        )

    # draw heatmap (will show weights on the right if provided)
    plot_correlation_details(ax, corr_df, weights=weights, show_values=show_values)
    return atm_df, corr_df, weights

def plot_correlation_details(
    ax: plt.Axes,
    corr_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    show_values: bool = True,
) -> None:
    """Heatmap of the correlation matrix; optionally show weights as annotation."""
    ax.clear()
    if corr_df is None or corr_df.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        return

    data = corr_df.to_numpy(dtype=float)
    finite_count = np.sum(np.isfinite(data))
    if finite_count == 0:
        ax.text(0.5, 0.5, "No valid correlations\n(insufficient overlapping data)",
                ha="center", va="center", fontsize=12)
        return
    if finite_count < data.size * 0.5:
        ax.text(0.5, 0.9, f"Warning: Limited data\n({finite_count}/{data.size} finite)",
                ha="center", va="top", fontsize=10, color="orange",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))

    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    if not hasattr(ax, "_colorbar_added"):
        try:
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax._colorbar_added = True
        except Exception:
            pass

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)
    ax.set_title("ATM Correlation (per pillars)")

    if show_values:
        n, m = data.shape
        for i in range(n):
            for j in range(m):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center",
                            fontsize=8, color=("white" if abs(val) > 0.5 else "black"),
                            weight="bold")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=7, color="gray", style="italic")

    if weights is not None and not weights.empty:
        w_sorted = weights.sort_values(ascending=False)
        txt = "ETF Weights (corr→ρ):\n" + "\n".join([f"{k}: {v:.3f}" for k, v in w_sorted.items()])
        ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

