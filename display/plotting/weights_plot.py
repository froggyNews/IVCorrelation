from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Mapping, Sequence


def plot_weights(ax: plt.Axes, weights: Mapping[str, float] | Sequence[float] | pd.Series) -> None:
    """Render weights as a sorted bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to draw on. It will be cleared.
    weights : mapping or sequence or pd.Series
        Mapping/series of label -> weight values or sequence of weights (labels
        will be numeric indices).
    """
    ax.clear()
    if weights is None:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center")
        return

    if isinstance(weights, pd.Series):
        s = weights.astype(float)
    elif isinstance(weights, Mapping):
        s = pd.Series(dict(weights), dtype=float)
    else:
        s = pd.Series(list(weights), dtype=float)

    if s.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center")
        return

    s = s.sort_values(ascending=False)
    x = range(len(s))
    bars = ax.bar(x, s.values, color="steelblue")
    ax.set_title("Index Weights")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, max(float(s.max()) * 1.1, 1.0))
    ax.set_xticks(list(x))
    ax.set_xticklabels(s.index.astype(str), rotation=45, ha="right")

    for idx, (bar, val) in enumerate(zip(bars, s.values)):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
