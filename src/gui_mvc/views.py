import numpy as np
import matplotlib.pyplot as plt

from .models import atm_normalize, to_moneyness


def plot_surface(stock: dict, median: dict, low: dict, high: dict,
                 normalize_atm: bool, use_moneyness: bool,
                 highlight_outliers: bool):
    """Return a matplotlib Figure and % of points inside the band."""
    if use_moneyness:
        stock, _ = to_moneyness(stock)
        median, _ = to_moneyness(median)
        low, _ = to_moneyness(low)
        high, _ = to_moneyness(high)

    if normalize_atm:
        stock = atm_normalize(stock)
        median = atm_normalize(median)
        low = atm_normalize(low)
        high = atm_normalize(high)

    K = np.asarray(stock["K"], dtype=float)
    T = np.asarray(stock["T"], dtype=float)
    Z = np.asarray(stock["IV"], dtype=float)

    inside_mask = (Z >= low["IV"]) & (Z <= high["IV"])
    inside_pct = 100.0 * inside_mask.sum() / Z.size

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    X, Y = np.meshgrid(K, T, indexing="ij")
    c = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    fig.colorbar(c, ax=ax, label="IV")

    ax.contour(X, Y, median["IV"], colors="white", linewidths=2)
    ax.contour(X, Y, low["IV"], colors="green", linestyles="dashed")
    ax.contour(X, Y, high["IV"], colors="red", linestyles="dashed")

    if highlight_outliers:
        out_y, out_x = np.where(~inside_mask)
        if len(out_x):
            ax.scatter(K[out_x], T[out_y], color="magenta", s=10, label="Outlier")

    ax.set_xlabel("Moneyness" if use_moneyness else "Strike")
    ax.set_ylabel("Maturity")
    ax.set_title("Volatility Surface")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, inside_pct
