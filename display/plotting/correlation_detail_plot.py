# display/plotting/correlation_detail_plot.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.pillars import DEFAULT_PILLARS_DAYS, detect_available_pillars
from analysis.correlation_utils import compute_atm_corr, corr_weights

# optional: fallback weights from persisted correlations (if you keep that path)
try:
    from analysis.beta_builder import peer_weights_from_correlations
    _HAS_PERSISTED_WEIGHTS = True
except Exception:
    _HAS_PERSISTED_WEIGHTS = False


# In-memory cache for pillar detection results
# Keyed by (sorted tickers tuple, asof)
_PILLAR_DETECTION_CACHE: Dict[Tuple[Tuple[str, ...], str], Optional[List[int]]] = {}

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
    min_pillars: int = 2,
    corr_method: str = "pearson",
    demean_rows: bool = False,
    show_values: bool = True,
    clip_negative: bool = True,
    power: float = 1.0,
    auto_detect_pillars: bool = True,
    min_tickers_per_pillar: int = 3,
    min_pillars_per_ticker: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """Compute correlation matrix and draw heatmap."""
    tickers = [t.upper() for t in tickers]
    key = (tuple(sorted(tickers)), asof)
    cached_pillars = _PILLAR_DETECTION_CACHE.get(key)
    if auto_detect_pillars:
        if cached_pillars is not None:
            if cached_pillars:
                pillars_days = cached_pillars
                print(f"📊 Using cached pillars: {pillars_days}")
            else:
                print(f"⚠️ Cached detection found no pillars, using defaults: {list(pillars_days)}")
        else:
            available_pillars = detect_available_pillars(
                get_smile_slice=get_smile_slice,
                tickers=tickers,
                asof=asof,
                candidate_pillars=pillars_days,
                min_tickers_per_pillar=max(2, len(tickers)//2),
                tol_days=tol_days,
            )
            _PILLAR_DETECTION_CACHE[key] = available_pillars
            if available_pillars:
                pillars_days = available_pillars
                print(f"📊 Auto-detected pillars with sufficient data: {pillars_days}")
            else:
                print(f"⚠️ No pillars found with sufficient data, using defaults: {list(pillars_days)}")
    else:
        if cached_pillars:
            pillars_days = cached_pillars
            print(f"⚡ Auto-detection skipped; using cached pillars: {pillars_days}")
    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        min_pillars=min_pillars,
        demean_rows=demean_rows,
        corr_method=corr_method,
        min_tickers_per_pillar=min_tickers_per_pillar,
        min_pillars_per_ticker=min_pillars_per_ticker,
    )
    weights = None
    if target and peers:
        weights = corr_weights(
            corr_df=corr_df,
            target=target,
            peers=list(peers),
            clip_negative=clip_negative,
            power=power,
        )
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
    total_elements = data.size
    
    if finite_count == 0:
        ax.text(0.5, 0.5, "No valid correlations\n(insufficient overlapping data)",
                ha="center", va="center", fontsize=12)
        return
    
    # ETF builder style data quality assessment
    data_quality = finite_count / total_elements if total_elements > 0 else 0
    
    if data_quality < 0.3:
        ax.text(0.5, 0.9, f"Poor data quality\n({finite_count}/{total_elements} finite, {data_quality:.1%})",
                ha="center", va="top", fontsize=10, color="red",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.3))
    elif data_quality < 0.7:
        ax.text(0.5, 0.9, f"Limited data quality\n({finite_count}/{total_elements} finite, {data_quality:.1%})",
                ha="center", va="top", fontsize=10, color="orange",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))
    else:
        ax.text(0.5, 0.9, f"Good data quality\n({finite_count}/{total_elements} finite, {data_quality:.1%})",
                ha="center", va="top", fontsize=10, color="green",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))

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


def scatter_corr_matrix(
    df_or_path: pd.DataFrame | str,
    columns: Optional[Iterable[str]] = None,
    *,
    plot: bool = True,
) -> pd.DataFrame:
    """Compute pairwise correlation coefficients for numeric columns and
    optionally draw a scatter-matrix plot.

    Parameters
    ----------
    df_or_path:
        Either a :class:`pandas.DataFrame` or path to a CSV file containing the
        data.
    columns:
        Optional iterable of column names to include.  Columns not present in
        the data are ignored, so the function continues to work if the caller
        has pre-filtered ("down-selected") the available columns.
    plot:
        If ``True`` (default) a scatter-matrix is produced using
        :func:`pandas.plotting.scatter_matrix`.

    Returns
    -------
    :class:`pandas.DataFrame`
        The correlation matrix of the selected numeric columns.  An empty
        DataFrame is returned if fewer than two numeric columns are available.
    """

    # ------------------------------------------------------------------
    # Load data & select columns
    # ------------------------------------------------------------------
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    if columns is not None:
        # Keep only columns that actually exist to avoid KeyError when the
        # caller has already removed some columns before plotting.
        cols = [c for c in columns if c in df.columns]
        df = df[cols]

    # Work only with numeric columns and drop rows with NA to avoid
    # spurious NaNs in the correlation calculation.
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    corr_df = num_df.corr()

    if plot:
        try:
            pd.plotting.scatter_matrix(num_df, figsize=(6, 6))
        except Exception:
            # The plot is best-effort; failure to plot should not crash the
            # correlation computation.
            pass

    return corr_df

