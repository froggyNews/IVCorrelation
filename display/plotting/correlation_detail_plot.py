# display/plotting/correlation_detail_plot.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.pillars import build_atm_matrix, DEFAULT_PILLARS_DAYS, detect_available_pillars
# surface construction

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
    min_pillars: int = 2,  # Reduced from 3 to 2 for more lenient correlation
    corr_method: str = "pearson",    # or "spearman" | "kendall"
    demean_rows: bool = False,
    show_values: bool = True,
    clip_negative: bool = True,
    power: float = 1.0,

    auto_detect_pillars: bool = True,  # New parameter
    min_tickers_per_pillar: int = 3,   # ETF builder style: need at least 3 tickers per pillar
    min_pillars_per_ticker: int = 2,   # ETF builder style: need at least 2 pillars per ticker

) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Build ATM-by-pillar matrix and correlation matrix; draw heatmap.
    If `target` and `peers` are provided, also compute and display corr-based weights.
    If ``auto_detect_pillars`` is ``True`` the function will look up a cached set of
    pillars for ``(tickers, asof)`` and only call :func:`detect_available_pillars`
    when no cached value is present.  When ``auto_detect_pillars`` is ``False`` the
    cached pillars are used if available, otherwise ``pillars_days`` is used as-is.
    Returns: (atm_df, corr_df, weights or None)
    """
    tickers = [t.upper() for t in tickers]

    key = (tuple(sorted(tickers)), asof)

    cached_pillars = _PILLAR_DETECTION_CACHE.get(key)

    if auto_detect_pillars:

        if cached_pillars is not None:
            if cached_pillars:
                pillars_days = cached_pillars
                print(f"📊 Using cached pillars: {pillars_days}")
            else:
                print(
                    f"⚠️ Cached detection found no pillars, using defaults: {list(pillars_days)}"
                )
        else:
            available_pillars = detect_available_pillars(
                get_smile_slice=get_smile_slice,
                tickers=tickers,
                asof=asof,
                candidate_pillars=pillars_days,
                min_tickers_per_pillar=max(2, len(tickers) // 2),  # At least half the tickers
                tol_days=tol_days,
            )
            _PILLAR_DETECTION_CACHE[key] = available_pillars
            if available_pillars:
                pillars_days = available_pillars
                print(
                    f"📊 Auto-detected pillars with sufficient data: {pillars_days}"
                )
            else:
                print(
                    f"⚠️ No pillars found with sufficient data, using defaults: {list(pillars_days)}"
                )
    else:
        if cached_pillars:
            pillars_days = cached_pillars
            print(
                f"⚡ Auto-detection skipped; using cached pillars: {pillars_days}"
            )


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
    
    # Apply ETF builder style filtering for better data quality
    if not atm_df.empty:
        # Filter columns (pillars) that have insufficient ticker coverage
        col_coverage = atm_df.count(axis=0)
        good_pillars = col_coverage[col_coverage >= min_tickers_per_pillar].index
        if len(good_pillars) >= 2:  # Need at least 2 pillars for correlation
            atm_df = atm_df[good_pillars]
            print(f"📊 Filtered to {len(good_pillars)} pillars with ≥{min_tickers_per_pillar} tickers each")
        
        # Filter rows (tickers) that have insufficient pillar coverage
        row_coverage = atm_df.count(axis=1)
        good_tickers = row_coverage[row_coverage >= min_pillars_per_ticker].index
        if len(good_tickers) >= 2:  # Need at least 2 tickers for correlation
            atm_df = atm_df.loc[good_tickers]
            print(f"📊 Filtered to {len(good_tickers)} tickers with ≥{min_pillars_per_ticker} pillars each")
        
        # Rebuild correlation matrix with filtered data
        if not atm_df.empty and atm_df.shape[0] >= 2 and atm_df.shape[1] >= 2:
            # Use ETF builder style correlation with ridge regularization
            atm_clean = atm_df.dropna()
            if not atm_clean.empty and atm_clean.shape[0] >= 2 and atm_clean.shape[1] >= 2:
                # Standardize and add ridge like ETF builder
                atm_std = (atm_clean - atm_clean.mean(axis=1).values.reshape(-1, 1)) / (atm_clean.std(axis=1).values.reshape(-1, 1) + 1e-8)
                corr_matrix = (atm_std @ atm_std.T) / max(atm_std.shape[1] - 1, 1)
                # Add ridge regularization for stability (ETF builder style)
                ridge = 1e-6
                corr_matrix = corr_matrix + ridge * np.eye(corr_matrix.shape[0])
                corr_df = pd.DataFrame(corr_matrix, index=atm_clean.index, columns=atm_clean.index)
            else:
                corr_df = pd.DataFrame(index=atm_df.index, columns=atm_df.index, dtype=float)
        else:
            corr_df = pd.DataFrame(index=atm_df.index, columns=atm_df.index, dtype=float)

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

