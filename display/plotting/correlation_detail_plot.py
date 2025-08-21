"""
Correlation plotting without pillars, rendering a relative weight matrix.

The module builds correlations across implied-volatility surfaces using the
first few expiries for each ticker (as opposed to fixed pillar days).  The
resulting correlation matrix is converted into a row‑normalised "relative
weight" matrix which is visualised as a heatmap.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.compute_or_load import compute_or_load


# ---------------------------------------------------------------------------
# Helpers to compute ATM curves and correlations without using fixed pillars
# ---------------------------------------------------------------------------


def _compute_atm_curve_simple(df: pd.DataFrame, atm_band: float = 0.05) -> pd.DataFrame:
    """
    Compute a simple ATM implied volatility per expiry using only the median
    of near‑ATM quotes.  This function avoids reliance on fixed pillar days.

    Parameters
    ----------
    df : pandas.DataFrame
        Slice of option quotes for a single ticker on a single date.
    atm_band : float, optional
        Relative band around ATM (moneyness ~ 1) used to compute medians.

    Returns
    -------
    pandas.DataFrame
        Columns ``T`` and ``atm_vol`` sorted by increasing maturity ``T``.
    """
    need_cols = {"T", "moneyness", "sigma"}
    if df is None or df.empty or not need_cols.issubset(df.columns):
        return pd.DataFrame(columns=["T", "atm_vol"])
    d = df.copy()
    d["T"] = pd.to_numeric(d["T"], errors="coerce")
    d["moneyness"] = pd.to_numeric(d["moneyness"], errors="coerce")
    d["sigma"] = pd.to_numeric(d["sigma"], errors="coerce")
    d = d.dropna(subset=["T", "moneyness", "sigma"])
    rows: List[dict[str, float]] = []
    for T_val, grp in d.groupby("T"):
        g = grp.dropna(subset=["moneyness", "sigma"])
        in_band = g.loc[(g["moneyness"] - 1.0).abs() <= atm_band]
        if not in_band.empty:
            atm_vol = float(in_band["sigma"].median())
        else:
            idx = int((g["moneyness"] - 1.0).abs().idxmin())
            atm_vol = float(g.loc[idx, "sigma"])
        rows.append({"T": float(T_val), "atm_vol": atm_vol})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def _corr_by_expiry_rank(
    get_slice,
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = 6,
    atm_band: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an ATM matrix and correlation matrix without using pillars.

    The matrix rows correspond to tickers, columns to expiry ranks (0,1,…).
    Correlations are computed across expiry ranks.
    """
    rows: List[pd.Series] = []
    for t in tickers:
        try:
            df = get_slice(
                t, asof_date=asof, T_target_years=None, call_put=None, nearest_by="T"
            )
        except Exception:
            df = None
        if df is None or df.empty:
            continue
        atm_df = _compute_atm_curve_simple(df, atm_band=atm_band)
        values: Dict[int, float] = {}
        for i in range(max_expiries):
            if i < len(atm_df):
                v = atm_df.at[i, "atm_vol"]
                values[i] = float(v) if pd.notna(v) else np.nan
            else:
                values[i] = np.nan
        rows.append(pd.Series(values, name=t.upper()))
    atm_rank_df = pd.DataFrame(rows)
    if atm_rank_df.empty or len(atm_rank_df.index) < 2:
        corr_df = pd.DataFrame(
            index=atm_rank_df.index, columns=atm_rank_df.index, dtype=float
        )
    else:
        corr_df = atm_rank_df.transpose().corr(method="pearson", min_periods=2)
    return atm_rank_df, corr_df


def _relative_weight_matrix(
    corr_df: pd.DataFrame,
    *,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.DataFrame:
    """Convert a correlation matrix into a relative weight matrix.

    Each row is normalised to sum to one after optional clipping of negative
    correlations and application of an exponent (``power``).
    """
    if corr_df is None or corr_df.empty:
        return pd.DataFrame()

    mat = corr_df.to_numpy(dtype=float)
    if clip_negative:
        mat = np.clip(mat, 0.0, None)
    if power is not None and float(power) != 1.0:
        mat = np.power(mat, float(power))

    np.fill_diagonal(mat, 0.0)
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.divide(mat, row_sums, where=row_sums != 0)

    return pd.DataFrame(mat, index=corr_df.index, columns=corr_df.columns)


# ---------------------------------------------------------------------------
# Correlation: compute and plot (optionally show weights)
# ---------------------------------------------------------------------------


def compute_and_plot_correlation(
    ax: plt.Axes,
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    *,
    target: Optional[str] = None,
    peers: Optional[Iterable[str]] = None,
    atm_band: float = 0.05,
    show_values: bool = True,
    clip_negative: bool = True,
    weight_power: float = 1.0,
    max_expiries: int = 6,
    weight_mode: str = "corr",
    **weight_config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlation and display the derived relative weight matrix.

    The correlation matrix is still returned for callers that need it, but the
    plot itself shows row-wise normalised weights derived from that correlation
    matrix.  Parameters related to ``target``/``peers`` and ``weight_mode`` are
    retained for backward compatibility but are not used for plotting.
    """
    tickers = [t.upper() for t in tickers]
    payload = {
        "tickers": sorted(tickers),
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "atm_band": atm_band,
        "max_expiries": max_expiries,
    }

    def _builder() -> Tuple[pd.DataFrame, pd.DataFrame]:
        return _corr_by_expiry_rank(
            get_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=max_expiries,
            atm_band=atm_band,
        )

    atm_df, corr_df = compute_or_load("corr", payload, _builder)

    weight_df = _relative_weight_matrix(
        corr_df, clip_negative=clip_negative, power=weight_power
    )
    plot_weight_matrix(ax, weight_df, show_values=show_values)
    return atm_df, corr_df, weight_df


def plot_weight_matrix(
    ax: plt.Axes,
    weight_df: pd.DataFrame,
    show_values: bool = True,
) -> None:
    """Render a heatmap of relative weights."""
    ax.clear()
    if weight_df is None or weight_df.empty:
        ax.text(0.5, 0.5, "No weight data", ha="center", va="center")
        return

    data = weight_df.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")

    if not hasattr(ax.figure, "_orig_position"):
        ax.figure._orig_position = ax.get_position().bounds
        sp = ax.figure.subplotpars
        ax.figure._orig_subplotpars = (sp.left, sp.right, sp.bottom, sp.top)

    if hasattr(ax.figure, "_correlation_colorbar"):
        try:
            ax.figure._correlation_colorbar.remove()
        except Exception:
            pass
        delattr(ax.figure, "_correlation_colorbar")

    try:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative Weight")
    except Exception:
        pass

    ax.set_xticks(range(len(weight_df.columns)))
    ax.set_yticks(range(len(weight_df.index)))
    ax.set_xticklabels(weight_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(weight_df.index, fontsize=9)
    ax.set_title("Relative Weight Matrix")

    if show_values:
        n, m = data.shape
        for i in range(n):
            for j in range(m):
                val = data[i, j]
                if np.isfinite(val) and val > 0:
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=("white" if val > 0.5 else "black"),
                        weight="bold",
                    )



def scatter_corr_matrix(
    df_or_path: pd.DataFrame | str,
    columns: Optional[Iterable[str]] = None,
    *,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise correlation coefficients for numeric columns and
    optionally draw a scatter‑matrix plot.

    If ``plot`` is True, the scatter-matrix uses pandas.plotting.scatter_matrix().
    """
    if isinstance(df_or_path, str):
        df = pd.read_parquet(df_or_path)
    else:
        df = df_or_path.copy()

    if columns is not None:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]

    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    corr_df = num_df.corr()

    if plot:
        try:
            pd.plotting.scatter_matrix(num_df, figsize=(6, 6))
        except Exception:
            pass

    return corr_df
