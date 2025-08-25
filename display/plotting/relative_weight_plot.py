"""
Correlation plotting without pillars, with configurable weighting modes.

This module computes correlations across implied-volatility surfaces using
the first few expiries for each ticker (as opposed to fixed pillar days).
It then provides a heatmap and optional ETF weight annotations.  You can
specify how weights are computed via the ``weight_mode`` parameter.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.beta_builder.unified_weights import compute_unified_weights
from analysis.model_params_logger import compute_or_load


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


def _maybe_compute_weights(
    target: Optional[str],
    peers: Optional[Iterable[str]],
    *,
    asof: str,
    weight_mode: str,
    weight_power: float,
    clip_negative: bool,
    **weight_config,
) -> Optional[pd.Series]:
    """Compute unified weights if ``target`` and ``peers`` are provided."""
    if not target or not peers:
        return None
    peers_list = list(peers)
    try:
        return compute_unified_weights(
            target=target,
            peers=peers_list,
            mode=weight_mode,
            asof=asof,
            clip_negative=clip_negative,
            power=weight_power,
            **weight_config,
        )
    except Exception:
        return None


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
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Compute a correlation matrix and draw a heatmap without relying on pillars.

    Parameters remain compatible with the upstream version but no longer accept
    pillar-related options.  The ``weight_mode`` is forwarded to
    :func:`analysis.unified_weights.compute_unified_weights`. Additional weight
    configuration such as ``weight_power`` and ``clip_negative`` can be
    supplied, along with any extra keyword arguments understood by the unified
    weight system.
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

    weights = _maybe_compute_weights(
        target=target,
        peers=peers,
        asof=asof,
        weight_mode=weight_mode,
        weight_power=weight_power,
        clip_negative=clip_negative,
        **weight_config,
    )

    plot_correlation_details(ax, corr_df, weights=weights, show_values=show_values)
    return atm_df, corr_df, weights


def plot_correlation_details(
    ax: plt.Axes,
    corr_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    show_values: bool = True,
) -> None:
    """
    Heatmap of the correlation matrix; optionally show weights as annotation.

    Adds a data-quality badge and, if supplied, lists the ETF weights on the
    right.  The heatmap is labelled “per expiries” to emphasize that no
    pillar selection is used.
    """
    ax.clear()
    if corr_df is None or corr_df.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        return

    data = corr_df.to_numpy(dtype=float)
    finite_count = np.sum(np.isfinite(data))
    total_elements = data.size

    if finite_count == 0:
        ax.text(
            0.5,
            0.5,
            "No valid correlations\n(insufficient overlapping data)",
            ha="center",
            va="center",
            fontsize=12,
        )
        return

    data_quality = finite_count / total_elements if total_elements > 0 else 0
    if data_quality < 0.3:
        dq_msg = f"Poor data quality\n({finite_count}/{total_elements} finite, {data_quality:.1%})"
        dq_color = "red"
        dq_face = "lightcoral"
    elif data_quality < 0.7:
        dq_msg = f"Limited data quality\n({finite_count}/{total_elements} finite, {data_quality:.1%})"
        dq_color = "orange"
        dq_face = "yellow"
    else:
        dq_msg = f"Good data quality\n({finite_count}/{total_elements} finite, {data_quality:.1%})"
        dq_color = "green"
        dq_face = "lightgreen"

    ax.text(
        0.5,
        1.02,
        dq_msg,
        ha="center",
        va="bottom",
        fontsize=10,
        color=dq_color,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor=dq_face, alpha=0.3),
        clip_on=False,
    )

    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")

    if not hasattr(ax.figure, "_orig_position"):
        ax.figure._orig_position = ax.get_position().bounds
        sp = ax.figure.subplotpars
        ax.figure._orig_subplotpars = (sp.left, sp.right, sp.bottom, sp.top)

    # Add or update correlation-specific colorbar
    if hasattr(ax.figure, "_correlation_colorbar"):
        try:
            ax.figure._correlation_colorbar.update_normal(im)
        except Exception:
            pass
        else:
            ax.figure._correlation_colorbar.set_label("Correlation (\u03c1)")
    else:
        try:
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation (\u03c1)")
            ax.figure._correlation_colorbar = cbar
        except Exception:
            pass

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)
    ax.set_title("Correlation and Relative Importance (per expiries)")

    if show_values:
        n, m = data.shape
        for i in range(n):
            for j in range(m):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=("white" if abs(val) > 0.5 else "black"),
                        weight="bold",
                    )
                else:
                    ax.text(
                        j,
                        i,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                        style="italic",
                    )

    if hasattr(ax.figure, "_corr_weight_ax"):
        try:
            ax.figure._corr_weight_ax.remove()
        except Exception:
            pass
        delattr(ax.figure, "_corr_weight_ax")

    if weights is not None and not weights.empty:
        w_sorted = weights.sort_values(ascending=False)
        bbox = ax.get_position()
        x0 = bbox.x1 + 0.02
        if hasattr(ax.figure, "_correlation_colorbar"):
            try:
                cbar_ax = ax.figure._correlation_colorbar.ax
                cbar_bbox = cbar_ax.get_position()
                x0 = cbar_bbox.x1 + 0.02
            except Exception:
                pass
        legend_ax = ax.figure.add_axes([x0, bbox.y0, 0.2, bbox.height])
        legend_ax.axis("off")
        legend_ax.set_title("Relative Importance", fontsize=10)
        legend_ax.text(
            0,
            1,
            "\n".join([f"{k}: {v:.3f}" for k, v in w_sorted.items()]),
            va="top",
            fontsize=9,
        )
        ax.figure._corr_weight_ax = legend_ax


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
