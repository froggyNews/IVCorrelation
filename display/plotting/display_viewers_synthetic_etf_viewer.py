"""
Matplotlib-based viewer for Synthetic ETF surfaces.

Features:
- Side-by-side target vs synthetic surface (moneyness × tenor × IV)
- Difference surface
- ATM relative value summary table (spread / z / pct_rank per pillar)
- Optional save to disk

Usage:
    from analysis.synthetic_etf import SyntheticETFBuilder, SyntheticETFConfig
    from display.viewers.synthetic_etf_viewer import show_synthetic_etf

    cfg = SyntheticETFConfig(target="SPY", peers=("QQQ","IWM"))
    builder = SyntheticETFBuilder(cfg)
    artifacts = builder.build_all()
    show_synthetic_etf(artifacts)

"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D plotting

from analysis.analysis_synthetic_etf import SyntheticETFArtifacts


def _as_float_index(idx) -> list[float]:
    out = []
    for x in idx:
        try:
            out.append(float(str(x).strip().split(":")[0]))
        except Exception:
            out.append(np.nan)
    return out


def _extract_latest(
    artifacts: SyntheticETFArtifacts, target: str
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[str],
    Optional[str],
]:
    """Return latest target and synthetic surfaces.

    Attempts to find a common date between target and synthetic surfaces. If none
    exists, falls back to the most recent date available for each side
    independently so the viewer can still render something.
    """

    if target not in artifacts.surfaces:
        return None, None, None, None

    target_dates = sorted(artifacts.surfaces[target].keys())
    synth_dates = sorted(artifacts.synthetic_surfaces.keys())
    common = sorted(set(target_dates).intersection(synth_dates))
    if common:
        d = common[-1]
        return artifacts.surfaces[target][d], artifacts.synthetic_surfaces[d], d, d

    d_tgt = target_dates[-1] if target_dates else None
    d_syn = synth_dates[-1] if synth_dates else None
    tgt_df = artifacts.surfaces[target].get(d_tgt) if d_tgt else None
    syn_df = artifacts.synthetic_surfaces.get(d_syn) if d_syn else None
    return tgt_df, syn_df, d_tgt, d_syn


def _plot_surface(ax, df: pd.DataFrame, title: str, cmap="viridis"):
    """Render a 3D volatility surface."""

    # df: rows mny bins (string labels), cols tenor-days
    mat = df.to_numpy(dtype=float)
    mny_vals = np.array(_as_float_index(df.index), dtype=float)
    tenors = np.array([float(c) for c in df.columns], dtype=float)
    T, M = np.meshgrid(tenors, mny_vals)
    surf = ax.plot_surface(T, M, mat, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Moneyness (K/S)")
    ax.set_zlabel("IV")
    return surf


def show_synthetic_etf(
    artifacts: SyntheticETFArtifacts,
    target: Optional[str] = None,
    save_path: Optional[str] = None,
    show_diff: bool = True,
    figsize=(14, 5),
):
    target = target or artifacts.meta.get("target")
    tgt_df, syn_df, tgt_date, syn_date = _extract_latest(artifacts, target)
    if tgt_df is None or syn_df is None:
        print("Missing surface data to plot synthetic ETF.")
        return

    ncols = 3 if show_diff else 2
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax0 = fig.add_subplot(1, ncols, 1, projection="3d")
    ax1 = fig.add_subplot(1, ncols, 2, projection="3d")
    surf0 = _plot_surface(ax0, tgt_df, f"{target} Surface ({tgt_date})")
    surf1 = _plot_surface(ax1, syn_df, f"Synthetic Surface ({syn_date})")

    fig.colorbar(surf0, ax=ax0, shrink=0.5, aspect=10)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    if show_diff:
        ax2 = fig.add_subplot(1, ncols, 3, projection="3d")
        diff = tgt_df.astype(float) - syn_df.astype(float)
        vmax = np.nanmax(np.abs(diff.to_numpy()))
        surf2 = _plot_surface(
            ax2,
            diff,
            "Target - Synthetic (Diff)",
            cmap="coolwarm",
        )
        surf2.set_clim(-vmax, vmax)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # Add RV metrics table as inset
    rv_df = artifacts.rv_metrics
    if (
        not rv_df.empty
        and "asof_date" in rv_df.columns
        and "pillar_days" in rv_df.columns
    ):
        rv_tail = rv_df.sort_values("asof_date").groupby("pillar_days").tail(1)
    else:
        rv_tail = pd.DataFrame()
    if not rv_tail.empty:
        cols = ["pillar_days", "iv_target", "iv_synth", "spread", "z", "pct_rank"]
        rv_show = rv_tail[cols].copy()
        rv_show["pillar_days"] = rv_show["pillar_days"].astype(int)
        txt = rv_show.to_string(index=False, float_format=lambda x: f"{x:0.4f}")
        fig.text(
            0.01,
            0.02,
            f"Latest RV Metrics\n{txt}",
            family="monospace",
            fontsize=8,
            va="bottom",
            ha="left",
        )

    if tgt_date != syn_date:
        fig.text(
            0.5,
            0.01,
            f"Note: target asof {tgt_date} vs synthetic {syn_date}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
