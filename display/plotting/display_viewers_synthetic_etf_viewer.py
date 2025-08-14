"""
Matplotlib-based viewer for Synthetic ETF surfaces.

Features:
- Side-by-side target vs synthetic surface (moneyness × tenor)
- Difference heatmap
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
from matplotlib.colors import TwoSlopeNorm

from analysis.analysis_synthetic_etf import SyntheticETFArtifacts


def _as_float_index(idx) -> list[float]:
    out = []
    for x in idx:
        try:
            out.append(float(str(x).strip().split(":")[0]))
        except Exception:
            out.append(np.nan)
    return out


def _extract_latest(artifacts: SyntheticETFArtifacts, target: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    if target not in artifacts.surfaces:
        return None, None, None
    target_dates = set(artifacts.surfaces[target].keys())
    synth_dates = set(artifacts.synthetic_surfaces.keys())
    common = sorted(target_dates.intersection(synth_dates))
    if not common:
        return None, None, None
    d = common[-1]
    return artifacts.surfaces[target][d], artifacts.synthetic_surfaces[d], d


def _plot_surface(ax, df: pd.DataFrame, title: str, cmap="viridis"):
    # df: rows mny bins (string labels), cols tenor-days
    mat = df.to_numpy(dtype=float)
    mny_vals = _as_float_index(df.index)
    tenors = [float(c) for c in df.columns]
    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        extent=[min(tenors), max(tenors), min(mny_vals), max(mny_vals)],
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Moneyness (K/S)")
    return im


def show_synthetic_etf(
    artifacts: SyntheticETFArtifacts,
    target: Optional[str] = None,
    save_path: Optional[str] = None,
    show_diff: bool = True,
    figsize=(14, 5),
):
    target = target or artifacts.meta.get("target")
    tgt_df, syn_df, date = _extract_latest(artifacts, target)
    if tgt_df is None or syn_df is None:
        print("No overlapping date between target and synthetic surfaces.")
        return

    fig, axes = plt.subplots(
        1, 3 if show_diff else 2, figsize=figsize, constrained_layout=True
    )
    ax0, ax1 = axes[0], axes[1]
    im0 = _plot_surface(ax0, tgt_df, f"{target} Surface ({date})")
    im1 = _plot_surface(ax1, syn_df, "Synthetic Surface")

    fig.colorbar(im0, ax=ax0, fraction=0.046)
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    if show_diff:
        ax2 = axes[2]
        diff = tgt_df.astype(float) - syn_df.astype(float)
        mat = diff.to_numpy()
        mny_vals = _as_float_index(diff.index)
        tenors = [float(c) for c in diff.columns]
        norm = TwoSlopeNorm(vcenter=0.0)
        im2 = ax2.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[min(tenors), max(tenors), min(mny_vals), max(mny_vals)],
            cmap="coolwarm",
            norm=norm,
        )
        ax2.set_title("Target - Synthetic (Diff)")
        ax2.set_xlabel("Tenor (days)")
        ax2.set_ylabel("Moneyness (K/S)")
        fig.colorbar(im2, ax=ax2, fraction=0.046)

    # Add RV metrics table as inset
    rv_tail = artifacts.rv_metrics.sort_values("asof_date").groupby("pillar_days").tail(1)
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

    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()