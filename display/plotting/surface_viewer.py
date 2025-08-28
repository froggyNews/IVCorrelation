# display/plotting/surface_viewer.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# Generic helpers
# -----------------------------

def _as_float_index(idx) -> np.ndarray:
    out = []
    for x in idx:
        try:
            if isinstance(x, pd.Interval):
                out.append((float(x.left) + float(x.right)) / 2.0)
            else:
                s = str(x).strip()
                if "-" in s:
                    lo, hi = s.split("-", 1)
                    out.append((float(lo) + float(hi)) / 2.0)
                else:
                    out.append(float(s))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)

def _prepare_grid(df: pd.DataFrame):
    """rows=moneyness bins, cols=tenor (days), values=IV"""
    Z = df.to_numpy(dtype=float)
    mny = _as_float_index(df.index)
    ten = np.asarray([float(c) for c in df.columns], dtype=float)
    return mny, ten, Z

def _align_grids_like(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer align (index/columns union); preserves values, introduces NaNs where missing."""
    a2 = a.copy()
    b2 = b.copy()
    a2.columns = [float(c) for c in a2.columns]
    b2.columns = [float(c) for c in b2.columns]
    cols = sorted(set(a2.columns) | set(b2.columns))
    idx = a2.index.union(b2.index)
    return a2.reindex(index=idx, columns=cols), b2.reindex(index=idx, columns=cols)

# -----------------------------
# Single-surface viewers
# -----------------------------

def show_surface_3d(df: pd.DataFrame, title: str = "IV Surface",
                    figsize=(8, 6), elev=25, azim=-60, cmap="viridis",
                    save_path: str | None = None):
    mny, ten, Z = _prepare_grid(df)
    M, T = np.meshgrid(mny, ten, indexing="ij")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, M, Z, cmap=cmap, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Moneyness")
    ax.set_zlabel("IV")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.65, aspect=14)
    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()
    return fig

def show_surface_heatmap(df: pd.DataFrame, title: str = "IV Surface (Heatmap)",
                         figsize=(8, 5), center: float | None = None, cmap: str = "viridis",
                         save_path: str | None = None):
    mny, ten, Z = _prepare_grid(df)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if center is not None:
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[ten.min() if len(ten) else 0, ten.max() if len(ten) else 1,
                mny.min() if len(mny) else 0, mny.max() if len(mny) else 1],
        cmap=cmap, norm=norm
    )
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Moneyness")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("IV")
    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()
    return fig

def show_whole_surface(surface_df: pd.DataFrame, *,
                       mode: str = "3d",
                       title: str = "IV Surface",
                       **kwargs):
    """
    Render the entire surface grid (one ticker, one date).
    mode: '3d' | 'heatmap'
    kwargs: passed to show_surface_3d / show_surface_heatmap
    """
    if mode == "heatmap":
        return show_surface_heatmap(surface_df, title=title, **kwargs)
    return show_surface_3d(surface_df, title=title, **kwargs)

# -----------------------------
# Comparison / diff viewers
# -----------------------------

def show_surfaces_compare(target_df: pd.DataFrame,
                          composite_df: pd.DataFrame,
                          *,
                          title_target: str = "Target Surface",
                          title_composite: str = "Composite Surface",
                          show_diff: bool = True,
                          figsize=(14, 5),
                          cmap_main: str = "viridis",
                          cmap_diff: str = "coolwarm",
                          save_path: str | None = None,
                          rv_table: pd.DataFrame | None = None):
    """
    Side-by-side 3D surfaces (target vs composite) with optional diff.
    - Aligns grids to outer union to avoid shape errors.
    - If show_diff=True, computes (target - composite) and centers colormap.
    - Optionally writes latest RV metrics (pillar_days / iv_target / iv_synth / spread / z / pct_rank)
      as a monospace text inset.
    """
    tgt, syn = _align_grids_like(target_df, composite_df)

    # figure layout
    ncols = 3 if show_diff else 2
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax0 = fig.add_subplot(1, ncols, 1, projection="3d")
    ax1 = fig.add_subplot(1, ncols, 2, projection="3d")

    # target
    surf0 = _surface_on_axes(ax0, tgt, title_target, cmap=cmap_main)
    cb0 = fig.colorbar(surf0, ax=ax0, shrink=0.6, aspect=14)

    # composite
    surf1 = _surface_on_axes(ax1, syn, title_composite, cmap=cmap_main)
    cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=14)

    # diff
    if show_diff:
        ax2 = fig.add_subplot(1, ncols, 3, projection="3d")
        diff = tgt.astype(float) - syn.astype(float)
        vmax = float(np.nanmax(np.abs(diff.to_numpy()))) if np.isfinite(diff.to_numpy()).any() else 0.0
        surf2 = _surface_on_axes(ax2, diff, "Target − Composite (Diff)", cmap=cmap_diff)
        if vmax > 0:
            surf2.set_clim(-vmax, vmax)
        fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=14)

    # optional RV inset
    if isinstance(rv_table, pd.DataFrame) and not rv_table.empty:
        needed = {"pillar_days", "iv_target", "iv_synth", "spread", "z", "pct_rank"}
        if needed.issubset(rv_table.columns):
            tail = rv_table.sort_values("asof_date").groupby("pillar_days", as_index=False).tail(1) \
                   if "asof_date" in rv_table.columns else rv_table.copy()
            tail = tail.loc[:, ["pillar_days", "iv_target", "iv_synth", "spread", "z", "pct_rank"]]
            tail["pillar_days"] = tail["pillar_days"].astype(int)
            txt = tail.to_string(index=False, float_format=lambda x: f"{x:0.4f}")
            fig.text(0.01, 0.02, f"Latest RV Metrics\n{txt}",
                     family="monospace", fontsize=8, va="bottom", ha="left")

    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()
    return fig

def _surface_on_axes(ax, df: pd.DataFrame, title: str, cmap: str):
    """Internal: render one 3D surface on provided axes."""
    mny, ten, Z = _prepare_grid(df)
    M, T = np.meshgrid(mny, ten, indexing="ij")
    surf = ax.plot_surface(T, M, Z, cmap=cmap, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("Tenor (days)")
    ax.set_ylabel("Moneyness")
    ax.set_zlabel("IV")
    return surf

# -----------------------------
# Slice (one tenor) overlay viewer
# -----------------------------

def show_smile_overlay(target_df: pd.DataFrame,
                       composite_df: pd.DataFrame,
                       T_days: float,
                       *,
                       title: str = "Smile Overlay",
                       linewidth: float = 1.8,
                       save_path: str | None = None):
    """
    Compare target vs composite smiles at the tenor nearest T_days (2D line overlay).
    Includes interpolation-to-common-grid fallback for mismatched moneyness bins.
    """
    # pick nearest tenor on each
    tgt_cols = np.asarray([float(c) for c in target_df.columns], float)
    syn_cols = np.asarray([float(c) for c in composite_df.columns], float)
    i_tgt = int(np.abs(tgt_cols - float(T_days)).argmin())
    i_syn = int(np.abs(syn_cols - float(T_days)).argmin())
    col_tgt = target_df.columns[i_tgt]
    col_syn = composite_df.columns[i_syn]

    xt = _as_float_index(target_df.index)
    yt = target_df[col_tgt].astype(float).to_numpy()
    xs = _as_float_index(composite_df.index)
    ys = composite_df[col_syn].astype(float).to_numpy()

    # interpolation to common grid
    def _interp_pair(xa, ya, xb, yb):
        va = np.isfinite(xa) & np.isfinite(ya)
        vb = np.isfinite(xb) & np.isfinite(yb)
        if va.sum() >= 2 and vb.sum() >= 2:
            lo = max(np.nanmin(xa[va]), np.nanmin(xb[vb]))
            hi = min(np.nanmax(xa[va]), np.nanmax(xb[vb]))
            if hi > lo:
                grid = np.linspace(lo, hi, 60)
                try:
                    from scipy.interpolate import interp1d
                    fa = interp1d(xa[va], ya[va], kind="linear", bounds_error=False, fill_value=np.nan)
                    fb = interp1d(xb[vb], yb[vb], kind="linear", bounds_error=False, fill_value=np.nan)
                    return grid, fa(grid), fb(grid)
                except Exception:
                    pass
        # fallback to target x with simple 1D interp
        yb_on_a = np.interp(xa, xb, yb, left=np.nan, right=np.nan)
        return xa, ya, yb_on_a

    x, y_t, y_s = _interp_pair(xt, yt, xs, ys)

    valid = np.isfinite(x) & np.isfinite(y_t) & np.isfinite(y_s)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if valid.sum() >= 2:
        ax.plot(x[valid], y_t[valid], "-", lw=linewidth, label=f"Target @ ~{int(T_days)}d")
        ax.plot(x[valid], y_s[valid], "--", lw=linewidth, label="Composite (weighted)")
    else:
        ax.text(0.5, 0.5, "Insufficient valid data for comparison", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()
    return fig
