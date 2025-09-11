#!/usr/bin/env python
"""
Generate a suite of plots (weights, relative-weight matrix, term structure,
term smile, and surface) for a target and peers.

Usage examples:
  python scripts/generate_plots.py --target SPY --peers QQQ,IVV --asof 2020-01-02 \
      --weight-mode corr_iv_atm --out-dir out --surface-mode heatmap

Notes:
- Uses non-interactive matplotlib backend and saves PNGs to --out-dir
- Falls back gracefully if some data is unavailable
"""


from __future__ import annotations
"""
python scripts/generate_plots.py --target IONQ --peers PLTR,MSFT,GOOGL --asof 2025-08-29 --weight-mode corr_iv_atm --slide10-smile --slide10-days 28 --slide10-ci 68 --out-dir outputs/IONQ_slide10_2025-08-29_fix
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analysis_pipeline import (
    get_smile_slice,
    get_most_recent_date_global,
    available_dates,
)
from analysis.beta_builder.unified_weights import compute_unified_weights
from display.plotting.weights_plot import plot_weights
from display.plotting.relative_weight_plot import compute_and_plot_relative_weight
from display.plotting.vol_structure_plots import (
    plot_atm_term_structure,
    plot_term_smile,
    plot_3d_vol_surface,
    create_vol_dashboard,
)
from display.plotting.smile_plot import (
    fit_smile_models_with_bands,
    fit_and_plot_smile,
    plot_composite_smile_overlay,
    plot_smile_with_composite,)
from volModel.sviFit import svi_smile_iv
from display.plotting.surface_viewer import show_surfaces_compare
from analysis.compositeIndexBuilder import combine_surfaces
from analysis.compositeIndexBuilder import build_surface_grids
from data.data_downloader import save_for_tickers
from data.db_utils import get_conn, ensure_initialized, check_db_health, DB_PATH
import shutil
from datetime import datetime


def _resolve_asof(target: str, asof: str | None) -> str | None:
    if asof:
        return asof
    try:
        d = get_most_recent_date_global()
        if d:
            return d
    except Exception:
        pass
    try:
        ds = available_dates(ticker=target, most_recent_only=True)
        if ds:
            return ds[0]
    except Exception:
        pass
    return None


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _clamp_max_expiries(x: int | None) -> int:
    """Clamp requested max_expiries to a conservative range.

    Some tickers have sparse surfaces; asking for too many expiries can
    produce unstable grids. We cap to [4, 8] with a default of 6.
    """
    try:
        if x is None:
            return 6
        xi = int(x)
        return max(4, min(8, xi))
    except Exception:
        return 6


def _resolve_surface_asof_for_tickers(tickers: list[str], requested_asof: str | None, max_expiries: int | None = None) -> str | None:
    """Find an asof date with available surface grids for the given tickers.

    Tries requested_asof first, then the most recent available per target.
    If nothing is available, attempts a fresh download and uses today's date.
    Returns an ISO date string or None if still unavailable.
    """
    tickers = [t.upper() for t in (tickers or [])]
    if not tickers:
        return requested_asof

    def _most_recent_for(t: str, surfaces: dict) -> str | None:
        if t not in surfaces or not surfaces[t]:
            return None
        dt = max(surfaces[t].keys())
        return pd.to_datetime(dt).date().isoformat()

    try:
        surfaces = build_surface_grids(tickers=tickers, use_atm_only=False, max_expiries=max_expiries)
    except Exception:
        surfaces = {}

    # If requested date works for the first ticker, keep it
    if requested_asof:
        req_ts = pd.to_datetime(requested_asof).normalize()
        t0 = tickers[0]
        if t0 in surfaces:
            norm_map = {pd.to_datetime(k).normalize(): k for k in surfaces[t0].keys()}
            if req_ts in norm_map:
                return pd.to_datetime(norm_map[req_ts]).date().isoformat()

    # Fallback to most recent available for the first ticker
    mr = _most_recent_for(tickers[0], surfaces)
    if mr:
        return mr

    # Try to pull fresh data
    try:
        save_for_tickers(tickers, max_expiries=_clamp_max_expiries(max_expiries))
        # Rebuild and use today's date if present
        surfaces = build_surface_grids(tickers=tickers, use_atm_only=False, max_expiries=max_expiries)
        mr = _most_recent_for(tickers[0], surfaces)
        if mr:
            return mr
    except Exception:
        pass
    return requested_asof


def _auto_repair_db_if_corrupt(required_tickers: list[str] | None, max_expiries: int) -> bool:
    """Detect 'malformed' SQLite DB and rebuild minimal schema + data for required tickers.

    Returns True if a repair was attempted (or not needed), False if unrecoverable.
    """
    try:
        conn = get_conn()
        try:
            check_db_health(conn)
            return True  # healthy
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        # Attempt basic recovery: backup and reinitialize
        try:
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup = f"{DB_PATH}.corrupt.{ts}"
            try:
                shutil.copyfile(DB_PATH, backup)
                print(f"Backed up corrupt DB to: {backup}")
            except Exception:
                pass
            # Reinitialize schema
            conn = get_conn()
            try:
                ensure_initialized(conn)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            # Reingest minimal set if provided
            if required_tickers:
                print(f"Rebuilding minimal data for: {sorted(set(required_tickers))}")
                try:
                    save_for_tickers(sorted(set(required_tickers)), max_expiries=_clamp_max_expiries(max_expiries))
                except Exception as e:
                    print(f"Warning: reingest failed: {e}")
            return True
        except Exception as e:
            print(f"Auto-repair failed: {e}")
            return False

def plot_peer_underlyings_1y(ax: plt.Axes, tickers: list[str], asof: str | None) -> pd.DataFrame:
    """Plot past-year underlying prices for tickers, rebased to 100.

    Returns price DataFrame (rows=dates, cols=tickers) used for plotting.
    """
    from data.db_utils import get_conn

    tickers = [t.upper() for t in (tickers or [])]
    if not tickers:
        ax.text(0.5, 0.5, "No tickers provided", ha="center", va="center"); return pd.DataFrame()

    # Determine end date
    if asof:
        end_dt = pd.to_datetime(asof).normalize()
    else:
        # try most recent across DB
        try:
            conn = get_conn()
            end_dt = pd.read_sql_query(
                "SELECT MAX(asof_date) AS d FROM underlying_prices", conn
            )["d"].dropna().pipe(lambda s: pd.to_datetime(s.iloc[0]) if not s.empty else pd.Timestamp.today()).normalize()
        except Exception:
            end_dt = pd.Timestamp.today().normalize()
    start_dt = end_dt - pd.Timedelta(days=365)

    conn = get_conn()
    start_s = start_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")

    # Try underlying_prices first
    try:
        placeholders = ",".join(["?"] * len(tickers))
        q = (
            "SELECT asof_date, ticker, close FROM underlying_prices "
            "WHERE ticker IN (" + placeholders + ") AND asof_date BETWEEN ? AND ?"
        )
        df = pd.read_sql_query(q, conn, params=tickers + [start_s, end_s])
    except Exception:
        df = pd.DataFrame()

    # Fallback: derive from options_quotes spot
    if df.empty:
        try:
            q = (
                "SELECT substr(asof_date,1,10) AS asof_date, ticker, median(spot) as close "
                "FROM options_quotes WHERE ticker IN (" + placeholders + ") "
                "AND substr(asof_date,1,10) BETWEEN ? AND ? GROUP BY 1,2"
            )
            df = pd.read_sql_query(q, conn, params=tickers + [start_s, end_s])
        except Exception:
            df = pd.DataFrame()
    try:
        conn.close()
    except Exception:
        pass

    if df.empty:
        ax.text(0.5, 0.5, "No underlying data", ha="center", va="center"); return pd.DataFrame()

    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.tz_localize(None)
    px = (
        df.groupby(["asof_date", "ticker"])['close']
          .median()
          .unstack('ticker')
          .sort_index()
    )
    # Filter to requested tickers order
    px = px.reindex(columns=tickers)
    # Forward fill to address missing days, then drop all-NaN
    px = px.ffill().dropna(how='all')

    # Rebase to 100 at start
    base = px.iloc[0]
    rebased = (px / base) * 100.0

    # Plot
    ax.clear()
    for t in rebased.columns:
        ax.plot(rebased.index, rebased[t].astype(float), label=t, lw=1.8)
    ax.set_title(f"Underlying Prices (1y) ending {end_dt.date()}")
    ax.set_ylabel("Rebased (start=100)")
    ax.set_xlabel("Date")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return px


def main() -> int:
    p = argparse.ArgumentParser(description="Generate plots for target and peers")
    p.add_argument("--target", required=True, help="Target ticker")
    p.add_argument("--peers", default="", help="Comma-separated peer tickers")
    p.add_argument("--asof", default=None, help="As-of date (YYYY-MM-DD)")
    p.add_argument("--weight-mode", default="corr_iv_atm", help="Weight mode (method_feature)")
    p.add_argument("--latest", action="store_true", help="Ignore --asof and use most recent available date")
    p.add_argument("--out-dir", default="plots_out", help="Output directory for PNGs")
    p.add_argument("--max-expiries", type=int, default=8, help="Max expiries used for ATM term structure, relative-weight matrix, and surfaces (not used for term smile)")
    p.add_argument("--atm-band", type=float, default=0.05, help="ATM band for matrix ATM extraction")
    p.add_argument("--t-days", type=float, default=30.0, help="Target days for term smile plot (nearest expiry)")
    p.add_argument("--tolerance-days", type=float, default=7.0, help="Tolerance for matching target maturity in term smile")
    p.add_argument("--surface-mode", choices=["3d", "heatmap"], default="heatmap", help="Surface rendering mode")
    p.add_argument("--no-cache", action="store_true", help="Bypass cache for relative-weight matrix build")
    p.add_argument("--weights-grids", action="store_true", help="Also render 1xN PCA and Correlation weights grids (e.g., iv_atm, surface_grid, ul)")
    p.add_argument("--features", default="iv_atm,surface_grid,ul", help="Comma-separated feature list for grids (e.g., iv_atm,surface_grid,ul)")
    # Slide 10: smile overlay figure
    p.add_argument("--slide10-smile", action="store_true", help="Render Slide10 smile overlay figure")
    p.add_argument("--slide10-peers", default=None, help="Optional override peers list for Slide10 smile (comma-separated)")
    p.add_argument("--slide10-days", type=float, default=28.0, help="Target days to expiry for Slide10 smile")
    p.add_argument("--slide10-ci", type=float, default=68.0, help="Confidence interval level for target smile (e.g., 68 or 0.68)")
    # Methods vs target smile overlay + accuracy table
    p.add_argument("--methods-smile", action="store_true", help="Overlay composite smiles for 5 methods vs target, with accuracy table")
    p.add_argument("--methods-days", type=float, default=28.0, help="Target days to expiry for methods smile overlay")
    # Synthetic overlays grid: Corr and PCA across ATM/Surface/UL
    p.add_argument("--synthetic-overlays", action="store_true", help="Render 2x3 grid of composite overlays: Corr/PCA x (ATM, Surface, UL)")
    p.add_argument("--synthetic-days", type=float, default=28.0, help="Target days to expiry for synthetic overlays grid")
    args = p.parse_args()

    target = (args.target or "").upper()
    peers = [t.strip().upper() for t in args.peers.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asof = _resolve_asof(target, None if args.latest else args.asof)

    # Pre-flight: repair DB if corrupted to avoid cascading failures
    required = [target] + ([t for t in args.peers.split(',') if t] if args.peers else [])
    if not _auto_repair_db_if_corrupt(required_tickers=required, max_expiries=int(args.max_expiries)):
        print("Fatal: database is corrupted and could not be auto-repaired.")
        return 2
    if not asof:
        print("No asof date available; some plots may be skipped.")

    # 1) Weights bar chart
    try:
        if peers:
            w = compute_unified_weights(target=target, peers=peers, mode=args.weight_mode, asof=asof)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_weights(ax, w)
            ax.set_title(f"Index Weights: {target} ({asof or 'latest'})")
            _savefig(fig, out_dir / "weights.png")
        else:
            print("No peers provided; skipping weights plot.")
    except Exception as e:
        print(f"Weights plot error: {e}")

    # 2) Relative-weight correlation matrix (skip for PCA modes)
    try:
        if asof:
            if args.weight_mode and args.weight_mode.lower().startswith("pca"):
                print("Skipping relative-weight matrix for PCA mode")
            else:
                tickers = [target] + peers if peers else [target]
                fig, ax = plt.subplots(figsize=(7.5, 6))
                # If using surface-based features, prefer a usable surface asof
                if args.weight_mode and (args.weight_mode.endswith("surface") or args.weight_mode.endswith("surface_grid")):
                    asof = _resolve_surface_asof_for_tickers(tickers, asof, args.max_expiries)
                # For surface features, render two variants: restricted and wide
                is_surface = args.weight_mode.endswith("surface") or args.weight_mode.endswith("surface_grid")
                if is_surface:
                    # Restricted: strict common date, strict intersection (min_coverage=1.0)
                    fig1, ax1 = plt.subplots(figsize=(7.5, 6))
                    feat1, corr1, w1 = compute_and_plot_relative_weight(
                        ax=ax1,
                        get_smile_slice=get_smile_slice,
                        tickers=tickers,
                        asof=asof,
                        target=target,
                        peers=peers,
                        atm_band=args.atm_band,
                        show_values=True,
                        clip_negative=True,
                        weight_power=1.0,
                        max_expiries=args.max_expiries,
                        weight_mode=args.weight_mode,
                        no_cache=bool(args.no_cache),
                        surface_min_coverage=1.0,
                        surface_strict_common_date=True,
                    )
                    ax1.set_title(f"Relative Weight Matrix (surface, restricted) {asof}")
                    _savefig(fig1, out_dir / "relative_weight_matrix_surface_restricted.png")

                    # Plot weights (restricted)
                    if w1 is not None:
                        figw1, aw1 = plt.subplots(figsize=(6, 4))
                        plot_weights(aw1, w1)
                        aw1.set_title("Surface Weights (restricted)")
                        _savefig(figw1, out_dir / "weights_surface_restricted.png")

                    # Wide: majority coverage (min_coverage=0.5), allow per‑ticker latest
                    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
                    feat2, corr2, w2 = compute_and_plot_relative_weight(
                        ax=ax2,
                        get_smile_slice=get_smile_slice,
                        tickers=tickers,
                        asof=asof,
                        target=target,
                        peers=peers,
                        atm_band=args.atm_band,
                        show_values=True,
                        clip_negative=True,
                        weight_power=1.0,
                        max_expiries=args.max_expiries,
                        weight_mode=args.weight_mode,
                        no_cache=bool(args.no_cache),
                        surface_min_coverage=0.5,
                        surface_strict_common_date=False,
                    )
                    ax2.set_title(f"Relative Weight Matrix (surface, wide) {asof}")
                    _savefig(fig2, out_dir / "relative_weight_matrix_surface_wide.png")

                    # Plot weights (wide)
                    if w2 is not None:
                        figw2, aw2 = plt.subplots(figsize=(6, 4))
                        plot_weights(aw2, w2)
                        aw2.set_title("Surface Weights (wide)")
                        _savefig(figw2, out_dir / "weights_surface_wide.png")
                else:
                    compute_and_plot_relative_weight(
                        ax=ax,
                        get_smile_slice=get_smile_slice,
                        tickers=tickers,
                        asof=asof,
                        target=target,
                        peers=peers,
                        atm_band=args.atm_band,
                        show_values=True,
                        clip_negative=True,
                        weight_power=1.0,
                        max_expiries=args.max_expiries,
                        weight_mode=args.weight_mode,
                        no_cache=bool(args.no_cache),
                    )
                    ax.set_title(f"Relative Weight Matrix ({asof})")
                    _savefig(fig, out_dir / "relative_weight_matrix.png")
        else:
            print("No asof resolved; skipping relative-weight matrix.")
    except Exception as e:
        print(f"Relative-weight plot error: {e}")

    # 3) ATM Term Structure (simple extractor) with peer overlays
    try:
        if asof:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            tgt_ts = plot_atm_term_structure(ax, target, asof, max_expiries=int(args.max_expiries))
            if peers:
                for p in peers:
                    try:
                        plot_atm_term_structure(
                            ax, p, asof,
                            max_expiries=int(args.max_expiries),
                            show_points=False,
                            line_kwargs={"alpha": 0.7, "label": p},
                        )
                    except Exception:
                        continue
                # Consolidated legend and title
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    ax.legend(loc="best", fontsize=8)
                ax.set_title(f"ATM Term Structure ({asof}) — {target} vs peers")
            _savefig(fig, out_dir / "term_structure.png")
        else:
            print("No asof resolved; skipping term structure.")
    except Exception as e:
        print(f"Term structure plot error: {e}")

    # 4) Term Smile at target days (overlay peers if provided)
    try:
        if asof:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            tgt_info = plot_term_smile(ax, target, asof, target_days=float(args.t_days), tolerance_days=float(args.tolerance_days))
            # Overlay peers as lines (no points) if provided
            if peers:
                for p in peers:
                    try:
                        plot_term_smile(
                            ax, p, asof,
                            target_days=float(args.t_days),
                            tolerance_days=float(args.tolerance_days),
                            show_points=False,
                            line_kwargs={"alpha": 0.7, "label": p},
                        )
                    except Exception:
                        continue
                # Reset a clean, combined title after overlays
                actual_days = None
                try:
                    actual_days = int(round(float(tgt_info.get("actual_days", args.t_days))))
                except Exception:
                    actual_days = int(float(args.t_days))
                ax.set_title(f"Term Smile ~{actual_days}d ({asof}) — {target} vs peers")
                # Ensure legend includes all lines
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    ax.legend(loc="best", fontsize=8)
            _savefig(fig, out_dir / "term_smile.png")
        else:
            print("No asof resolved; skipping term smile.")
    except Exception as e:
        print(f"Term smile plot error: {e}")

    # 5) Surface (3D or heatmap)
    try:
        if asof:
            # Resolve a usable surface asof for the target, pull if missing
            asof_surf = _resolve_surface_asof_for_tickers([target], asof, args.max_expiries)
            fig = plot_3d_vol_surface(target, asof_surf, mode=args.surface_mode, max_expiries=args.max_expiries)
            if fig is not None:
                _savefig(fig, out_dir / f"surface_{args.surface_mode}.png")
            else:
                print("Surface plot returned None; skipping save.")
        else:
            print("No asof resolved; skipping surface.")
    except Exception as e:
        print(f"Surface plot error: {e}")

    # 6) Volatility Dashboard (composite view)
    try:
        if asof:
            fig = create_vol_dashboard(target, asof, target_days=float(args.t_days))
            _savefig(fig, out_dir / "vol_dashboard.png")
        else:
            print("No asof resolved; skipping dashboard.")
    except Exception as e:
        print(f"Dashboard error: {e}")

    # 7) Underlying price history (1y) for target+peers
    try:
        tickers_all = [target] + peers if peers else [target]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_peer_underlyings_1y(ax, tickers_all, asof)
        _savefig(fig, out_dir / "underlyings_1y.png")
    except Exception as e:
        print(f"Underlying 1y plot error: {e}")

    # 8) 1x3 grids of weights for PCA and Correlation methods
    try:
        if peers and args.weights_grids:
            features = [f.strip() for f in args.features.split(',') if f.strip()]
            # PCA grid
            fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
            axes = np.atleast_1d(axes)
            for ax, feat in zip(axes, features):
                mode = f"pca_{feat}"
                try:
                    w = compute_unified_weights(target=target, peers=peers, mode=mode, asof=asof, **({"pca_ul_via_factors": True} if mode == "pca_ul" else {}))
                    plot_weights(ax, w)
                    ax.set_title(f"PCA Weights: {feat}")
                except Exception as e:
                    ax.clear(); ax.text(0.5,0.5,f"Error: {e}",ha='center',va='center')
            fig.suptitle(f"{target} PCA Weights ({asof or 'latest'})")
            _savefig(fig, out_dir / "weights_grid_pca.png")

            # Correlation grid
            fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
            axes = np.atleast_1d(axes)
            for ax, feat in zip(axes, features):
                mode = f"corr_{feat}"
                try:
                    w = compute_unified_weights(target=target, peers=peers, mode=mode, asof=asof, **({"pca_ul_via_factors": True} if mode == "pca_ul" else {}))
                    plot_weights(ax, w)
                    ax.set_title(f"Corr Weights: {feat}")
                except Exception as e:
                    ax.clear(); ax.text(0.5,0.5,f"Error: {e}",ha='center',va='center')
            fig.suptitle(f"{target} Correlation Weights ({asof or 'latest'})")
            _savefig(fig, out_dir / "weights_grid_corr.png")
        elif args.weights_grids and not peers:
            print("No peers provided; skipping weights grids.")
    except Exception as e:
        print(f"Weights grids error: {e}")

    # 9) Slide 10 grids: PCA and Correlation for (iv_atm, surface_grid, ul) with blue theme
    try:
        if getattr(args, "slide10", False):
            slide_target = target
            slide_peers = [t.strip().upper() for t in (getattr(args, "slide10_peers", "RGTI, QUBT, QBTS, CVS,T") or "").split(",") if t.strip()]
            features = [("iv_atm", "ATM"), ("surface_grid", "Surface"), ("ul", "UL")]

            import matplotlib as mpl
            plt.style.use("seaborn-v0_8-whitegrid")
            mpl.rcParams.update({
                "figure.facecolor": "#f6f9ff",
                "axes.facecolor": "#f6f9ff",
                "axes.labelcolor": "#0d2b52",
                "axes.titlecolor": "#0d2b52",
            })

            # PCA grid
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
            for ax, (feat, label) in zip(axes, features):
                try:
                    mode = f"pca_{feat}"
                    w = compute_unified_weights(target=slide_target, peers=slide_peers, mode=mode, asof=asof)
                    plot_weights(ax, w)
                    ax.set_title(f"PCA • {label}")
                except Exception as e:
                    ax.clear(); ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            fig.suptitle(f"PCA Weights • {slide_target} @ {asof or 'latest'}")
            _savefig(fig, out_dir / f"slide10_weights_grid_pca.png")
            plt.close(fig)

            # Correlation grid
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
            for ax, (feat, label) in zip(axes, features):
                try:
                    mode = f"corr_{feat}"
                    w = compute_unified_weights(target=slide_target, peers=slide_peers, mode=mode, asof=asof)
                    plot_weights(ax, w)
                    ax.set_title(f"Correlation • {label}")
                except Exception as e:
                    ax.clear(); ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            fig.suptitle(f"Correlation Weights • {slide_target} @ {asof or 'latest'}")
            _savefig(fig, out_dir / f"slide10_weights_grid_corr.png")
            plt.close(fig)
    except Exception as e:
        print(f"Slide10 grids error: {e}")

        # 10) Slide 10: Smile overlay using the same fitter as GUI (plot_smile_with_composite)
    try:
        if getattr(args, "slide10_smile", False):
            slide_peers = []
            if args.slide10_peers is not None:
                slide_peers = [t.strip().upper() for t in (args.slide10_peers or "").split(',') if t.strip()]
            else:
                slide_peers = peers

            T_days = float(args.slide10_days or 28.0)
            ci_in = float(args.slide10_ci)
            ci_level = ci_in / 100.0 if ci_in > 1.0 else (ci_in if 0.0 < ci_in < 1.0 else 0.68)

            df_tgt = get_smile_slice(target, asof, T_target_years=T_days / 365.25, max_expiries=None) if asof else pd.DataFrame()
            if df_tgt is None or df_tgt.empty:
                print("Slide10 smile: no target smile data; skipping.")
            else:
                fig, ax = plt.subplots(figsize=(7.5, 4.8))
                # Build weights and surfaces for composite overlay
                try:
                    w_map = compute_unified_weights(target=target, peers=slide_peers, mode=args.weight_mode, asof=asof) if slide_peers else {}
                    if isinstance(w_map, pd.Series):
                        w_series = w_map.dropna().astype(float)
                        w_map = {str(k).upper(): float(v) for k, v in w_series.items() if np.isfinite(v) and float(v) > 0.0}
                    elif isinstance(w_map, dict):
                        w_map = {str(k).upper(): float(v) for k, v in dict(w_map).items() if np.isfinite(v) and float(v) > 0.0}
                    tickers_for_surf = [target] + slide_peers
                    surfaces = build_surface_grids(tickers=tickers_for_surf, use_atm_only=False, max_expiries=args.max_expiries)
                except Exception:
                    w_map = {}
                    surfaces = {}

                # GUI-consistent smile fit and optional composite overlay
                try:
                    plot_smile_with_composite(
                        ax=ax,
                        df=df_tgt,
                        target=target,
                        asof=asof,
                        model="svi",
                        T_days=T_days,
                        ci=ci_level,
                        overlay_composite=bool(w_map),
                        surfaces=surfaces if w_map else None,
                        weights=w_map if w_map else None,
                    )
                except Exception as e:
                    print(f"Slide10 smile: GUI-style smile plot failed: {e}")

                # Thin peer SVI lines (no points/CI)
                if slide_peers:
                    for p in slide_peers:
                        try:
                            df_p = get_smile_slice(p, asof, T_target_years=T_days / 365.25, max_expiries=None)
                            if df_p is None or df_p.empty:
                                continue
                            S_p = float(pd.to_numeric(df_p["S"], errors="coerce").median())
                            K_p = pd.to_numeric(df_p["K"], errors="coerce").to_numpy(dtype=float, copy=False)
                            IV_p = pd.to_numeric(df_p["sigma"], errors="coerce").to_numpy(dtype=float, copy=False)
                            T_p = float(pd.to_numeric(df_p["T"], errors="coerce").median())
                            m_grid_p = np.linspace(0.7, 1.3, 121)
                            K_grid_p = m_grid_p * float(S_p if np.isfinite(S_p) else 1.0)
                            res_p = fit_smile_models_with_bands(float(S_p), K_p, float(T_p), IV_p, K_grid_p, None)
                            params_p = res_p.get('models', {}).get('svi', {}) or {}
                            fit_and_plot_smile(
                                ax,
                                S=float(S_p),
                                K=K_p,
                                T=float(T_p),
                                iv=IV_p,
                                model="svi",
                                params=params_p,
                                bands=None,
                                moneyness_grid=(0.7, 1.3, 121),
                                show_points=False,
                                line_kwargs={"lw": 1.1, "alpha": 0.7, "label": p},
                                label=p,
                            )
                        except Exception:
                            continue

                _savefig(fig, out_dir / "slide10_smile_overlay.png")
                plt.close(fig)
    except Exception as e:
        print(f"Slide10 smile figure error: {e}")# 11) Methods: overlay 5 methods' composite smiles vs target + accuracy table (MAE, cosine)
    try:
        if getattr(args, "methods_smile", False) and peers:
            T_days = float(args.methods_days or 28.0)
            # Pull target data near requested days
            df_tgt = get_smile_slice(target, asof, T_target_years=T_days / 365.25, max_expiries=None) if asof else pd.DataFrame()
            if df_tgt is None or df_tgt.empty:
                print("Methods smile: no target smile data; skipping.")
            else:
                # Target arrays
                try:
                    S = float(pd.to_numeric(df_tgt["S"], errors="coerce").median())
                    K = pd.to_numeric(df_tgt["K"], errors="coerce").to_numpy(dtype=float, copy=False)
                    IV = pd.to_numeric(df_tgt["sigma"], errors="coerce").to_numpy(dtype=float, copy=False)
                    T_used = float(pd.to_numeric(df_tgt["T"], errors="coerce").median())
                except Exception:
                    S = np.nan; K = np.array([]); IV = np.array([]); T_used = np.nan
                if not np.isfinite(S) or K.size == 0 or IV.size == 0 or not np.isfinite(T_used):
                    print("Methods smile: invalid target arrays; skipping.")
                else:
                    # Fit SVI for target and build evaluation grid
                    m_grid = np.linspace(0.7, 1.3, 121)
                    K_grid = m_grid * float(S)
                    # Fit using points near the display window to avoid tail-driven misfits
                    try:
                        m_vals_all = K / float(S)
                    except Exception:
                        m_vals_all = np.full_like(K, np.nan, dtype=float)
                    mask_fit = np.isfinite(m_vals_all) & np.isfinite(IV) & (m_vals_all >= float(m_grid.min())) & (m_vals_all <= float(m_grid.max()))
                    K_fit = K[mask_fit] if np.count_nonzero(mask_fit) >= 5 else K
                    IV_fit = IV[mask_fit] if np.count_nonzero(mask_fit) >= 5 else IV
                    try:
                        res = fit_smile_models_with_bands(float(S), K_fit, float(T_used), IV_fit, K_grid, None)
                        p_svi = res.get('models', {}).get('svi', {}) or {}
                    except Exception:
                        p_svi = {}
                    # If no params, bail
                    if not p_svi:
                        print("Methods smile: SVI fit failed; skipping.")
                    else:
                        y_tgt = svi_smile_iv(float(S), K_grid, float(T_used), p_svi)

                        # Build surfaces once
                        tickers_for_surf = [target] + peers
                        try:
                            surfaces = build_surface_grids(tickers=tickers_for_surf, use_atm_only=False, max_expiries=args.max_expiries)
                        except Exception:
                            surfaces = {}
                        # Choose target date (prefer requested asof normalized)
                        date_use = None
                        if target in surfaces and surfaces[target]:
                            if asof:
                                asof_ts = pd.to_datetime(asof).normalize()
                                norm_map = {pd.to_datetime(k).normalize(): k for k in surfaces[target].keys()}
                                date_use = norm_map.get(asof_ts, max(surfaces[target].keys()))
                            else:
                                date_use = max(surfaces[target].keys())

                        # Helper: parse column labels to days
                        def _cols_to_days_local(cols):
                            out = []
                            for c in cols:
                                try:
                                    if isinstance(c, str) and c.endswith('d'):
                                        out.append(float(c[:-1]))
                                    else:
                                        out.append(float(c))
                                except Exception:
                                    out.append(np.nan)
                            return np.asarray(out, float)
                        # Helper: moneyness from index labels
                        def _mny_from_index_local(index):
                            vals = []
                            for lab in index:
                                s = str(lab)
                                try:
                                    if '-' in s:
                                        a, b = s.split('-', 1)
                                        vals.append((float(a) + float(b)) / 2.0)
                                    elif s.startswith('m'):
                                        vals.append(float(s[1:]))
                                    else:
                                        vals.append(float(s))
                                except Exception:
                                    vals.append(np.nan)
                            return np.asarray(vals, float)

                        # Methods to evaluate
                        methods = [
                            ("corr_iv_atm", "Correlation"),
                            ("pca_iv_atm", "PCA"),
                            ("cosine_iv_atm", "Cosine"),
                            ("equal", "Equal"),
                            ("oi", "OpenInterest"),
                        ]
                        colors = {
                            "Correlation": "tab:blue",
                            "PCA": "tab:orange",
                            "Cosine": "tab:green",
                            "Equal": "tab:red",
                            "OpenInterest": "tab:purple",
                        }
                        metrics = []  # (label, mae, cosine)

                        # Figure with table on right
                        fig = plt.figure(figsize=(10.5, 5.2))
                        gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0])
                        ax = fig.add_subplot(gs[0, 0])
                        ax_tbl = fig.add_subplot(gs[0, 1])
                        ax_tbl.axis('off')

                        # Plot target
                        ax.plot(m_grid, y_tgt, label=f"{target} SVI", lw=2.4, color="black")

                        # For each method, build composite and overlay + compute metrics
                        for mode, label in methods:
                            try:
                                w = compute_unified_weights(target=target, peers=peers, mode=mode, asof=asof, **({"pca_ul_via_factors": True} if mode == "pca_ul" else {}))
                                if isinstance(w, pd.Series):
                                    w = w.dropna().astype(float)
                                    w = {k.upper(): float(v) for k, v in w.items() if np.isfinite(v) and float(v) > 0}
                                elif isinstance(w, dict):
                                    w = {str(k).upper(): float(v) for k, v in dict(w).items() if np.isfinite(v) and float(v) > 0}
                                else:
                                    w = {}
                                if not w:
                                    continue
                                # Compose surfaces
                                comp_surfaces = {t: surfaces.get(t, {}) for t in w.keys() if t in surfaces}
                                if not comp_surfaces:
                                    continue
                                comp_by_date = combine_surfaces(comp_surfaces, w)
                                # pick date aligned to target
                                if date_use not in comp_by_date:
                                    if comp_by_date:
                                        date_c = max(comp_by_date.keys())
                                    else:
                                        continue
                                else:
                                    date_c = date_use
                                tgt_grid = surfaces.get(target, {}).get(date_use)
                                comp_grid = comp_by_date.get(date_c)
                                if tgt_grid is None or comp_grid is None:
                                    continue
                                # Overlay line (visual)
                                try:
                                    comp_df_map2 = {t: comp_surfaces[t][date_c] for t in comp_surfaces if date_c in comp_surfaces[t]}
                                    plot_composite_smile_overlay(ax, tgt_grid, comp_grid, T_days, label=label, composite_surfaces=comp_df_map2, weights=w)
                                except Exception:
                                    pass
                                # Compute metrics on common x grid (m_grid)
                                # Take nearest tenor from composite grid
                                syn_days = _cols_to_days_local(comp_grid.columns)
                                if not np.isfinite(syn_days).any():
                                    continue
                                i_syn = int(np.nanargmin(np.abs(syn_days - T_days)))
                                col = comp_grid.columns[i_syn]
                                pmny = _mny_from_index_local(comp_grid.index)
                                piv = pd.to_numeric(comp_grid[col], errors="coerce").to_numpy(float)
                                mask = np.isfinite(pmny) & np.isfinite(piv)
                                if mask.sum() < 2:
                                    continue
                                # Interpolate composite IV to m_grid
                                order = np.argsort(pmny[mask])
                                x_src = pmny[mask][order]
                                y_src = piv[mask][order]
                                # Restrict to overlap range
                                lo, hi = float(x_src.min()), float(x_src.max())
                                mask_eval = (m_grid >= lo) & (m_grid <= hi)
                                if mask_eval.sum() < 2:
                                    continue
                                y_comp = np.interp(m_grid[mask_eval], x_src, y_src)
                                y_ref = y_tgt[mask_eval]
                                # MAE
                                mae = float(np.nanmean(np.abs(y_comp - y_ref)))
                                # Cosine similarity on demeaned vectors
                                u = y_ref - float(np.nanmean(y_ref))
                                v = y_comp - float(np.nanmean(y_comp))
                                denom = float(np.linalg.norm(u) * np.linalg.norm(v))
                                cos = float(np.dot(u, v) / denom) if denom > 0 else np.nan
                                metrics.append((label, mae, cos))
                            except Exception:
                                continue

                        # Finalize plot
                        ax.set_xlabel("Moneyness (K/S)")
                        ax.set_ylabel("Implied Volatility")
                        try:
                            actual_days = int(round(float(T_used) * 365.25))
                        except Exception:
                            actual_days = int(T_days)
                        ax.set_title(f"Composite by Method vs Target (~{actual_days}d, {asof or 'latest'})")
                        handles, labels = ax.get_legend_handles_labels()
                        if handles and labels:
                            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8)
                        ax.grid(True, alpha=0.3)

                        # Build accuracy table
                        if metrics:
                            dfm = pd.DataFrame(metrics, columns=["Method", "MAE", "Cosine"])
                            # Sort by MAE ascending
                            dfm = dfm.sort_values(["MAE", "Cosine"], ascending=[True, False])
                            cell_text = [[r["Method"], f"{r['MAE']:.4f}", f"{r['Cosine']:.3f}"] for _, r in dfm.iterrows()]
                            table = ax_tbl.table(cellText=cell_text, colLabels=["Method", "MAE", "Cosine"], loc='center')
                            table.auto_set_font_size(False)
                            table.set_fontsize(9)
                            table.scale(1, 1.3)
                            ax_tbl.set_title("Accuracy vs Target")
                            # Save metrics CSV
                            try:
                                dfm.to_csv(out_dir / "methods_smile_metrics.csv", index=False)
                            except Exception:
                                pass
                        _savefig(fig, out_dir / "methods_smile_overlay.png")
                        plt.close(fig)
        elif getattr(args, "methods_smile", False) and not peers:
            print("Methods smile requested but no peers provided; skipping.")
    except Exception as e:
        print(f"Methods smile figure error: {e}")

    # 12) Synthetic overlays grid: Corr/PCA x (ATM, Surface, UL)
    try:
        if getattr(args, "synthetic_overlays", False) and peers:
            T_days = float(args.synthetic_days or 28.0)
            # Pull and fit target SVI once
            df_tgt = get_smile_slice(target, asof, T_target_years=T_days / 365.25, max_expiries=None) if asof else pd.DataFrame()
            if df_tgt is None or df_tgt.empty:
                print("Synthetic overlays: no target smile data; skipping.")
            else:
                try:
                    S = float(pd.to_numeric(df_tgt["S"], errors="coerce").median())
                    K = pd.to_numeric(df_tgt["K"], errors="coerce").to_numpy(dtype=float, copy=False)
                    IV = pd.to_numeric(df_tgt["sigma"], errors="coerce").to_numpy(dtype=float, copy=False)
                    T_used = float(pd.to_numeric(df_tgt["T"], errors="coerce").median())
                except Exception:
                    S = np.nan; K = np.array([]); IV = np.array([]); T_used = np.nan
                if not (np.isfinite(S) and K.size and IV.size and np.isfinite(T_used)):
                    print("Synthetic overlays: invalid target arrays; skipping.")
                else:
                    m_grid = np.linspace(0.7, 1.3, 121)
                    K_grid = m_grid * float(S)
                    try:
                        # Restrict fit to in-window quotes to avoid tail-driven misfits
                        try:
                            m_vals_all = K / float(S)
                        except Exception:
                            m_vals_all = np.full_like(K, np.nan, dtype=float)
                        mask_fit = (
                            np.isfinite(m_vals_all)
                            & np.isfinite(IV)
                            & (m_vals_all >= float(m_grid.min()))
                            & (m_vals_all <= float(m_grid.max()))
                        )
                        K_fit = K[mask_fit] if np.count_nonzero(mask_fit) >= 5 else K
                        IV_fit = IV[mask_fit] if np.count_nonzero(mask_fit) >= 5 else IV
                        res = fit_smile_models_with_bands(float(S), K_fit, float(T_used), IV_fit, K_grid, None)
                        p_svi = res.get('models', {}).get('svi', {}) or {}
                    except Exception:
                        p_svi = {}
                    if not p_svi:
                        print("Synthetic overlays: SVI fit failed; skipping.")
                    else:
                        y_tgt = svi_smile_iv(float(S), K_grid, float(T_used), p_svi)
                        # Build surfaces once
                        tickers_for_surf = [target] + peers
                        try:
                            surfaces = build_surface_grids(tickers=tickers_for_surf, use_atm_only=False, max_expiries=args.max_expiries)
                        except Exception:
                            surfaces = {}

                        # Prepare figure grid
                        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
                        grid_modes = [
                            [("corr_iv_atm", "Corr — ATM"), ("corr_surface_grid", "Corr — Surface"), ("corr_ul", "Corr — UL")],
                            [("pca_iv_atm", "PCA — ATM"), ("pca_surface_grid", "PCA — Surface"), ("pca_ul", "PCA — UL")],
                        ]
                        weights_rows = []  # collect per-mode weights for diagnostics
                        for r in range(2):
                            for c in range(3):
                                ax = axes[r][c]
                                mode, title = grid_modes[r][c]
                                # plot target
                                ax.plot(m_grid, y_tgt, color="black", lw=2.0, label=f"{target} SVI")
                                # light scatter of observed target points within view
                                try:
                                    m_sc = K / float(S)
                                    mask_sc = np.isfinite(m_sc) & np.isfinite(IV) & (m_sc >= 0.6) & (m_sc <= 1.4)
                                    if np.any(mask_sc):
                                        ax.scatter(m_sc[mask_sc], IV[mask_sc], s=12, alpha=0.35, color="black", linewidths=0, label="_nolegend_")
                                except Exception:
                                    pass
                                try:
                                    ax.set_xlim(0.6, 1.4)
                                except Exception:
                                    pass
                                try:
                                    w = compute_unified_weights(target=target, peers=peers, mode=mode, asof=asof, **({"pca_ul_via_factors": True} if mode == "pca_ul" else {}))
                                    if isinstance(w, pd.Series):
                                        w = w.dropna().astype(float)
                                        w = {k.upper(): float(v) for k, v in w.items() if np.isfinite(v) and float(v) > 0}
                                    elif isinstance(w, dict):
                                        w = {str(k).upper(): float(v) for k, v in dict(w).items() if np.isfinite(v) and float(v) > 0}
                                    else:
                                        w = {}
                                    # record weights for diagnostics
                                    if w:
                                        for tkr, val in w.items():
                                            weights_rows.append({"mode": mode, "ticker": tkr, "weight": float(val)})
                                    if w:
                                        comp_surfaces = {t: surfaces.get(t, {}) for t in w.keys() if t in surfaces}
                                        if comp_surfaces:
                                            comp_by_date = combine_surfaces(comp_surfaces, w)
                                            # Pick a date aligned to target or fallback to composite latest
                                            date_use = None
                                            if target in surfaces and surfaces[target]:
                                                if asof:
                                                    asof_ts = pd.to_datetime(asof).normalize()
                                                    norm_map = {pd.to_datetime(k).normalize(): k for k in surfaces[target].keys()}
                                                    date_use = norm_map.get(asof_ts, max(surfaces[target].keys()))
                                                else:
                                                    date_use = max(surfaces[target].keys())
                                            date_c = date_use if (date_use in comp_by_date) else (max(comp_by_date.keys()) if comp_by_date else None)
                                            tgt_grid = surfaces.get(target, {}).get(date_use)
                                            comp_grid = comp_by_date.get(date_c)
                                            if tgt_grid is not None and comp_grid is not None:
                                                comp_df_map = {t: comp_surfaces[t][date_c] for t in comp_surfaces if date_c in comp_surfaces[t]}
                                                plot_composite_smile_overlay(ax, tgt_grid, comp_grid, T_days, label="Composite", composite_surfaces=comp_df_map, weights=w)
                                except Exception as e:
                                    ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                                ax.set_title(title)
                                ax.set_xlabel("Moneyness (K/S)")
                                ax.set_ylabel("Implied Volatility")
                                handles, labels = ax.get_legend_handles_labels()
                                if handles and labels:
                                    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1))
                                ax.grid(True, alpha=0.3)

                        try:
                            actual_days = int(round(float(T_used) * 365.25))
                        except Exception:
                            actual_days = int(T_days)
                        fig.suptitle(f"Synthetic Overlays: Corr/PCA vs Target (~{actual_days}d, {asof or 'latest'}) — {target}")
                        _savefig(fig, out_dir / "synthetic_overlays_grid.png")
                        plt.close(fig)

                        # Dump per-mode weights for verification (shows recomputation per mode)
                        try:
                            if weights_rows:
                                pd.DataFrame(weights_rows).to_csv(out_dir / "synthetic_overlays_weights.csv", index=False)
                        except Exception:
                            pass

                        # Also render all 6 overlays on a single plot
                        try:
                            fig_all, ax_all = plt.subplots(figsize=(9.5, 5.5))
                            ax_all.plot(m_grid, y_tgt, color="black", lw=2.4, label=f"{target} SVI")
                            # light scatter of observed target points within view
                            try:
                                m_sc = K / float(S)
                                mask_sc = np.isfinite(m_sc) & np.isfinite(IV) & (m_sc >= 0.6) & (m_sc <= 1.4)
                                if np.any(mask_sc):
                                    ax_all.scatter(m_sc[mask_sc], IV[mask_sc], s=12, alpha=0.35, color="black", linewidths=0, label="_nolegend_")
                            except Exception:
                                pass
                            try:
                                ax_all.set_xlim(0.6, 1.4)
                            except Exception:
                                pass
                            overlay_list = [
                                ("corr_iv_atm", "Corr — ATM"),
                                ("corr_surface_grid", "Corr — Surface"),
                                ("corr_ul", "Corr — UL"),
                                ("pca_iv_atm", "PCA — ATM"),
                                ("pca_surface_grid", "PCA — Surface"),
                                ("pca_ul", "PCA — UL"),
                            ]
                            metrics_rows = []
                            for mode, label in overlay_list:
                                try:
                                    w = compute_unified_weights(target=target, peers=peers, mode=mode, asof=asof, **({"pca_ul_via_factors": True} if mode == "pca_ul" else {}))
                                    if isinstance(w, pd.Series):
                                        w = w.dropna().astype(float)
                                        w = {k.upper(): float(v) for k, v in w.items() if np.isfinite(v) and float(v) > 0}
                                    elif isinstance(w, dict):
                                        w = {str(k).upper(): float(v) for k, v in dict(w).items() if np.isfinite(v) and float(v) > 0}
                                    else:
                                        w = {}
                                    if not w:
                                        continue
                                    comp_surfaces = {t: surfaces.get(t, {}) for t in w.keys() if t in surfaces}
                                    if not comp_surfaces:
                                        continue
                                    comp_by_date = combine_surfaces(comp_surfaces, w)
                                    # Align date to target or fallback to composite latest
                                    date_use = None
                                    if target in surfaces and surfaces[target]:
                                        if asof:
                                            asof_ts = pd.to_datetime(asof).normalize()
                                            norm_map = {pd.to_datetime(k).normalize(): k for k in surfaces[target].keys()}
                                            date_use = norm_map.get(asof_ts, max(surfaces[target].keys()))
                                        else:
                                            date_use = max(surfaces[target].keys())
                                    date_c = date_use if (date_use in comp_by_date) else (max(comp_by_date.keys()) if comp_by_date else None)
                                    tgt_grid = surfaces.get(target, {}).get(date_use)
                                    comp_grid = comp_by_date.get(date_c)
                                    if tgt_grid is None or comp_grid is None:
                                        continue
                                    comp_df_map = {t: comp_surfaces[t][date_c] for t in comp_surfaces if date_c in comp_surfaces[t]}
                                    plot_composite_smile_overlay(ax_all, tgt_grid, comp_grid, T_days, label=label, composite_surfaces=comp_df_map, weights=w)
                                    # Compute accuracy metrics vs target SVI on m_grid
                                    try:
                                        # local helpers
                                        def _cols_to_days_local(cols):
                                            arr = []
                                            for c in cols:
                                                try:
                                                    if isinstance(c, str) and c.endswith('d'):
                                                        arr.append(float(c[:-1]))
                                                    else:
                                                        arr.append(float(c))
                                                except Exception:
                                                    arr.append(np.nan)
                                            return np.asarray(arr, float)
                                        def _mny_from_index_local(index):
                                            out = []
                                            for lab in index:
                                                s = str(lab)
                                                try:
                                                    if '-' in s:
                                                        a, b = s.split('-', 1)
                                                        out.append((float(a) + float(b)) / 2.0)
                                                    elif s.startswith('m'):
                                                        out.append(float(s[1:]))
                                                    else:
                                                        out.append(float(s))
                                                except Exception:
                                                    out.append(np.nan)
                                            return np.asarray(out, float)
                                        syn_days = _cols_to_days_local(comp_grid.columns)
                                        if np.isfinite(syn_days).any():
                                            i_syn = int(np.nanargmin(np.abs(syn_days - T_days)))
                                            col = comp_grid.columns[i_syn]
                                            pmny = _mny_from_index_local(comp_grid.index)
                                            piv = pd.to_numeric(comp_grid[col], errors='coerce').to_numpy(float)
                                            valid = np.isfinite(pmny) & np.isfinite(piv)
                                            if np.sum(valid) >= 2:
                                                order = np.argsort(pmny[valid])
                                                x_src = pmny[valid][order]
                                                y_src = piv[valid][order]
                                                lo, hi = float(x_src.min()), float(x_src.max())
                                                m_eval = m_grid[(m_grid >= max(lo, 0.6)) & (m_grid <= min(hi, 1.4))]
                                                if m_eval.size >= 3:
                                                    y_comp = np.interp(m_eval, x_src, y_src)
                                                    mask_idx = np.isin(m_grid, m_eval)
                                                    y_ref = y_tgt[mask_idx]
                                                    mae = float(np.nanmean(np.abs(y_comp - y_ref)))
                                                    u = y_ref - float(np.nanmean(y_ref))
                                                    v = y_comp - float(np.nanmean(y_comp))
                                                    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
                                                    cos = float(np.dot(u, v) / denom) if denom > 0 else np.nan
                                                    # Pearson correlation on the sampled vectors (shape similarity)
                                                    try:
                                                        r = float(np.corrcoef(y_ref, y_comp)[0, 1])
                                                    except Exception:
                                                        r = np.nan
                                                    # Persist sampled vectors for inspection per mode
                                                    try:
                                                        df_vec = pd.DataFrame({'m': m_eval, 'target_svi': y_ref, 'composite': y_comp})
                                                        safe = str(mode).replace('/', '_')
                                                        df_vec.to_csv(out_dir / f"synthetic_overlays_vectors_{safe}.csv", index=False)
                                                    except Exception:
                                                        pass
                                                    # Log concise metrics line
                                                    try:
                                                        print(f"METRICS {mode}: n={m_eval.size} MAE={mae:.4f} Cos={cos:.3f} r={r:.3f}")
                                                    except Exception:
                                                        pass
                                                    metrics_rows.append({'Method': label, 'MAE': mae, 'Cosine': cos, 'Correlation': r})
                                    except Exception:
                                        pass
                                except Exception:
                                    continue
                            ax_all.set_title(f"Synthetic Overlays (All) — {target} @ {asof or 'latest'} (~{actual_days}d)")
                            ax_all.set_xlabel("Moneyness (K/S)")
                            ax_all.set_ylabel("Implied Volatility")
                            handles, labels = ax_all.get_legend_handles_labels()
                            if handles and labels:
                                ax_all.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8)
                            ax_all.grid(True, alpha=0.3)
                            # Always save metrics CSV if computed (even if table is not drawn)
                            try:
                                if metrics_rows:
                                    dfm = pd.DataFrame(metrics_rows)
                                    # allow NaNs in correlation; sort by MAE asc, then by |Correlation| desc
                                    if not dfm.empty:
                                        dfm = dfm.sort_values(['MAE','Cosine'], ascending=[True, False])
                                        dfm.to_csv(out_dir / 'synthetic_overlays_metrics.csv', index=False)
                            except Exception:
                                pass
                            _savefig(fig_all, out_dir / "synthetic_overlays_all.png")
                            plt.close(fig_all)
                        except Exception:
                            pass
        elif getattr(args, "synthetic_overlays", False) and not peers:
            print("Synthetic overlays requested but no peers provided; skipping.")
    except Exception as e:
        print(f"Synthetic overlays figure error: {e}")

    print(f"Done. Plots saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










