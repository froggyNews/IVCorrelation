# display/gui/gui_plot_manager.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path to ensure our display package is found first
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from display.plotting.correlation_detail_plot import (
    compute_and_plot_correlation,   # draws the corr heatmap
)
from analysis.correlation_utils import corr_weights
from analysis.beta_builder import pca_weights, pca_weights_from_atm_matrix
from display.plotting.smile_plot import fit_and_plot_smile
from display.plotting.term_plot import plot_atm_term_structure
from analysis.model_params_logger import append_params

# surfaces & synth
from analysis.syntheticETFBuilder import build_surface_grids, combine_surfaces




from analysis.analysis_pipeline import get_smile_slice, relative_value_atm_report_corrweighted
from analysis.pillars import compute_atm_by_expiry

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from analysis.pillars import atm_curve_for_ticker_on_date
from display.plotting.anim_utils import animate_surface_timesweep
DEFAULT_ATM_BAND = ATM_BAND = 0.05
def _cols_to_days(cols) -> np.ndarray:
    out = []
    for c in cols:
        try:
            out.append(int(float(c)))
        except Exception:
            s = "".join(ch for ch in str(c) if ch.isdigit() or ch in ".-")
            out.append(int(float(s)) if s else 0)
    return np.array(out, dtype=int)

def _nearest_tenor_idx(tenor_days, target_days: float) -> int:
    arr = np.array(list(tenor_days), dtype=float)
    return int(np.argmin(np.abs(arr - float(target_days))))

def _mny_from_index_labels(idx) -> np.ndarray:
    out = []
    for label in idx:
        s = str(label)
        if "-" in s:
            try:
                a, b = s.split("-", 1)
                out.append((float(a) + float(b)) / 2.0)
                continue
            except Exception:
                pass
        try:
            out.append(float(s))
        except Exception:
            num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
            out.append(float(num) if num else np.nan)
    return np.array(out, dtype=float)

# at top of class PlotManager
class PlotManager:
    def __init__(self):
        self.get_smile_slice = get_smile_slice
        self.last_corr_df: pd.DataFrame | None = None
        self.last_corr_meta: dict = {}
        # Current max_expiries setting
        self._current_max_expiries = None
        # --- click-through TTE state ---
        self.canvas = None
        self._cid_click = None
        self._current_plot_type = None
        self._smile_ctx = None   # dict storing precomputed arrays + current index + overlay bits
        
        # --- animation state ---
        self._animation = None
        self._animation_paused = False
        self._animation_speed = 120  # ms between frames

    def attach_canvas(self, canvas):
        self.canvas = canvas
        if self._cid_click is not None:
            try:
                self.canvas.mpl_disconnect(self._cid_click)
            except Exception:
                pass
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _clear_correlation_colorbar(self, ax):
        """Remove any existing correlation colorbar to prevent it from appearing on non-correlation plots."""
        try:
            if hasattr(ax.figure, '_correlation_colorbar'):
                ax.figure._correlation_colorbar.remove()
                delattr(ax.figure, '_correlation_colorbar')
                if hasattr(ax.figure, '_orig_position'):
                    ax.set_position(ax.figure._orig_position)
                if hasattr(ax.figure, '_orig_subplotpars'):
                    l, r, b, t = ax.figure._orig_subplotpars
                    ax.figure.subplots_adjust(left=l, right=r, bottom=b, top=t)
        except Exception:
            pass

    # ---- main entry ----
    def plot(self, ax: plt.Axes, settings: dict):
        plot_type     = settings["plot_type"]
        target        = settings["target"]
        asof          = settings["asof"]
        model         = settings["model"]
        T_days        = settings["T_days"]
        ci            = settings["ci"]
        x_units       = settings["x_units"]
        weight_mode   = settings["weight_mode"]
        overlay_synth = settings.get("overlay_synth", False)
        overlay_peers = settings.get("overlay_peers", False)
        peers         = settings["peers"]
        pillars       = settings["pillars"]
        max_expiries  = settings.get("max_expiries", 6)

        # remember current plot type for the click handler
        self._current_plot_type = plot_type
        self._current_max_expiries = max_expiries

        # create a bounded get_smile_slice function with current max_expiries
        def bounded_get_smile_slice(ticker, asof_date=None, T_target_years=None, call_put=None, nearest_by="T"):
            return get_smile_slice(ticker, asof_date, T_target_years, call_put, nearest_by, max_expiries=self._current_max_expiries)
        self.get_smile_slice = bounded_get_smile_slice

        ax.clear()

        # --- Smile: click-through path (NO df_all needed here) ---
        if plot_type.startswith("Smile"):
            # Clear any correlation colorbar for non-correlation plots
            self._clear_correlation_colorbar(ax)
            # load all expiries so we can click through them
            chain_df = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
            if chain_df is None or chain_df.empty:
                ax.set_title("No data"); return

            # precompute arrays for fast slicing
            T_arr = pd.to_numeric(chain_df["T"], errors="coerce").to_numpy(float)
            K_arr = pd.to_numeric(chain_df["K"], errors="coerce").to_numpy(float)
            sigma_arr = pd.to_numeric(chain_df["sigma"], errors="coerce").to_numpy(float)
            S_arr = pd.to_numeric(chain_df["S"], errors="coerce").to_numpy(float)
            expiry_arr = pd.to_datetime(chain_df.get("expiry"), errors="coerce").to_numpy()

            Ts = np.sort(np.unique(T_arr[np.isfinite(T_arr)]))
            if Ts.size == 0:
                ax.set_title("No expiries"); return
            target_T = float(T_days) / 365.25
            idx0 = int(np.argmin(np.abs(Ts - target_T)))

            weights = None
            tgt_surface = None
            syn_surface = None
            peer_slices: dict[str, dict] = {}

            if overlay_synth and peers:
                weights = self._weights_from_ui_or_matrix(
                    target, peers, weight_mode, asof=asof,
                    pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None,
                )
                try:
                    tickers = list({target, *peers})
                    surfaces = build_surface_grids(
                        tickers=tickers,
                        tenors=None,
                        mny_bins=None,
                        use_atm_only=False,
                        max_expiries=max_expiries,
                    )
                    if target in surfaces and asof in surfaces[target]:
                        tgt_surface = surfaces[target][asof]
                    peer_surfaces = {p: surfaces[p] for p in peers if p in surfaces}
                    synth_by_date = combine_surfaces(peer_surfaces, weights.to_dict())
                    syn_surface = synth_by_date.get(asof)
                except Exception:
                    tgt_surface = None
                    syn_surface = None
                for p in peers:
                    df_p = get_smile_slice(p, asof, T_target_years=None, max_expiries=max_expiries)
                    if df_p is None or df_p.empty:
                        continue
                    T_p = pd.to_numeric(df_p["T"], errors="coerce").to_numpy(float)
                    K_p = pd.to_numeric(df_p["K"], errors="coerce").to_numpy(float)
                    sigma_p = pd.to_numeric(df_p["sigma"], errors="coerce").to_numpy(float)
                    S_p = pd.to_numeric(df_p["S"], errors="coerce").to_numpy(float)
                    peer_slices[p.upper()] = {
                        "T_arr": T_p,
                        "K_arr": K_p,
                        "sigma_arr": sigma_p,
                        "S_arr": S_p,
                    }

            self._smile_ctx = {
                "ax": ax,
                "T_arr": T_arr,
                "K_arr": K_arr,
                "sigma_arr": sigma_arr,
                "S_arr": S_arr,
                "Ts": Ts,
                "idx": idx0,
                "settings": settings,
                "weights": weights,
                "tgt_surface": tgt_surface,
                "syn_surface": syn_surface,
                "peer_slices": peer_slices,
                "expiry_arr": expiry_arr,
            }
            self._render_smile_at_index()
            return

        # --- Term: needs all expiries for this day ---
        elif plot_type.startswith("Term"):
            # Clear any correlation colorbar for non-correlation plots
            self._clear_correlation_colorbar(ax)
            df_all = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
            if df_all is None or df_all.empty:
                ax.set_title("No data"); return
            self._plot_term(ax, df_all, target, asof, x_units, ci, overlay_synth, peers, weight_mode)
            return

        # --- Corr Matrix: doesn't need df_all ---
        elif plot_type.startswith("Corr Matrix"):
            self._plot_corr_matrix(ax, target, peers, asof, pillars, weight_mode)
            return

        # --- Synthetic Surface: doesn't need df_all here ---
        elif plot_type.startswith("Synthetic Surface"):
            # Clear any correlation colorbar for non-correlation plots
            self._clear_correlation_colorbar(ax)
            self._plot_synth_surface(ax, target, peers, asof, T_days, weight_mode)
            return

        # --- ETF Weights only ---
        elif plot_type.startswith("ETF Weights"):
            # Clear any correlation colorbar for non-correlation plots
            self._clear_correlation_colorbar(ax)
            if not peers:
                ax.text(0.5, 0.5, "No peers", ha="center", va="center")
                return
            weights = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=pillars)
            if weights is None or weights.empty:
                ax.text(0.5, 0.5, "No weights", ha="center", va="center")
                return
            from display.plotting.weights_plot import plot_weights
            plot_weights(ax, weights)
            return

        else:
            ax.text(0.5, 0.5, f"Unknown plot: {plot_type}", ha="center", va="center")

    # ---- implementations ----
    def _on_click(self, event):
        if not self.is_smile_active() or event.inaxes is None:
            return
        if event.inaxes is not self._smile_ctx["ax"]:
            return

        if event.button == 1:
            self.next_expiry()
        elif event.button in (3, 2):
            self.prev_expiry()

    def _weights_from_ui_or_matrix(self, target: str, peers: list[str], weight_mode: str, asof=None, pillars=None):
        """
        Compute weights using the selected mode (including PCA).
        Priority: PCA modes > cached correlation matrix > legacy correlation methods
        """
        import numpy as np
        import pandas as pd

        target = (target or "").upper()
        peers = [p.upper() for p in (peers or [])]

        # 0) PCA modes: compute directly and return
        if isinstance(weight_mode, str) and weight_mode.startswith("pca"):
            try:
                # Use cached ATM matrix if available and from a correlation matrix plot
                if (hasattr(self, 'last_atm_df') and 
                    isinstance(self.last_atm_df, pd.DataFrame) and 
                    not self.last_atm_df.empty and
                    weight_mode.startswith("pca_atm")):
                    w = pca_weights_from_atm_matrix(self.last_atm_df, target, peers)
                else:
                    # Fallback to computing fresh ATM matrix
                    w = pca_weights(
                        get_smile_slice=self.get_smile_slice,
                        mode=weight_mode,
                        target=target,
                        peers=peers,
                        asof=asof,
                        pillars_days=pillars or [7,30,60,90,180,365],
                        # you can pass tenors/mny_bins if you've customized them
                        k=None, nonneg=True, standardize=True,
                    )
                if not w.empty and np.isfinite(w.to_numpy(float)).any():
                    return (w / w.sum()).astype(float)
            except Exception as e:
                print(f"PCA weights failed: {e}")
                pass  # fall through to corr/legacy

        # 0b) surface_grid mode: return the full grid of betas (dict of Series)
        if weight_mode == "surface_grid":
            try:
                from analysis.analysis_pipeline import compute_peer_weights
                grid_betas = compute_peer_weights(
                    target=target,
                    peers=peers,
                    weight_mode="surface_grid",
                )
                return grid_betas  # dict: key=grid cell, value=Series of betas
            except Exception as e:
                print(f"surface_grid weights failed: {e}")
                return None

        # 1) Use cached Corr Matrix from the Corr Matrix plot (only if weight_mode unchanged)
        if (
            isinstance(self.last_corr_df, pd.DataFrame)
            and not self.last_corr_df.empty
            and self.last_corr_meta.get("weight_mode") == weight_mode
        ):
            try:
                w = corr_weights(self.last_corr_df, target, peers, clip_negative=True, power=1.0)
                if w is not None and not w.empty and np.isfinite(w.to_numpy(dtype=float)).any():
                    # Normalize and keep only peers that had a column in the matrix
                    w = w.dropna().astype(float)
                    w = w[w.index.isin(peers)]
                    if not w.empty and np.isfinite(w.sum()):
                        return (w / w.sum()).astype(float)
            except Exception:
                pass

        # 2) Fallback: compute weights directly using the requested mode
        try:
            from analysis.analysis_pipeline import compute_peer_weights

            w = compute_peer_weights(
                target=target,
                peers=peers,
                weight_mode=weight_mode,
                asof=asof,
                pillar_days=pillars or [7, 30, 60, 90, 180, 365],
            )
            if w is not None and not w.empty and np.isfinite(w.to_numpy(dtype=float)).any():
                w = w.dropna().astype(float)
                w = w[w.index.isin(peers)]
                if not w.empty and np.isfinite(w.sum()):
                    return (w / w.sum()).astype(float)
        except Exception as e:
            print(f"compute_peer_weights failed: {e}")
            return None

    def _plot_smile(self, ax, df, target, asof, model, T_days, ci,
                    overlay_synth, peers, weight_mode):
        """
        Draw target smile; if synthetic overlay is ON, draw a horizontal line at the
        corr-matrix synthetic ATM for the nearest tenor to T_days (no recomputation of
        correlations).
        """
        import numpy as np
        import pandas as pd
        from display.plotting.smile_plot import fit_and_plot_smile

        dfe = df.copy()
        S = float(dfe["S"].median())
        K = dfe["K"].to_numpy(float)
        IV = dfe["sigma"].to_numpy(float)
        T_used = float(dfe["T"].median())

        info = fit_and_plot_smile(
            ax, S=S, K=K, T=T_used, iv=IV,
            model=model, moneyness_grid=(0.7, 1.3, 121), ci_level=ci, show_points=True,
            enable_svi_toggles=(model == "svi")  # Only enable toggles for SVI model
        )
        title = f"{target}  {asof}  T≈{T_used:.3f}y  RMSE={info['rmse']:.4f}"

        # log the parameters so they can be viewed later
        try:
            expiry_dt = None
            if "expiry" in dfe.columns and not dfe["expiry"].empty:
                expiry_dt = dfe["expiry"].iloc[0]
            append_params(
                asof_date=asof,
                ticker=target,
                expiry=str(expiry_dt) if expiry_dt is not None else None,
                model=model,
                params=info.get("params", {}),
                meta={"rmse": info.get("rmse")}
            )
        except Exception:
            pass


        if overlay_synth and peers:
            try:
                w = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof,
                                                   pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None)

                # build target + peers surfaces, combine peers using matrix weights
                tickers = list({target, *peers})
                surfaces = build_surface_grids(tickers=tickers, tenors=None, mny_bins=None, use_atm_only=False, max_expiries=self._current_max_expiries)
                if target in surfaces and asof in surfaces[target]:
                    peer_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
                    synth_by_date = combine_surfaces(peer_surfaces, w.to_dict())
                    if asof in synth_by_date:
                        tgt_grid = surfaces[target][asof]   # index: mny labels, cols: tenor-days
                        syn_grid = synth_by_date[asof]

                        tgt_cols = _cols_to_days(tgt_grid.columns)
                        syn_cols = _cols_to_days(syn_grid.columns)
                        i_tgt = _nearest_tenor_idx(tgt_cols, T_days)
                        i_syn = _nearest_tenor_idx(syn_cols, T_days)
                        col_tgt = tgt_grid.columns[i_tgt]
                        col_syn = syn_grid.columns[i_syn]

                        # align moneyness bins
                        x_mny = _mny_from_index_labels(tgt_grid.index)
                        y_syn = syn_grid[col_syn].astype(float).to_numpy()
                        if not tgt_grid.index.equals(syn_grid.index):
                            common = tgt_grid.index.intersection(syn_grid.index)
                            if len(common) >= 3:
                                x_mny = _mny_from_index_labels(common)
                                y_syn = syn_grid.loc[common, col_syn].astype(float).to_numpy()

                        ax.plot(x_mny, y_syn, "--", linewidth=1.6, alpha=0.95,
                                label="Synthetic smile (corr-matrix)")
                        ax.legend(loc="best", fontsize=8)
            except Exception:
                pass

        ax.set_title(title)
    def _plot_synth_surface(self, ax, target, peers, asof, T_days, weight_mode):
        peers = [p for p in peers if p]
        if not peers:
            ax.text(0.5, 0.5, "Provide peers to build synthetic surface", ha="center", va="center")
            return

        # weights from live corr matrix (preferred) or fallback
        w = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None)

        try:
            # Build target and peer surfaces, then combine peers using correlation weights
            tickers = list({target, *peers})
            surfaces = build_surface_grids(tickers=tickers, use_atm_only=False, max_expiries=self._current_max_expiries)

            if target not in surfaces or asof not in surfaces[target]:
                ax.text(0.5, 0.5, "No target surface for date", ha="center", va="center")
                ax.set_title(f"Synthetic Surface - {target} vs peers")
                return

            peer_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
            synth_by_date = combine_surfaces(peer_surfaces, w.to_dict())
            if asof not in synth_by_date:
                ax.text(0.5, 0.5, "No synthetic surface for date", ha="center", va="center")
                ax.set_title(f"Synthetic Surface - {target} vs peers")
                return

            tgt_grid = surfaces[target][asof]
            syn_grid = synth_by_date[asof]
            tgt_cols_days = _cols_to_days(tgt_grid.columns)
            syn_cols_days = _cols_to_days(syn_grid.columns)
            i_tgt = _nearest_tenor_idx(tgt_cols_days, T_days)
            i_syn = _nearest_tenor_idx(syn_cols_days, T_days)
            col_tgt = tgt_grid.columns[i_tgt]
            col_syn = syn_grid.columns[i_syn]

            x_mny = _mny_from_index_labels(tgt_grid.index)
            y_tgt = tgt_grid[col_tgt].astype(float).to_numpy()
            y_syn = syn_grid[col_syn].astype(float).to_numpy()

            # align on common moneyness buckets if necessary
            if not tgt_grid.index.equals(syn_grid.index):
                common = tgt_grid.index.intersection(syn_grid.index)
                if len(common) >= 3:
                    x_mny = _mny_from_index_labels(common)
                    y_tgt = tgt_grid.loc[common, col_tgt].astype(float).to_numpy()
                    y_syn = syn_grid.loc[common, col_syn].astype(float).to_numpy()

            ax.plot(x_mny, y_tgt, "-", linewidth=1.8, label=f"{target} smile @ ~{int(T_days)}d")
            ax.plot(x_mny, y_syn, "--", linewidth=1.8, label="Synthetic (corr-matrix)")
            ax.set_xlabel("Moneyness (K/S)")
            ax.set_ylabel("Implied Vol")
            ax.set_title(f"Synthetic Construction vs {target} | asof={asof} | ~{int(T_days)}d")
            ax.legend(loc="best", fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, "Synthetic surface plotting failed", ha="center", va="center")
            ax.set_title(f"Synthetic Surface - {target} vs peers")

    def _render_smile_at_index(self):
        if not self._smile_ctx:
            return
        ax = self._smile_ctx["ax"]
        T_arr = self._smile_ctx["T_arr"]
        K_arr = self._smile_ctx["K_arr"]
        sigma_arr = self._smile_ctx["sigma_arr"]
        S_arr = self._smile_ctx["S_arr"]
        Ts = self._smile_ctx["Ts"]
        i = int(np.clip(self._smile_ctx["idx"], 0, len(Ts) - 1))
        self._smile_ctx["idx"] = i  # keep in range

        settings = self._smile_ctx["settings"]
        target = settings["target"]; asof = settings["asof"]
        model = settings["model"]; ci = settings["ci"]

        T_sel = float(Ts[i])
        # pick nearest slice to T_sel
        j = int(np.nanargmin(np.abs(T_arr - T_sel)))
        T0 = float(T_arr[j])
        mask = np.isclose(T_arr, T0)
        if not np.any(mask):
            tol = 1e-6
            mask = (T_arr >= T0 - tol) & (T_arr <= T0 + tol)
        if not np.any(mask):
            ax.clear()
            self._clear_correlation_colorbar(ax)  # Clear correlation colorbar for smile plots
            ax.set_title("No data")
            if self.canvas is not None:
                self.canvas.draw_idle()
            return

        ax.clear()
        self._clear_correlation_colorbar(ax)  # Clear correlation colorbar for smile plots
        S = float(np.nanmedian(S_arr[mask]))
        K = K_arr[mask]
        IV = sigma_arr[mask]

        info = fit_and_plot_smile(
            ax, S=S, K=K, T=T0, iv=IV,
            model=model, moneyness_grid=(0.7, 1.3, 121), ci_level=ci, show_points=True,
            label=f"{target} {model.upper()}",
            enable_svi_toggles=(model == "svi")  # Enable toggles for SVI model
        )

        try:
            expiry_dt = None
            expiry_arr = self._smile_ctx.get("expiry_arr")
            if expiry_arr is not None:
                try:
                    exp_sel = expiry_arr[mask]
                    if getattr(exp_sel, "size", 0) > 0:
                        expiry_dt = exp_sel[0]
                except Exception:
                    pass
            append_params(
                asof_date=asof,
                ticker=target,
                expiry=str(expiry_dt) if expiry_dt is not None else None,
                model=model,
                params=info.get("params", {}),
                meta={"rmse": info.get("rmse")}
            )
        except Exception:
            pass

        # overlay: synthetic smile (corr-matrix) at this T
        syn_surface = self._smile_ctx.get("syn_surface")
        tgt_surface = self._smile_ctx.get("tgt_surface")
        if syn_surface is not None:
            syn_cols_days = _cols_to_days(syn_surface.columns)
            jx = _nearest_tenor_idx(syn_cols_days, T0 * 365.25)
            col_syn = syn_surface.columns[jx]
            x_mny = _mny_from_index_labels(syn_surface.index)
            y_syn = syn_surface[col_syn].astype(float).to_numpy()
            if tgt_surface is not None and not tgt_surface.index.equals(syn_surface.index):
                common = tgt_surface.index.intersection(syn_surface.index)
                if len(common) >= 3:
                    x_mny = _mny_from_index_labels(common)
                    y_syn = syn_surface.loc[common, col_syn].astype(float).to_numpy()
            ax.plot(x_mny, y_syn, linestyle="--", linewidth=1.5, alpha=0.9,
                    label="Synthetic smile (corr-matrix)")

        peer_slices = self._smile_ctx.get("peer_slices") or {}
        if peer_slices:
            for p, d in peer_slices.items():
                T_p = d["T_arr"]
                K_p = d["K_arr"]
                sigma_p = d["sigma_arr"]
                S_p = d["S_arr"]
                if T_p.size == 0:
                    continue
                jp = int(np.nanargmin(np.abs(T_p - T0)))
                T0p = float(T_p[jp])
                maskp = np.isclose(T_p, T0p)
                if not np.any(maskp):
                    tol = 1e-6
                    maskp = (T_p >= T0p - tol) & (T_p <= T0p + tol)
                if not np.any(maskp):
                    continue
                Sp = float(np.nanmedian(S_p[maskp]))
                Kp = K_p[maskp]
                IVp = sigma_p[maskp]
                fit_and_plot_smile(
                    ax, S=Sp, K=Kp, T=T0p, iv=IVp,
                    model=model, moneyness_grid=(0.7, 1.3, 121), ci_level=0,
                    show_points=False, label=p, line_kwargs={"alpha":0.7}
                )

        ax.legend(loc="best", fontsize=8)

        days = int(round(T0 * 365.25))
        ax.set_title(
            f"{target}  {asof}  T≈{T0:.3f}y (~{days}d)  RMSE={info['rmse']:.4f}\n"
            f"(Use buttons or click: L=next, R=prev)"
        )
        if self.canvas is not None:
            self.canvas.draw_idle()

    def is_smile_active(self) -> bool:
        return (
            self._current_plot_type is not None
            and self._current_plot_type.startswith("Smile")
            and self._smile_ctx is not None
        )

    def next_expiry(self):
        if not self.is_smile_active():
            return
        Ts = self._smile_ctx["Ts"]
        self._smile_ctx["idx"] = min(self._smile_ctx["idx"] + 1, len(Ts) - 1)
        self._render_smile_at_index()

    def prev_expiry(self):
        if not self.is_smile_active():
            return
        Ts = self._smile_ctx["Ts"]
        self._smile_ctx["idx"] = max(self._smile_ctx["idx"] - 1, 0)
        self._render_smile_at_index()

    def _plot_term(self, ax, df, target, asof, x_units, ci,
                    overlay_synth, peers, weight_mode):
        """
        Plot target ATM term structure; optionally overlay corr-matrix synthetic ATM curve
        built from peers on the SAME date (using cached matrix weights when available).
        """
        import numpy as np
        from analysis.pillars import compute_atm_by_expiry
        from display.plotting.term_plot import plot_atm_term_structure

        # Bootstrap confidence intervals are expensive and block the GUI when
        # executed on the main thread.  Disable bootstrapping for interactive
        # plots so the term‑structure view remains responsive.
        atm_target = compute_atm_by_expiry(
            df,
            method="fit",
            model="auto",
            vega_weighted=True,
            n_boot=0,
            ci_level=ci,
        )
        plot_atm_term_structure(
            ax,
            atm_target,
            x_units=x_units,
            connect=True,
            smooth=True,
            show_ci=False,  # Disable CI to avoid DPI errors when bootstrapping is disabled
            ci_level=ci,
            generate_ci=False,  # no auto-bootstrap
        )
        title = f"{target}  {asof}  ATM Term Structure  (N={len(atm_target)})"

        if overlay_synth and peers:
            try:
                w = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None)
                synth_curve = self._corr_weighted_synth_atm_curve(
                    asof=asof, peers=peers, weights=w, atm_band=ATM_BAND, t_tolerance_days=10.0
                )
                if synth_curve is not None and not synth_curve.empty:
                    # Ensure synthetic curve has same expiries as raw curve for consistency
                    raw_T = atm_target["T"].to_numpy(float)
                    synth_T = synth_curve["T"].to_numpy(float)
                    
                    # Find common expiries within tolerance
                    # Use percentage-based tolerance: 10% of the shorter expiry, with minimum 2 days
                    tol_days = max(2.0 / 365.25, 0.1 * min(raw_T.min(), synth_T.min()))
                    common_T = []
                    for rt in raw_T:
                        for st in synth_T:
                            if abs(rt - st) <= tol_days:
                                common_T.append(rt)
                                break
                    
                    if len(common_T) > 0:
                        # Filter both curves to common expiries
                        common_T = np.array(common_T)
                        common_T.sort()
                        
                        # Filter raw curve to common expiries
                        raw_filtered = atm_target[atm_target["T"].apply(lambda x: any(abs(x - ct) <= tol_days for ct in common_T))]
                        
                        # Filter synthetic curve to common expiries
                        synth_filtered = synth_curve[synth_curve["T"].apply(lambda x: any(abs(x - ct) <= tol_days for ct in common_T))]
                        
                        # Debug info (can be removed in production)
                        print(f"Term plot: Found {len(common_T)} common expiries between raw ({len(atm_target)}) and synthetic ({len(synth_curve)}) curves")
                        print(f"  Tolerance: {tol_days*365.25:.1f} days")
                        print(f"  Raw filtered: {len(raw_filtered)} points")
                        print(f"  Synthetic filtered: {len(synth_filtered)} points")
                        
                        # Safety check: ensure we have data after filtering
                        if len(raw_filtered) == 0 or len(synth_filtered) == 0:
                            print("Warning: Filtering resulted in empty curves, falling back to original")
                            # Fallback to original synthetic curve
                            x = synth_curve["T"].to_numpy(float)
                            if x_units == "days":
                                x = x * 365.25
                            y = synth_curve["atm_vol"].to_numpy(float)
                            order = np.argsort(x)
                            ax.plot(x[order], y[order], linestyle="--", linewidth=1.6, alpha=0.9,
                                    label="Synthetic ATM (corr-matrix)")
                            ax.scatter(x, y, s=18, alpha=0.8)
                            ax.legend(loc="best", fontsize=8)
                        else:
                            # Re-plot raw curve with filtered data to ensure consistency
                            ax.clear()
                            plot_atm_term_structure(
                                ax,
                                raw_filtered,
                                x_units=x_units,
                                connect=True,
                                smooth=True,
                                show_ci=False,  # Disable CI to avoid DPI errors
                                ci_level=ci,
                                generate_ci=False,
                            )
                            
                            # Plot synthetic curve with same expiries
                            x = synth_filtered["T"].to_numpy(float)
                            if x_units == "days":
                                x = x * 365.25
                            y = synth_filtered["atm_vol"].to_numpy(float)
                            order = np.argsort(x)
                            ax.plot(x[order], y[order], linestyle="--", linewidth=1.6, alpha=0.9,
                                    label="Synthetic ATM (corr-matrix)")
                            ax.scatter(x, y, s=18, alpha=0.8)
                            
                            # Add confidence bands for synthetic curve if enough points
                            # Disabled to avoid DPI errors in GUI
                            # if len(x) >= 3:
                            #     try:
                            #         from display.plotting.term_plot import generate_term_structure_confidence_bands
                            #         T_orig = synth_filtered["T"].to_numpy(float)
                            #         T_grid, ci_lo, ci_hi = generate_term_structure_confidence_bands(
                            #             T=T_orig,
                            #             atm_vol=y,
                            #             level=ci,
                            #             n_boot=0,
                            #         )
                            #         if len(T_grid) > 0:
                            #             if x_units == "days":
                            #                 T_grid_plot = T_grid * 365.25
                            #             else:
                            #                 T_grid_plot = T_grid
                            #             ax.fill_between(T_grid_plot, ci_lo, ci_hi, alpha=0.1, 
                            #                            color='orange', label=f"Synthetic CI ({ci:.0%})")
                            #     except Exception:
                            #         pass
                            
                            # Update title to reflect filtered data
                            title = f"{target}  {asof}  ATM Term Structure  (N={len(raw_filtered)}) - Synthetic Overlay (N={len(synth_filtered)})"
                            ax.legend(loc="best", fontsize=8)
                    else:
                        # Fallback: plot original synthetic curve if no common expiries
                        x = synth_curve["T"].to_numpy(float)
                        if x_units == "days":
                            x = x * 365.25
                        y = synth_curve["atm_vol"].to_numpy(float)
                        order = np.argsort(x)
                        ax.plot(x[order], y[order], linestyle="--", linewidth=1.6, alpha=0.9,
                                label="Synthetic ATM (corr-matrix)")
                        ax.scatter(x, y, s=18, alpha=0.8)
                        ax.legend(loc="best", fontsize=8)
            except Exception:
                pass

        ax.set_title(title)

    def _plot_corr_matrix(
        self, ax, target, peers, asof, pillars, weight_mode,  # weight_mode unused here
        corr_method: str = "pearson", demean_rows: bool = False,
    ):
        tickers = [t for t in [target] + peers if t]
        if not tickers:
            ax.set_title("No tickers")
            return

        try:
            # Use new expiry-rank correlation logic without pillar detection
            atm_df, corr_df, weights = compute_and_plot_correlation(
                ax=ax,
                get_smile_slice=self.get_smile_slice,
                tickers=tickers,
                asof=asof,
                pillars_days=pillars,
                atm_band=DEFAULT_ATM_BAND,
                tol_days=7.0,
                min_pillars=3,
                corr_method=corr_method,
                demean_rows=demean_rows,
                show_values=True,
                target=target,
                peers=peers,
                auto_detect_pillars=False,  # Disable pillar auto-detection as requested
                min_tickers_per_pillar=3,
                min_pillars_per_ticker=2,
                max_expiries=self._current_max_expiries or 6,
                weight_mode=weight_mode,  # Pass weight_mode argument
            )
        except TypeError:
            # Fallback - also use new expiry-rank logic
            atm_df, corr_df = compute_and_plot_correlation(
                ax=ax,
                get_smile_slice=self.get_smile_slice,
                tickers=tickers,
                asof=asof,
                pillars_days=pillars,
                atm_band=DEFAULT_ATM_BAND,
                tol_days=7.0,
                min_pillars=3,
                corr_method=corr_method,
                demean_rows=demean_rows,
                show_values=True,
                auto_detect_pillars=False,  # Disable pillar auto-detection
                max_expiries=self._current_max_expiries or 6,
                weight_mode=weight_mode,  # Pass weight_mode argument
            )

        # cache for other plots (including PCA)
        self.last_corr_df = corr_df
        self.last_atm_df = atm_df  # Cache ATM matrix for PCA
        self.last_corr_meta = {
            "asof": asof,
            "tickers": list(tickers),
            "pillars": list(pillars),
            "corr_method": corr_method,
            "demean_rows": bool(demean_rows),
            "weight_mode": weight_mode,
        }

    def _corr_weighted_synth_atm_curve(
        self,
        asof: str,
        peers: list[str],
        weights: pd.Series | None,
        atm_band: float = 0.05,
        t_tolerance_days: float = 10.0
    ) -> pd.DataFrame:
        

        if weights is None or weights.empty:
            weights = pd.Series({p: 1.0 for p in peers}, dtype=float)
        weights = (weights / weights.sum()).astype(float)
        peers = [p for p in weights.index if p in peers]

        curves = {}
        for p in peers:
            c = atm_curve_for_ticker_on_date(
                self.get_smile_slice, p, asof, 
                atm_band=atm_band,
                method="median",        # <<--- IMPORTANT: fast & robust
                model="auto",
                vega_weighted=False
            )
            if not c.empty:
                curves[p] = c
        if not curves:
            return pd.DataFrame(columns=["T", "atm_vol"])

        grid_days = np.unique(
            np.concatenate([(c["T"].to_numpy() * 365.25) for c in curves.values()])
        )
        grid_days.sort()

        tol = float(t_tolerance_days)
        out_T, out_iv = [], []
        for td in grid_days:
            contrib, wts = [], []
            for p, c in curves.items():
                arr_days = c["T"].to_numpy() * 365.25
                if arr_days.size == 0:
                    continue
                j = int(np.argmin(np.abs(arr_days - td)))
                if np.abs(arr_days[j] - td) <= tol:
                    ivp = float(c["atm_vol"].iloc[j])
                    if np.isfinite(ivp):
                        wp = float(weights.get(p, 0.0))
                        if wp > 0:
                            contrib.append(ivp * wp)
                            wts.append(wp)
            if wts:
                out_T.append(td / 365.25)
                out_iv.append(sum(contrib) / sum(wts))

        if not out_T:
            return pd.DataFrame(columns=["T", "atm_vol"])
        return pd.DataFrame(
            {"T": np.array(out_T, float), "atm_vol": np.array(out_iv, float)}
        ).sort_values("T")

    # ---- animation control methods ----
    
    def has_animation_support(self, plot_type: str) -> bool:
        """Check if a plot type supports animation."""
        # Smile animations are disabled, only synthetic surface animations are supported
        return plot_type.startswith("Synthetic Surface")
    
    def is_animation_active(self) -> bool:
        """Check if an animation is currently active."""
        return self._animation is not None
    
    def start_animation(self) -> bool:
        """Start or resume animation. Returns True if animation was started."""
        if not self._animation:
            return False
        if self._animation_paused:
            self._animation.resume()
            self._animation_paused = False
        return True
    
    def pause_animation(self) -> bool:
        """Pause animation. Returns True if animation was paused."""
        if not self._animation:
            return False
        if not self._animation_paused:
            self._animation.pause()
            self._animation_paused = True
        return True
    
    def stop_animation(self):
        """Stop and cleanup current animation."""
        if self._animation:
            try:
                self._animation.pause()
                self._animation = None
                self._animation_paused = False
            except Exception:
                pass
    
    def set_animation_speed(self, interval_ms: int):
        """Set animation speed in milliseconds between frames."""
        self._animation_speed = max(50, min(2000, interval_ms))
        if self._animation:
            self._animation.event_source.interval = self._animation_speed
    
    def get_animation_speed(self) -> int:
        """Get current animation speed in milliseconds."""
        return self._animation_speed
    
    def plot_animated(self, ax: plt.Axes, settings: dict) -> bool:
        """
        Create animated version of plot if supported.
        Returns True if animation was created, False if not supported.
        """
        plot_type = settings["plot_type"]
        
        if not self.has_animation_support(plot_type):
            return False
            
        self.stop_animation()  # cleanup any existing animation
        
        try:
            if plot_type.startswith("Smile"):
                return self._create_animated_smile(ax, settings)
            elif plot_type.startswith("Synthetic Surface"):
                return self._create_animated_surface(ax, settings)
        except Exception as e:
            print(f"Failed to create animation: {e}")
            return False
        
        return False
    
    def _create_animated_smile(self, ax: plt.Axes, settings: dict) -> bool:
        """Create animated smile plot over time/expiries."""
        target = settings["target"]
        asof = settings["asof"]
        max_expiries = settings.get("max_expiries", 6)
        
        try:
            # First try to animate over different dates
            if self._try_animate_smile_over_dates(ax, settings):
                return True
                
            # Fall back to animating over expiries for the given date
            return self._try_animate_smile_over_expiries(ax, settings)
            
        except Exception as e:
            print(f"Error creating animated smile: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _try_animate_smile_over_dates(self, ax: plt.Axes, settings: dict) -> bool:
        """Try to create smile animation over multiple dates."""
        target = settings["target"]
        max_expiries = settings.get("max_expiries", 6)
        T_days = settings.get("T_days", 30)
        
        from analysis.analysis_pipeline import available_dates
        
        dates = available_dates(target)
        if len(dates) < 2:
            return False  # Need multiple dates for animation
            
        # Use up to 10 most recent dates for animation
        animation_dates = dates[-10:] if len(dates) > 10 else dates
        
        # Collect data for each date
        k_data, iv_data = [], []
        valid_dates = []
        
        for date in animation_dates:
            df = get_smile_slice(target, date, T_target_years=T_days/365.25, max_expiries=max_expiries)
            if df is None or df.empty:
                continue
                
            # Extract K and sigma for this date
            K_arr = pd.to_numeric(df["K"], errors="coerce").to_numpy(float)
            sigma_arr = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(float)
            S_arr = pd.to_numeric(df["S"], errors="coerce").to_numpy(float)
            
            # Convert to moneyness
            S = np.nanmedian(S_arr)
            if not np.isfinite(S) or S <= 0:
                continue
            k = K_arr / S
            
            # Filter valid data
            valid_mask = np.isfinite(k) & np.isfinite(sigma_arr)
            if not np.any(valid_mask):
                continue
                
            k_clean = k[valid_mask]
            iv_clean = sigma_arr[valid_mask]
            
            # Sort by moneyness
            sort_idx = np.argsort(k_clean)
            k_data.append(k_clean[sort_idx])
            iv_data.append(iv_clean[sort_idx])
            valid_dates.append(date)
        
        if len(valid_dates) < 2:
            return False
            
        return self._create_smile_animation(ax, k_data, iv_data, valid_dates, f"{target} Smile Over Time")
    
    def _try_animate_smile_over_expiries(self, ax: plt.Axes, settings: dict) -> bool:
        """Try to create smile animation over different expiries for single date."""
        target = settings["target"]
        asof = settings["asof"]
        max_expiries = settings.get("max_expiries", 6)
        
        # Load all expiries for this date
        df = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
        if df is None or df.empty:
            return False
            
        # Group by expiry
        T_arr = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
        K_arr = pd.to_numeric(df["K"], errors="coerce").to_numpy(float)
        sigma_arr = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(float)
        S_arr = pd.to_numeric(df["S"], errors="coerce").to_numpy(float)
        
        # Get unique expiries
        Ts = np.sort(np.unique(T_arr[np.isfinite(T_arr)]))
        if len(Ts) < 2:
            return False
            
        # Collect data for each expiry
        k_data, iv_data = [], []
        valid_expiries = []
        
        for T in Ts:
            mask = np.isclose(T_arr, T, atol=1e-6)
            if not np.any(mask):
                continue
                
            K_T = K_arr[mask]
            sigma_T = sigma_arr[mask] 
            S_T = S_arr[mask]
            
            # Convert to moneyness
            S = np.nanmedian(S_T)
            if not np.isfinite(S) or S <= 0:
                continue
            k = K_T / S
            
            # Filter valid data
            valid_mask = np.isfinite(k) & np.isfinite(sigma_T)
            if not np.any(valid_mask):
                continue
                
            k_clean = k[valid_mask]
            iv_clean = sigma_T[valid_mask]
            
            # Sort by moneyness
            sort_idx = np.argsort(k_clean)
            k_data.append(k_clean[sort_idx])
            iv_data.append(iv_clean[sort_idx])
            
            # Format expiry label
            days = int(round(T * 365.25))
            valid_expiries.append(f"T={T:.3f}y ({days}d)")
        
        if len(valid_expiries) < 2:
            return False
            
        return self._create_smile_animation(ax, k_data, iv_data, valid_expiries, f"{target} Smile Over Expiries - {asof}")
    
    def _create_smile_animation(self, ax: plt.Axes, k_data: list, iv_data: list, labels: list, base_title: str) -> bool:
        """Create the actual smile animation from prepared data."""
        # Create common moneyness grid
        all_k = np.concatenate(k_data)
        k_min, k_max = np.nanpercentile(all_k, [5, 95])
        k_grid = np.linspace(k_min, k_max, 50)
        
        # Interpolate all curves to common grid
        iv_grid_data = []
        for k_points, iv_points in zip(k_data, iv_data):
            if len(k_points) > 1:
                iv_interp = np.interp(k_grid, k_points, iv_points, left=np.nan, right=np.nan)
            else:
                iv_interp = np.full_like(k_grid, np.nan)
            iv_grid_data.append(iv_interp)
        
        iv_tk = np.array(iv_grid_data)
        
        # Clear the axes and set up the plot
        ax.clear()
        
        # Create the animated line
        line, = ax.plot(k_grid, iv_tk[0], label="Smile", lw=2, color='blue')
        ax.set_xlim(k_grid.min(), k_grid.max())
        iv_min, iv_max = np.nanpercentile(iv_tk, [1, 99])
        iv_range = iv_max - iv_min
        ax.set_ylim(iv_min - 0.1*iv_range, iv_max + 0.1*iv_range)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def update_frame(i):
            line.set_ydata(iv_tk[i])
            ax.set_title(f"{base_title} - {labels[i]}")
            return [line]
        
        # Create animation on the current figure
        fig = ax.figure
        self._animation = FuncAnimation(
            fig, update_frame, frames=len(labels),
            interval=self._animation_speed, blit=True, repeat=True
        )
        
        # Start with initial frame
        update_frame(0)
        
        return True
    
    def _create_animated_surface(self, ax: plt.Axes, settings: dict) -> bool:
        """Create animated surface plot over time."""
        target = settings["target"]
        asof = settings["asof"]
        peers = settings.get("peers", [])
        weight_mode = settings.get("weight_mode", "iv_atm")
        
        try:
            # Load surface data for animation
            from analysis.analysis_pipeline import available_dates
            from analysis.syntheticETFBuilder import build_surface_grids
            
            dates = available_dates(target)
            if len(dates) < 2:
                return False
                
            # Use up to 8 most recent dates for surface animation
            animation_dates = dates[-8:] if len(dates) > 8 else dates
            
            # Build surface grids for each date
            surfaces_data = []
            valid_dates = []
            
            for date in animation_dates:
                try:
                    # Build surface for this date
                    surfaces = build_surface_grids(
                        tickers=[target] + peers if peers else [target],
                        asof_dates=[date],
                        max_expiries=settings.get("max_expiries", 6)
                    )
                    
                    if target in surfaces and date in surfaces[target]:
                        surface = surfaces[target][date]
                        if not surface.empty:
                            surfaces_data.append(surface)
                            valid_dates.append(date)
                            
                except Exception:
                    continue
            
            if len(valid_dates) < 2:
                return False
                
            # Extract common grid parameters
            first_surface = surfaces_data[0]
            tau_days = first_surface.columns.values
            k_levels = first_surface.index.values
            
            # Convert to numpy arrays for animation
            tau = np.array([float(t) for t in tau_days])
            k = np.array([float(k_str.split('-')[0]) if '-' in str(k_str) else float(k_str) for k_str in k_levels])
            
            # Stack all surface data
            iv_tktau = []
            for surface in surfaces_data:
                # Align surface to common grid
                aligned_surface = surface.reindex(index=k_levels, columns=tau_days, fill_value=np.nan)
                iv_tktau.append(aligned_surface.values)
                
            iv_tktau = np.array(iv_tktau)
            
            # Clear axes and create surface plot
            ax.clear()
            
            # Create initial surface image
            vmin, vmax = np.nanpercentile(iv_tktau, [1, 99])
            im = ax.imshow(
                iv_tktau[0],
                origin="lower",
                aspect="auto", 
                extent=[tau.min(), tau.max(), k.min(), k.max()],
                vmin=vmin,
                vmax=vmax,
                animated=True
            )
            
            ax.set_xlabel("Time to Expiry (days)")
            ax.set_ylabel("Moneyness")
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label("Implied Volatility")
            
            def update_surface(i):
                im.set_array(iv_tktau[i])
                ax.set_title(f"{target} IV Surface Animation - {valid_dates[i]}")
                return [im]
            
            # Create animation
            fig = ax.figure
            self._animation = FuncAnimation(
                fig, update_surface, frames=len(valid_dates),
                interval=self._animation_speed, blit=True, repeat=True
            )
            
            # Show initial frame
            update_surface(0)
            
            return True
            
        except Exception as e:
            print(f"Error creating animated surface: {e}")
            import traceback
            traceback.print_exc()
            return False

