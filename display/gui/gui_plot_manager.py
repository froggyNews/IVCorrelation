# display/gui/gui_plot_manager.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk

# --- Ensure project root on path so local packages resolve first ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# GUI input panel
from display.gui.gui_input import InputPanel

# Plot helpers
from display.plotting.relative_weight_plot import (
    compute_and_plot_relative_weight,   # draws the heatmap
    _relative_weight_by_expiry_rank,
)
from display.plotting.smile_plot import (
    fit_and_plot_smile, 
    plot_smile_with_composite,
    _cols_to_days,
    _nearest_tenor_idx, 
    _mny_from_index_labels
)
from display.plotting.term_plot import (
    plot_composite_index_term_structure,
    plot_atm_term_structure as plot_atm_term_structure_df,
)
from display.plotting.weights_plot import plot_weights

# Surfaces & composite construction
from analysis.compositeIndexBuilder import build_surface_grids, combine_surfaces
from display.plotting.surface_viewer import (
    show_whole_surface,         # single surface (3D or heatmap)
    show_smile_overlay,         # 2D smile overlay at nearest tenor
)
from display.plotting.vol_structure_plots import (
    plot_atm_term_structure as plot_atm_term_structure_simple,    # ATM IV vs TTE
    plot_term_smile,            # IV vs Strike for fixed maturity
    plot_3d_vol_surface,        # 3D surface for single ticker
)
# Data/analysis utilities
from analysis.analysis_pipeline import get_smile_slice, prepare_smile_data, prepare_term_data
from analysis.model_params_logger import compute_or_load, WarmupWorker

from analysis.pillars import get_target_expiry_pillars, get_feasible_expiry_pillars
from volModel.sviFit import fit_svi_slice
from volModel.sabrFit import fit_sabr_slice
from volModel.polyFit import fit_tps_slice
from volModel.multi_model_cache import (
    get_cached_model_params,
    fit_all_models_cached,
    get_cached_confidence_bands,
)
from volModel.termFit import fit_term_structure, term_structure_iv
from volModel.term_structure_cache import get_cached_term_structure_data
from analysis.confidence_bands import (
    generate_term_structure_confidence_bands,
    svi_confidence_bands,
    sabr_confidence_bands,
    tps_confidence_bands,
)
from display.gui.gui_input import InputPanel
from display.gui.gui_plot_weights import weights_from_ui_or_matrix
DEFAULT_ATM_BAND = 0.05
DEFAULT_WEIGHT_METHOD = "corr"
DEFAULT_FEATURE_MODE = "iv_atm"

from display.gui.gui_plot_events import is_smile_active, next_expiry, prev_expiry
from display.gui.gui_plot_cache import get_target_pillars, get_surface_grids

    # -------------------- correlation matrix --------------------
# ---------------------------------------------------------------------------
# Plot Manager
# ---------------------------------------------------------------------------
class PlotManager:
    def __init__(self):
        # get_smile_slice is rebound per-plot to respect max_expiries
        self.get_smile_slice = get_smile_slice

        # caches
        self.last_relative_weight_df: pd.DataFrame | None = None
        self.last_atm_df: pd.DataFrame | None = None
        self.last_relative_weight_meta: dict = {}

        # ui state
        self._current_max_expiries = None
        self.canvas = None
        self._cid_click = None
        self._current_plot_type = None

        # smile click-through state (for fast re-render without re-query)
        self._smile_ctx = None  # dict storing arrays + current index + overlay bits

        # preserve last settings for weight computation context
        self.last_settings: dict = {}
        # store latest fit parameters for UI parameter tab
        self.last_fit_info: dict | None = None

        # cache for surface grids: key is (tickers tuple, max_expiries)
        self._surface_cache: dict[tuple[tuple[str, ...], int], dict] = {}

        # background cache warmer
        self._warm = WarmupWorker()

    # -------------------- canvas wiring --------------------
    def attach_canvas(self, canvas):
        self.canvas = canvas
        if self._cid_click is not None:
            try:
                self.canvas.mpl_disconnect(self._cid_click)
            except Exception:
                pass
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _clear_session(self):
        """Clear the current session."""
        self.ent_target.delete(0, tk.END)
        self.ent_peers.delete(0, tk.END)
        self.cmb_presets.set("")
        self.cmb_r_presets.set("")
        self.ent_r.delete(0, tk.END)
        self._sync_settings()
        self._refresh_presets()
        self._refresh_interest_rates()
        self._refresh_plot()

    def _validate_date_and_target(self, asof: str, target: str) -> tuple[bool, str]:
        """
        Validate that date and target are available for plotting.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not target or target.strip() == "":
            return False, "Please select a target ticker"
            
        if not asof or asof.strip() == "":
            return False, "Please select a valid date"
            
        # Check if data exists for this ticker/date combination
        try:
            df = self.get_smile_slice(target, asof)
            if df.empty:
                return False, f"No data available for {target} on {asof}"
            return True, ""
        except Exception as e:
            return False, f"Error checking data: {e}"
    
    def _clear_correlation_colorbar(self, ax: plt.Axes):
        """Remove any existing correlation colorbar to prevent it from
        appearing on non-correlation plots. Keeps axes geometry consistent
        with tests that assert position stability.
        """
        try:
            if hasattr(ax.figure, "_correlation_colorbar"):
                cbar = getattr(ax.figure, "_correlation_colorbar")
                try:
                    cbar.remove()
                except Exception:
                    pass
                delattr(ax.figure, "_correlation_colorbar")
                if hasattr(ax.figure, "_orig_position"):
                    ax.set_position(ax.figure._orig_position)
                if hasattr(ax.figure, "_orig_subplotpars"):
                    l, r, b, t = ax.figure._orig_subplotpars
                    ax.figure.subplots_adjust(left=l, right=r, bottom=b, top=t)
        except Exception:
            pass

    def invalidate_surface_cache(self):
        """Clear any cached surface grids."""
        self._surface_cache.clear()

    def refresh_data_connections(self):
        """Refresh database connections and clear caches after data ingestion."""
        try:
            # Clear surface cache
            self.invalidate_surface_cache()
            
            # Clear correlation/weight caches
            self.last_corr_df = None
            self.last_atm_df = None
            self.last_corr_meta = {}
            
            # Force close any cached database connections
            try:
                from data.db_utils import close_all_connections
                close_all_connections()
            except ImportError:
                # Function doesn't exist, just clear any connection caches
                pass
            
            print("✅ Data connections refreshed, caches cleared")
            
        except Exception as e:
            print(f"Warning: Error refreshing data connections: {e}")



    def plot(self, ax: plt.Axes, settings: dict):
        plot_type = settings["plot_type"]
        self.target = settings["target"]
        asof = settings["asof"]
        model = settings["model"]
        T_days = settings["T_days"]
        ci = settings["ci"]
        x_units = settings["x_units"]
        # Use default ATM band (UI control removed)
        atm_band = DEFAULT_ATM_BAND
        weight_method = settings.get("weight_method", DEFAULT_WEIGHT_METHOD)
        feature_mode = settings.get("feature_mode", DEFAULT_FEATURE_MODE)
        # backward compatibility: allow legacy weight_mode field
        legacy = settings.get("weight_mode")
        if legacy and not settings.get("weight_method"):
            if legacy == "oi":
                weight_method, feature_mode = "oi", DEFAULT_FEATURE_MODE
            elif "_" in legacy:
                weight_method, feature_mode = legacy.split("_", 1)
            else:
                weight_method, feature_mode = legacy, DEFAULT_FEATURE_MODE
        weight_mode = (
            "oi" if weight_method == "oi" else f"{weight_method}_{feature_mode}"
        )
        overlay_composite = settings.get("overlay_composite", True)
        overlay_peers = settings.get("overlay_peers", False)
        self.peers = settings["peers"]
        max_expiries = settings.get("max_expiries", 10)

        # invalidate surface cache if tickers or max_expiries changed
        prev = getattr(self, "last_settings", {}) or {}
        prev_tickers = set([prev.get("target", "")] + prev.get("peers", []))
        curr_tickers = set([self.target] + self.peers)
        if prev_tickers != curr_tickers or prev.get("max_expiries") != max_expiries:
            self.invalidate_surface_cache()

        # reset last-fit info before plotting
        self.last_fit_info = None
        self.tickers = [t for t in [self.target] + self.peers if t]
        # remember for other helpers
        self._current_plot_type = plot_type
        self._current_max_expiries = max_expiries
        self.last_settings = settings
        self.target_pillars= get_target_pillars(self, self.target, asof, max_expiries)
        self.weights = weights_from_ui_or_matrix(self,self.target, self.peers, weight_mode, asof=asof, pillars=self.target_pillars)
        # create a bounded get_smile_slice with current max_expiries
        def bounded_get_smile_slice(ticker, asof_date=None, T_target_years=None, call_put=None):
            return get_smile_slice(
                ticker,
                asof_date=asof_date,
                T_target_years=T_target_years,
                call_put=call_put,
                max_expiries=self._current_max_expiries,
            )

        self.get_smile_slice = bounded_get_smile_slice

        ax.clear()

        # --- Smile: click-through (preload all expiries for date) ---
        if plot_type.startswith("Smile"):
            self._clear_correlation_colorbar(ax)

            weights = None
            if self.peers:
                # Use feasible expiries that work across target and peers
                weights = self.weights

            payload = {
                "target": self.target,
                "asof": pd.to_datetime(asof).floor("min").isoformat(),
                "T_days": float(T_days),
                "overlay_composite": overlay_composite,
                "peers": sorted(self.peers),
                "weights": weights.to_dict() if weights is not None else None,
                "overlay_peers": overlay_peers,
                "max_expiries": max_expiries,
                "ci": float(ci),
            }

            def _builder():
                return prepare_smile_data(
                    target=self.target,
                    asof=asof,
                    T_days=T_days,
                    model=model,
                    ci=ci,
                    overlay_composite=overlay_composite,
                    peers=self.peers,
                    weights=weights.to_dict() if weights is not None else None,
                    overlay_peers=overlay_peers,
                    max_expiries=max_expiries,
                )

            if hasattr(self, "_warm"):
                try:
                    self._warm.enqueue("smile", payload, _builder)
                except Exception:
                    pass

            data = compute_or_load("smile", payload, _builder)

            if not data:
                ax.set_title("No data")
                return

            fit_map = data.get("fit_by_expiry", {})
            self._smile_ctx = {
                "ax": ax,
                "T_arr": data["T_arr"],
                "K_arr": data["K_arr"],
                "sigma_arr": data["sigma_arr"],
                "S_arr": data["S_arr"],
                "Ts": data["Ts"],
                "idx": data["idx0"],
                "settings": settings,
                "weights": weights,
                "tgt_surface": data.get("tgt_surface"),
                "syn_surface": data.get("syn_surface"),
                "composite_slices": data.get("composite_slices", {}),
                "expiry_arr": data.get("expiry_arr"),
                "fit_by_expiry": fit_map,
            }
            self.last_fit_info = {
                "ticker": self.target,
                "asof": asof,
                "fit_by_expiry": fit_map,
            }
            self._render_smile_at_index()
            return

        # --- Term: needs all expiries for this day ---
        elif plot_type.startswith("Term"):
            self._clear_correlation_colorbar(ax)

            weights = None
            if self.peers:
                weights = self.weights

            payload = {
                "target": self.target,
                "asof": pd.to_datetime(asof).floor("min").isoformat(),
                "ci": ci,
                "overlay_composite": overlay_composite,
                "peers": sorted(self.peers),
                "weights": weights.to_dict() if weights is not None else None,
                "atm_band": atm_band,
                "max_expiries": max_expiries,
                "weight_mode": weight_mode,
            }

            def _builder():
                return prepare_term_data(
                    target=self.target,
                    asof=asof,
                    ci=ci,
                    overlay_composite=overlay_composite,
                    peers=self.peers,
                    weights=weights.to_dict() if weights is not None else None,
                    atm_band=atm_band,
                    max_expiries=max_expiries,
                    get_slice=self.get_smile_slice,
                )

            if hasattr(self, "_warm"):
                try:
                    self._warm.enqueue("term", payload, _builder)
                except Exception:
                    pass

            data = compute_or_load("term", payload, _builder)

            atm_curve = data.get("atm_curve") if data else None
            if atm_curve is None or atm_curve.empty:
                ax.set_title("No data")
                return

            # Prepare data for parameter summary tab
            try:
                fit_map: dict = {}
                for _, row in atm_curve.iterrows():
                    T_val = float(row.get("T", np.nan))
                    if not np.isfinite(T_val):
                        continue
                    entry: dict = {}
                    exp = row.get("expiry")
                    if pd.notna(exp):
                        entry["expiry"] = str(exp)
                    sens = {
                        k: row[k]
                        for k in ("atm_vol", "skew", "curv")
                        if k in row and pd.notna(row[k])
                    }
                    if sens:
                        entry["sens"] = sens
                    if entry:
                        fit_map[T_val] = entry
                self.last_fit_info = {
                    "ticker": self.target,
                    "asof": asof,
                    "fit_by_expiry": fit_map,
                }
            except Exception:
                self.last_fit_info = None

            self._plot_term(
                ax,
                data,
                self.target,
                asof,
                x_units,
                ci,
            )
            return

        # --- Relative Weight Matrix ---
        elif plot_type.startswith("Relative Weight Matrix"):
            # Use feasible expiries that work across target and peers for relative-weight matrix
            self._plot_relative_weight_matrix(ax, self.target, self.peers, asof, self.target_pillars, weight_mode, atm_band)
            return

        # Removed: Target vs Composite (no longer available in UI)

        # --- Index Weights only ---
        elif plot_type.startswith("Index Weights"):
            self._clear_correlation_colorbar(ax)
            plot_weights(ax, self.weights)
            return

        # --- ATM Term Structure ---
        elif plot_type.startswith("ATM Term Structure"):
            self._clear_correlation_colorbar(ax)
            
            # Validate date and target
            is_valid, error_msg = self._validate_date_and_target(asof, self.target)
            if not is_valid:
                ax.text(0.5, 0.5, error_msg, ha="center", va="center")
                ax.set_title("ATM Term Structure - Validation Error")
                return
                
            atm_info = plot_atm_term_structure_simple(ax, self.target, asof)
            return

        # --- Term Smile ---
        elif plot_type.startswith("Term Smile"):
            self._clear_correlation_colorbar(ax)
            
            # Validate date and target
            is_valid, error_msg = self._validate_date_and_target(asof, self.target)
            if not is_valid:
                ax.text(0.5, 0.5, error_msg, ha="center", va="center")
                ax.set_title("Term Smile - Validation Error")
                return
                
            smile_info = plot_term_smile(ax, self.target, asof, T_days)
            return

        # --- 3D Vol Surface ---
        elif plot_type.startswith("3D Vol Surface"):
            self._clear_correlation_colorbar(ax)
            
            # Validate date and target
            is_valid, error_msg = self._validate_date_and_target(asof, self.target)
            if not is_valid:
                ax.text(0.5, 0.5, error_msg, ha="center", va="center")
                ax.set_title("3D Vol Surface - Validation Error")
                return
                
            # Pop separate window for 3D surface
            try:
                print(f"Creating 3D surface for {self.target} on {asof}")
                fig = plot_3d_vol_surface(
                    self.target, 
                    asof, 
                    mode="3d",
                    max_expiries=self._current_max_expiries or 12
                )
                if fig:
                    print(f"3D surface created successfully: {type(fig)}")
                    ax.set_title("3D Surface opened (separate window)")
                else:
                    print("3D surface creation returned None")
                    ax.text(0.5, 0.5, f"No surface data available for {self.target} on {asof}", ha="center", va="center")
                    ax.set_title("3D Vol Surface - No Data")
            except Exception as e:
                print(f"Error creating 3D surface: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                ax.set_title("3D Vol Surface - Error")
            return


        else:
            ax.text(0.5, 0.5, f"Unknown plot: {plot_type}", ha="center", va="center")

    # -------------------- event handlers --------------------
    def _on_click(self, event):
        if not is_smile_active(self) or event.inaxes is None:
            return
        if event.inaxes is not self._smile_ctx["ax"]:
            return

        if event.button == 1:
            next_expiry(self)
        elif event.button in (3, 2):
            prev_expiry(self)
    

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


    # -------------------- specific plotters --------------------
    def _plot_smile(self, ax, df, target, asof, model, T_days, ci, overlay_composite, peers, weight_mode):
        # Prepare surfaces and weights for composite overlay if needed
        surfaces = None
        weights = None

        if overlay_composite and peers:
            try:
                # Use target ticker's actual expiries for weight computation
                w = self.weights
                tickers = list({self.target, *self.peers})
                surfaces = build_surface_grids(
                    tickers=tickers, use_atm_only=False, max_expiries=self._current_max_expiries
                )
                weights = {p: w.get(p, 0.0) for p in self.peers}
            except Exception:
                pass

        # Use the refactored smile plotting function
        info, last_fit_info = plot_smile_with_composite(
            ax=ax,
            df=df,
            target=target,
            asof=asof,
            model=model,
            T_days=T_days,
            ci=ci,
            overlay_composite=overlay_composite,
            surfaces=surfaces,
            weights=weights
        )
        
        # Store fit info for later access
        self.last_fit_info = last_fit_info

    
    # -------------------- smile click-through renderer --------------------
    

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
        self._smile_ctx["idx"] = i

        settings = self._smile_ctx["settings"]
        target = settings["target"]
        asof = settings["asof"]
        model = settings["model"]
        ci = settings["ci"]

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
            self._clear_correlation_colorbar(ax)
            ax.set_title("No data")
            if self.canvas is not None:
                self.canvas.draw_idle()
            return

        ax.clear()
        self._clear_correlation_colorbar(ax)
        S = float(np.nanmedian(S_arr[mask]))
        K = K_arr[mask]
        IV = sigma_arr[mask]

        fit_map = self._smile_ctx.get("fit_by_expiry", {})
        pre = fit_map.get(T0)
        # Ensure we have a complete set of model params cached for this expiry
        if isinstance(pre, dict) and any(k not in pre for k in ("svi", "sabr", "tps")):
            try:
                all_params = fit_all_models_cached(S, K, T0, IV, use_cache=True)
                pre.update(all_params)
                fit_map[T0] = pre
            except Exception:
                pass
        pre_params = pre.get(model) if isinstance(pre, dict) else None
        if pre_params is None:
            # Use cached multi-model fitting
            try:
                all_params = fit_all_models_cached(S, K, T0, IV, use_cache=True)
                pre_params = all_params.get(model, {})
                # Store all models in cache for future use
                if isinstance(pre, dict):
                    pre.update(all_params)
                    fit_map[T0] = pre
                elif pre is None:
                    fit_map[T0] = all_params
            except Exception:
                # Fallback to individual model fitting
                if model == "svi":
                    pre_params = fit_svi_slice(S, K, T0, IV)
                elif model == "sabr":
                    pre_params = fit_sabr_slice(S, K, T0, IV)
                else:
                    pre_params = fit_tps_slice(S, K, T0, IV)
                if isinstance(pre, dict):
                    pre[model] = pre_params
                    fit_map[T0] = pre
        bands = None
        bands_map = pre.get("bands") if isinstance(pre, dict) else None
        if ci and ci > 0:
            if isinstance(bands_map, dict) and model in bands_map:
                bands = bands_map[model]
            else:
                m_grid = np.linspace(0.7, 1.3, 121)
                K_grid = m_grid * S
                # Try cached confidence bands first
                try:
                    bands = get_cached_confidence_bands(
                        S, K, T0, IV, K_grid, model, level=float(ci), use_cache=True
                    )
                except Exception:
                    # Fallback to direct computation
                    try:
                        if model == "svi":
                            bands = svi_confidence_bands(S, K, T0, IV, K_grid, level=float(ci))
                        elif model == "sabr":
                            bands = sabr_confidence_bands(S, K, T0, IV, K_grid, level=float(ci))
                        else:
                            bands = tps_confidence_bands(S, K, T0, IV, K_grid, level=float(ci))
                    except Exception:
                        bands = None

                # Store computed bands in local cache
                if bands is not None and isinstance(pre, dict):
                    if not isinstance(bands_map, dict):
                        bands_map = {}
                    bands_map[model] = bands
                    pre["bands"] = bands_map
                    fit_map[T0] = pre
        info = fit_and_plot_smile(
            ax,
            S=S,
            K=K,
            T=T0,
            iv=IV,
            model=model,
            params=pre_params,
            bands=bands,
            moneyness_grid=(0.7, 1.3, 121),
            show_points=True,
            label=f"{target} {model.upper()}",
        )

        if fit_map:
            self.last_fit_info = {
                "ticker": target,
                "asof": asof,
                "fit_by_expiry": fit_map,
            }

        # overlay: composite smile at this T
        syn_surface = self._smile_ctx.get("syn_surface")
        tgt_surface = self._smile_ctx.get("tgt_surface")
        if settings.get("overlay_composite"):
            if syn_surface is None or tgt_surface is None:
                try:
                    weights = self._smile_ctx.get("weights")
                    surfaces = get_surface_grids(self, self.tickers, self._current_max_expiries)
                    if tgt_surface is None and target in surfaces and asof in surfaces[target]:
                        tgt_surface = surfaces[target][asof]
                        self._smile_ctx["tgt_surface"] = tgt_surface
                    if weights is not None:
                        composite_surfaces = {p: surfaces[p] for p in (settings.get("peers") or []) if p in surfaces}
                        composite_by_date = combine_surfaces(composite_surfaces, weights.to_dict())
                        syn_surface = composite_by_date.get(asof)
                        self._smile_ctx["syn_surface"] = syn_surface
                except Exception:
                    syn_surface = None
 

        composite_slices = self._smile_ctx.get("composite_slices") or {}
        if settings.get("overlay_peers") and composite_slices:
            for p, d in composite_slices.items():
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
                # Use cached model parameters for peer overlays
                try:
                    p_params = get_cached_model_params(Sp, Kp, T0p, IVp, model)
                except Exception:
                    # Fallback to individual model fitting
                    if model == "svi":
                        p_params = fit_svi_slice(Sp, Kp, T0p, IVp)
                    elif model == "sabr":
                        p_params = fit_sabr_slice(Sp, Kp, T0p, IVp)
                    else:
                        p_params = fit_tps_slice(Sp, Kp, T0p, IVp)
                fit_and_plot_smile(
                    ax,
                    S=Sp,
                    K=Kp,
                    T=T0p,
                    iv=IVp,
                    model=model,
                    params=p_params,
                    moneyness_grid=(0.7, 1.3, 121),
                    show_points=False,
                    label=p,
                    line_kwargs={"alpha": 0.7},
                )

        # Add legend only if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best", fontsize=8)
        days = int(round(T0 * 365.25))
        ax.set_title(f"{target}  {asof}  T≈{T0:.3f}y (~{days}d)  RMSE={info['rmse']:.4f}\n(Use buttons or click: L=next, R=prev)")

        # Ensure smile plot axis labels are set correctly (fix for label pollution from term plots)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Vol")

        # Ensure canvas and figure are valid before drawing
        if self.canvas is not None and ax.figure is not None:
            self.canvas.draw_idle()

    # -------------------- smile state helpers --------------------

    def _plot_relative_weight_matrix(
        self,
        ax,
        target,
        peers,
        asof,
        pillars,
        weight_mode,  # passed through to compute_and_plot_relative_weight
        atm_band,
    ):
        """Plot relative weight correlation matrix with improved error reporting."""

        # UI no longer exposes weight power or clip-negative; use fixed defaults

        max_exp = self._current_max_expiries or 6

        payload = {
            "tickers": sorted(self.tickers),
            "asof": pd.to_datetime(asof).floor("min").isoformat(),
            "atm_band": atm_band,
            "max_expiries": max_exp,
        }

        def _builder():
            return _relative_weight_by_expiry_rank(
                get_slice=self.get_smile_slice,  # Use bounded slicer
                tickers=self.tickers,
                asof=asof,
                max_expiries=max_exp,
                atm_band=atm_band,
            )

        if hasattr(self, "_warm"):
            try:
                self._warm.enqueue("relative_weight", payload, _builder)
            except Exception:
                pass

        try:
            # For surface modes, show two variants side-by-side: restricted and wide
            if isinstance(weight_mode, str) and (
                weight_mode.endswith("surface") or weight_mode.endswith("surface_grid")
            ):
                fig = ax.figure
                # Clear any existing axes content
                fig.clf()
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)

                # Restricted: strict common date + strict intersection
                atm_df, weight_df, _ = compute_and_plot_relative_weight(
                    ax=ax1,
                    get_smile_slice=self.get_smile_slice,
                    tickers=self.tickers,
                    asof=asof,
                    atm_band=atm_band,
                    show_values=True,
                    clip_negative=True,
                    weight_power=1.0,
                    target=target,
                    peers=self.peers,
                    max_expiries=max_exp,
                    weight_mode=weight_mode,
                    surface_min_coverage=1.0,
                    surface_strict_common_date=True,
                    surface_mny_range=(0.9, 1.1),
                )
                ax1.set_title("Surface Corr (restricted)")
                # Annotate restriction summary

                # Wide: majority coverage, allow per‑ticker latest
                atm_df2, weight_df2, _ = compute_and_plot_relative_weight(
                    ax=ax2,
                    get_smile_slice=self.get_smile_slice,
                    tickers=self.tickers,
                    asof=asof,
                    atm_band=atm_band,
                    show_values=True,
                    clip_negative=True,
                    weight_power=1.0,
                    target=target,
                    peers=self.peers,
                    max_expiries=max_exp,
                    weight_mode=weight_mode,
                    surface_min_coverage=0.5,
                    surface_strict_common_date=False,
                    surface_mny_range=None,
                )
                ax2.set_title("Surface Corr (wide)")
                try:
                    ax2.text(0.01, 1.02, "per-ticker latest, min_coverage=0.5, mny=ALL",
                              transform=ax2.transAxes, fontsize=8, va="bottom")
                except Exception:
                    pass
                # For caching/inspection, store the wide variant
                weight_df = weight_df2
                atm_df = atm_df2
            else:
                atm_df, weight_df, _ = compute_and_plot_relative_weight(
                    ax=ax,
                    get_smile_slice=self.get_smile_slice,  # Use bounded slicer
                    tickers=self.tickers,
                    asof=asof,
                    atm_band=atm_band,
                    show_values=True,
                    clip_negative=True,
                    weight_power=1.0,
                    target=target,
                    peers=self.peers,
                    max_expiries=max_exp,
                    weight_mode=weight_mode,
                )

            # cache for other plots (keep both legacy and new names for compatibility)
            self.last_corr_df = weight_df
            self.last_atm_df = atm_df
            self.last_corr_meta = {
                "asof": asof,
                "tickers": list(self.tickers),
                "pillars": list(pillars or []),  # These are now dynamic target expiry pillars
                "weight_mode": weight_mode,
            }
            # Back-compat for weights_from_ui_or_matrix fallback
            self.last_relative_weight_df = weight_df
            self.last_relative_weight_meta = dict(self.last_corr_meta)

            # Better error reporting for empty results
            if atm_df is None or atm_df.empty:
                error_msg = f"Empty ATM panel for {self.tickers} on {asof}"
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: max_expiries={max_exp}, atm_band={atm_band}")
                print(f"DEBUG: Using bounded slicer: {type(self.get_smile_slice)}")
                ax.text(0.5, 0.5, error_msg, ha="center", va="center", fontsize=10)
                ax.set_title("Relative Weight Matrix - No ATM Data")
                return

            if weight_df is None or weight_df.empty:
                error_msg = f"Empty correlation matrix for {self.tickers} on {asof}"
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: ATM panel shape: {atm_df.shape if atm_df is not None else 'None'}")
                ax.text(0.5, 0.5, error_msg, ha="center", va="center", fontsize=10)
                ax.set_title("Relative Weight Matrix - No Correlation Data")
                return

        except Exception as e:
            error_msg = f"Error computing relative weights: {e}"
            print(f"DEBUG: {error_msg}")
            ax.text(0.5, 0.5, error_msg, ha="center", va="center", fontsize=10)
            ax.set_title("Relative Weight Matrix - Error")
            return

    # -------------------- term structure --------------------
    def _plot_term(self, ax, data, target, asof, x_units, ci):
        """Plot precomputed ATM term structure and optional composite overlay."""
        atm_curve = data.get("atm_curve")

        # Use cached term structure fitting for performance
        fit_x = fit_y = None
        if atm_curve is not None and not atm_curve.empty:
            term_data = get_cached_term_structure_data(atm_curve, fit_points=100, use_cache=True)
            fit_x = term_data.get("fit_x")
            fit_y = term_data.get("fit_y")

        plot_atm_term_structure_df(
            ax,
            atm_curve,
            fit_x=fit_x,
            fit_y=fit_y,
            show_ci=bool(
                ci
                and ci > 0
                and isinstance(atm_curve, pd.DataFrame)
                and {"atm_lo", "atm_hi"}.issubset(atm_curve.columns)
            ),
        )
        title = f"{target}  {asof}  ATM Term Structure  (N={len(atm_curve)})"
        composite_bands = data.get("composite_bands")
        if composite_bands is not None:
            bands = composite_bands
            if x_units != "days":
                # Convert x-axis from days to years for the bands
                bands = type(composite_bands)(
                    x=composite_bands.x / 365.25,
                    mean=composite_bands.mean,
                    lo=composite_bands.lo,
                    hi=composite_bands.hi,
                    level=composite_bands.level,
                )
            plot_composite_index_term_structure(ax, bands)
            # restore axis labels overridden by composite plot
            ax.set_xlabel("Time to Expiry (days)" if x_units == "days" else "Time to Expiry (years)")
            ax.set_ylabel("Implied Vol (ATM)")
            title += f" - composite Overlay (N={len(bands.x)})"
