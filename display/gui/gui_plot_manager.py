# display/gui/gui_plot_manager.py
from __future__ import annotations
import numpy as np
import pandas as pd

from display.plotting.correlation_detail_plot import (
    compute_and_plot_correlation,   # draws the corr heatmap
    corr_weights_from_matrix,       # converts a column of the matrix to weights
)
from analysis.pca_builder import pca_weights, pca_weights_from_atm_matrix
from display.plotting.smile_plot import fit_and_plot_smile
from display.plotting.term_plot import plot_atm_term_structure

# surfaces & synth
from analysis.syntheticETFBuilder import build_surface_grids, combine_surfaces

# fallback only (used if no live corr matrix has been computed yet)
from analysis.correlation_builder import peer_weights_from_correlations


from analysis.analysis_pipeline import get_smile_slice, relative_value_atm_report_corrweighted
from analysis.pillars import compute_atm_by_expiry
from analysis.correlation_builder import peer_weights_from_correlations
import matplotlib.pyplot as plt

from analysis.pillars import atm_curve_for_ticker_on_date
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
        # --- click-through TTE state ---
        self.canvas = None
        self._cid_click = None
        self._current_plot_type = None
        self._smile_ctx = None   # dict storing chain + current index + overlay bits

    def attach_canvas(self, canvas):
        self.canvas = canvas
        if self._cid_click is not None:
            try:
                self.canvas.mpl_disconnect(self._cid_click)
            except Exception:
                pass
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_click)

    # ---- main entry ----
    def plot(self, ax: plt.Axes, settings: dict):
        plot_type  = settings["plot_type"]
        target     = settings["target"]
        asof       = settings["asof"]
        model      = settings["model"]
        T_days     = settings["T_days"]
        ci         = settings["ci"]
        x_units    = settings["x_units"]
        weight_mode= settings["weight_mode"]
        overlay    = settings["overlay"]
        peers      = settings["peers"]
        pillars    = settings["pillars"]

        # remember current plot type for the click handler
        self._current_plot_type = plot_type

        ax.clear()

        # --- Smile: click-through path (NO df_all needed here) ---
        if plot_type.startswith("Smile"):
            # load all expiries so we can click through them
            chain_df = get_smile_slice(target, asof, T_target_years=None)
            if chain_df is None or chain_df.empty:
                ax.set_title("No data"); return

            Ts = np.sort(np.unique(pd.to_numeric(chain_df["T"], errors="coerce").dropna().to_numpy(float)))
            if Ts.size == 0:
                ax.set_title("No expiries"); return
            target_T = float(T_days) / 365.25
            idx0 = int(np.argmin(np.abs(Ts - target_T)))

            weights = None
            synth_curve = None
            if overlay and peers:
                weights = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None)
                synth_curve = self._corr_weighted_synth_atm_curve(
                    asof=asof, peers=peers, weights=weights, atm_band=ATM_BAND, t_tolerance_days=10.0
                )

            self._smile_ctx = {
                "ax": ax,
                "chain_df": chain_df,
                "Ts": Ts,
                "idx": idx0,
                "settings": settings,
                "weights": weights,
                "synth_curve": synth_curve,
            }
            self._render_smile_at_index()
            return

        # --- Term: needs all expiries for this day ---
        elif plot_type.startswith("Term"):
            df_all = get_smile_slice(target, asof, T_target_years=None)
            if df_all is None or df_all.empty:
                ax.set_title("No data"); return
            self._plot_term(ax, df_all, target, asof, x_units, ci, overlay, peers, weight_mode)
            return

        # --- Corr Matrix: doesn't need df_all ---
        elif plot_type.startswith("Corr Matrix"):
            self._plot_corr_matrix(ax, target, peers, asof, pillars, weight_mode)
            return

        # --- Synthetic Surface: doesn't need df_all here ---
        elif plot_type.startswith("Synthetic Surface"):
            self._plot_synth_surface(ax, target, peers, asof, T_days, weight_mode)
            return

        else:
            ax.text(0.5, 0.5, f"Unknown plot: {plot_type}", ha="center", va="center")

    # ---- implementations ----
    def _on_click(self, event):
        if self._current_plot_type is None or not self._current_plot_type.startswith("Smile"):
            return
        if self._smile_ctx is None or event.inaxes is None:
            return
        if event.inaxes is not self._smile_ctx["ax"]:
            return

        if event.button == 1:   # left click -> next expiry
            self._smile_ctx["idx"] = min(self._smile_ctx["idx"] + 1, len(self._smile_ctx["Ts"]) - 1)
        elif event.button in (3, 2):  # right/middle -> previous expiry
            self._smile_ctx["idx"] = max(self._smile_ctx["idx"] - 1, 0)
        else:
            return

        self._render_smile_at_index()

    def _weights_from_ui_or_matrix(self, target: str, peers: list[str], weight_mode: str, asof=None, pillars=None) -> pd.Series:
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
                    w = pca_weights_from_atm_matrix(self.last_atm_df, target, peers, verbose=False)
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

        # 1) Use cached Corr Matrix from the Corr Matrix plot
        if isinstance(self.last_corr_df, pd.DataFrame) and not self.last_corr_df.empty:
            try:
                w = corr_weights_from_matrix(self.last_corr_df, target, peers, clip_negative=True, power=1.0)
                if w is not None and not w.empty and np.isfinite(w.to_numpy(dtype=float)).any():
                    # Normalize and keep only peers that had a column in the matrix
                    w = w.dropna().astype(float)
                    w = w[w.index.isin(peers)]
                    if not w.empty and np.isfinite(w.sum()):
                        return (w / w.sum()).astype(float)
            except Exception:
                pass

        # 2) Fallback: compute from correlation_builder with the UI's weight mode
        w = peer_weights_from_correlations(
            benchmark=target,
            peers=peers,
            mode=weight_mode,
            pillar_days=pillars or [7, 30, 60, 90, 180, 365],
            clip_negative=True,
            power=1.0,
        )
        if w is None or w.empty:
            # equal weights if we really have nothing
            return pd.Series(1.0 / max(len(peers), 1), index=peers, dtype=float)
        w = w.dropna().astype(float)
        w = w[w.index.isin(peers)]
        if w.empty:
            return pd.Series(1.0 / max(len(peers), 1), index=peers, dtype=float)
        return (w / w.sum()).astype(float)


    def _plot_smile(self, ax, df, target, asof, model, T_days, ci, overlay, peers, weight_mode):
        """
        Draw target smile; if overlay is ON, draw a horizontal line at the corr-matrix
        synthetic ATM for the nearest tenor to T_days (no recomputation of correlations).
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
            model=model, moneyness_grid=(0.7, 1.3, 121), ci_level=ci, show_points=True
        )
        title = f"{target}  {asof}  T≈{T_used:.3f}y  RMSE={info['rmse']:.4f}"


        # inside _plot_smile(...), replace the horizontal line section with:
        if overlay and peers:
            try:
                w = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None)

                # build target + peers surfaces, combine peers using matrix weights
                tickers = list({target, *peers})
                surfaces = build_surface_grids(tickers=tickers, tenors=None, mny_bins=None, use_atm_only=False)
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

        # Temporary: synthetic surface plotting disabled due to missing helper functions
        ax.text(0.5, 0.5, "Synthetic Surface plotting\ntemporarily disabled\n(missing helper functions)", 
                ha="center", va="center", fontsize=10)
        ax.set_title(f"Synthetic Surface - {target} vs peers")
        return

        # TODO: Re-implement synthetic surface plotting when helper functions are available
        """
        try:
            from analysis.syntheticETFBuilder import combine_surfaces, build_surface_grids, DEFAULT_TENORS, DEFAULT_MNY_BINS
            tickers = list({target, *peers})
            surfaces = build_surface_grids(tickers=tickers, tenors=DEFAULT_TENORS, mny_bins=DEFAULT_MNY_BINS, use_atm_only=False)

            if target not in surfaces or asof not in surfaces[target]:
                ax.text(0.5, 0.5, "No target surface for date", ha="center", va="center"); return

            peer_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
            synth_by_date = combine_surfaces(peer_surfaces, w.to_dict())
            if asof not in synth_by_date:
                ax.text(0.5, 0.5, "No synthetic surface for date", ha="center", va="center"); return

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
        """

    def _render_smile_at_index(self):
        if not self._smile_ctx: 
            return
        ax = self._smile_ctx["ax"]
        chain_df = self._smile_ctx["chain_df"]
        Ts = self._smile_ctx["Ts"]
        i = int(np.clip(self._smile_ctx["idx"], 0, len(Ts)-1))
        self._smile_ctx["idx"] = i  # keep in range

        settings = self._smile_ctx["settings"]
        target = settings["target"]; asof = settings["asof"]
        model = settings["model"]; ci = settings["ci"]

        T_sel = float(Ts[i])
        # pick nearest slice to T_sel
        g = chain_df.copy()
        j = int(np.nanargmin(np.abs(g["T"].to_numpy(float) - T_sel)))
        T0 = float(g["T"].iloc[j])
        slice_df = g[np.isclose(g["T"].to_numpy(float), T0)].copy()
        if slice_df.empty:
            # fallback: take a small window around nearest
            tol = 1e-6
            slice_df = g[(g["T"] >= T0 - tol) & (g["T"] <= T0 + tol)].copy()

        ax.clear()
        S = float(slice_df["S"].median())
        K = slice_df["K"].to_numpy(float)
        IV = slice_df["sigma"].to_numpy(float)

        info = fit_and_plot_smile(
            ax, S=S, K=K, T=T0, iv=IV,
            model=model, moneyness_grid=(0.7, 1.3, 121), ci_level=ci, show_points=True
        )

        # overlay: horizontal synthetic ATM (from cached curve) at this T
        synth_curve = self._smile_ctx.get("synth_curve")
        if synth_curve is not None and not synth_curve.empty:
            x_days = synth_curve["T"].to_numpy(float) * 365.25
            y = synth_curve["atm_vol"].to_numpy(float)
            jx = int(np.argmin(np.abs(x_days - T0 * 365.25)))
            iv_synth = float(y[jx])
            ax.axhline(iv_synth, linestyle="--", linewidth=1.5, alpha=0.9, label="Synthetic ATM (corr-matrix)")
            ax.legend(loc="best", fontsize=8)

        days = int(round(T0 * 365.25))
        ax.set_title(f"{target}  {asof}  T≈{T0:.3f}y (~{days}d)  RMSE={info['rmse']:.4f}\n"
                    f"(Click: L=next expiry, R=prev)")
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _plot_term(self, ax, df, target, asof, x_units, ci, overlay, peers, weight_mode):
        """
        Plot target ATM term structure; optionally overlay corr-matrix synthetic ATM curve
        built from peers on the SAME date (using cached matrix weights when available).
        """
        import numpy as np
        from analysis.pillars import compute_atm_by_expiry
        from display.plotting.term_plot import plot_atm_term_structure

        atm_target = compute_atm_by_expiry(df, method="fit", model="auto", vega_weighted=True, 
                                          n_boot=50, ci_level=ci)  # Enable bootstrap CI (fewer boots for speed)
        plot_atm_term_structure(ax, atm_target, x_units=x_units, connect=True, smooth=True, 
                               show_ci=True, ci_level=ci, generate_ci=True)
        title = f"{target}  {asof}  ATM Term Structure  (N={len(atm_target)})"

        if overlay and peers:
            try:
                w = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None)
                synth_curve = self._corr_weighted_synth_atm_curve(
                    asof=asof, peers=peers, weights=w, atm_band=ATM_BAND, t_tolerance_days=10.0
                )
                if synth_curve is not None and not synth_curve.empty:
                    x = synth_curve["T"].to_numpy(float)
                    if x_units == "days":
                        x = x * 365.25
                    y = synth_curve["atm_vol"].to_numpy(float)
                    order = np.argsort(x)
                    ax.plot(x[order], y[order], linestyle="--", linewidth=1.6, alpha=0.9,
                            label="Synthetic ATM (corr-matrix)")
                    ax.scatter(x, y, s=18, alpha=0.8)
                    
                    # Add confidence bands for synthetic curve if enough points
                    if len(x) >= 3:
                        try:
                            from display.plotting.term_plot import generate_term_structure_confidence_bands
                            T_orig = synth_curve["T"].to_numpy(float)  # Keep in years for CI calculation
                            T_grid, ci_lo, ci_hi = generate_term_structure_confidence_bands(
                                T=T_orig, atm_vol=y, level=ci, n_boot=50  # Fewer bootstrap for synthetic
                            )
                            if len(T_grid) > 0:
                                if x_units == "days":
                                    T_grid_plot = T_grid * 365.25
                                else:
                                    T_grid_plot = T_grid
                                ax.fill_between(T_grid_plot, ci_lo, ci_hi, alpha=0.1, 
                                               color='orange', label=f"Synthetic CI ({ci:.0%})")
                        except Exception:
                            pass  # CI generation failed, continue without
                    
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
            # if your compute_and_plot_correlation now returns (atm_df, corr_df, weights)
            atm_df, corr_df, _ = compute_and_plot_correlation(
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
                target=target,       # ok if ignored by the 2-return version
                peers=peers,
            )
        except TypeError:
            # older 2-return version
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

