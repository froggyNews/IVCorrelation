# This file is based on the upstream IVCorrelation project but has been
# modified to improve GUI responsiveness. The changes revolve around
# running potentially long-running operations (database queries and
# ingestion) in background threads and then marshaling UI updates back
# to the Tkinter main thread via `after()`. These modifications help
# prevent the UI from freezing while data is downloaded or dates are
# fetched.

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from pathlib import Path
import argparse

# add near the very top of your entry script (e.g., browser.py)
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)  # raise instead of warn
warnings.filterwarnings("error", category=FutureWarning)   # optional

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analysis_pipeline import available_tickers, available_dates, ingest_and_process
from display.gui.gui_input import InputPanel
from display.gui.gui_plot_manager import PlotManager
from display.gui.spillover_gui import launch_spillover, SpilloverFrame
from display.gui.parameters_tab import ParametersTab


class BrowserApp(tk.Tk):
    def __init__(self, *, overlay_synth: bool = True, overlay_peers: bool = True,
                 ci_percent: float = 68.0):
        super().__init__()
        self.title("Implied Volatility Browser")
        self.geometry("1200x820")
        self.minsize(800, 600)

        # Notebook with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ---- Main exploration tab ----
        self.tab_browser = ttk.Frame(self.notebook)
        # Clarify purpose: this tab lets users explore IV surfaces
        self.notebook.add(self.tab_browser, text="Parameter Explorer")

        # Inputs
        self.inputs = InputPanel(self.tab_browser, overlay_synth=overlay_synth,
                                 overlay_peers=overlay_peers,
                                 ci_percent=ci_percent)
        # Bind events
        self.inputs.bind_download(self._on_download)
        self.inputs.bind_plot(self._refresh_plot)
        self.inputs.bind_target_change(self._on_target_change)
        self.inputs.bind_session_clear(self._on_session_clear)

        # Expiry navigation controls
        nav = ttk.Frame(self.tab_browser)
        nav.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        self.btn_prev = ttk.Button(nav, text="Prev Expiry", command=self._prev_expiry)
        self.btn_prev.pack(side=tk.LEFT, padx=4)
        self.btn_next = ttk.Button(nav, text="Next Expiry", command=self._next_expiry)
        self.btn_next.pack(side=tk.LEFT, padx=4)

        ttk.Separator(nav, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.btn_spill = ttk.Button(nav, text="Spillover", command=self._open_spillover)
        self.btn_spill.pack(side=tk.LEFT, padx=4)

        # Canvas
        self.fig = plt.Figure(figsize=(11.2, 6.6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_browser)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_mgr = PlotManager()
        self.plot_mgr.attach_canvas(self.canvas)

        # ---- Parameter summary tab ----
        self.tab_params = ParametersTab(self.notebook)
        self.notebook.add(self.tab_params, text="Parameter Summary")

        # ---- Spillover tab ----
        self.tab_spillover = SpilloverFrame(self.notebook)
        self.notebook.add(self.tab_spillover, text="Spillover")

        # Status bar for user feedback
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Default target suggestion
        tickers = self._load_tickers()
        if tickers and not self.inputs.get_target():
            self.inputs.ent_target.insert(0, tickers[0])
            # Perform initial date load asynchronously
            self._on_target_change()

        self._update_nav_buttons()

        self.spill_win = None

    # ---------- events ----------
    def _on_target_change(self, *_):
        """
        Handle changes to the target ticker. To avoid thread-affinity issues
        with SQLite, this method spawns a worker thread that opens a fresh
        database connection and queries the available dates for the current
        ticker. The results are marshalled back to the Tkinter main thread
        using ``after()`` to safely update the UI without blocking the event
        loop.
        """
        t = self.inputs.get_target()
        if not t:
            return

        # Indicate loading in status bar
        self.status.config(text="Loading available dates...")

        def worker():
            from data.db_utils import get_conn
            import pandas as pd
            dates: list[str] = []
            conn = None
            try:
                conn = get_conn()
                if t:
                    # Query available dates for a specific ticker
                    df = pd.read_sql_query(
                        "SELECT DISTINCT asof_date FROM options_quotes WHERE ticker = ? ORDER BY 1",
                        conn,
                        params=[t],
                    )
                else:
                    # Query all available dates
                    df = pd.read_sql_query(
                        "SELECT DISTINCT asof_date FROM options_quotes ORDER BY 1",
                        conn,
                    )
                dates = df["asof_date"].tolist()
            except Exception:
                dates = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            # Schedule UI update on main thread
            def update_ui():
                self.inputs.set_dates(dates)
                self.status.config(text="Ready")

            self.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _on_download(self):
        """
        Trigger ingestion of data. This can take a long time due to
        network/database operations. Run ingestion in a background thread
        and marshal UI updates to the main thread when complete. Also
        disable the download button while work is in progress to prevent
        multiple concurrent ingestions.
        """
        target = self.inputs.get_target()
        peers = self.inputs.get_peers()
        universe = [x for x in [target] + peers if x]
        if not universe:
            messagebox.showerror("No tickers", "Enter a target and/or peers first.")
            self.status.config(text="No tickers specified")
            return
        max_exp = self.inputs.get_max_exp()
        r, q = self.inputs.get_rates()

        # Provide immediate feedback and disable download button
        self.status.config(text="Downloading data...")
        self.inputs.btn_download.config(state=tk.DISABLED)

        def worker():
            try:
                inserted = ingest_and_process(universe, max_expiries=max_exp, r=r, q=q)
                # On success, schedule UI updates
                def done():
                    messagebox.showinfo("Download complete", f"Ingested rows: {inserted}\nTickers: {', '.join(universe)}")
                    self.status.config(text=f"Downloaded data for {', '.join(universe)}")
                    # Refresh available dates now that new data may be present
                    self._on_target_change()
                    self.inputs.btn_download.config(state=tk.NORMAL)
                self.after(0, done)
            except Exception as e:
                def handle():
                    messagebox.showerror("Download error", str(e))
                    self.status.config(text="Download failed")
                    self.inputs.btn_download.config(state=tk.NORMAL)
                self.after(0, handle)

        threading.Thread(target=worker, daemon=True).start()

    def _on_session_clear(self):
        """Handle session clear button click - clear all session state."""
        try:
            # Clear plot manager session state
            if hasattr(self, 'plot_mgr'):
                # Clear cached data in plot manager
                self.plot_mgr.last_atm_df = None
                # updated name after renaming correlation -> relative_weight
                if hasattr(self.plot_mgr, 'last_relative_weight_meta'):
                    self.plot_mgr.last_relative_weight_meta = {}
                self.plot_mgr.last_settings = {}
                self.plot_mgr.last_fit_info = None
                self.plot_mgr._smile_ctx = None
                self.plot_mgr.invalidate_surface_cache()

            # Clear the plot
            if hasattr(self, 'ax'):
                self.ax.clear()
                self.ax.set_title("Session Cleared")
                self.ax.text(0.5, 0.5, "Session cleared - ready for new analysis",
                             ha="center", va="center", transform=self.ax.transAxes)
                self.canvas.draw()

            # Update status
            if hasattr(self, 'status'):
                self.status.config(text="Session cleared - ready for new analysis")

            print("✅ All session state cleared!")

        except Exception as e:
            print(f"Error clearing session state: {e}")

    def _refresh_plot(self):
        settings = dict(
            plot_type=self.inputs.get_plot_type(),
            target=self.inputs.get_target(),
            asof=self.inputs.get_asof(),
            model=self.inputs.get_model(),
            T_days=self.inputs.get_T_days(),
            ci=self.inputs.get_ci(),
            x_units=self.inputs.get_x_units(),
            atm_band=self.inputs.get_atm_band(),
            weight_method=self.inputs.get_weight_method(),
            feature_mode=self.inputs.get_feature_mode(),
            overlay_synth=self.inputs.get_overlay_synth(),
            overlay_peers=self.inputs.get_overlay_peers(),
            peers=self.inputs.get_peers(),
            pillars=self.inputs.get_pillars(),
            max_expiries=self.inputs.get_max_exp(),
        )
        if not settings["target"] or not settings["asof"]:
            self.status.config(text="Enter target and date to plot")
            return

        self.status.config(text="Loading...")

        def worker():
            try:
                # Always static plot (animation removed)
                self.plot_mgr.plot(self.ax, settings)

                self.canvas.draw()
                self.after(0, self._update_nav_buttons)
                # Refresh parameter table with latest fit info
                self.after(0, lambda: self.tab_params.update(self.plot_mgr.last_fit_info))

            except Exception as e:
                def handle_err(exc: Exception):
                    messagebox.showerror("Plot error", str(exc))
                    self.status.config(text="Plot failed")
                    self._update_nav_buttons()
                self.after(0, lambda exc=e: handle_err(exc))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- helpers ----------
    def _prev_expiry(self):
        self.plot_mgr.prev_expiry()
        self.canvas.draw()

    def _next_expiry(self):
        self.plot_mgr.next_expiry()
        self.canvas.draw()

    def _update_nav_buttons(self):
        state = tk.NORMAL if self.plot_mgr.is_smile_active() else tk.DISABLED
        self.btn_prev.config(state=state)
        self.btn_next.config(state=state)

    def _open_spillover(self):
        """Open spillover analysis window."""
        if self.spill_win is None or not self.spill_win.winfo_exists():
            self.spill_win = launch_spillover(self)
        else:
            self.spill_win.lift()

    def _load_tickers(self):
        try:
            return available_tickers()
        except Exception:
            return []


def main():
    parser = argparse.ArgumentParser(description="Vol Browser")
    parser.add_argument("--overlay-synth", action="store_true", help="Overlay synthetic curves")
    parser.add_argument("--overlay-peers", action="store_true", help="Overlay peer curves")
    parser.add_argument("--ci", type=float, default=68.0,
                        help="Confidence interval percentage (e.g. 95 for 95%%)")
    args = parser.parse_args()
    app = BrowserApp(overlay_synth=args.overlay_synth,
                     overlay_peers=args.overlay_peers,
                     ci_percent=args.ci)
    app.mainloop()


if __name__ == "__main__":
    main()
