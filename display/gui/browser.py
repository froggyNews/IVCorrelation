# display/gui/browser.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from pathlib import Path
import argparse

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analysis_pipeline import available_tickers, available_dates, ingest_and_process
from display.gui.gui_input import InputPanel
from display.gui.gui_plot_manager import PlotManager


class BrowserApp(tk.Tk):
    def __init__(self, *, overlay: bool = True, ci_percent: float = 68.0):
        super().__init__()
        self.title("Implied Volatility Browser")
        self.geometry("1200x820")
        self.minsize(800, 600)

        # Inputs
        self.inputs = InputPanel(self, overlay=overlay, ci_percent=ci_percent)
        self.inputs.bind_download(self._on_download)
        self.inputs.bind_plot(self._refresh_plot)
        self.inputs.bind_target_change(self._on_target_change)

        # Canvas
        self.fig = plt.Figure(figsize=(11.2, 6.6))
        self.ax = self.fig.add_subplot(1,1,1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_mgr = PlotManager()
        self.plot_mgr.attach_canvas(self.canvas)

        # Status bar for user feedback
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Default target suggestion
        tickers = self._load_tickers()
        if tickers and not self.inputs.get_target():
            self.inputs.ent_target.insert(0, tickers[0])
            self._on_target_change()

    # ---------- events ----------
    def _on_target_change(self, *_):
        t = self.inputs.get_target()
        if not t:
            return
        try:
            dates = available_dates(t)
        except Exception:
            dates = []
        self.inputs.set_dates(dates)

    def _on_download(self):
        target = self.inputs.get_target()
        peers  = self.inputs.get_peers()
        universe = [x for x in [target] + peers if x]
        if not universe:
            messagebox.showerror("No tickers", "Enter a target and/or peers first.")
            self.status.config(text="No tickers specified")
            return
        max_exp = self.inputs.get_max_exp()
        r, q    = self.inputs.get_rates()
        self.status.config(text="Downloading data...")
        try:
            inserted = ingest_and_process(universe, max_expiries=max_exp, r=r, q=q)
            messagebox.showinfo("Download complete", f"Ingested rows: {inserted}\nTickers: {', '.join(universe)}")
            self.status.config(text=f"Downloaded data for {', '.join(universe)}")
            self._on_target_change()
        except Exception as e:
            messagebox.showerror("Download error", str(e))
            self.status.config(text="Download failed")

    def _refresh_plot(self):
        settings = dict(
            plot_type  = self.inputs.get_plot_type(),
            target     = self.inputs.get_target(),
            asof       = self.inputs.get_asof(),
            model      = self.inputs.get_model(),
            T_days     = self.inputs.get_T_days(),
            ci         = self.inputs.get_ci(),
            x_units    = self.inputs.get_x_units(),
            weight_mode= self.inputs.get_weight_mode(),
            overlay    = self.inputs.get_overlay(),
            peers      = self.inputs.get_peers(),
            pillars    = self.inputs.get_pillars(),
        )
        if not settings["target"] or not settings["asof"]:
            self.status.config(text="Enter target and date to plot")
            return
        self.plot_mgr.plot(self.ax, settings)
        self.canvas.draw()
        self.status.config(text="Plot updated")

    # ---------- helpers ----------
    def _load_tickers(self):
        try:
            return available_tickers()
        except Exception:
            return []

def main():
    parser = argparse.ArgumentParser(description="Vol Browser")
    parser.add_argument("--overlay", action="store_true", help="Overlay synthetic curves")
    parser.add_argument("--ci", type=float, default=68.0,
                        help="Confidence interval percentage (e.g. 95 for 95%)")
    args = parser.parse_args()
    app = BrowserApp(overlay=args.overlay, ci_percent=args.ci)
    app.mainloop()

if __name__ == "__main__":
    main()
