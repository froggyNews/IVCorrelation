import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model_params_logger import load_model_params


class ModelParamsFrame(ttk.Frame):
    """Frame providing a simple interface for model parameter time-series."""

    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl, text="Ticker:").grid(row=0, column=0, sticky=tk.W)
        self.ent_ticker = ttk.Entry(ctrl, width=12)
        self.ent_ticker.grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Model:").grid(row=0, column=2, sticky=tk.W)
        self.cmb_model = ttk.Combobox(ctrl, values=["svi", "sabr", "tps"], width=8, state="readonly")
        self.cmb_model.set("svi")
        self.cmb_model.grid(row=0, column=3, padx=4)

        ttk.Label(ctrl, text="As of ≤:").grid(row=0, column=4, sticky=tk.W)
        self.ent_asof = ttk.Entry(ctrl, width=12)
        self.ent_asof.grid(row=0, column=5, padx=4)

        btn_plot = ttk.Button(ctrl, text="Plot", command=self._plot)
        btn_plot.grid(row=0, column=6, padx=4)

        self.fig = plt.Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot(self):
        ticker = self.ent_ticker.get().strip().upper()
        model = self.cmb_model.get().strip().lower()
        asof = self.ent_asof.get().strip()

        self.ax.clear()
        if not ticker:
            self.ax.text(0.5, 0.5, "Enter ticker", ha="center", va="center")
            self.canvas.draw()
            return

        df = load_model_params()
        df = df[(df["ticker"] == ticker) & (df["model"] == model)]
        if asof:
            try:
                cutoff = pd.to_datetime(asof)
                df = df[df["asof_date"] <= cutoff]
            except Exception:
                messagebox.showerror("Input error", "Invalid asof date")
                self.canvas.draw()
                return
        if df.empty:
            self.ax.text(0.5, 0.5, "No parameter data", ha="center", va="center")
            self.ax.set_title("Model Parameters")
            self.canvas.draw()
            return

        df = df.sort_values("asof_date")
        for param_name in df["param"].unique():
            sub = df[df["param"] == param_name]
            self.ax.plot(sub["asof_date"], sub["value"], label=param_name)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Parameter value")
        self.ax.set_title(f"{ticker} {model.upper()} parameter trends")
        self.ax.legend(loc="best", fontsize=8)
        self.ax.tick_params(axis="x", rotation=45)
        self.canvas.draw()


class ModelParamsApp(tk.Tk):
    """Standalone application wrapper for :class:`ModelParamsFrame`."""

    def __init__(self):
        super().__init__()
        self.title("Model Parameters Viewer")
        self.geometry("900x700")
        panel = ModelParamsFrame(self)
        panel.pack(fill=tk.BOTH, expand=True)


def launch_model_params(parent=None):
    """Launch the model parameters viewer window."""
    if parent is None:
        return ModelParamsApp()
    else:
        window = tk.Toplevel(parent)
        window.title("Model Parameters Viewer")
        window.geometry("900x700")
        panel = ModelParamsFrame(window)
        panel.pack(fill=tk.BOTH, expand=True)
        return window


if __name__ == "__main__":
    app = launch_model_params()
    app.mainloop()
