import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ------------------------- Data Helpers -------------------------

def load_data(theme: str, ticker: str, month: str):
    """Load stock and ETF volatility surfaces from pickle files."""
    base = Path("data/vol_surfaces") / theme
    stock_path = base / f"{ticker}_{month}.pkl"
    med_path = base / f"ETF_MEDIAN_{month}.pkl"
    low_path = base / f"ETF_LOW_{month}.pkl"
    high_path = base / f"ETF_HIGH_{month}.pkl"

    def _load(path: Path):
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    stock = _load(stock_path)
    median = _load(med_path)
    low = _load(low_path)
    high = _load(high_path)
    return stock, median, low, high


def _atm_normalize(surface: dict) -> dict:
    K = np.asarray(surface["K"], dtype=float)
    IV = np.asarray(surface["IV"], dtype=float)
    idx_atm = np.argmin(np.abs(K - np.median(K)))
    atm = IV[idx_atm, :]
    IV = IV / atm
    return {"K": K, "T": surface["T"], "IV": IV}


def _to_moneyness(surface: dict) -> Tuple[dict, float]:
    K = np.asarray(surface["K"], dtype=float)
    s0 = np.median(K)
    return {"K": K / s0, "T": surface["T"], "IV": surface["IV"]}, s0


# ------------------------- Plotting -------------------------

def plot_surface(stock: dict, median: dict, low: dict, high: dict,
                 normalize_atm: bool, use_moneyness: bool,
                 highlight_outliers: bool) -> Tuple[plt.Figure, float]:
    """Return a matplotlib Figure with the volatility surface."""
    if use_moneyness:
        stock, _ = _to_moneyness(stock)
        median, _ = _to_moneyness(median)
        low, _ = _to_moneyness(low)
        high, _ = _to_moneyness(high)

    if normalize_atm:
        stock = _atm_normalize(stock)
        median = _atm_normalize(median)
        low = _atm_normalize(low)
        high = _atm_normalize(high)

    K = np.asarray(stock["K"], dtype=float)
    T = np.asarray(stock["T"], dtype=float)
    Z = np.asarray(stock["IV"], dtype=float)

    inside_mask = (Z >= low["IV"]) & (Z <= high["IV"])
    inside_pct = 100.0 * inside_mask.sum() / Z.size

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    X, Y = np.meshgrid(K, T, indexing="ij")
    c = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    fig.colorbar(c, ax=ax, label="IV")

    ax.contour(X, Y, median["IV"], colors="white", linewidths=2)
    ax.contour(X, Y, low["IV"], colors="green", linestyles="dashed")
    ax.contour(X, Y, high["IV"], colors="red", linestyles="dashed")

    if highlight_outliers:
        out_y, out_x = np.where(~inside_mask)
        if len(out_x):
            ax.scatter(K[out_x], T[out_y], color="magenta", s=10, label="Outlier")

    ax.set_xlabel("Moneyness" if use_moneyness else "Strike")
    ax.set_ylabel("Maturity")
    ax.set_title("Volatility Surface")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, inside_pct


# ------------------------- GUI Application -------------------------

class VolatilitySurfaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Volatility Surface Explorer")
        self.geometry("900x600")

        # Variables
        self.theme_var = tk.StringVar(value="Quantum")
        self.ticker_var = tk.StringVar(value="QUBT")
        self.month_var = tk.StringVar(value="2024-01")
        self.norm_var = tk.BooleanVar()
        self.mny_var = tk.BooleanVar()
        self.outlier_var = tk.BooleanVar()

        self.figure = None
        self.canvas = None

        self._build_ui()
        self.draw_plot()

    def _build_ui(self):
        control = ttk.Frame(self)
        control.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        ttk.Label(control, text="Theme").pack(anchor="w")
        ttk.OptionMenu(control, self.theme_var, self.theme_var.get(),
                       "Quantum", "Crypto", "Clean Energy").pack(anchor="w", fill="x")

        ttk.Label(control, text="Ticker").pack(anchor="w")
        ttk.OptionMenu(control, self.ticker_var, self.ticker_var.get(),
                       "QUBT", "IONQ", "QBTS").pack(anchor="w", fill="x")

        months = [f"2024-{i:02d}" for i in range(1, 13)]
        ttk.Label(control, text="Month").pack(anchor="w")
        ttk.OptionMenu(control, self.month_var, self.month_var.get(), *months).pack(anchor="w", fill="x")

        ttk.Checkbutton(control, text="Normalize by ATM", variable=self.norm_var,
                        command=self.draw_plot).pack(anchor="w")
        ttk.Checkbutton(control, text="Use moneyness", variable=self.mny_var,
                        command=self.draw_plot).pack(anchor="w")
        ttk.Checkbutton(control, text="Highlight outliers", variable=self.outlier_var,
                        command=self.draw_plot).pack(anchor="w")

        ttk.Button(control, text="Update", command=self.draw_plot).pack(anchor="w", pady=(5, 0))
        ttk.Button(control, text="Export PNG", command=self.export_png).pack(anchor="w", pady=(2, 0))
        ttk.Button(control, text="Play animation", command=self.play_animation).pack(anchor="w", pady=(2, 0))

        self.metric_label = ttk.Label(control, text="% inside band: --")
        self.metric_label.pack(anchor="w", pady=(10, 0))

        # Plot area
        self.plot_frame = plot_frame

    # ---------------- Plot and Data Logic -----------------
    def draw_plot(self):
        theme = self.theme_var.get()
        ticker = self.ticker_var.get()
        month = self.month_var.get()
        data = load_data(theme, ticker, month)
        if None in data:
            messagebox.showerror("Error", "Data not found for selection")
            return

        fig, pct = plot_surface(*data, self.norm_var.get(),
                                self.mny_var.get(), self.outlier_var.get())
        self.metric_label.config(text=f"% inside band: {pct:.1f}%")

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_png(self):
        if not self.figure:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png")])
        if filepath:
            self.figure.savefig(filepath)

    # ---------------- Animation -----------------
    def play_animation(self):
        self.anim_months = [f"2024-{i:02d}" for i in range(1, 13)]
        self.anim_index = 0
        self._animate_step()

    def _animate_step(self):
        if self.anim_index >= len(self.anim_months):
            return
        self.month_var.set(self.anim_months[self.anim_index])
        self.draw_plot()
        self.anim_index += 1
        self.after(1000, self._animate_step)


# ------------------------- Entry Point -------------------------

def main():
    app = VolatilitySurfaceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
