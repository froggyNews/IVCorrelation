import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.spillover.vol_spillover import run_spillover


class SpilloverApp(tk.Tk):
    """Simple GUI to run spillover analysis and visualise results."""

    def __init__(self):
        super().__init__()
        self.title("IV Spillover Explorer")
        self.geometry("900x700")

        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl, text="Tickers:").grid(row=0, column=0, sticky=tk.W)
        self.ent_tickers = ttk.Entry(ctrl, width=40)
        self.ent_tickers.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(ctrl, text="Lookback:").grid(row=1, column=0, sticky=tk.W)
        self.ent_lookback = ttk.Entry(ctrl, width=5)
        self.ent_lookback.insert(0, "60")
        self.ent_lookback.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(ctrl, text="Threshold (%):").grid(row=0, column=2, sticky=tk.W)
        self.ent_threshold = ttk.Entry(ctrl, width=5)
        self.ent_threshold.insert(0, "10")
        self.ent_threshold.grid(row=0, column=3, sticky=tk.W)

        ttk.Label(ctrl, text="Peers K:").grid(row=1, column=2, sticky=tk.W)
        self.ent_topk = ttk.Entry(ctrl, width=5)
        self.ent_topk.insert(0, "3")
        self.ent_topk.grid(row=1, column=3, sticky=tk.W)

        ttk.Label(ctrl, text="Horizons:").grid(row=0, column=4, sticky=tk.W)
        self.ent_horizons = ttk.Entry(ctrl, width=10)
        self.ent_horizons.insert(0, "1,3,5")
        self.ent_horizons.grid(row=0, column=5, sticky=tk.W)

        self.var_raw = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Use raw IV", variable=self.var_raw).grid(row=1, column=4, columnspan=2, sticky=tk.W)

        btn = ttk.Button(ctrl, text="Run", command=self.run)
        btn.grid(row=0, column=6, rowspan=2, padx=4)

        # Event table
        self.tree = ttk.Treeview(self, columns=("date", "ticker", "chg"), show="headings", height=8)
        self.tree.heading("date", text="Date")
        self.tree.heading("ticker", text="Ticker")
        self.tree.heading("chg", text="Rel Change")
        self.tree.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Plot
        self.fig = plt.Figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(1,1,1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.results = None

    def run(self):
        tickers = [t.strip().upper() for t in self.ent_tickers.get().split(',') if t.strip()]
        if not tickers:
            messagebox.showerror("No tickers", "Enter at least one ticker")
            return
        try:
            lookback = int(self.ent_lookback.get())
            thr = float(self.ent_threshold.get()) / 100.0
            topk = int(self.ent_topk.get())
            horizons = [int(h) for h in self.ent_horizons.get().split(',') if h]
        except ValueError:
            messagebox.showerror("Input error", "Invalid numeric input")
            return
        iv_path = ROOT / "data" / "iv_daily.parquet"
        if not iv_path.exists():
            messagebox.showerror("Data missing", f"Cannot find {iv_path}")
            return
        self.results = run_spillover(
            str(iv_path),
            tickers=tickers,
            threshold=thr,
            lookback=lookback,
            top_k=topk,
            horizons=horizons,
            use_raw=self.var_raw.get(),
        )
        self._populate_events()
        self._plot_response()

    def _populate_events(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        events = self.results["events"].sort_values("date", ascending=False).head(20)
        for _, row in events.iterrows():
            self.tree.insert("", tk.END, values=(row["date"].date(), row["ticker"], f"{row['rel_change']:.2%}"))

    def _plot_response(self):
        self.ax.clear()
        summary = self.results["summary"]
        if summary.empty:
            self.canvas.draw()
            return
        # average peer response across tickers
        grp = summary.groupby("h")["median_resp"].mean()
        self.ax.plot(grp.index, grp.values, marker="o", label="Median resp")
        self.ax.set_xlabel("Horizon (days)")
        self.ax.set_ylabel("Peer IV change")
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    app = SpilloverApp()
    app.mainloop()
