import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .models import load_data
from .views import plot_surface


class VolatilitySurfaceApp(tk.Tk):
    """Simple Tkinter GUI wired up using MVC components."""

    def __init__(self):
        super().__init__()
        self.title("Volatility Surface Explorer")
        self.geometry("900x600")

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

        self.plot_frame = plot_frame

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


def main():
    app = VolatilitySurfaceApp()
    app.mainloop()

