# display/gui/browser.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import threading
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
from display.gui.spillover_gui import launch_spillover, SpilloverFrame
from display.gui.model_params_gui import ModelParamsFrame



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

        # ---- Main browser tab ----
        self.tab_browser = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_browser, text="Browser")

        # Inputs
        self.inputs = InputPanel(self.tab_browser, overlay_synth=overlay_synth,
                                 overlay_peers=overlay_peers,
                                 ci_percent=ci_percent)
        self.inputs.bind_download(self._on_download)
        self.inputs.bind_plot(self._refresh_plot)
        self.inputs.bind_target_change(self._on_target_change)

        # Expiry navigation and animation controls
        nav = ttk.Frame(self.tab_browser); nav.pack(side=tk.TOP, fill=tk.X, pady=(0,4))

        # Expiry navigation (existing)
        self.btn_prev = ttk.Button(nav, text="Prev Expiry", command=self._prev_expiry)
        self.btn_prev.pack(side=tk.LEFT, padx=4)
        self.btn_next = ttk.Button(nav, text="Next Expiry", command=self._next_expiry)
        self.btn_next.pack(side=tk.LEFT, padx=4)

        # Animation controls (new)
        ttk.Separator(nav, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.var_animated = tk.BooleanVar(value=False)
        self.chk_animated = ttk.Checkbutton(nav, text="Animate", variable=self.var_animated,
                                           command=self._toggle_animation_mode)
        self.chk_animated.pack(side=tk.LEFT, padx=4)

        self.btn_play_pause = ttk.Button(nav, text="Play", command=self._toggle_animation)
        self.btn_play_pause.pack(side=tk.LEFT, padx=2)

        self.btn_stop = ttk.Button(nav, text="Stop", command=self._stop_animation)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        ttk.Label(nav, text="Speed:").pack(side=tk.LEFT, padx=(8,2))
        self.speed_var = tk.IntVar(value=500)  # Default speed
        self.speed_scale = ttk.Scale(nav, from_=100, to=2000, variable=self.speed_var,
                                    orient=tk.HORIZONTAL, length=100,
                                    command=self._on_speed_change)
        self.speed_scale.pack(side=tk.LEFT, padx=2)

        ttk.Separator(nav, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.btn_spill = ttk.Button(nav, text="Spillover", command=self._open_spillover)
        self.btn_spill.pack(side=tk.LEFT, padx=4)

        # Canvas
        self.fig = plt.Figure(figsize=(11.2, 6.6))
        self.ax = self.fig.add_subplot(1,1,1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_browser)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_mgr = PlotManager()
        self.plot_mgr.attach_canvas(self.canvas)

        # ---- Spillover tab ----
        self.tab_spillover = SpilloverFrame(self.notebook)
        self.notebook.add(self.tab_spillover, text="Spillover")

        # ---- Model Params tab ----
        self.tab_modelparams = ModelParamsFrame(self.notebook)
        self.notebook.add(self.tab_modelparams, text="Model Params")

        # Status bar for user feedback
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Default target suggestion
        tickers = self._load_tickers()
        if tickers and not self.inputs.get_target():
            self.inputs.ent_target.insert(0, tickers[0])
            self._on_target_change()

        self._update_nav_buttons()
        self._update_animation_buttons()

        self.spill_win = None

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
        settings = self.inputs.get_settings()
        if not settings["target"] or not settings["asof"]:
            self.status.config(text="Enter target and date to plot")
            return

        self.status.config(text="Loading...")

        def worker():
            try:
                # Check if animation is requested and supported
                if (self.var_animated.get() and 
                    self.plot_mgr.has_animation_support(settings["plot_type"])):
                    
                    # Try to create animated plot
                    if self.plot_mgr.plot_animated(self.ax, settings):
                        self.after(0, lambda: self.status.config(text="Animated plot created"))
                    else:
                        # Fall back to static plot
                        self.plot_mgr.plot(self.ax, settings)
                        self.after(0, lambda: self.status.config(text="Animation failed - using static plot"))
                else:
                    # Create static plot
                    self.plot_mgr.stop_animation()  # Stop any existing animation
                    self.plot_mgr.plot(self.ax, settings)
                    
                self.canvas.draw()
                self.after(0, self._update_nav_buttons)
                self.after(0, self._update_animation_buttons)
                
            except Exception as e:
                def handle_err(exc: Exception):
                    messagebox.showerror("Plot error", str(e))
                    self.status.config(text="Plot failed")
                    self._update_nav_buttons()
                    self._update_animation_buttons()
                self.after(0, handle_err(e))

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
    
    def _update_animation_buttons(self):
        """Update animation control button states."""
        plot_type = self.inputs.get_plot_type()
        has_anim_support = self.plot_mgr.has_animation_support(plot_type)
        is_animated = self.var_animated.get()
        is_anim_active = self.plot_mgr.is_animation_active()
        
        # Enable/disable animation checkbox based on plot type support
        anim_state = tk.NORMAL if has_anim_support else tk.DISABLED
        self.chk_animated.config(state=anim_state)
        
        # Enable/disable animation controls based on animation state
        control_state = tk.NORMAL if (is_animated and is_anim_active) else tk.DISABLED
        self.btn_play_pause.config(state=control_state)
        self.btn_stop.config(state=control_state)
        self.speed_scale.config(state=control_state)
        
        # Update play/pause button text
        if is_anim_active and not self.plot_mgr._animation_paused:
            self.btn_play_pause.config(text="Pause")
        else:
            self.btn_play_pause.config(text="Play")
    
    def _toggle_animation_mode(self):
        """Handle animation checkbox toggle."""
        # Refresh plot when animation mode changes
        self._refresh_plot()
    
    def _toggle_animation(self):
        """Toggle animation play/pause."""
        if self.plot_mgr.is_animation_active():
            if self.plot_mgr._animation_paused:
                self.plot_mgr.start_animation()
            else:
                self.plot_mgr.pause_animation()
            self._update_animation_buttons()
    
    def _stop_animation(self):
        """Stop animation."""
        self.plot_mgr.stop_animation()
        self._update_animation_buttons()
    
    def _on_speed_change(self, value):
        """Handle animation speed change."""
        try:
            speed_ms = int(2100 - float(value))  # Invert scale (higher value = faster)
            self.plot_mgr.set_animation_speed(speed_ms)
        except Exception:
            pass

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
                        help="Confidence interval percentage (e.g. 95 for 95%)")
    args = parser.parse_args()
    app = BrowserApp(overlay_synth=args.overlay_synth,
                     overlay_peers=args.overlay_peers,
                     ci_percent=args.ci)
    app.mainloop()

if __name__ == "__main__":
    main()
