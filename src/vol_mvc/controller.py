"""Controller for the volatility surface demo."""

from . import models, views


def run_analysis_with_gui(target_ticker, reference_tickers, options):
    print(f"Target Ticker: {target_ticker}")
    print(f"Reference Tickers: {', '.join(reference_tickers)}")
    target_df = models.download_option_data(target_ticker, max_expiries=8)
    if target_df is None or len(target_df) == 0:
        print("No data for target ticker")
        return
    if options.get("run_sabr"):
        result = models.simple_sabr_stats(target_df)
        if len(result) > 0:
            views.visualize_surface(result, target_df, target_ticker)
    if options.get("run_calls_puts"):
        views.interactive_sabr_smile_browser(target_df)
    if options.get("run_etf"):
        tickers = [target_ticker] + reference_tickers
        all_data = models.download_multiple_tickers(tickers, max_expiries=8)
        if len(all_data) > 1:
            weights, corr = models.compute_correlation_weights(all_data)
            synthetic = models.construct_synthetic_etf(all_data, weights)
            if synthetic is not None and options.get("run_interactive"):
                views.interactive_sabr_smile_browser(synthetic)


def main():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Vol Surface Demo")
    root.geometry("400x200")

    target_var = tk.StringVar(value="IONQ")
    ttk.Entry(root, textvariable=target_var).pack(fill=tk.X, padx=10, pady=5)
    refs_var = tk.StringVar(value="ARQQ,FORM")
    ttk.Entry(root, textvariable=refs_var).pack(fill=tk.X, padx=10, pady=5)

    def start():
        t = target_var.get().strip().upper()
        r = [x.strip().upper() for x in refs_var.get().split(',') if x.strip()]
        opts = {"run_sabr": True, "run_calls_puts": False, "run_etf": False, "run_interactive": False}
        root.destroy()
        run_analysis_with_gui(t, r, opts)

    ttk.Button(root, text="Start", command=start).pack(pady=10)
    root.mainloop()
