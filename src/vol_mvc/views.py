# Visualization helpers for the volatility surface demo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from sabr import hagan_lognormal_vol
from compute_volatility import fit_sabr_smile


def visualize_surface(result, original_df, ticker):
    """Visualize SABR smiles for each maturity."""
    valid = result.dropna(subset=["sabr_alpha"])
    if len(valid) == 0:
        print("No valid SABR fits found for visualization")
        return
    n_maturities = len(valid)
    cols = min(3, n_maturities)
    rows = (n_maturities + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    axes = axes.flatten()
    for i, (_, row) in enumerate(valid.iterrows()):
        T = row["T"]
        alpha, beta, rho, nu = row["sabr_alpha"], row["sabr_beta"], row["sabr_rho"], row["sabr_nu"]
        ax = axes[i]
        maturity_data = original_df[original_df["T"] == T]
        strikes = maturity_data["K"].values
        vols = maturity_data["sigma"].values
        ax.scatter(strikes, vols, color="blue", alpha=0.6, s=20, label="Observed")
        if not np.isnan(alpha):
            f = strikes.mean()
            k_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted = [hagan_lognormal_vol(f, k, T, alpha, beta, rho, nu) for k in k_range]
            ax.plot(k_range, fitted, "r-", label="SABR Fit", linewidth=2)
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"T={T:.3f} years")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for i in range(n_maturities, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.suptitle(f"SABR Smile Fits for {ticker}", y=1.02)
    plt.show()


def interactive_sabr_smile_browser(df):
    """Simple interactive browser for SABR smile fits."""
    calls_df = df[df["type"] == "call"]
    puts_df = df[df["type"] == "put"]
    call_maturities = sorted(calls_df["T"].unique())
    put_maturities = sorted(puts_df["T"].unique())
    call_valid = [T for T in call_maturities if len(calls_df[calls_df["T"] == T]) >= 2]
    put_valid = [T for T in put_maturities if len(puts_df[puts_df["T"] == T]) >= 2]
    if not call_valid and not put_valid:
        print("No maturities with sufficient data")
        return
    fig, (ax_calls, ax_puts) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.25, top=0.9)
    btn_next_ax = plt.axes((0.81, 0.15, 0.1, 0.075))
    btn_prev_ax = plt.axes((0.7, 0.15, 0.1, 0.075))
    btn_next = Button(btn_next_ax, "Next")
    btn_prev = Button(btn_prev_ax, "Prev")
    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    ci_slider = Slider(slider_ax, "CI Level", 0.50, 0.99, valinit=0.95, valstep=0.01)
    all_maturities = sorted(set(call_valid + put_valid))
    current = [0]

    def plot_smile(idx, ci_level=0.95):
        if idx < 0 or idx >= len(all_maturities):
            return
        T = all_maturities[idx]
        ax_calls.clear()
        ax_puts.clear()
        calls = calls_df[calls_df["T"] == T]
        if len(calls) >= 2:
            m = calls["moneyness"].values
            v = calls["sigma"].values
            f = m.mean()
            params = fit_sabr_smile(m, v, f, T)
            plot_smile_with_ci_bands(ax_calls, m, v, f, T, params, "Calls", ci_level)
        puts = puts_df[puts_df["T"] == T]
        if len(puts) >= 2:
            m = puts["moneyness"].values
            v = puts["sigma"].values
            f = m.mean()
            params = fit_sabr_smile(m, v, f, T)
            plot_smile_with_ci_bands(ax_puts, m, v, f, T, params, "Puts", ci_level)
        ax_calls.set_xlabel("Moneyness (K/S)")
        ax_calls.set_ylabel("Implied Volatility")
        ax_calls.legend()
        ax_calls.grid(True, alpha=0.3)
        ax_puts.set_xlabel("Moneyness (K/S)")
        ax_puts.legend()
        ax_puts.grid(True, alpha=0.3)
        fig.suptitle(f"SABR Smile Browser - {idx+1}/{len(all_maturities)}", fontsize=14)
        plt.draw()

    def next_event(event):
        current[0] = (current[0] + 1) % len(all_maturities)
        plot_smile(current[0], ci_slider.val)

    def prev_event(event):
        current[0] = (current[0] - 1) % len(all_maturities)
        plot_smile(current[0], ci_slider.val)

    btn_next.on_clicked(next_event)
    btn_prev.on_clicked(prev_event)
    ci_slider.on_changed(lambda v: plot_smile(current[0], v))
    if all_maturities:
        plot_smile(0, ci_slider.val)
    plt.show()


def evaluate_sabr_fit(strikes, vols, f, t, sabr_params):
    """Evaluate the quality of a SABR fit."""
    if sabr_params is None:
        return None
    alpha, beta, rho, nu = sabr_params
    fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strikes]
    fitted_vols = np.array(fitted_vols)
    absolute_errors = np.abs(vols - fitted_vols)
    relative_errors = np.abs((vols - fitted_vols) / vols)
    mae = np.mean(absolute_errors)
    mre = np.mean(relative_errors)
    max_error = np.max(absolute_errors)
    max_rel_error = np.max(relative_errors)
    ss_res = np.sum((vols - fitted_vols) ** 2)
    ss_tot = np.sum((vols - np.mean(vols)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return {
        "mae": mae,
        "mre": mre,
        "max_error": max_error,
        "max_rel_error": max_rel_error,
        "r_squared": r_squared,
        "fitted_vols": fitted_vols,
    }


def add_sabr_confidence_bands(ax, strikes, vols, f, t, sabr_params, confidence_level=0.95):
    if sabr_params is None:
        return
    from scipy import stats
    fitted = [hagan_lognormal_vol(f, k, t, *sabr_params) for k in strikes]
    residuals = vols - fitted
    std_error = np.std(residuals)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * std_error
    k_range = np.linspace(strikes.min(), strikes.max(), 100)
    fitted_range = [hagan_lognormal_vol(f, k, t, *sabr_params) for k in k_range]
    upper = np.array(fitted_range) + margin
    lower = np.array(fitted_range) - margin
    ax.fill_between(k_range, lower, upper, alpha=0.3, color="gray",
                    label=f"{confidence_level*100:.0f}% Confidence Band")


def plot_smile_with_ci_bands(ax, strikes, vols, f, t, sabr_params, title_prefix="", ci_level=0.95):
    ax.scatter(strikes, vols, color="blue", alpha=0.6, s=20, label="Observed")
    if sabr_params is not None:
        alpha, beta, rho, nu = sabr_params
        quality = evaluate_sabr_fit(strikes, vols, f, t, sabr_params)
        if quality is not None:
            k_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in k_range]
            ax.plot(k_range, fitted, "r-", label="SABR Fit", linewidth=2)
            add_sabr_confidence_bands(ax, strikes, vols, f, t, sabr_params, ci_level)
            ax.text(0.02, 0.98, f"R²={quality['r_squared']:.3f}", transform=ax.transAxes,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
    ax.set_title(f"{title_prefix}")

