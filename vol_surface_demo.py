#!/usr/bin/env python3
"""
Volatility Surface Demo with Real Option Data

This script demonstrates the complete pipeline:
1. Download real option data from Yahoo Finance
2. Compute weighted volatility statistics
3. Fit SABR models to volatility smiles
4. Visualize the resulting volatility surface
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from compute_volatility import weighted_stats, fit_sabr_smile
from pysabr import Hagan2002LognormalSABR
from sabr import hagan_lognormal_vol
from matplotlib.widgets import Button
from scipy.stats import pearsonr


def download_option_data(ticker='AAPL', max_expiries=8):
    """Download option chain data from Yahoo Finance with spot price normalization."""
    print(f"Downloading option data for {ticker}...")
    
    tk = yf.Ticker(ticker)
    expiries = tk.options
    
    if not expiries:
        print(f"No option data available for {ticker}")
        return None
    
    # Get current spot price
    try:
        spot_price = tk.info.get('regularMarketPrice', None)
        if spot_price is None:
            # Try to get from history
            hist = tk.history(period='1d')
            if not hist.empty:
                spot_price = hist['Close'].iloc[-1]
            else:
                print(f"Could not get spot price for {ticker}")
                return None
        print(f"Spot price for {ticker}: ${spot_price:.2f}")
    except Exception as e:
        print(f"Error getting spot price for {ticker}: {e}")
        return None
    
    print(f"Available expiries: {expiries[:10]} ...")
    print(f"Current date: {datetime.now()}")
    
    option_data = []
    for expiry in expiries[:max_expiries]:
        try:
            opt = tk.option_chain(expiry)
            for df, opt_type in [(opt.calls, 'call'), (opt.puts, 'put')]:
                print(f"Processing {expiry} {opt_type}: {df.shape[0]} rows")
                
                # Calculate time to maturity
                expiry_date = pd.to_datetime(expiry)
                current_date = pd.to_datetime(datetime.now())
                ttm = (expiry_date - current_date).days / 365.25
                
                print(f"  Debug: expiry={expiry_date}, current={current_date}, ttm={ttm:.6f}")
                
                # Filter for valid data
                valid_iv = df['impliedVolatility'].notna() & (df['impliedVolatility'] > 0)
                valid_ttm = ttm > 0
                
                print(f"  Valid IV: {valid_iv.sum()}, Valid TTM: {valid_ttm}")
                
                if valid_iv.sum() > 0 and valid_ttm:
                    for _, row in df[valid_iv].iterrows():
                        strike = row['strike']
                        moneyness = strike / spot_price  # K/S ratio
                        
                        option_data.append({
                            'K': strike,
                            'S': spot_price,
                            'moneyness': moneyness,
                            'T': ttm,
                            'sigma': row['impliedVolatility'],
                            'volume': row.get('volume', 0),
                            'type': opt_type
                        })
                        
        except Exception as e:
            print(f"Error processing {expiry}: {e}")
            continue
    
    if not option_data:
        print("No valid option data found")
        return None
    
    df = pd.DataFrame(option_data)
    print(f"Downloaded {len(df)} option records")
    
    # Filter out unrealistic IV values
    df = df[df['sigma'] > 0.01]  # Remove very low IV
    df = df[df['sigma'] < 2.0]   # Remove very high IV (>200%)
    
    # Filter out very short maturities
    df = df[df['T'] > 0.01]  # At least ~3.6 days
    
    # Filter moneyness range (reasonable range around spot)
    df = df[df['moneyness'] > 0.1]  # K/S > 0.1
    df = df[df['moneyness'] < 10.0]  # K/S < 10
    
    # Filter to only keep maturities with enough data for SABR fitting
    maturity_counts = df.groupby('T').size()
    valid_maturities = maturity_counts[maturity_counts >= 3].index  # Lowered from 5 to 3
    df = df[df['T'].isin(valid_maturities)]
    
    # Show data statistics
    if len(df) > 0:
        print(f"Maturity range: {df['T'].min():.3f} - {df['T'].max():.3f} years")
        print(f"Strike range: ${df['K'].min():.2f} - ${df['K'].max():.2f}")
        print(f"Moneyness range: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
        print(f"IV range: {df['sigma'].min():.3f} - {df['sigma'].max():.3f}")
        print(f"Unique maturities: {df['T'].nunique()}")
        print(f"Unique strikes: {df['K'].nunique()}")
        
        print(f"\nMaturity counts:")
        for T, count in maturity_counts.items():
            print(f"  T={T:.4f}: {count} options")
        
        print(f"\nSample data structure:")
        print(df.head())
        
        # Check for duplicate (K,T) combinations
        duplicates = df.groupby(['K', 'T']).size()
        print(f"\nDuplicate (K,T) combinations: {len(duplicates[duplicates > 1])} records")
        print(f"Unique (K,T) combinations: {len(duplicates)}")
        
        print(f"\nMaturities with sufficient data for SABR fitting:")
        for T in valid_maturities:
            count = len(df[df['T'] == T])
            print(f"  T={T:.4f}: {count} options")
    else:
        print("No valid data after filtering")
        return None
    
    return df


def compute_weights(df):
    """Compute correlation and liquidity weights."""
    print("Computing weights...")
    
    # Set correlation weights to 1 (can be customized based on correlation analysis)
    df['corr_weight'] = 1.0
    
    # Use volume as liquidity weight, normalized within each maturity
    df['liq_weight'] = df.groupby('T')['volume'].transform(
        lambda x: x / (x.sum() if x.sum() > 0 else 1)
    )
    
    return df


def test_weighted_stats(df):
    """Test the weighted statistics computation."""
    print("Computing weighted statistics and SABR parameters...")
    
    result = weighted_stats(df)
    
    print(f"Computed statistics for {len(result)} (K,T) pairs")
    print("\nSample results:")
    print(result.head())
    
    # Check for any NaN values in SABR parameters
    sabr_nan_count = result[['sabr_alpha', 'sabr_beta', 'sabr_rho', 'sabr_nu']].isna().sum()
    print(f"\nSABR parameter NaN counts:")
    print(sabr_nan_count)
    
    return result


def visualize_surface(result, original_df, ticker):
    """Visualize SABR smiles for each maturity."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sabr import hagan_lognormal_vol
    
    print("Creating SABR smile visualization...")
    
    # Filter out rows with NaN SABR parameters
    valid_result = result.dropna(subset=['sabr_alpha'])
    
    if len(valid_result) == 0:
        print("No valid SABR fits found for visualization")
        return
    
    # Create subplots for each maturity
    n_maturities = len(valid_result)
    cols = min(3, n_maturities)
    rows = (n_maturities + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (_, row) in enumerate(valid_result.iterrows()):
        T = row['T']
        alpha = row['sabr_alpha']
        beta = row['sabr_beta']
        rho = row['sabr_rho']
        nu = row['sabr_nu']
        
        ax = axes[i]
        
        # Get original data for this maturity
        maturity_data = original_df[original_df['T'] == T]
        strikes = maturity_data['K'].values
        vols = maturity_data['sigma'].values
        
        # Plot observed data
        ax.scatter(strikes, vols, color='blue', alpha=0.6, s=20, label='Observed')
        
        # Plot SABR fit
        if not np.isnan(alpha):
            f = strikes.mean()  # Use mean strike as forward
            strike_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted_vols = [hagan_lognormal_vol(f, k, T, alpha, beta, rho, nu) for k in strike_range]
            ax.plot(strike_range, fitted_vols, 'r-', label='SABR Fit', linewidth=2)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'T={T:.3f} years ({len(maturity_data)} options)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add SABR parameters as text
        if not np.isnan(alpha):
            param_text = f'α={alpha:.3f}\nβ={beta:.1f}\nρ={rho:.3f}\nν={nu:.3f}'
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_maturities, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'SABR Smile Fits for {ticker}', y=1.02)
    plt.show()
    
    # Also show SABR parameters vs maturity
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(valid_result['T'], valid_result['sabr_alpha'], 'o-')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Alpha')
    plt.title('SABR Alpha vs Maturity')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(valid_result['T'], valid_result['sabr_rho'], 'o-')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Rho')
    plt.title('SABR Rho vs Maturity')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(valid_result['T'], valid_result['sabr_nu'], 'o-')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Nu')
    plt.title('SABR Nu vs Maturity')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(valid_result['T'], valid_result['sigma_med'], 'o-')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Median IV')
    plt.title('Median IV vs Maturity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_sabr_fitting(df, result):
    """Test SABR fitting on individual smiles using moneyness."""
    print("Testing SABR fitting on individual smiles...")
    
    # Filter out rows with NaN SABR parameters
    valid_result = result.dropna(subset=['sabr_alpha'])
    
    for i, (_, row) in enumerate(valid_result.head(3).iterrows()):
        T = row['T']
        alpha = row['sabr_alpha']
        beta = row['sabr_beta']
        rho = row['sabr_rho']
        nu = row['sabr_nu']
        
        print(f"Testing SABR fit for maturity T={T:.3f}")
        print(f"  alpha={alpha:.4f}, beta={beta:.4f}, rho={rho:.4f}, nu={nu:.4f}")
        
        # Get original data for this maturity
        maturity_data = df[df['T'] == T]
        moneyness = maturity_data['moneyness'].values
        vols = maturity_data['sigma'].values
        
        # Use moneyness for SABR fitting
        f = moneyness.mean()  # Forward moneyness
        sabr_params = fit_sabr_smile(moneyness, vols, f, T)
        
        if sabr_params is not None:
            print(f"  SABR fit successful")
        else:
            print(f"  SABR fit failed")


def plot_all_sabr_smiles(df):
    """Plot SABR fit for each unique maturity (T) in the data with interactive navigation."""
    unique_T = sorted(df['T'].unique())
    print(f"Available maturities (years): {[f'{t:.4f}' for t in unique_T]}")
    print(f"Total maturities to view: {len(unique_T)}")
    
    for i, T in enumerate(unique_T):
        smile = df[df['T'] == T]
        if len(smile) < 3:
            print(f"Skipping T={T:.4f} (insufficient data: {len(smile)} points)")
            continue
            
        strikes = smile['K'].values
        vols = smile['sigma'].values
        weights = smile['corr_weight'].values * smile['liq_weight'].values
        fwd = np.average(strikes, weights=weights)
        
        print(f"\n--- Maturity {i+1}/{len(unique_T)}: T={T:.4f} years ---")
        print(f"Number of data points: {len(smile)}")
        print(f"Strike range: ${strikes.min():.2f} - ${strikes.max():.2f}")
        print(f"IV range: {vols.min():.3f} - {vols.max():.3f}")
        
        sabr_params = fit_sabr_smile(strikes, vols, fwd, T)
        if sabr_params[0] is None:
            print(f"SABR fit failed for T={T:.4f}")
            continue
            
        alpha, beta, rho, nu = sabr_params
        print(f"SABR parameters: alpha={alpha:.4f}, beta={beta:.4f}, rho={rho:.4f}, nu={nu:.4f}")
        
        sabr = Hagan2002LognormalSABR(fwd, beta)
        fitted_vols = [hagan_lognormal_vol(fwd, K, T, alpha, beta, rho, nu) for K in strikes]
        
        # Calculate fit quality
        mse = np.mean((np.array(vols) - np.array(fitted_vols))**2)
        print(f"Mean squared error: {mse:.6f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, vols, 'o', label='Observed', markersize=8)
        plt.plot(strikes, fitted_vols, '-', label='SABR Fit', linewidth=2)
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Implied Volatility')
        plt.title(f'SABR Smile Fit - Maturity {i+1}/{len(unique_T)}\nT={T:.4f} years ({T*365:.1f} days)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add parameter text box
        param_text = f'α={alpha:.4f}\nβ={beta:.4f}\nρ={rho:.4f}\nν={nu:.4f}\nMSE={mse:.6f}'
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        if i < len(unique_T) - 1:
            response = input(f"Press Enter for next maturity ({i+2}/{len(unique_T)}) or 'q' to quit: ")
            if response.lower() == 'q':
                print("Stopping SABR plot navigation.")
                break
        else:
            print("Finished viewing all maturities.")
    
    return unique_T


def interactive_sabr_smile_browser(df):
    """Interactive browser for SABR smile fits by maturity with keyboard and GUI navigation."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    import numpy as np
    from compute_volatility import fit_sabr_smile
    from sabr import hagan_lognormal_vol

    # Separate calls and puts
    calls_df = df[df['type'] == 'call']
    puts_df = df[df['type'] == 'put']
    
    print(f"Total options: {len(df)}")
    print(f"Calls: {len(calls_df)}")
    print(f"Puts: {len(puts_df)}")
    
    # Get maturities with sufficient data for each type
    call_maturities = sorted(calls_df['T'].unique())
    put_maturities = sorted(puts_df['T'].unique())
    
    call_valid = [T for T in call_maturities if len(calls_df[calls_df['T'] == T]) >= 2]  # Lowered from 3 to 2
    put_valid = [T for T in put_maturities if len(puts_df[puts_df['T'] == T]) >= 2]  # Lowered from 3 to 2
    
    print(f"Call maturities with sufficient data: {len(call_valid)}")
    print(f"Put maturities with sufficient data: {len(put_valid)}")
    
    if not call_valid and not put_valid:
        print("No maturities with sufficient data.")
        return
    
    # Create figure with subplots for calls and puts
    fig, (ax_calls, ax_puts) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.25, top=0.9)
    
    # Navigation buttons
    btn_next_ax = plt.axes((0.81, 0.15, 0.1, 0.075))
    btn_prev_ax = plt.axes((0.7, 0.15, 0.1, 0.075))
    btn_next = Button(btn_next_ax, 'Next')
    btn_prev = Button(btn_prev_ax, 'Prev')
    
    # CI Slider
    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    ci_slider = Slider(slider_ax, 'CI Level', 0.50, 0.99, valinit=0.95, valstep=0.01)
    
    # Store text artists for proper cleanup
    text_artists = []
    
    # Combine valid maturities and track current
    all_maturities = sorted(set(call_valid + put_valid))
    current = [0]  # mutable index
    
    def plot_smile(idx, ci_level=0.95):
        if idx < 0 or idx >= len(all_maturities):
            return
        
        T = all_maturities[idx]
        
        # Get data for this maturity
        maturity_data = df[df['T'] == T]
        calls_data = maturity_data[maturity_data['type'] == 'call']
        puts_data = maturity_data[maturity_data['type'] == 'put']
        
        # Clear previous plots
        ax_calls.clear()
        ax_puts.clear()
        
        # Plot calls
        if len(calls_data) > 0:
            call_moneyness = calls_data['moneyness'].values
            call_vols = calls_data['sigma'].values
            
            # Try SABR fit for calls
            if len(call_moneyness) >= 3:  # Lowered from 5 to 3
                f = call_moneyness.mean()
                sabr_params = fit_sabr_smile(call_moneyness, call_vols, f, T)
                
                # Plot with CI bands
                plot_smile_with_ci_bands(ax_calls, call_moneyness, call_vols, f, T, sabr_params, "Calls", ci_level)
            else:
                ax_calls.scatter(call_moneyness, call_vols, color='blue', alpha=0.6, s=20, label='Observed')
                ax_calls.text(0.02, 0.98, 'Insufficient data for SABR fit', transform=ax_calls.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot puts
        if len(puts_data) > 0:
            put_moneyness = puts_data['moneyness'].values
            put_vols = puts_data['sigma'].values
            
            # Try SABR fit for puts
            if len(put_moneyness) >= 3:  # Lowered from 5 to 3
                f = put_moneyness.mean()
                sabr_params = fit_sabr_smile(put_moneyness, put_vols, f, T)
                
                # Plot with CI bands
                plot_smile_with_ci_bands(ax_puts, put_moneyness, put_vols, f, T, sabr_params, "Puts", ci_level)
            else:
                ax_puts.scatter(put_moneyness, put_vols, color='red', alpha=0.6, s=20, label='Observed')
                ax_puts.text(0.02, 0.98, 'Insufficient data for SABR fit', transform=ax_puts.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Set labels and title
        ax_calls.set_xlabel('Moneyness (K/S)')
        ax_calls.set_ylabel('Implied Volatility')
        ax_calls.set_title(f'Calls - T={T:.3f} years')
        ax_calls.legend()
        ax_calls.grid(True, alpha=0.3)
        
        ax_puts.set_xlabel('Moneyness (K/S)')
        ax_puts.set_ylabel('Implied Volatility')
        ax_puts.set_title(f'Puts - T={T:.3f} years')
        ax_puts.legend()
        ax_puts.grid(True, alpha=0.3)
        
        # Update title with CI level
        fig.suptitle(f'SABR Smile Browser with CI Slider - Maturity {idx+1}/{len(all_maturities)} (T={T:.3f} years, CI={ci_level*100:.0f}%)', fontsize=14)
        
        plt.draw()

    def next_event(event):
        current[0] = (current[0] + 1) % len(all_maturities)
        plot_smile(current[0], ci_slider.val)

    def prev_event(event):
        current[0] = (current[0] - 1) % len(all_maturities)
        plot_smile(current[0], ci_slider.val)

    def key_event(event):
        if event.key == 'right' or event.key == 'down' or event.key == 'enter':
            next_event(None)
        elif event.key == 'left' or event.key == 'up':
            prev_event(None)
        elif event.key == 'q':
            plt.close()

    def ci_slider_update(val):
        plot_smile(current[0], val)

    # Connect events
    btn_next.on_clicked(next_event)
    btn_prev.on_clicked(prev_event)
    fig.canvas.mpl_connect('key_press_event', key_event)
    ci_slider.on_changed(ci_slider_update)

    # Show first plot
    if all_maturities:
        plot_smile(0, ci_slider.val)
    
    print("Interactive SABR Browser with CI Slider:")
    print("- Use arrow keys or Enter to navigate")
    print("- Click Next/Prev buttons")
    print("- Adjust CI level with slider (50%-99%)")
    print("- Press 'q' to quit")
    print("- Left plot: Calls, Right plot: Puts")
    
    plt.show()


def analyze_calls_vs_puts(df):
    """Analyze calls and puts separately."""
    calls_df = df[df['type'] == 'call']
    puts_df = df[df['type'] == 'put']
    
    print(f"\n=== Calls vs Puts Analysis ===")
    print(f"Total options: {len(df)}")
    print(f"Calls: {len(calls_df)} ({len(calls_df)/len(df)*100:.1f}%)")
    print(f"Puts: {len(puts_df)} ({len(puts_df)/len(df)*100:.1f}%)")
    
    # Maturity analysis
    call_maturities = sorted(calls_df['T'].unique())
    put_maturities = sorted(puts_df['T'].unique())
    
    print(f"\nCall maturities: {len(call_maturities)}")
    print(f"Put maturities: {len(put_maturities)}")
    
    # IV statistics
    print(f"\nCall IV stats:")
    print(f"  Mean: {calls_df['sigma'].mean():.3f}")
    print(f"  Std: {calls_df['sigma'].std():.3f}")
    print(f"  Min: {calls_df['sigma'].min():.3f}")
    print(f"  Max: {calls_df['sigma'].max():.3f}")
    
    print(f"\nPut IV stats:")
    print(f"  Mean: {puts_df['sigma'].mean():.3f}")
    print(f"  Std: {puts_df['sigma'].std():.3f}")
    print(f"  Min: {puts_df['sigma'].min():.3f}")
    print(f"  Max: {puts_df['sigma'].max():.3f}")
    
    # Strike analysis
    print(f"\nCall strike range: ${calls_df['K'].min():.2f} - ${calls_df['K'].max():.2f}")
    print(f"Put strike range: ${puts_df['K'].min():.2f} - ${puts_df['K'].max():.2f}")
    
    return calls_df, puts_df


def evaluate_sabr_fit(strikes, vols, f, t, sabr_params):
    """Evaluate the quality of SABR fit."""
    if sabr_params is None:
        return None
    
    alpha, beta, rho, nu = sabr_params
    
    try:
        # Calculate fitted volatilities
        fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strikes]
        fitted_vols = np.array(fitted_vols)
        
        # Calculate error metrics
        absolute_errors = np.abs(vols - fitted_vols)
        relative_errors = np.abs((vols - fitted_vols) / vols)
        
        mae = np.mean(absolute_errors)
        mre = np.mean(relative_errors)
        max_error = np.max(absolute_errors)
        max_rel_error = np.max(relative_errors)
        
        # Calculate R-squared
        ss_res = np.sum((vols - fitted_vols) ** 2)
        ss_tot = np.sum((vols - np.mean(vols)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': mae,
            'mre': mre,
            'max_error': max_error,
            'max_rel_error': max_rel_error,
            'r_squared': r_squared,
            'fitted_vols': fitted_vols
        }
    except:
        return None

def plot_smile_with_ci_bands(ax, strikes, vols, f, t, sabr_params, title_prefix="", ci_level=0.95):
    """Plot smile with SABR fit and adjustable confidence bands."""
    # Plot observed data
    ax.scatter(strikes, vols, color='blue', alpha=0.6, s=20, label='Observed')
    
    # Try SABR fit
    if sabr_params is not None:
        alpha, beta, rho, nu = sabr_params
        
        # Evaluate fit quality
        quality = evaluate_sabr_fit(strikes, vols, f, t, sabr_params)
        
        if quality is not None:
            # Plot fitted curve
            strike_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
            ax.plot(strike_range, fitted_vols, 'r-', label='SABR Fit', linewidth=2)
            
            # Add confidence bands
            add_sabr_confidence_bands(ax, strikes, vols, f, t, sabr_params, ci_level)
            
            # Add quality metrics
            quality_text = f'R²={quality["r_squared"]:.3f}\nMAE={quality["mae"]:.4f}\nMRE={quality["mre"]:.3f}'
            ax.text(0.02, 0.98, quality_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Add SABR parameters
            param_text = f'α={alpha:.3f}\nβ={beta:.1f}\nρ={rho:.3f}\nν={nu:.3f}'
            ax.text(0.02, 0.85, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add CI level info
            ci_text = f'CI Level: {ci_level*100:.0f}%'
            ax.text(0.02, 0.70, ci_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            return quality
        else:
            ax.text(0.02, 0.98, 'SABR fit failed', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    return None

def plot_smile_with_correlation_confidence(ax, strikes, vols, f, t, sabr_params, all_data, weights, title_prefix=""):
    """Plot smile with SABR fit and correlation-based confidence bands."""
    # Plot observed data
    ax.scatter(strikes, vols, color='blue', alpha=0.6, s=20, label='Observed')
    
    # Try SABR fit
    if sabr_params is not None:
        alpha, beta, rho, nu = sabr_params
        
        # Evaluate fit quality
        quality = evaluate_sabr_fit(strikes, vols, f, t, sabr_params)
        
        if quality is not None:
            # Plot fitted curve
            strike_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
            ax.plot(strike_range, fitted_vols, 'r-', label='SABR Fit', linewidth=2)
            
            # Add correlation-based confidence bands
            add_correlation_confidence_bands(ax, strikes, vols, f, t, sabr_params, all_data, weights)
            
            # Add quality metrics
            quality_text = f'R²={quality["r_squared"]:.3f}\nMAE={quality["mae"]:.4f}\nMRE={quality["mre"]:.3f}'
            ax.text(0.02, 0.98, quality_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Add SABR parameters
            param_text = f'α={alpha:.3f}\nβ={beta:.1f}\nρ={rho:.3f}\nν={nu:.3f}'
            ax.text(0.02, 0.85, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            return quality
        else:
            ax.text(0.02, 0.98, 'SABR fit failed', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    return None

def download_multiple_tickers(tickers=['ARQQ', 'FORM', 'IONQ', 'QBTS', 'QMCO', 'QTUM', 'QUBT', 'RGTI', 'SKYT', 'WIMI'], max_expiries=8):
    """Download option data for multiple tickers."""
    print(f"Downloading data for {len(tickers)} tickers: {tickers}")
    
    all_data = {}
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            df = download_option_data(ticker, max_expiries)
            if df is not None and len(df) > 0:
                all_data[ticker] = df
                print(f"  {ticker}: {len(df)} options, {df['T'].nunique()} maturities")
            else:
                print(f"  {ticker}: No valid data")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    return all_data

def construct_synthetic_etf(all_data, weights=None, correlation_matrix=None):
    """Construct synthetic ETF from multiple tickers using real data with calls/puts separation."""
    if not all_data:
        print("No data available for synthetic ETF construction")
        return None
    
    tickers = list(all_data.keys())
    print(f"\nConstructing synthetic ETF from {len(tickers)} tickers")
    
    # Default equal weights if not provided
    if weights is None:
        weights = {ticker: 1.0/len(tickers) for ticker in tickers}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {ticker: w/total_weight for ticker, w in weights.items()}
    
    print("Weights:", weights)
    
    # Find common maturities across all tickers
    all_maturities = set()
    for ticker, df in all_data.items():
        all_maturities.update(df['T'].unique())
    
    common_maturities = sorted(all_maturities)
    print(f"Common maturities: {len(common_maturities)}")
    
    # Construct synthetic ETF surface separately for calls and puts
    synthetic_data = []
    
    for T in common_maturities:
        # Get data for this maturity from all tickers, separated by type
        maturity_data_calls = {}
        maturity_data_puts = {}
        
        for ticker, df in all_data.items():
            ticker_data = df[df['T'] == T]
            if len(ticker_data) > 0:
                calls_data = ticker_data[ticker_data['type'] == 'call']
                puts_data = ticker_data[ticker_data['type'] == 'put']
                
                if len(calls_data) > 0:
                    maturity_data_calls[ticker] = calls_data
                if len(puts_data) > 0:
                    maturity_data_puts[ticker] = puts_data
        
        # Process calls
        if len(maturity_data_calls) >= 2:
            synthetic_data.extend(construct_synthetic_surface(maturity_data_calls, T, 'call', weights))
        
        # Process puts
        if len(maturity_data_puts) >= 2:
            synthetic_data.extend(construct_synthetic_surface(maturity_data_puts, T, 'put', weights))
    
    if not synthetic_data:
        print("No synthetic data generated")
        return None
    
    synthetic_df = pd.DataFrame(synthetic_data)
    print(f"Synthetic ETF: {len(synthetic_df)} options, {synthetic_df['T'].nunique()} maturities")
    
    return synthetic_df


def construct_synthetic_surface(maturity_data, T, option_type, weights):
    """Construct synthetic surface for a specific maturity and option type."""
    synthetic_data = []
    
    # Find common moneyness range
    all_moneyness = set()
    for ticker_data in maturity_data.values():
        all_moneyness.update(ticker_data['moneyness'].unique())
    
    # For each moneyness level, compute weighted average IV
    for moneyness in sorted(all_moneyness):
        weighted_iv = 0
        total_weight_for_moneyness = 0
        
        for ticker, ticker_data in maturity_data.items():
            moneyness_data = ticker_data[ticker_data['moneyness'] == moneyness]
            if len(moneyness_data) > 0:
                # Use median IV for this moneyness level
                iv = moneyness_data['sigma'].median()
                weight = weights[ticker]
                weighted_iv += weight * iv
                total_weight_for_moneyness += weight
        
        if total_weight_for_moneyness > 0:
            # Use average spot price for this moneyness level
            avg_spot = np.mean([ticker_data['S'].iloc[0] for ticker_data in maturity_data.values()])
            synthetic_strike = moneyness * avg_spot
            
            synthetic_data.append({
                'K': synthetic_strike,
                'S': avg_spot,
                'moneyness': moneyness,
                'T': T,
                'sigma': weighted_iv / total_weight_for_moneyness,
                'type': option_type,
                'volume': 0  # Synthetic has no volume
            })
    
    return synthetic_data

def add_sabr_confidence_bands(ax, strikes, vols, f, t, sabr_params, confidence_level=0.95):
    """Add confidence bands to SABR fit."""
    if sabr_params is None:
        return
    
    alpha, beta, rho, nu = sabr_params
    
    try:
        import numpy as np
        from scipy import stats
        
        # Calculate fitted volatilities
        fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strikes]
        fitted_vols = np.array(fitted_vols)
        
        # Calculate residuals
        residuals = vols - fitted_vols
        
        # Estimate standard error of residuals
        std_error = np.std(residuals)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_error
        
        # Generate confidence bands
        strike_range = np.linspace(strikes.min(), strikes.max(), 100)
        fitted_range = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
        
        upper_band = np.array(fitted_range) + margin_of_error
        lower_band = np.array(fitted_range) - margin_of_error
        
        # Plot confidence bands
        ax.fill_between(strike_range, lower_band, upper_band, alpha=0.3, color='gray', 
                       label=f'{confidence_level*100:.0f}% Confidence Band')
        
        # Add confidence band info
        ax.text(0.02, 0.70, f'Std Error: {std_error:.4f}\nConfidence: {confidence_level*100:.0f}%', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
    except Exception as e:
        print(f"Warning: Could not add confidence bands: {e}")

def plot_smile_with_confidence(ax, strikes, vols, f, t, sabr_params, title_prefix="", confidence_level=0.95):
    """Plot smile with SABR fit and confidence bands."""
    # Plot observed data
    ax.scatter(strikes, vols, color='blue', alpha=0.6, s=20, label='Observed')
    
    # Try SABR fit
    if sabr_params is not None:
        alpha, beta, rho, nu = sabr_params
        
        # Evaluate fit quality
        quality = evaluate_sabr_fit(strikes, vols, f, t, sabr_params)
        
        if quality is not None:
            # Plot fitted curve
            strike_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
            ax.plot(strike_range, fitted_vols, 'r-', label='SABR Fit', linewidth=2)
            
            # Add confidence bands
            add_sabr_confidence_bands(ax, strikes, vols, f, t, sabr_params, confidence_level)
            
            # Add quality metrics
            quality_text = f'R²={quality["r_squared"]:.3f}\nMAE={quality["mae"]:.4f}\nMRE={quality["mre"]:.3f}'
            ax.text(0.02, 0.98, quality_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Add SABR parameters
            param_text = f'α={alpha:.3f}\nβ={beta:.1f}\nρ={rho:.3f}\nν={nu:.3f}'
            ax.text(0.02, 0.85, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            return quality
        else:
            ax.text(0.02, 0.98, 'SABR fit failed', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    return None

def compute_correlation_weights(all_data, correlation_matrix=None):
    """Compute correlation-based weights for synthetic ETF construction."""
    tickers = list(all_data.keys())
    
    if correlation_matrix is None:
        # Compute correlation matrix from IV data
        print("Computing correlation matrix from IV data...")
        correlation_matrix = compute_iv_correlation_matrix(all_data)
    
    # Convert correlation to weights (higher correlation = higher weight)
    weights = {}
    for i, ticker in enumerate(tickers):
        # Use average correlation with other tickers as weight
        correlations = [correlation_matrix[i][j] for j in range(len(tickers)) if i != j]
        avg_correlation = np.mean(correlations) if correlations else 0.5
        weights[ticker] = avg_correlation
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {ticker: w/total_weight for ticker, w in weights.items()}
    
    print("Correlation-based weights:", weights)
    return weights, correlation_matrix


def compute_iv_correlation_matrix(all_data):
    """Compute correlation matrix from IV data with color-coded scatter plot."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    
    tickers = list(all_data.keys())
    n_tickers = len(tickers)
    
    # Create correlation matrix
    correlation_matrix = np.zeros((n_tickers, n_tickers))
    
    # Collect IV data for each ticker
    iv_data = {}
    for i, ticker in enumerate(tickers):
        df = all_data[ticker]
        # Use median IV for each maturity
        iv_by_maturity = df.groupby('T')['sigma'].median()
        iv_data[ticker] = iv_by_maturity
    
    # Compute correlations
    for i in range(n_tickers):
        for j in range(n_tickers):
            if i == j:
                correlation_matrix[i][j] = 1.0
            else:
                ticker1, ticker2 = tickers[i], tickers[j]
                iv1, iv2 = iv_data[ticker1], iv_data[ticker2]
                
                # Find common maturities
                common_maturities = iv1.index.intersection(iv2.index)
                if len(common_maturities) >= 2:
                    corr, _ = pearsonr(iv1[common_maturities], iv2[common_maturities])
                    correlation_matrix[i][j] = corr
                else:
                    correlation_matrix[i][j] = 0.0
    
    # Create color-coded scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette for tickers
    colors = plt.cm.tab10(np.linspace(0, 1, n_tickers))
    
    # Plot 1: Correlation heatmap
    im = ax1.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(n_tickers))
    ax1.set_yticks(range(n_tickers))
    ax1.set_xticklabels(tickers, rotation=45)
    ax1.set_yticklabels(tickers)
    ax1.set_title('IV Correlation Matrix')
    
    # Add correlation values as text
    for i in range(n_tickers):
        for j in range(n_tickers):
            text = ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Plot 2: Color-coded scatter plot
    # Create pairs of tickers for scatter plot
    ticker_pairs = []
    correlations = []
    colors_scatter = []
    
    for i in range(n_tickers):
        for j in range(i+1, n_tickers):
            ticker1, ticker2 = tickers[i], tickers[j]
            iv1, iv2 = iv_data[ticker1], iv_data[ticker2]
            
            # Find common maturities
            common_maturities = iv1.index.intersection(iv2.index)
            if len(common_maturities) >= 2:
                # Create scatter plot for this pair
                x_vals = iv1[common_maturities].values
                y_vals = iv2[common_maturities].values
                
                # Color code by ticker1 (primary ticker)
                color = colors[i]
                
                # Plot with ticker labels
                ax2.scatter(x_vals, y_vals, c=[color], alpha=0.7, s=50, 
                           label=f'{ticker1} vs {ticker2}', edgecolors='black', linewidth=0.5)
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                ax2.plot(x_vals, p(x_vals), color=color, alpha=0.5, linewidth=2)
    
    ax2.set_xlabel('IV (Ticker 1)')
    ax2.set_ylabel('IV (Ticker 2)')
    ax2.set_title('IV Correlation Scatter Plot (Color-coded by Primary Ticker)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation statistics
    corr_stats = []
    for i in range(n_tickers):
        for j in range(i+1, n_tickers):
            corr_val = correlation_matrix[i][j]
            corr_stats.append(f'{tickers[i]}-{tickers[j]}: {corr_val:.3f}')
    
    stats_text = '\n'.join(corr_stats[:8])  # Show first 8 correlations
    ax2.text(0.02, 0.98, f'Top Correlations:\n{stats_text}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix


def create_correlation_scatter_plot(all_data, correlation_matrix):
    """Create a detailed color-coded correlation scatter plot."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    tickers = list(all_data.keys())
    n_tickers = len(tickers)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed IV Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_tickers))
    
    # Plot 1: Correlation heatmap
    im1 = axes[0,0].imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    axes[0,0].set_xticks(range(n_tickers))
    axes[0,0].set_yticks(range(n_tickers))
    axes[0,0].set_xticklabels(tickers, rotation=45)
    axes[0,0].set_yticklabels(tickers)
    axes[0,0].set_title('IV Correlation Matrix')
    
    # Add correlation values
    for i in range(n_tickers):
        for j in range(n_tickers):
            text = axes[0,0].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=axes[0,0], label='Correlation')
    
    # Plot 2: Color-coded scatter matrix
    # Select top 4 most correlated pairs
    correlations = []
    for i in range(n_tickers):
        for j in range(i+1, n_tickers):
            correlations.append((i, j, correlation_matrix[i][j]))
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = correlations[:4]
    
    for idx, (i, j, corr) in enumerate(top_pairs):
        ticker1, ticker2 = tickers[i], tickers[j]
        df1, df2 = all_data[ticker1], all_data[ticker2]
        
        # Get common maturities
        common_maturities = set(df1['T'].unique()) & set(df2['T'].unique())
        
        if len(common_maturities) >= 2:
            # Aggregate IV by maturity
            iv1 = df1[df1['T'].isin(common_maturities)].groupby('T')['sigma'].median()
            iv2 = df2[df2['T'].isin(common_maturities)].groupby('T')['sigma'].median()
            
            # Align data
            aligned_data = pd.DataFrame({'IV1': iv1, 'IV2': iv2}).dropna()
            
            if len(aligned_data) >= 2:
                row, col = idx // 2, idx % 2
                ax = axes[row, col]
                
                # Color code by maturity (time to maturity)
                scatter = ax.scatter(aligned_data['IV1'], aligned_data['IV2'], 
                                   c=aligned_data.index, cmap='viridis', s=100, alpha=0.8)
                
                # Add trend line
                z = np.polyfit(aligned_data['IV1'], aligned_data['IV2'], 1)
                p = np.poly1d(z)
                ax.plot(aligned_data['IV1'], p(aligned_data['IV1']), 'r--', alpha=0.7)
                
                ax.set_xlabel(f'{ticker1} IV')
                ax.set_ylabel(f'{ticker2} IV')
                ax.set_title(f'{ticker1} vs {ticker2}\nCorrelation: {corr:.3f}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar for maturity
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Time to Maturity (years)')
    
    # Plot 3: IV distribution by ticker
    ax_dist = axes[1,0]
    for i, ticker in enumerate(tickers):
        df = all_data[ticker]
        ax_dist.hist(df['sigma'], bins=20, alpha=0.6, label=ticker, 
                    color=colors[i], density=True)
    
    ax_dist.set_xlabel('Implied Volatility')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title('IV Distribution by Ticker')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # Plot 4: Correlation summary
    ax_summary = axes[1,1]
    
    # Create correlation pairs for bar plot
    pair_names = []
    corr_values = []
    colors_bars = []
    
    for i, j, corr in correlations[:8]:  # Top 8 correlations
        pair_names.append(f'{tickers[i]}-{tickers[j]}')
        corr_values.append(corr)
        colors_bars.append('red' if corr < 0 else 'blue')
    
    bars = ax_summary.bar(range(len(pair_names)), corr_values, color=colors_bars, alpha=0.7)
    ax_summary.set_xticks(range(len(pair_names)))
    ax_summary.set_xticklabels(pair_names, rotation=45, ha='right')
    ax_summary.set_ylabel('Correlation')
    ax_summary.set_title('Top IV Correlations')
    ax_summary.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_summary.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, corr_values):
        height = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def add_correlation_confidence_bands(ax, strikes, vols, f, t, sabr_params, all_data, weights, confidence_level=0.95):
    """Add confidence bands based on correlation-weighted uncertainty."""
    if sabr_params is None:
        return
    
    alpha, beta, rho, nu = sabr_params
    
    try:
        import numpy as np
        from scipy import stats
        
        # Calculate fitted volatilities
        fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strikes]
        fitted_vols = np.array(fitted_vols)
        
        # Calculate base residuals
        residuals = vols - fitted_vols
        base_std_error = np.std(residuals)
        
        # Adjust uncertainty based on correlation weights
        # Higher correlation = lower uncertainty (more confidence)
        avg_correlation = np.mean(list(weights.values()))
        correlation_factor = 1.0 - avg_correlation  # Higher correlation = lower uncertainty
        
        adjusted_std_error = base_std_error * (1.0 + correlation_factor)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * adjusted_std_error
        
        # Generate confidence bands
        strike_range = np.linspace(strikes.min(), strikes.max(), 100)
        fitted_range = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
        
        upper_band = np.array(fitted_range) + margin_of_error
        lower_band = np.array(fitted_range) - margin_of_error
        
        # Plot confidence bands
        ax.fill_between(strike_range, lower_band, upper_band, alpha=0.3, color='orange', 
                       label=f'Correlation-Weighted {confidence_level*100:.0f}% Confidence')
        
        # Add correlation info
        ax.text(0.02, 0.70, f'Avg Correlation: {avg_correlation:.3f}\nAdjusted Std: {adjusted_std_error:.4f}', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
    except Exception as e:
        print(f"Warning: Could not add correlation confidence bands: {e}")

def interactive_correlated_etf_browser(all_data, synthetic_df, weights=None):
    """Interactive browser for correlated ETF analysis with calls/puts separation and CI slider."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    import numpy as np
    from compute_volatility import fit_sabr_smile
    from sabr import hagan_lognormal_vol

    if synthetic_df is None:
        print("No synthetic ETF data available")
        return
    
    # Separate calls and puts
    calls_df = synthetic_df[synthetic_df['type'] == 'call']
    puts_df = synthetic_df[synthetic_df['type'] == 'put']
    
    print(f"Synthetic ETF - Calls: {len(calls_df)}, Puts: {len(puts_df)}")
    
    # Get maturities with sufficient data for each type
    call_maturities = sorted(calls_df['T'].unique())
    put_maturities = sorted(puts_df['T'].unique())
    
    call_valid = [T for T in call_maturities if len(calls_df[calls_df['T'] == T]) >= 3]
    put_valid = [T for T in put_maturities if len(puts_df[puts_df['T'] == T]) >= 3]
    
    if not call_valid and not put_valid:
        print("No maturities with sufficient data for synthetic ETF")
        return
    
    # Create figure with subplots for calls and puts
    fig, (ax_calls, ax_puts) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.25, top=0.9)
    
    # Navigation buttons
    btn_next_ax = plt.axes((0.81, 0.15, 0.1, 0.075))
    btn_prev_ax = plt.axes((0.7, 0.15, 0.1, 0.075))
    btn_next = Button(btn_next_ax, 'Next')
    btn_prev = Button(btn_prev_ax, 'Prev')
    
    # CI Slider
    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    ci_slider = Slider(slider_ax, 'CI Level', 0.50, 0.99, valinit=0.95, valstep=0.01)
    
    # Combine valid maturities and track current
    all_maturities = sorted(set(call_valid + put_valid))
    current = [0]  # mutable index
    
    def plot_maturity(idx, ci_level=0.95):
        """Plot the synthetic ETF calls and puts for the given maturity."""
        if idx < 0 or idx >= len(all_maturities):
            return
        
        T = all_maturities[idx]
        
        # Clear previous plots
        ax_calls.clear()
        ax_puts.clear()
        
        # Plot synthetic ETF calls
        calls_data = calls_df[calls_df['T'] == T]
        if len(calls_data) > 0:
            call_moneyness = calls_data['moneyness'].values
            call_vols = calls_data['sigma'].values
            
            # Try SABR fit for calls
            if len(call_moneyness) >= 3:
                f = call_moneyness.mean()
                sabr_params = fit_sabr_smile(call_moneyness, call_vols, f, T)
                
                # Plot with correlation-based confidence bands
                plot_smile_with_correlation_ci_bands(ax_calls, call_moneyness, call_vols, f, T, sabr_params, all_data, weights, "Synthetic ETF Calls", ci_level)
            else:
                ax_calls.scatter(call_moneyness, call_vols, color='blue', alpha=0.6, s=20, label='Synthetic ETF Calls')
                ax_calls.text(0.02, 0.98, 'Insufficient data for SABR fit', transform=ax_calls.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot synthetic ETF puts
        puts_data = puts_df[puts_df['T'] == T]
        if len(puts_data) > 0:
            put_moneyness = puts_data['moneyness'].values
            put_vols = puts_data['sigma'].values
            
            # Try SABR fit for puts
            if len(put_moneyness) >= 3:
                f = put_moneyness.mean()
                sabr_params = fit_sabr_smile(put_moneyness, put_vols, f, T)
                
                # Plot with correlation-based confidence bands
                plot_smile_with_correlation_ci_bands(ax_puts, put_moneyness, put_vols, f, T, sabr_params, all_data, weights, "Synthetic ETF Puts", ci_level)
            else:
                ax_puts.scatter(put_moneyness, put_vols, color='red', alpha=0.6, s=20, label='Synthetic ETF Puts')
                ax_puts.text(0.02, 0.98, 'Insufficient data for SABR fit', transform=ax_puts.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Set labels and title
        ax_calls.set_xlabel('Moneyness (K/S)')
        ax_calls.set_ylabel('Implied Volatility')
        ax_calls.set_title(f'Synthetic ETF Calls - T={T:.3f} years')
        ax_calls.legend()
        ax_calls.grid(True, alpha=0.3)
        
        ax_puts.set_xlabel('Moneyness (K/S)')
        ax_puts.set_ylabel('Implied Volatility')
        ax_puts.set_title(f'Synthetic ETF Puts - T={T:.3f} years')
        ax_puts.legend()
        ax_puts.grid(True, alpha=0.3)
        
        # Update title with CI level
        fig.suptitle(f'Correlated ETF Browser with CI Slider - Maturity {idx+1}/{len(all_maturities)} (T={T:.3f} years, CI={ci_level*100:.0f}%)', fontsize=14)
        
        plt.draw()

    def next_event(event):
        current[0] = (current[0] + 1) % len(all_maturities)
        plot_maturity(current[0], ci_slider.val)

    def prev_event(event):
        current[0] = (current[0] - 1) % len(all_maturities)
        plot_maturity(current[0], ci_slider.val)

    def key_event(event):
        if event.key == 'right' or event.key == 'down' or event.key == 'enter':
            next_event(None)
        elif event.key == 'left' or event.key == 'up':
            prev_event(None)
        elif event.key == 'q':
            plt.close()

    def ci_slider_update(val):
        plot_maturity(current[0], val)

    # Connect events
    btn_next.on_clicked(next_event)
    btn_prev.on_clicked(prev_event)
    fig.canvas.mpl_connect('key_press_event', key_event)
    ci_slider.on_changed(ci_slider_update)

    # Show first plot
    if all_maturities:
        plot_maturity(0, ci_slider.val)
    
    print("Interactive Correlated ETF Browser with CI Slider:")
    print("- Use arrow keys or Enter to navigate")
    print("- Click Next/Prev buttons")
    print("- Adjust CI level with slider (50%-99%)")
    print("- Press 'q' to quit")
    print("- Left plot: Synthetic ETF Calls, Right plot: Synthetic ETF Puts")
    
    plt.show()


def plot_smile_with_correlation_ci_bands(ax, strikes, vols, f, t, sabr_params, all_data, weights, title_prefix="", ci_level=0.95):
    """Plot smile with SABR fit and correlation-based confidence bands with adjustable CI."""
    # Plot observed data
    ax.scatter(strikes, vols, color='blue', alpha=0.6, s=20, label='Observed')
    
    # Try SABR fit
    if sabr_params is not None:
        alpha, beta, rho, nu = sabr_params
        
        # Evaluate fit quality
        quality = evaluate_sabr_fit(strikes, vols, f, t, sabr_params)
        
        if quality is not None:
            # Plot fitted curve
            strike_range = np.linspace(strikes.min(), strikes.max(), 100)
            fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
            ax.plot(strike_range, fitted_vols, 'r-', label='SABR Fit', linewidth=2)
            
            # Add correlation-based confidence bands with adjustable CI
            add_correlation_confidence_bands_adjustable(ax, strikes, vols, f, t, sabr_params, all_data, weights, ci_level)
            
            # Add quality metrics
            quality_text = f'R²={quality["r_squared"]:.3f}\nMAE={quality["mae"]:.4f}\nMRE={quality["mre"]:.3f}'
            ax.text(0.02, 0.98, quality_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Add SABR parameters
            param_text = f'α={alpha:.3f}\nβ={beta:.1f}\nρ={rho:.3f}\nν={nu:.3f}'
            ax.text(0.02, 0.85, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add CI level info
            ci_text = f'CI Level: {ci_level*100:.0f}%'
            ax.text(0.02, 0.70, ci_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            return quality
        else:
            ax.text(0.02, 0.98, 'SABR fit failed', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    return None


def add_correlation_confidence_bands_adjustable(ax, strikes, vols, f, t, sabr_params, all_data, weights, confidence_level=0.95):
    """Add confidence bands based on correlation-weighted uncertainty with adjustable CI."""
    if sabr_params is None:
        return
    
    alpha, beta, rho, nu = sabr_params
    
    try:
        import numpy as np
        from scipy import stats
        
        # Calculate fitted volatilities
        fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strikes]
        fitted_vols = np.array(fitted_vols)
        
        # Calculate base residuals
        residuals = vols - fitted_vols
        base_std_error = np.std(residuals)
        
        # Adjust uncertainty based on correlation weights
        # Higher correlation = lower uncertainty (more confidence)
        avg_correlation = np.mean(list(weights.values()))
        correlation_factor = 1.0 - avg_correlation  # Higher correlation = lower uncertainty
        
        adjusted_std_error = base_std_error * (1.0 + correlation_factor)
        
        # Calculate confidence interval with adjustable level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * adjusted_std_error
        
        # Generate confidence bands
        strike_range = np.linspace(strikes.min(), strikes.max(), 100)
        fitted_range = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strike_range]
        
        upper_band = np.array(fitted_range) + margin_of_error
        lower_band = np.array(fitted_range) - margin_of_error
        
        # Plot confidence bands
        ax.fill_between(strike_range, lower_band, upper_band, alpha=0.3, color='orange', 
                       label=f'Correlation-Weighted {confidence_level*100:.0f}% Confidence')
        
        # Add correlation info
        ax.text(0.02, 0.55, f'Avg Correlation: {avg_correlation:.3f}\nAdjusted Std: {adjusted_std_error:.4f}\nCI Level: {confidence_level*100:.0f}%', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
    except Exception as e:
        print(f"Warning: Could not add correlation confidence bands: {e}")

def simple_sabr_stats(df):
    """Compute SABR parameters for each maturity in a single-ticker DataFrame."""
    import pandas as pd
    from compute_volatility import fit_sabr_smile
    import numpy as np
    
    results = []
    for T, group in df.groupby('T'):
        strikes = group['K'].values
        vols = group['sigma'].values
        f = strikes.mean() if len(strikes) > 0 else float('nan')
        sabr_params = fit_sabr_smile(strikes, vols, f, T)
        if sabr_params is not None:
            alpha, beta, rho, nu = sabr_params
        else:
            alpha = beta = rho = nu = float('nan')
        results.append({
            'T': T,
            'sigma_med': np.median(vols),
            'sigma_std': np.std(vols),
            'sigma_low': np.min(vols),
            'sigma_high': np.max(vols),
            'sabr_alpha': alpha,
            'sabr_beta': beta,
            'sabr_rho': rho,
            'sabr_nu': nu,
        })
    return pd.DataFrame(results)

def main():
    """Main demo function with GUI interface for ticker selection."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    import pandas as pd
    
    # Create GUI window
    root = tk.Tk()
    root.title("Volatility Surface Analysis - Ticker Selection")
    root.geometry("800x600")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Volatility Surface Analysis", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
    
    # Target Ticker Section
    ttk.Label(main_frame, text="Target Ticker:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
    target_ticker_var = tk.StringVar(value="IONQ")
    target_entry = ttk.Entry(main_frame, textvariable=target_ticker_var, width=20)
    target_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
    
    # Reference Tickers Section
    ttk.Label(main_frame, text="Reference Tickers (comma-separated):", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
    reference_tickers_var = tk.StringVar(value="ARQQ,FORM,QBTS,QTUM,QUBT,RGTI,SKYT,WIMI")
    reference_entry = ttk.Entry(main_frame, textvariable=reference_tickers_var, width=50)
    reference_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 5))
    
    # Group Selection Section
    ttk.Label(main_frame, text="Analysis Group:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
    group_var = tk.StringVar(value="tech_quantum")
    group_combo = ttk.Combobox(main_frame, textvariable=group_var, width=20)
    group_combo['values'] = [
        "tech_quantum", 
        "tech_ai", 
        "tech_blockchain", 
        "tech_semiconductor",
        "custom"
    ]
    group_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 5))
    
    # Preset Groups Info
    groups_info = """
    Preset Groups:
    • tech_quantum: Quantum computing and quantum tech
    • tech_ai: Artificial intelligence and machine learning
    • tech_blockchain: Blockchain and cryptocurrency
    • tech_semiconductor: Semiconductor and chip companies
    • custom: Use your own ticker list
    """
    info_label = ttk.Label(main_frame, text=groups_info, font=("Arial", 9), foreground="gray")
    info_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(10, 20))
    
    # Analysis Options
    ttk.Label(main_frame, text="Analysis Options:", font=("Arial", 12, "bold")).grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
    
    # Checkboxes for analysis types
    run_sabr_var = tk.BooleanVar(value=True)
    run_calls_puts_var = tk.BooleanVar(value=True)
    run_etf_var = tk.BooleanVar(value=True)
    run_interactive_var = tk.BooleanVar(value=True)
    
    ttk.Checkbutton(main_frame, text="SABR Fitting Analysis", variable=run_sabr_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
    ttk.Checkbutton(main_frame, text="Calls vs Puts Analysis", variable=run_calls_puts_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)
    ttk.Checkbutton(main_frame, text="Correlated ETF Analysis", variable=run_etf_var).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=2)
    ttk.Checkbutton(main_frame, text="Interactive Browsers", variable=run_interactive_var).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=10, column=0, columnspan=3, pady=(20, 0))
    
    def start_analysis():
        """Start the analysis with selected parameters."""
        target = target_ticker_var.get().strip().upper()
        reference_str = reference_tickers_var.get().strip()
        group = group_var.get()
        
        if not target:
            messagebox.showerror("Error", "Please enter a target ticker")
            return
        
        # Parse reference tickers
        if group == "custom":
            reference_tickers = [t.strip().upper() for t in reference_str.split(",") if t.strip()]
        else:
            # Use preset groups
            preset_groups = {
                "tech_quantum": ["IONQ", "ARQQ", "QBTS", "QUBT"],
                "tech_ai": ["FORM", "SKYT", "WIMI", "RGTI"],
                "tech_blockchain": ["QTUM", "QUBT", "RGTI"],
                "tech_semiconductor": ["IONQ", "QBTS", "SKYT"]
            }
            reference_tickers = preset_groups.get(group, [])
        
        if not reference_tickers:
            messagebox.showerror("Error", "Please enter reference tickers or select a valid group")
            return
        
        # Close GUI and start analysis
        root.destroy()
        
        # Run the analysis
        run_analysis_with_gui(target, reference_tickers, {
            'run_sabr': run_sabr_var.get(),
            'run_calls_puts': run_calls_puts_var.get(),
            'run_etf': run_etf_var.get(),
            'run_interactive': run_interactive_var.get()
        })
    
    def preview_tickers():
        """Preview the tickers that will be analyzed."""
        target = target_ticker_var.get().strip().upper()
        reference_str = reference_tickers_var.get().strip()
        group = group_var.get()
        
        if group == "custom":
            reference_tickers = [t.strip().upper() for t in reference_str.split(",") if t.strip()]
        else:
            preset_groups = {
                "tech_quantum": ["IONQ", "ARQQ", "QBTS", "QUBT"],
                "tech_ai": ["FORM", "SKYT", "WIMI", "RGTI"],
                "tech_blockchain": ["QTUM", "QUBT", "RGTI"],
                "tech_semiconductor": ["IONQ", "QBTS", "SKYT"]
            }
            reference_tickers = preset_groups.get(group, [])
        
        preview_text = f"Target: {target}\nReference: {', '.join(reference_tickers)}\nGroup: {group}"
        messagebox.showinfo("Ticker Preview", preview_text)
    
    def load_preset():
        """Load preset tickers based on selected group."""
        group = group_var.get()
        preset_groups = {
            "tech_quantum": ["IONQ", "ARQQ", "QBTS", "QUBT"],
            "tech_ai": ["FORM", "SKYT", "WIMI", "RGTI"],
            "tech_blockchain": ["QTUM", "QUBT", "RGTI"],
            "tech_semiconductor": ["IONQ", "QBTS", "SKYT"]
        }
        
        if group in preset_groups:
            reference_tickers_var.set(", ".join(preset_groups[group]))
    
    # Bind group selection to load preset
    group_combo.bind('<<ComboboxSelected>>', lambda e: load_preset())
    
    ttk.Button(button_frame, text="Preview Tickers", command=preview_tickers).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="Start Analysis", command=start_analysis).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="Cancel", command=root.destroy).pack(side=tk.LEFT)
    
    # Status bar
    status_var = tk.StringVar(value="Ready to analyze volatility surfaces")
    status_label = ttk.Label(main_frame, textvariable=status_var, font=("Arial", 9), foreground="blue")
    status_label.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))
    
    # Start GUI
    root.mainloop()


def run_analysis_with_gui(target_ticker, reference_tickers, options):
    """Run the analysis with parameters from GUI."""
    print(f"\n{'='*60}")
    print(f"VOLATILITY SURFACE ANALYSIS")
    print(f"{'='*60}")
    print(f"Target Ticker: {target_ticker}")
    print(f"Reference Tickers: {', '.join(reference_tickers)}")
    print(f"Analysis Options: {options}")
    print(f"{'='*60}\n")
    
    # Download target ticker data
    print(f"Downloading target ticker data for {target_ticker}...")
    target_df = download_option_data(target_ticker, max_expiries=8)
    
    if target_df is None or len(target_df) == 0:
        print(f"Error: No data available for target ticker {target_ticker}")
        return
    
    print(f"Target ticker data: {len(target_df)} options, {target_df['T'].nunique()} maturities")
    
    # Run SABR analysis if selected
    if options['run_sabr']:
        print(f"\n{'='*40}")
        print("SABR FITTING ANALYSIS")
        print(f"{'='*40}")
        
        # Compute SABR parameters
        result = simple_sabr_stats(target_df)
        
        if result is not None and len(result) > 0:
            print(f"Computed SABR parameters for {len(result)} maturities")
            
            # Test SABR fitting
            test_sabr_fitting(target_df, result)
            
            # Create SABR visualization
            visualize_surface(result, target_df, target_ticker)
        else:
            print("No valid SABR parameters computed")
    
    # Run calls vs puts analysis if selected
    if options['run_calls_puts']:
        print(f"\n{'='*40}")
        print("CALLS VS PUTS ANALYSIS")
        print(f"{'='*40}")
        analyze_calls_vs_puts(target_df)
    
    # Run correlated ETF analysis if selected
    if options['run_etf']:
        print(f"\n{'='*40}")
        print("CORRELATED ETF ANALYSIS")
        print(f"{'='*40}")
        
        # Download reference ticker data
        all_tickers = [target_ticker] + reference_tickers
        all_data = download_multiple_tickers(all_tickers, max_expiries=8)
        
        if all_data and len(all_data) > 1:
            # Compute correlation-based weights and get correlation matrix
            weights, correlation_matrix = compute_correlation_weights(all_data)
            
            # Create detailed correlation visualization
            print("Creating detailed correlation analysis...")
            create_correlation_scatter_plot(all_data, correlation_matrix)
            
            # Construct synthetic ETF
            synthetic_df = construct_synthetic_etf(all_data, weights)
            
            if synthetic_df is not None:
                print(f"Synthetic ETF constructed: {len(synthetic_df)} options")
                
                # Analyze synthetic ETF
                analyze_synthetic_etf(synthetic_df)
                
                # Run interactive browsers if selected
                if options['run_interactive']:
                    print(f"\n{'='*40}")
                    print("INTERACTIVE BROWSERS")
                    print(f"{'='*40}")
                    
                    # Interactive SABR browser for target ticker
                    print("Opening interactive SABR browser for target ticker...")
                    interactive_sabr_smile_browser(target_df)
                    
                    # Interactive correlated ETF browser
                    print("Opening interactive correlated ETF browser...")
                    interactive_correlated_etf_browser(all_data, synthetic_df, weights)
            else:
                print("Failed to construct synthetic ETF")
        else:
            print("Insufficient data for correlated ETF analysis")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


def analyze_synthetic_etf(synthetic_df):
    """Analyze the synthetic ETF data."""
    print(f"\nSynthetic ETF Analysis:")
    print(f"Total options: {len(synthetic_df)}")
    
    # Separate calls and puts
    calls_df = synthetic_df[synthetic_df['type'] == 'call']
    puts_df = synthetic_df[synthetic_df['type'] == 'put']
    
    print(f"Calls: {len(calls_df)} ({len(calls_df)/len(synthetic_df)*100:.1f}%)")
    print(f"Puts: {len(puts_df)} ({len(puts_df)/len(synthetic_df)*100:.1f}%)")
    
    print(f"Call maturities: {calls_df['T'].nunique()}")
    print(f"Put maturities: {puts_df['T'].nunique()}")
    
    # IV statistics
    print(f"\nCall IV stats:")
    print(f"  Mean: {calls_df['sigma'].mean():.3f}")
    print(f"  Std: {calls_df['sigma'].std():.3f}")
    print(f"  Min: {calls_df['sigma'].min():.3f}")
    print(f"  Max: {calls_df['sigma'].max():.3f}")
    
    print(f"\nPut IV stats:")
    print(f"  Mean: {puts_df['sigma'].mean():.3f}")
    print(f"  Std: {puts_df['sigma'].std():.3f}")
    print(f"  Min: {puts_df['sigma'].min():.3f}")
    print(f"  Max: {puts_df['sigma'].max():.3f}")
    
    # Strike ranges
    print(f"\nCall strike range: ${calls_df['K'].min():.2f} - ${calls_df['K'].max():.2f}")
    print(f"Put strike range: ${puts_df['K'].min():.2f} - ${puts_df['K'].max():.2f}")


if __name__ == "__main__":
    main()