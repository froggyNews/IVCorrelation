# display/plotting/vol_structure_plots.py
"""
Volatility structure plotting functions:
- ATM smile (IV vs TTE) 
- Term smile (IV vs Strike for fixed maturity)
- 3D surface integration with existing surface_viewer
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from pathlib import Path
import sys

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analysis_pipeline import get_smile_slice
from display.plotting.surface_viewer import show_surface_3d, show_surface_heatmap


def extract_atm_term_structure(ticker: str, asof: str, 
                              atm_tolerance: float = 0.05,
                              max_expiries: int = 12) -> pd.DataFrame:
    """
    Extract ATM implied volatility term structure.
    
    Args:
        ticker: Target ticker symbol
        asof: As-of date string
        atm_tolerance: Tolerance for ATM selection (K/S within 1±tolerance)
        max_expiries: Maximum number of expiries to include
        
    Returns:
        DataFrame with columns: ['expiry', 'T_years', 'atm_iv', 'atm_strike', 'spot']
    """
    try:
        # Get all smile data for the ticker
        df = get_smile_slice(ticker, asof, max_expiries=max_expiries)
        if df.empty:
            return pd.DataFrame()
        
        # Calculate moneyness
        df = df.copy()
        df['moneyness'] = df['K'] / df['S']
        
        # Filter for ATM options (moneyness close to 1.0)
        atm_mask = np.abs(df['moneyness'] - 1.0) <= atm_tolerance
        atm_df = df[atm_mask].copy()
        
        if atm_df.empty:
            return pd.DataFrame()
        
        # Group by expiry and select the closest to ATM for each expiry
        atm_by_expiry = []
        for expiry, group in atm_df.groupby('expiry'):
            # Find the row closest to moneyness = 1.0
            closest_idx = (group['moneyness'] - 1.0).abs().idxmin()
            closest_row = group.loc[closest_idx]
            
            atm_by_expiry.append({
                'expiry': expiry,
                'T_years': closest_row['T'],
                'atm_iv': closest_row['sigma'],
                'atm_strike': closest_row['K'],
                'spot': closest_row['S'],
                'moneyness': closest_row['moneyness']
            })
        
        result = pd.DataFrame(atm_by_expiry)
        
        # Sort by time to expiry
        result = result.sort_values('T_years').reset_index(drop=True)
        
        return result
        
    except Exception as e:
        print(f"Error extracting ATM term structure for {ticker}: {e}")
        return pd.DataFrame()


def plot_atm_term_structure(ax: plt.Axes, ticker: str, asof: str,
                           atm_tolerance: float = 0.05,
                           max_expiries: int = 12,
                           show_points: bool = True,
                           line_kwargs: Optional[Dict] = None) -> Dict:
    """
    Plot ATM implied volatility term structure (IV vs Time to Expiry).
    
    Args:
        ax: matplotlib axes
        ticker: Target ticker symbol  
        asof: As-of date string
        atm_tolerance: Tolerance for ATM selection
        max_expiries: Maximum number of expiries
        show_points: Whether to show individual data points
        line_kwargs: Additional arguments for line plotting
        
    Returns:
        Dict with plotting info and data
    """
    # Extract ATM term structure
    atm_data = extract_atm_term_structure(ticker, asof, atm_tolerance, max_expiries)
    
    if atm_data.empty:
        ax.text(0.5, 0.5, f"No ATM data for {ticker}", 
                ha="center", va="center", transform=ax.transAxes)
        return {"data": atm_data, "success": False}
    
    # Prepare plotting data
    T_years = atm_data['T_years'].to_numpy()
    iv = atm_data['atm_iv'].to_numpy()
    
    # Filter valid data
    valid = np.isfinite(T_years) & np.isfinite(iv)
    if not np.any(valid):
        ax.text(0.5, 0.5, f"No valid ATM data for {ticker}", 
                ha="center", va="center", transform=ax.transAxes)
        return {"data": atm_data, "success": False}
    
    T_years = T_years[valid]
    iv = iv[valid]
    
    # Plot line
    line_kwargs = line_kwargs or {}
    line_kwargs.setdefault("linewidth", 2.0)
    line_kwargs.setdefault("alpha", 0.8)
    line_kwargs.setdefault("label", f"{ticker} ATM Term Structure")
    
    ax.plot(T_years, iv, "-", **line_kwargs)
    
    # Plot points if requested
    if show_points:
        ax.scatter(T_years, iv, s=40, alpha=0.7, zorder=5, 
                  label="_nolegend_")
    
    # Formatting
    ax.set_xlabel("Time to Expiry (years)")
    ax.set_ylabel("ATM Implied Volatility")
    ax.set_title(f"ATM Term Structure - {ticker} ({asof})")
    ax.grid(True, alpha=0.3)
    
    # Legend
    if not ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best", fontsize=8)
    
    return {
        "data": atm_data,
        "success": True,
        "T_years": T_years,
        "iv": iv,
        "n_points": len(T_years)
    }


def extract_term_smile(ticker: str, asof: str, target_days: float,
                      tolerance_days: float = 7.0) -> pd.DataFrame:
    """
    Extract smile for a specific maturity (term smile: IV vs Strike).
    
    Args:
        ticker: Target ticker symbol
        asof: As-of date string  
        target_days: Target maturity in days
        tolerance_days: Tolerance for maturity matching
        
    Returns:
        DataFrame with smile data for the target maturity
    """
    try:
        # Get smile data
        df = get_smile_slice(ticker, asof)
        if df.empty:
            return pd.DataFrame()
        
        # Convert T to days for easier matching
        df = df.copy()
        df['T_days'] = df['T'] * 365.25
        
        # Filter for target maturity with tolerance
        maturity_mask = np.abs(df['T_days'] - target_days) <= tolerance_days
        term_df = df[maturity_mask].copy()
        
        if term_df.empty:
            return pd.DataFrame()
        
        # If multiple expiries within tolerance, pick the closest one
        if 'expiry' in term_df.columns:
            # Group by expiry and pick the one closest to target
            expiry_distances = term_df.groupby('expiry')['T_days'].first().apply(
                lambda x: abs(x - target_days)
            )
            closest_expiry = expiry_distances.idxmin()
            term_df = term_df[term_df['expiry'] == closest_expiry].copy()
        
        # Calculate moneyness and sort by strike
        term_df['moneyness'] = term_df['K'] / term_df['S']
        term_df = term_df.sort_values('K').reset_index(drop=True)
        
        return term_df
        
    except Exception as e:
        print(f"Error extracting term smile for {ticker}: {e}")
        return pd.DataFrame()


def plot_term_smile(ax: plt.Axes, ticker: str, asof: str, target_days: float,
                   tolerance_days: float = 7.0,
                   show_points: bool = True,
                   x_axis: str = "moneyness",  # "moneyness" or "strike"
                   line_kwargs: Optional[Dict] = None) -> Dict:
    """
    Plot term smile (IV vs Strike/Moneyness for fixed maturity).
    
    Args:
        ax: matplotlib axes
        ticker: Target ticker symbol
        asof: As-of date string
        target_days: Target maturity in days
        tolerance_days: Tolerance for maturity matching
        show_points: Whether to show individual data points
        x_axis: "moneyness" or "strike" for x-axis
        line_kwargs: Additional arguments for line plotting
        
    Returns:
        Dict with plotting info and data
    """
    # Extract term smile
    term_data = extract_term_smile(ticker, asof, target_days, tolerance_days)
    
    if term_data.empty:
        ax.text(0.5, 0.5, f"No data for {ticker} ~{int(target_days)}d", 
                ha="center", va="center", transform=ax.transAxes)
        return {"data": term_data, "success": False}
    
    # Prepare plotting data
    if x_axis == "moneyness":
        x_data = term_data['moneyness'].to_numpy()
        x_label = "Moneyness (K/S)"
    else:
        x_data = term_data['K'].to_numpy()
        x_label = "Strike"
    
    iv = term_data['sigma'].to_numpy()
    actual_days = term_data['T_days'].iloc[0] if len(term_data) > 0 else target_days
    
    # Filter valid data
    valid = np.isfinite(x_data) & np.isfinite(iv)
    if not np.any(valid):
        ax.text(0.5, 0.5, f"No valid smile data for {ticker}", 
                ha="center", va="center", transform=ax.transAxes)
        return {"data": term_data, "success": False}
    
    x_data = x_data[valid]
    iv = iv[valid]
    
    # Plot line
    line_kwargs = line_kwargs or {}
    line_kwargs.setdefault("linewidth", 2.0)
    line_kwargs.setdefault("alpha", 0.8)
    line_kwargs.setdefault("label", f"{ticker} ~{int(actual_days)}d")
    
    ax.plot(x_data, iv, "-", **line_kwargs)
    
    # Plot points if requested
    if show_points:
        ax.scatter(x_data, iv, s=40, alpha=0.7, zorder=5, 
                  label="_nolegend_")
    
    # Add ATM line if using moneyness
    if x_axis == "moneyness":
        ax.axvline(1.0, color="grey", linestyle="--", alpha=0.6, 
                  linewidth=1, label="_nolegend_")
    
    # Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Term Smile - {ticker} ~{int(actual_days)}d ({asof})")
    ax.grid(True, alpha=0.3)
    
    # Legend
    if not ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best", fontsize=8)
    
    return {
        "data": term_data,
        "success": True,
        "x_data": x_data,
        "iv": iv,
        "actual_days": actual_days,
        "n_points": len(x_data)
    }


def plot_3d_vol_surface(ticker: str, asof: str, 
                       mode: str = "3d",  # "3d" or "heatmap"
                       max_expiries: int = 12,
                       figsize: Tuple[float, float] = (10, 7),
                       save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot 3D volatility surface for a single ticker.
    
    Args:
        ticker: Target ticker symbol
        asof: As-of date string
        mode: "3d" for 3D surface or "heatmap" for 2D heatmap
        max_expiries: Maximum number of expiries
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure or None if failed
    """
    try:
        from analysis.compositeETFBuilder import build_surface_grids
        
        # Build surface grid
        surfaces = build_surface_grids(
            tickers=[ticker],
            use_atm_only=False,
            max_expiries=max_expiries
        )
        
        if ticker not in surfaces:
            print(f"No surface data available for {ticker}")
            return None
        
        # Get surface for the specified date
        asof_dt = pd.to_datetime(asof)
        if asof_dt not in surfaces[ticker]:
            print(f"No surface data for {ticker} on {asof_dt}")
            return None
        
        surface_df = surfaces[ticker][asof_dt]
        
        if surface_df.empty:
            print(f"Empty surface for {ticker} on {asof}")
            return None
        
        # Plot surface
        title = f"{ticker} Vol Surface ({asof_dt.date()})"
        
        if mode == "3d":
            fig = show_surface_3d(
                surface_df, 
                title=title,
                figsize=figsize,
                save_path=save_path
            )
        else:
            fig = show_surface_heatmap(
                surface_df,
                title=title, 
                figsize=figsize,
                save_path=save_path
            )
        
        # Debug: check if figure was created successfully
        if fig is None:
            print(f"Warning: {mode} surface plot returned None figure")
            return None
            
        # Ensure figure has proper attributes
        if not hasattr(fig, 'dpi'):
            print(f"Warning: Figure missing dpi attribute")
            fig.dpi = 100  # Set default DPI
            
        return fig
        
    except Exception as e:
        print(f"Error plotting 3D surface for {ticker}: {e}")
        return None


def create_vol_dashboard(ticker: str, asof: str,
                        target_days: float = 30.0,
                        figsize: Tuple[float, float] = (15, 10),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive volatility dashboard with multiple views.
    
    Args:
        ticker: Target ticker symbol
        asof: As-of date string
        target_days: Target days for term smile
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure with subplots
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    
    # Create subplots: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ATM term structure (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    atm_info = plot_atm_term_structure(ax1, ticker, asof)
    
    # Term smile (top right)  
    ax2 = fig.add_subplot(gs[0, 1])
    smile_info = plot_term_smile(ax2, ticker, asof, target_days)
    
    # 3D surface (bottom, spans both columns)
    ax3 = fig.add_subplot(gs[1, :], projection='3d')
    
    try:
        from analysis.compositeETFBuilder import build_surface_grids
        surfaces = build_surface_grids(tickers=[ticker], use_atm_only=False, max_expiries=12)
        
        if ticker in surfaces:
            asof_dt = pd.to_datetime(asof)
            if asof_dt in surfaces[ticker]:
                surface_df = surfaces[ticker][asof_dt]
                
                # Plot 3D surface on existing axes
                from display.plotting.surface_viewer import _surface_on_axes
                surf = _surface_on_axes(ax3, surface_df, f"{ticker} Vol Surface", "viridis")
                fig.colorbar(surf, ax=ax3, shrink=0.6, aspect=20)
            else:
                ax3.text(0.5, 0.5, 0.5, f"No surface data for {asof}", 
                        ha="center", va="center")
        else:
            ax3.text(0.5, 0.5, 0.5, f"No surface data for {ticker}", 
                    ha="center", va="center")
            
    except Exception as e:
        ax3.text(0.5, 0.5, 0.5, f"Error: {e}", ha="center", va="center")
    
    # Overall title
    fig.suptitle(f"Volatility Analysis Dashboard - {ticker} ({asof})", 
                 fontsize=16, fontweight='bold')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    
    return fig
