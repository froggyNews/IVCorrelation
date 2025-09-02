# display/plotting/smile_plot.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from volModel.sviFit import svi_smile_iv, fit_svi_slice
from volModel.sabrFit import sabr_smile_iv, fit_sabr_slice
from volModel.polyFit import tps_smile_iv, fit_tps_slice
from volModel.multi_model_cache import fit_all_models_with_bands_cached
from analysis.confidence_bands import Bands, svi_confidence_bands, sabr_confidence_bands, tps_confidence_bands
from analysis.model_params_logger import append_params
from analysis.pillars import _fit_smile_get_atm
from display.plotting.anim_utils import  add_legend_toggles
import pandas as pd
from scipy.interpolate import interp1d

ModelName = Literal["svi", "sabr", "tps"]


# ------------------
# helpers
# ------------------
def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return a.ravel() if a.ndim > 1 else a


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    mask = None
    for a in arrs:
        a = np.asarray(a, float)
        m = np.isfinite(a)
        mask = m if mask is None else (mask & m)
    return mask if mask is not None else np.array([], dtype=bool)


def _cols_to_days(cols):
    """Convert column labels to days for tenor matching."""
    result = []
    for c in cols:
        try:
            if isinstance(c, str) and c.endswith('d'):
                result.append(float(c[:-1]))
            else:
                result.append(float(c))
        except (ValueError, TypeError):
            result.append(np.nan)
    return np.array(result)


def _nearest_tenor_idx(tenor_days, target_days):
    """Find index of nearest tenor to target."""
    tenor_days = np.asarray(tenor_days, dtype=float)
    valid = np.isfinite(tenor_days)
    if not np.any(valid):
        return 0
    diffs = np.abs(tenor_days[valid] - float(target_days))
    min_idx = np.argmin(diffs)
    return np.where(valid)[0][min_idx]


def _mny_from_index_labels(index):
    """Extract moneyness values from index labels."""
    result = []
    for label in index:
        try:
            if isinstance(label, str) and label.startswith('m'):
                result.append(float(label[1:]))
            else:
                result.append(float(label))
        except (ValueError, TypeError):
            result.append(np.nan)
    return np.array(result)


# ------------------
# main
# ------------------
def fit_and_plot_smile(
    ax: plt.Axes,
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    *,
    model: ModelName = "svi",
    params: Dict,
    bands: Optional[Bands] = None,
    moneyness_grid: Tuple[float, float, int] = (0.8, 1.2, 121),
    show_points: bool = True,
    beta: float = 0.5,              # SABR beta
    label: Optional[str] = None,
    line_kwargs: Optional[Dict] = None,
    enable_toggles: bool = True,       # legend/keyboard toggles (all models)
    use_checkboxes: bool = False,      # keep False by default; legend is primary
) -> Dict:
    """
    Plot observed points, model fit, and optional CI on moneyness (K/S).
    
    Supports SVI, SABR, and TPS models with interactive legend toggles.
    All operations are logged to .txt file via db_logger.
    Returns dict: {params, rmse, T, S, series_map or None}
    """

    # ---- safety check: ensure axes has valid figure
    if ax is None or ax.figure is None:
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": float(S), "series_map": None}

    # ---- sanitize
    S = float(S)
    K = _as_1d(K)
    iv = _as_1d(iv)

    m = _finite_mask(K, iv)
    if not np.any(m):
        # Ensure axes has a valid figure before adding text
        if ax.figure is not None:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": S, "series_map": None}

    K = K[m]
    iv = iv[m]

    # ---- grid in strike space via moneyness
    mlo, mhi, n = moneyness_grid
    m_grid = np.linspace(float(mlo), float(mhi), int(n))
    K_grid = m_grid * S

    # ---- artists map for legend toggles
    series_map: Dict[str, list] = {}

    # ---- observed points
    if show_points:
        pts = ax.scatter(K / S, iv, s=20, alpha=0.85, label="Observed")
        if enable_toggles:
            series_map["Observed Points"] = [pts]

    # ---- fit + optional CI
    if not params:
        raise ValueError("fit parameters must be provided")
    fit_params = params
    if model == "svi":
        y_fit = svi_smile_iv(S, K_grid, T, fit_params)
    elif model == "sabr":
        y_fit = sabr_smile_iv(S, K_grid, T, fit_params)
    else:
        y_fit = tps_smile_iv(S, K_grid, T, fit_params)

    # ---- fit line
    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 2 if show_points else 1.6)
    fit_lbl = label or (f"{model.upper()} fit")
    fit_line = ax.plot(K_grid / S, y_fit, label=fit_lbl, lw=2.2, alpha=1.0)  # More defined fit line
    if enable_toggles:
        series_map[f"{model.upper()} Fit"] = list(fit_line)

    # ---- confidence bands
    if bands is not None:
        ci_fill = ax.fill_between(bands.x / S, bands.lo, bands.hi, alpha=0.20, label=f"{int(bands.level*100)}% CI")
        ci_mean = ax.plot(bands.x / S, bands.mean, lw=1.8, alpha=0.8, linestyle="--")  # More defined CI mean line
        if enable_toggles:
            series_map[f"{model.upper()} Confidence Interval"] = [ci_fill, *ci_mean]

    # ---- ATM marker (not part of toggles / legend)
    ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.85, label="_nolegend_")

    # ---- axes / legend
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Implied Vol")
    if not ax.get_legend():
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best", fontsize=8)

    # ---- legend-first toggle system (primary), keyboard helpers
    if enable_toggles and series_map and ax.figure is not None:
        add_legend_toggles(ax, series_map)  # your improved legend system
        # checkboxes are optional; keep off unless explicitly asked


    # ---- fit quality
    rmse = float(fit_params.get("rmse", np.nan)) if isinstance(fit_params, dict) else np.nan

    return {
        "params": fit_params,
        "rmse": rmse,
        "T": float(T),
        "S": float(S),
        "series_map": series_map if enable_toggles else None,
    }


def plot_composite_etf_smile(
    ax: plt.Axes,
    bands: Bands,
    *,
    label: Optional[str] = "composite ETF",
    line_kwargs: Optional[Dict] = None,
) -> Bands:
    """Plot composite ETF smile using pre-computed confidence bands."""

    ax.fill_between(bands.x, bands.lo, bands.hi, alpha=0.20, label=f"CI ({int(bands.level*100)}%)")
    # add a line for the mean - make it more defined/prominent
    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 2.5)  # Increased line width for better visibility
    line_kwargs.setdefault("alpha", 1.0)  # Full opacity
    line_kwargs.setdefault("linestyle", "-")  # Solid line
    ax.plot(bands.x, bands.mean, label=label, **line_kwargs)

    ax.set_xlabel("Strike / Moneyness")
    ax.set_ylabel("Implied Vol")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc="best", fontsize=8)

    return bands


def fit_smile_models_with_bands(S: float, K: np.ndarray, T: float, IV: np.ndarray, 
                                 K_grid: np.ndarray, ci_level: Optional[float] = None) -> Dict:
    """
    Fit all smile models (SVI, SABR, TPS) with optional confidence bands.
    
    Returns dict with structure:
    {
        'models': {'svi': params, 'sabr': params, 'tps': params},
        'bands': {'svi': bands, 'sabr': bands, 'tps': bands}  # if ci_level provided
    }
    """
    result = {'models': {}, 'bands': {}}
    
    # Try cached multi-model fitting first
    try:
        all_results = fit_all_models_with_bands_cached(
            S, K, T, IV, K_grid=K_grid, ci_level=ci_level, use_cache=True
        )
        result['models'] = all_results.get('models', {})
        if ci_level and 'bands' in all_results:
            result['bands'] = all_results['bands']
        return result
    except Exception:
        pass
    
    # Fallback to individual fits
    try:
        result['models']['svi'] = fit_svi_slice(S, K, T, IV)
    except Exception:
        result['models']['svi'] = {}
        
    try:
        result['models']['sabr'] = fit_sabr_slice(S, K, T, IV)
    except Exception:
        result['models']['sabr'] = {}
        
    try:
        result['models']['tps'] = fit_tps_slice(S, K, T, IV)
    except Exception:
        result['models']['tps'] = {}
    
    # Fallback confidence bands computation
    if ci_level and ci_level > 0:
        try:
            result['bands']['svi'] = svi_confidence_bands(S, K, T, IV, K_grid, level=float(ci_level))
        except Exception:
            pass
            
        try:
            result['bands']['sabr'] = sabr_confidence_bands(S, K, T, IV, K_grid, level=float(ci_level))
        except Exception:
            pass
            
        try:
            result['bands']['tps'] = tps_confidence_bands(S, K, T, IV, K_grid, level=float(ci_level))
        except Exception:
            pass
    
    return result


def plot_composite_smile_overlay(
    ax: plt.Axes,
    target_grid: pd.DataFrame,
    synthetic_grid: pd.DataFrame,
    T_days: float,
    label: str = "composite smile (relative-weight)",
) -> bool:
    """
    Plot composite smile overlay on existing smile plot.
    
    Args:
        ax: matplotlib axes
        target_grid: Target ticker's surface grid
        synthetic_grid: Synthetic/composite surface grid  
        T_days: Target tenor in days
        label: Label for the composite smile line
        
    Returns:
        bool: True if overlay was successfully plotted
    """
    try:
        # Determine tenor selection/interpolation
        tgt_cols = _cols_to_days(target_grid.columns)
        syn_cols = _cols_to_days(synthetic_grid.columns)
        i_tgt = _nearest_tenor_idx(tgt_cols, T_days)
        col_tgt = target_grid.columns[i_tgt]

        # SVI-based interpolation across tenor for synthetic grid
        x_mny = _mny_from_index_labels(target_grid.index)
        syn_days = np.asarray(syn_cols, dtype=float)
        # Default: nearest column
        i_syn_near = _nearest_tenor_idx(syn_cols, T_days)
        col_syn_near = synthetic_grid.columns[i_syn_near]
        y_syn = synthetic_grid[col_syn_near].astype(float).to_numpy()

        # Try bracketing columns for interpolation
        try:
            ty = float(T_days) / 365.25
            # indices sorted by days
            order = np.argsort(syn_days)
            syn_days_sorted = syn_days[order]
            cols_sorted = [synthetic_grid.columns[i] for i in order]
            # find lower and upper neighbors
            lo_idx = np.searchsorted(syn_days_sorted, float(T_days), side="right") - 1
            hi_idx = lo_idx + 1
            if lo_idx >= 0 and hi_idx < len(syn_days_sorted):
                d_lo = syn_days_sorted[lo_idx]
                d_hi = syn_days_sorted[hi_idx]
                if d_hi > d_lo:
                    alpha = (float(T_days) - d_lo) / (d_hi - d_lo)
                    # Build synthetic grid moneyness numeric
                    syn_mny = _mny_from_index_labels(synthetic_grid.index)
                    # Fetch IV vectors
                    y_lo = synthetic_grid[cols_sorted[lo_idx]].astype(float).to_numpy()
                    y_hi = synthetic_grid[cols_sorted[hi_idx]].astype(float).to_numpy()
                    # Fit SVI on each neighbor and interpolate params
                    from volModel.sviFit import fit_svi_slice, svi_smile_iv
                    S = 1.0
                    T_lo = float(d_lo) / 365.25
                    T_hi = float(d_hi) / 365.25
                    mask_lo = np.isfinite(syn_mny) & np.isfinite(y_lo)
                    mask_hi = np.isfinite(syn_mny) & np.isfinite(y_hi)
                    if np.sum(mask_lo) >= 5 and np.sum(mask_hi) >= 5:
                        p_lo = fit_svi_slice(S, syn_mny[mask_lo] * S, T_lo, y_lo[mask_lo])
                        p_hi = fit_svi_slice(S, syn_mny[mask_hi] * S, T_hi, y_hi[mask_hi])
                        p_int = (1.0 - alpha) * np.asarray(p_lo, dtype=float) + alpha * np.asarray(p_hi, dtype=float)
                        y_syn_full = svi_smile_iv(S, syn_mny * S, ty, p_int)
                        y_syn = np.asarray(y_syn_full, dtype=float)
                    else:
                        # Fallback to linear IV interpolation in tenor
                        y_syn = (1.0 - alpha) * y_lo + alpha * y_hi
        except Exception:
            pass
        
        # Improved grid alignment for composite smile
        if not target_grid.index.equals(synthetic_grid.index):
            try:
                syn_mny = _mny_from_index_labels(synthetic_grid.index)
                syn_iv = synthetic_grid[col_syn].astype(float).to_numpy()
                
                syn_valid = np.isfinite(syn_mny) & np.isfinite(syn_iv)
                tgt_valid = np.isfinite(x_mny)
                
                if np.sum(syn_valid) >= 2 and np.sum(tgt_valid) >= 2:
                    syn_mny_clean = syn_mny[syn_valid]
                    syn_iv_clean = syn_iv[syn_valid]
                    tgt_mny_clean = x_mny[tgt_valid]
                    
                    # Interpolate within range
                    min_syn = np.min(syn_mny_clean)
                    max_syn = np.max(syn_mny_clean)
                    interp_mask = (tgt_mny_clean >= min_syn) & (tgt_mny_clean <= max_syn)
                    
                    if np.sum(interp_mask) >= 2:
                        tgt_mny_interp = tgt_mny_clean[interp_mask]
                        f_interp = interp1d(syn_mny_clean, syn_iv_clean,
                                          kind='linear', bounds_error=False, fill_value=np.nan)
                        syn_iv_interp = f_interp(tgt_mny_interp)
                        x_mny = tgt_mny_interp
                        y_syn = syn_iv_interp
                        
            except (ImportError, Exception):
                # Fallback to intersection-based alignment
                common = target_grid.index.intersection(synthetic_grid.index)
                if len(common) >= 3:
                    x_mny = _mny_from_index_labels(common)
                    y_syn = synthetic_grid.loc[common, col_syn].astype(float).to_numpy()
        
        # Filter final valid data before plotting
        final_valid = np.isfinite(x_mny) & np.isfinite(y_syn)
        if np.sum(final_valid) >= 2:
            ax.plot(
                x_mny[final_valid],
                y_syn[final_valid],
                "--",
                linewidth=1.6,
                alpha=0.95,
                label=label,
            )
            return True
            
    except Exception:
        pass
    
    return False


def log_smile_parameters(asof: str, target: str, expiry_dt, model_results: Dict, 
                        S: float, T_used: float, df: pd.DataFrame) -> Optional[Dict]:
    """
    Log fitted smile parameters for all models and compute sensitivities.
    
    Args:
        asof: As-of date string
        target: Target ticker
        expiry_dt: Expiry datetime
        model_results: Results from fit_smile_models_with_bands
        S: Spot price
        T_used: Time to expiry in years
        df: Original dataframe with smile data
        
    Returns:
        Dict with fit info for storage, or None if failed
    """
    try:
        models = model_results.get('models', {})
        svi_params = models.get('svi', {})
        sabr_params = models.get('sabr', {})
        tps_params = models.get('tps', {})
        
        # Compute sensitivities
        dfe2 = df.copy()
        dfe2["moneyness"] = dfe2["K"].astype(float) / float(S)
        sens = _fit_smile_get_atm(dfe2, model="auto")
        sens_params = {k: sens[k] for k in ("atm_vol", "skew", "curv") if k in sens}
        
        # Log all parameters
        append_params(
            asof_date=asof,
            ticker=target,
            expiry=str(expiry_dt) if expiry_dt is not None else None,
            model="svi",
            params=svi_params,
            meta={"rmse": svi_params.get("rmse")},
        )
        append_params(
            asof_date=asof,
            ticker=target,
            expiry=str(expiry_dt) if expiry_dt is not None else None,
            model="sabr",
            params=sabr_params,
            meta={"rmse": sabr_params.get("rmse")},
        )
        append_params(
            asof_date=asof,
            ticker=target,
            expiry=str(expiry_dt) if expiry_dt is not None else None,
            model="tps",
            params=tps_params,
            meta={"rmse": tps_params.get("rmse")},
        )
        append_params(
            asof_date=asof,
            ticker=target,
            expiry=str(expiry_dt) if expiry_dt is not None else None,
            model="sens",
            params=sens_params,
        )
        
        fit_map = {
            float(T_used): {
                "expiry": str(expiry_dt.date()) if expiry_dt is not None else None,
                "svi": svi_params,
                "sabr": sabr_params,
                "tps": tps_params,
                "sens": sens_params,
            }
        }
        
        return {
            "ticker": target,
            "asof": asof,
            "fit_by_expiry": fit_map,
        }
        
    except Exception:
        return None


def plot_smile_with_composite(ax: plt.Axes, df: pd.DataFrame, target: str, asof: str, 
                             model: ModelName, T_days: float, ci: Optional[float] = None,
                             overlay_synth: bool = False, surfaces: Optional[Dict] = None,
                             weights: Optional[Dict] = None) -> Tuple[Dict, Optional[Dict]]:
    """
    Complete smile plotting function with model fitting, confidence bands, and composite overlay.
    
    This is the main function that can replace most of the logic in gui_plot_manager._plot_smile.
    
    Args:
        ax: matplotlib axes
        df: DataFrame with smile data (columns: S, K, sigma, T, expiry)
        target: Target ticker symbol
        asof: As-of date string
        model: Model to use for fitting ('svi', 'sabr', 'tps')
        T_days: Target tenor in days
        ci: Confidence interval level (0-1), None to disable
        overlay_synth: Whether to overlay composite smile
        surfaces: Pre-computed surface grids for composite
        weights: Weights for composite construction
        
    Returns:
        Tuple of (fit_info_dict, last_fit_info_dict)
    """
    # Prepare data
    dfe = df.copy()
    S = float(dfe["S"].median())
    K = dfe["K"].to_numpy(float)
    IV = dfe["sigma"].to_numpy(float)
    T_used = float(dfe["T"].median())
    
    # Create moneyness grid
    m_grid = np.linspace(0.7, 1.3, 121)
    K_grid = m_grid * S
    
    # Fit all models with confidence bands
    ci_level = float(ci) if ci and ci > 0 else None
    all_results = fit_smile_models_with_bands(S, K, T_used, IV, K_grid, ci_level)
    
    # Get specific model parameters and bands
    fit_params = all_results['models'].get(model, {})
    bands = all_results['bands'].get(model) if ci_level else None
    
    # Plot the main smile
    info = fit_and_plot_smile(
        ax,
        S=S,
        K=K,
        T=T_used,
        iv=IV,
        model=model,
        params=fit_params,
        bands=bands,
        moneyness_grid=(0.7, 1.3, 121),
        show_points=True,
        enable_toggles=True,
    )
    
    # Set title
    title = f"{target}  {asof}  T≈{T_used:.3f}y  RMSE={info['rmse']:.4f}"
    
    # Ensure correct axis labels (fix for label pollution from term plots)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol")
    
    # Log parameters
    expiry_dt = None
    if "expiry" in dfe.columns and not dfe["expiry"].empty:
        expiry_dt = dfe["expiry"].iloc[0]
    
    last_fit_info = log_smile_parameters(asof, target, expiry_dt, all_results, S, T_used, dfe)
    
    # Add composite overlay if requested
    if overlay_synth and surfaces and weights and target in surfaces:
        try:
            import pandas as pd
            # Resolve date keys robustly (surfaces dict uses pd.Timestamp keys)
            asof_ts = pd.to_datetime(asof).normalize()

            def _resolve_key(dct):
                # map normalized day -> actual key
                norm_map = {pd.to_datetime(k).normalize(): k for k in dct.keys()}
                return norm_map.get(asof_ts, (max(dct.keys()) if dct else None))

            target_key = _resolve_key(surfaces.get(target, {}))
            # Build peer surfaces mapping with per-ticker resolved keys
            peer_tickers = [t for t in weights.keys() if t in surfaces]
            peers_map = {}
            for t in peer_tickers:
                k = _resolve_key(surfaces[t])
                if k is not None and k in surfaces[t]:
                    # combine_surfaces expects dict[ticker][date]->DF
                    peers_map[t.upper()] = {k: surfaces[t][k]}

            if peers_map and target_key is not None and target_key in surfaces[target]:
                from analysis.compositeETFBuilder import combine_surfaces
                synth_by_date = combine_surfaces(peers_map, weights)

                # Choose date to use: prefer target_key if available; else latest in synth
                date_used = target_key if target_key in synth_by_date else (max(synth_by_date.keys()) if synth_by_date else None)
                if date_used is not None and date_used in surfaces[target]:
                    target_grid = surfaces[target][date_used]
                    synthetic_grid = synth_by_date.get(date_used)
                    if synthetic_grid is not None:
                        success = plot_composite_smile_overlay(ax, target_grid, synthetic_grid, T_days)
                        if success:
                            ax.legend(loc="best", fontsize=8)
        except Exception:
            pass
    
    ax.set_title(title)
    
    return info, last_fit_info

