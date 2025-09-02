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


def _cols_to_days(cols) -> np.ndarray:
    """Convert column labels to days for tenor matching (robust).

    Accepts numeric, strings like '30d'/'30days', and pandas/NumPy timedeltas.
    Timestamps are returned as NaN (caller must resolve relative to asof).
    """
    out = []
    for c in cols:
        try:
            # pandas Timedelta or numpy timedelta64
            if hasattr(c, "components"):
                out.append(float(pd.to_timedelta(c).days))
            elif isinstance(c, (np.timedelta64,)):
                out.append(float(pd.to_timedelta(c).days))
            # pandas Timestamp / numpy datetime64: ambiguous without asof
            elif isinstance(c, (pd.Timestamp, np.datetime64)):
                out.append(np.nan)
            elif isinstance(c, str):
                s = c.strip().lower()
                if s.endswith("days"):
                    out.append(float(s[:-4]))
                elif s.endswith("d"):
                    out.append(float(s[:-1]))
                else:
                    out.append(float(s))
            else:
                out.append(float(c))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _nearest_tenor_idx(tenor_days, target_days):
    """Find index of nearest tenor to target."""
    tenor_days = np.asarray(tenor_days, dtype=float)
    valid = np.isfinite(tenor_days)
    if not np.any(valid):
        return 0
    diffs = np.abs(tenor_days[valid] - float(target_days))
    min_idx = np.argmin(diffs)
    return np.where(valid)[0][min_idx]


def _mny_from_index_labels(index) -> np.ndarray:
    """Extract numeric moneyness values from index labels (robust)."""
    out = []
    for lab in list(index):
        try:
            # pandas Interval or similar: use midpoint
            if hasattr(lab, "mid"):
                try:
                    out.append(float(getattr(lab, "mid")))
                    continue
                except Exception:
                    try:
                        out.append((float(lab.left) + float(lab.right)) / 2.0)
                        continue
                    except Exception:
                        pass
            if isinstance(lab, str):
                # normalize unicode dashes and spaces
                s = lab.strip().lower().replace("\u2013", "-").replace("\u2014", "-").replace(" ", "")
                # Handle range bins like "0.90-1.10" by midpoint
                if "-" in s and not s.startswith("m-"):
                    a, b = s.split("-", 1)
                    out.append((float(a) + float(b)) / 2.0)
                    continue
                if s.startswith("m"):
                    out.append(float(s[1:]))
                elif s.startswith("k/s="):
                    out.append(float(s.split("=", 1)[1]))
                else:
                    out.append(float(s))
            else:
                out.append(float(lab))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _lin_inpaint(y: np.ndarray) -> np.ndarray:
    """Simple linear inpaint of NaNs along a 1D vector."""
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0:
        return y
    m = np.isfinite(y)
    if m.sum() >= 2:
        x = np.arange(y.size, dtype=float)
        y[~m] = np.interp(x[~m], x[m], y[m])
    elif m.sum() == 1:
        y[~m] = y[m].item()
    return y


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
    moneyness_grid: Tuple[float, float, int] = (0.7, 1.4, 121),
    show_points: bool = True,
    beta: float = 0.5,              # SABR beta
    label: Optional[str] = None,
    line_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Plot observed points, model fit, and optional CI on moneyness (K/S).
    
    Supports SVI, SABR, and TPS models.
    Returns dict: {params, rmse, T, S}
    """

    # ---- safety check: ensure axes has valid figure
    if ax is None or ax.figure is None:
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": float(S)}

    # ---- sanitize
    S = float(S)
    K = _as_1d(K)
    iv = _as_1d(iv)

    m = _finite_mask(K, iv)
    if not np.any(m):
        # Ensure axes has a valid figure before adding text
        if ax.figure is not None:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": S}

    K = K[m]
    iv = iv[m]

    # ---- grid in strike space via moneyness
    mlo, mhi, n = moneyness_grid
    m_grid = np.linspace(float(mlo), float(mhi), int(n))
    K_grid = m_grid * S

    # ---- observed points
    if show_points:
        pts = ax.scatter(K / S, iv, s=20, alpha=0.85, label="Observed")

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

    # ---- target confidence bands disabled per UX (composite CI shown instead)
    if False and bands is not None:
        ci_fill = ax.fill_between(bands.x / S, bands.lo, bands.hi, alpha=0.20, label=f"{int(bands.level*100)}% CI")
        ci_mean = ax.plot(bands.x / S, bands.mean, lw=1.8, alpha=0.8, linestyle="--")

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

    # ---- fit quality
    rmse = float(fit_params.get("rmse", np.nan)) if isinstance(fit_params, dict) else np.nan

    return {
        "params": fit_params,
        "rmse": rmse,
        "T": float(T),
        "S": float(S),
    }


def plot_composite_index_smile(
    ax: plt.Axes,
    bands: Bands,
    *,
    label: Optional[str] = "composite index",
    line_kwargs: Optional[Dict] = None,
) -> Bands:
    """Plot composite index smile using pre-computed confidence bands."""

    ax.fill_between(bands.x, bands.lo, bands.hi, alpha=0.20, label=f"CI ({int(bands.level*100)}%)")
    # add a line for the mean - make it more defined/prominent
    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 2.5)  # Increased line width for better visibility
    line_kwargs.setdefault("alpha", 1.0)  # Full opacity
    line_kwargs.setdefault("linestyle", "-")  # Solid line
    ax.plot(bands.x, bands.mean, label=label, **line_kwargs)

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc="best", fontsize=8)

    return bands

# Back-compat alias (old name)
plot_composite_etf_smile = plot_composite_index_smile


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



def plot_composite_smile_overlay(ax: plt.Axes,
                                 target_grid: pd.DataFrame,
                                 composite_grid: pd.DataFrame,
                                 T_days: float,
                                 label: str = "Composite overlay",
                                 *,
                                 composite_surfaces: Optional[Dict[str, pd.DataFrame]] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 level: float = 0.68) -> bool:
    """
    Plot composite smile overlay on existing smile plot.
    
    Args:
        ax: matplotlib axes
        target_grid: Target ticker's surface grid
        composite_grid: composite/composite surface grid  
        T_days: Target tenor in days
        label: Label for the composite smile line
        
    Returns:
        bool: True if overlay was successfully plotted
    """
    try:
        def _dbg(name: str, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
            try:
                x = np.asarray(x, float)
                y_arr = None if y is None else np.asarray(y, float)
                fin = np.isfinite(x) & (np.isfinite(y_arr) if y_arr is not None else True)
                xr_lo = float(np.nanmin(x)) if x.size else np.nan
                xr_hi = float(np.nanmax(x)) if x.size else np.nan
                if y_arr is not None and y_arr.size:
                    yr_lo = float(np.nanmin(y_arr))
                    yr_hi = float(np.nanmax(y_arr))
                    y_part = f", yrange=[{yr_lo:.3f},{yr_hi:.3f}]"
                else:
                    y_part = ""
                print(f"DEBUG {name}: n={x.size}, finite={int(np.sum(fin))}, xrange=[{xr_lo:.3f},{xr_hi:.3f}]{y_part}")
            except Exception:
                pass
        # Ensure monotonic moneyness order for stability
        def _sort_by_mny(df: pd.DataFrame) -> pd.DataFrame:
            m = _mny_from_index_labels(df.index)
            try:
                order = np.argsort(m)
                return df.iloc[order]
            except Exception:
                return df
        target_grid = _sort_by_mny(target_grid)
        composite_grid = _sort_by_mny(composite_grid)
        # Determine tenor selection/interpolation
        tgt_cols = _cols_to_days(target_grid.columns)
        syn_cols = _cols_to_days(composite_grid.columns)
        i_tgt = _nearest_tenor_idx(tgt_cols, T_days)
        col_tgt = target_grid.columns[i_tgt]

        # SVI-based interpolation across tenor for compositeetic grid
        x_mny = _mny_from_index_labels(target_grid.index)
        syn_days = np.asarray(syn_cols, dtype=float)
        # Default: nearest column
        i_syn_near = _nearest_tenor_idx(syn_cols, T_days)
        col_syn_near = composite_grid.columns[i_syn_near]
        y_syn = composite_grid[col_syn_near].astype(float).to_numpy()

        # Try bracketing columns for interpolation
        try:
            ty = float(T_days) / 365.25
            # indices sorted by days
            order = np.argsort(syn_days)
            syn_days_sorted = syn_days[order]
            cols_sorted = [composite_grid.columns[i] for i in order]
            # find lower and upper neighbors
            lo_idx = np.searchsorted(syn_days_sorted, float(T_days), side="right") - 1
            hi_idx = lo_idx + 1
            if lo_idx >= 0 and hi_idx < len(syn_days_sorted):
                d_lo = syn_days_sorted[lo_idx]
                d_hi = syn_days_sorted[hi_idx]
                if d_hi > d_lo:
                    alpha = (float(T_days) - d_lo) / (d_hi - d_lo)
                    # Build compositeetic grid moneyness numeric
                    syn_mny = _mny_from_index_labels(composite_grid.index)
                    # Fetch IV vectors
                    y_lo = composite_grid[cols_sorted[lo_idx]].astype(float).to_numpy()
                    y_hi = composite_grid[cols_sorted[hi_idx]].astype(float).to_numpy()
                    # Inpaint NaNs to avoid propagation
                    y_lo = _lin_inpaint(y_lo)
                    y_hi = _lin_inpaint(y_hi)
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
        
        # Deterministic alignment: sort, overlap, interpolate to target grid or coarse grid
        syn_mny_all = _mny_from_index_labels(composite_grid.index)
        tgt_mny_all = _mny_from_index_labels(target_grid.index)
        # Ensure we have a y_syn from chosen/bracketed tenor
        if y_syn is None or (isinstance(y_syn, np.ndarray) and y_syn.size == 0):
            i_syn_near = _nearest_tenor_idx(syn_cols, T_days)
            y_syn = composite_grid.iloc[:, i_syn_near].astype(float).to_numpy()
        y_syn = _lin_inpaint(np.asarray(y_syn, float))

        valid_syn = np.isfinite(syn_mny_all) & np.isfinite(y_syn)
        syn_x = syn_mny_all[valid_syn]
        syn_y = y_syn[valid_syn]
        order = np.argsort(syn_x)
        syn_x = syn_x[order]
        syn_y = syn_y[order]

        valid_tgt = np.isfinite(tgt_mny_all)
        tgt_x = tgt_mny_all[valid_tgt]

        _dbg("syn_raw", syn_x, syn_y)
        _dbg("tgt_raw", tgt_x, None)

        MIN_POINTS = 5
        if syn_x.size >= 2 and tgt_x.size >= 2:
            lo = max(float(np.min(syn_x)), float(np.min(tgt_x)))
            hi = min(float(np.max(syn_x)), float(np.max(tgt_x)))
            if lo < hi:
                mask_eval = (tgt_x >= lo) & (tgt_x <= hi)
                tgt_eval = tgt_x[mask_eval]
                if tgt_eval.size >= MIN_POINTS:
                    f = interp1d(syn_x, syn_y, kind="linear", bounds_error=False,
                                 fill_value=(syn_y[0], syn_y[-1]))
                    y_eval = f(tgt_eval)
                    x_mny = tgt_eval
                    y_syn = y_eval
                else:
                    grid = np.linspace(lo, hi, max(MIN_POINTS, 21))
                    y_eval = np.interp(grid, syn_x, syn_y)
                    x_mny = grid
                    y_syn = y_eval
            else:
                print("DEBUG: composite overlay skipped (no moneyness overlap)")
                return False
        else:
            print(f"DEBUG: insufficient points pre-align: syn={syn_x.size}, tgt={tgt_x.size}")
            return False

        _dbg("eval", x_mny, y_syn)
        
        # Filter final valid data before plotting
        final_valid = np.isfinite(x_mny) & np.isfinite(y_syn)
        if np.sum(final_valid) >= 2:
            ax.plot(
                x_mny[final_valid],
                y_syn[final_valid],
                "-",
                linewidth=2.6,
                alpha=1.0,
                label=label,
                zorder=3,
            )
            # Optional composite CI using peer dispersion around weighted mean
            try:
                if composite_surfaces and weights:
                    # normalize weights across available peers
                    peers = [p for p in weights.keys() if p in composite_surfaces]
                    if len(peers) >= 2:
                        w_vals = np.array([float(weights[p]) for p in peers], dtype=float)
                        w_vals = np.where(np.isfinite(w_vals) & (w_vals > 0), w_vals, 0.0)
                        if w_vals.sum() > 0:
                            w = w_vals / w_vals.sum()
                            # helper: sample peer at nearest tenor and align to x_mny
                            def sample_peer(df_peer: pd.DataFrame) -> np.ndarray:
                                cols = _cols_to_days(df_peer.columns)
                                idx = _nearest_tenor_idx(cols, T_days)
                                col = df_peer.columns[idx]
                                if df_peer.index.equals(target_grid.index):
                                    y = df_peer[col].astype(float).to_numpy()
                                    return y
                                # align by interpolation to x_mny
                                try:
                                    pmny = _mny_from_index_labels(df_peer.index)
                                    piv = df_peer[col].astype(float).to_numpy()
                                    valid = np.isfinite(pmny) & np.isfinite(piv)
                                    if np.sum(valid) >= 2:
                                        f = interp1d(pmny[valid], piv[valid], kind='linear', bounds_error=False, fill_value=np.nan)
                                        return f(x_mny)
                                except Exception:
                                    pass
                                # fallback: NaNs
                                return np.full_like(x_mny, np.nan)

                            Y = np.vstack([sample_peer(composite_surfaces[p]) for p in peers])  # [n_peers x n_points]
                            # compute weighted mean (should match y_syn) and weighted std dev
                            mu = np.nansum((w[:, None] * Y), axis=0)
                            # population variance with weights: sum(w*(x-mu)^2)/sum(w)
                            var = np.nansum(w[:, None] * (Y - mu) ** 2, axis=0)
                            std = np.sqrt(np.maximum(var, 0.0))
                            # z-multiplier approx from level
                            if level >= 0.99:
                                z = 2.58
                            elif level >= 0.95:
                                z = 1.96
                            elif level >= 0.90:
                                z = 1.64
                            else:
                                z = 1.0
                            lo = mu - z * std
                            hi = mu + z * std
                            mask = final_valid & np.isfinite(lo) & np.isfinite(hi)
                            if np.sum(mask) >= 2:
                                ax.fill_between(x_mny[mask], lo[mask], hi[mask], alpha=0.20,
                                                label=f"Composite CI ({int(level*100)}%)")
            except Exception as e:
                print(f"DEBUG: composite CI failed: {e}")
            return True
        else:
            print("DEBUG: composite overlay skipped (insufficient valid points after alignment)")
    except Exception:
        print("DEBUG: composite overlay failed (exception)")
    
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
                             overlay_composite: bool = False, surfaces: Optional[Dict] = None,
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
        overlay_composite: Whether to overlay composite smile
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

    # Prefer fitting in-window to avoid tail-driven misfits
    try:
        m_vals_all = K / float(S)
    except Exception:
        m_vals_all = np.full_like(K, np.nan, dtype=float)
    mask_fit = (
        np.isfinite(m_vals_all)
        & np.isfinite(IV)
        & (m_vals_all >= float(m_grid.min()))
        & (m_vals_all <= float(m_grid.max()))
    )
    if np.count_nonzero(mask_fit) >= 5:
        K_fit = K[mask_fit]
        IV_fit = IV[mask_fit]
    else:
        K_fit = K
        IV_fit = IV
    
    # Fit all models with confidence bands
    ci_level = float(ci) if ci and ci > 0 else None
    all_results = fit_smile_models_with_bands(S, K_fit, T_used, IV_fit, K_grid, ci_level)
    
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
    )
    # Title and caption
    T_used_days = int(round(T_used * 365.25))
    title = f"{target}  {asof}  T≈{T_used:.3f}y (~{T_used_days}d)"
    
    # Ensure correct axis labels (fix for label pollution from term plots)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol")
    
    # Log parameters
    expiry_dt = None
    if "expiry" in dfe.columns and not dfe["expiry"].empty:
        expiry_dt = dfe["expiry"].iloc[0]
    
    last_fit_info = log_smile_parameters(asof, target, expiry_dt, all_results, S, T_used, dfe)
    
    # Add composite overlay if requested
    if overlay_composite and surfaces and weights and target in surfaces:
        try:
            import pandas as pd
            # Resolve date keys robustly (surfaces dict uses pd.Timestamp keys)
            asof_ts = pd.to_datetime(asof).normalize()

            def _resolve_key(dct):
                # map normalized day -> actual key
                norm_map = {pd.to_datetime(k).normalize(): k for k in dct.keys()}
                return norm_map.get(asof_ts, (max(dct.keys()) if dct else None))

            target_key = _resolve_key(surfaces.get(target, {}))
            # Build peer surfaces mapping (per-ticker DataFrame at target_key)
            peer_tickers = [t for t in weights.keys() if t in surfaces]
            peers_map = {}
            for t in peer_tickers:
                k = _resolve_key(surfaces[t])
                if k is not None and k in surfaces[t]:
                    peers_map[t.upper()] = {k: surfaces[t][k]}

            if peers_map and target_key is not None and target_key in surfaces[target]:
                from analysis.compositeETFBuilder import combine_surfaces
                composite_by_date = combine_surfaces(peers_map, weights)
                
                if target_key in composite_by_date:
                    target_grid = surfaces[target][target_key]
                    composite_grid = composite_by_date[target_key]
                    # Reduce to DataFrame map for CI sampling (only peers with target_key available)
                    peer_df_map = {t: d.get(target_key) for t, d in peers_map.items() if d.get(target_key) is not None}
                    
                    success = plot_composite_smile_overlay(
                        ax, target_grid, composite_grid, T_days,
                        label="Composite overlay",
                        composite_surfaces=peer_df_map,
                        weights=weights,
                        level=ci_level or 0.68
                    )
                    if success:
                        ax.legend(loc="best", fontsize=8)
                else:
                    print("DEBUG: composite overlay skipped (no composite grid for target date)")
            else:
                print("DEBUG: composite overlay skipped (no peer surfaces)")
                        

        except Exception:
            print("DEBUG: composite overlay failed (exception in combine or overlay)")
    
    ax.set_title(title)
    # Add RMSE as a small caption below the plot
    try:
        ax.text(0.5, -0.12, f"RMSE={info['rmse']:.4f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    except Exception:
        pass

    return info, last_fit_info
