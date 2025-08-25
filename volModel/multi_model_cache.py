"""Multi-model volatility fitting with aggressive caching.

This module provides cached volatility model fitting that computes and stores
all three models (SVI, SABR, TPS) when any one is requested, dramatically
reducing redundant computation in GUI interactions.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .sviFit import fit_svi_slice
from .sabrFit import fit_sabr_slice
from .polyFit import fit_tps_slice

# Import confidence band functions
try:
    from analysis.confidence_bands import (
        svi_confidence_bands,
        sabr_confidence_bands, 
        tps_confidence_bands,
    )
    _CONFIDENCE_BANDS_AVAILABLE = True
except ImportError:
    _CONFIDENCE_BANDS_AVAILABLE = False


def fit_all_models_cached(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    beta: float = 0.5,
    use_cache: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Fit all three volatility models (SVI, SABR, TPS) and cache the results.
    
    This function computes all models together to amortize the cost of data
    preparation and validation across all three fits.
    
    Parameters
    ----------
    S : float
        Spot price
    K : np.ndarray
        Strike prices
    T : float
        Time to expiry in years
    iv_obs : np.ndarray
        Observed implied volatilities
    beta : float, optional
        SABR beta parameter, by default 0.5
    use_cache : bool, optional
        Whether to use caching, by default True
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with keys 'svi', 'sabr', 'tps' containing model parameters
    """
    if not use_cache:
        return _fit_all_models_direct(S, K, T, iv_obs, beta)
    
    try:
        from analysis.model_params_logger import compute_or_load
        
        # Create cache payload
        payload = {
            "S": float(S),
            "K": K.tolist() if hasattr(K, 'tolist') else list(K),
            "T": float(T),
            "iv_obs": iv_obs.tolist() if hasattr(iv_obs, 'tolist') else list(iv_obs),
            "beta": float(beta),
        }
        
        def _builder():
            return _fit_all_models_direct(S, K, T, iv_obs, beta)
        
        return compute_or_load("multi_vol_models", payload, _builder, ttl_sec=3600)  # 1hr cache
    except Exception:
        # Fallback to direct computation if caching fails
        return _fit_all_models_direct(S, K, T, iv_obs, beta)


def _fit_all_models_direct(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    beta: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Direct computation of all three models without caching."""
    # Common data validation and preparation
    K = np.asarray(K, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)
    mask = np.isfinite(K) & np.isfinite(iv_obs)
    K_clean = K[mask]
    iv_clean = iv_obs[mask]
    
    results = {}
    
    # Fit SVI
    try:
        results['svi'] = fit_svi_slice(S, K_clean, T, iv_clean)
    except Exception:
        results['svi'] = {
            "a": np.nan, "b": np.nan, "rho": np.nan, "m": np.nan, "sigma": np.nan,
            "rmse": np.nan, "n": int(len(K_clean))
        }
    
    # Fit SABR
    try:
        results['sabr'] = fit_sabr_slice(S, K_clean, T, iv_clean, beta=beta)
    except Exception:
        results['sabr'] = {
            "alpha": np.nan, "beta": float(beta), "rho": np.nan, "nu": np.nan,
            "rmse": np.nan, "n": int(len(K_clean))
        }
    
    # Fit TPS
    try:
        results['tps'] = fit_tps_slice(S, K_clean, T, iv_clean)
    except Exception:
        results['tps'] = {
            "rmse": np.nan, "n": int(len(K_clean)), "coeffs": []
        }
    
    return results


def get_cached_model_params(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    model: str,
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    Get parameters for a specific model, using cached multi-model results if available.
    
    This is the main interface that should replace individual model fitting calls
    in the GUI to take advantage of multi-model caching.
    
    Parameters
    ----------
    S : float
        Spot price
    K : np.ndarray
        Strike prices
    T : float
        Time to expiry in years
    iv_obs : np.ndarray
        Observed implied volatilities
    model : str
        Model name: 'svi', 'sabr', or 'tps'
    beta : float, optional
        SABR beta parameter, by default 0.5
        
    Returns
    -------
    Dict[str, float]
        Model parameters for the requested model
    """
    model = model.lower()
    if model not in ['svi', 'sabr', 'tps']:
        raise ValueError(f"Unknown model: {model}. Must be 'svi', 'sabr', or 'tps'")
    
    # Get all models (from cache if available)
    all_results = fit_all_models_cached(S, K, T, iv_obs, beta)
    
    return all_results.get(model, {})


def invalidate_model_cache() -> None:
    """Clear the multi-model volatility cache."""
    try:
        from analysis.model_params_logger import clear_cache
        cleared = clear_cache("multi_vol_models")
        print(f"Cleared {cleared} multi-model cache entries")
    except Exception as e:
        print(f"Failed to clear model cache: {e}")


def get_cached_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    K_grid: np.ndarray,
    model: str,
    level: float = 0.68,
    beta: float = 0.5,
    use_cache: bool = True,
) -> Optional[Any]:
    """
    Get confidence bands for a specific model, using cache if available.
    
    Parameters
    ----------
    S : float
        Spot price
    K : np.ndarray
        Strike prices (observed data)
    T : float
        Time to expiry in years
    iv_obs : np.ndarray
        Observed implied volatilities
    K_grid : np.ndarray
        Grid of strike prices for confidence bands
    model : str
        Model name: 'svi', 'sabr', or 'tps'
    level : float, optional
        Confidence level (e.g., 0.68 for 68%), by default 0.68
    beta : float, optional
        SABR beta parameter, by default 0.5
    use_cache : bool, optional
        Whether to use caching, by default True
        
    Returns
    -------
    Optional[Any]
        Confidence bands object or None if computation fails
    """
    if not _CONFIDENCE_BANDS_AVAILABLE:
        return None
        
    model = model.lower()
    if model not in ['svi', 'sabr', 'tps']:
        return None
    
    if not use_cache:
        return _compute_confidence_bands_direct(S, K, T, iv_obs, K_grid, model, level, beta)
    
    try:
        from analysis.model_params_logger import compute_or_load
        
        # Create cache payload
        payload = {
            "S": float(S),
            "K": K.tolist() if hasattr(K, 'tolist') else list(K),
            "T": float(T),
            "iv_obs": iv_obs.tolist() if hasattr(iv_obs, 'tolist') else list(iv_obs),
            "K_grid": K_grid.tolist() if hasattr(K_grid, 'tolist') else list(K_grid),
            "model": model,
            "level": float(level),
            "beta": float(beta),
        }
        
        def _builder():
            return _compute_confidence_bands_direct(S, K, T, iv_obs, K_grid, model, level, beta)
        
        return compute_or_load("confidence_bands", payload, _builder, ttl_sec=3600)  # 1hr cache
    except Exception:
        # Fallback to direct computation if caching fails
        return _compute_confidence_bands_direct(S, K, T, iv_obs, K_grid, model, level, beta)


def _compute_confidence_bands_direct(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    K_grid: np.ndarray,
    model: str,
    level: float,
    beta: float,
) -> Optional[Any]:
    """Direct computation of confidence bands without caching."""
    if not _CONFIDENCE_BANDS_AVAILABLE:
        return None
    
    try:
        # Clean the data
        K = np.asarray(K, dtype=float)
        iv_obs = np.asarray(iv_obs, dtype=float)
        mask = np.isfinite(K) & np.isfinite(iv_obs)
        K_clean = K[mask]
        iv_clean = iv_obs[mask]
        
        if len(K_clean) < 3:
            return None
        
        # Compute confidence bands based on model
        if model == "svi":
            return svi_confidence_bands(S, K_clean, T, iv_clean, K_grid, level=level)
        elif model == "sabr":
            return sabr_confidence_bands(S, K_clean, T, iv_clean, K_grid, level=level)
        elif model == "tps":
            return tps_confidence_bands(S, K_clean, T, iv_clean, K_grid, level=level)
        else:
            return None
    except Exception:
        return None


def fit_all_models_with_bands_cached(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    K_grid: Optional[np.ndarray] = None,
    ci_level: Optional[float] = None,
    beta: float = 0.5,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Fit all models and compute confidence bands, using cache for both.
    
    This is an optimized version that computes models and confidence bands
    together when CI is requested, or just models when CI is not needed.
    
    Parameters
    ----------
    S : float
        Spot price
    K : np.ndarray
        Strike prices
    T : float
        Time to expiry in years
    iv_obs : np.ndarray
        Observed implied volatilities
    K_grid : np.ndarray, optional
        Grid for confidence bands. If None and ci_level is provided,
        a default grid will be created.
    ci_level : float, optional
        Confidence level for bands (e.g., 0.68). If None, no bands computed.
    beta : float, optional
        SABR beta parameter, by default 0.5
    use_cache : bool, optional
        Whether to use caching, by default True
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with model parameters and optionally confidence bands
    """
    # Get model parameters (cached)
    models = fit_all_models_cached(S, K, T, iv_obs, beta, use_cache)
    
    result = {"models": models}
    
    # Add confidence bands if requested
    if ci_level is not None and ci_level > 0:
        if K_grid is None:
            # Create default grid around current spot
            m_grid = np.linspace(0.7, 1.3, 121)
            K_grid = m_grid * S
        
        bands = {}
        for model_name in ['svi', 'sabr', 'tps']:
            try:
                band = get_cached_confidence_bands(
                    S, K, T, iv_obs, K_grid, model_name, ci_level, beta, use_cache
                )
                if band is not None:
                    bands[model_name] = band
            except Exception:
                pass
        
        if bands:
            result["bands"] = bands
    
    return result


def get_model_cache_stats() -> Dict[str, Any]:
    """Get statistics about the multi-model cache."""
    try:
        from analysis.model_params_logger import cache_stats, _load_cache_df
        
        stats = cache_stats()
        df = _load_cache_df()
        
        if df.empty:
            return {"total_entries": 0, "multi_model_entries": 0, "confidence_bands_entries": 0}
        
        multi_model_entries = len(df[df['kind'] == 'multi_vol_models'])
        bands_entries = len(df[df['kind'] == 'confidence_bands'])
        
        return {
            "total_entries": stats.get("entries", 0),
            "multi_model_entries": multi_model_entries,
            "confidence_bands_entries": bands_entries,
            "kinds": stats.get("kinds", []),
        }
    except Exception:
        return {"error": "Could not retrieve cache stats"}
