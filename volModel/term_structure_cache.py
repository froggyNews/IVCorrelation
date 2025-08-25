"""
Caching for term structure fits to improve GUI performance.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from analysis.model_params_logger import compute_or_load
from volModel.termFit import fit_term_structure, term_structure_iv


def _create_term_cache_key(T_data: np.ndarray, vol_data: np.ndarray) -> str:
    """Create a cache key for term structure data."""
    # Create a deterministic hash of the input data
    data_dict = {
        "T": T_data.tolist() if hasattr(T_data, 'tolist') else list(T_data),
        "vol": vol_data.tolist() if hasattr(vol_data, 'tolist') else list(vol_data)
    }
    data_str = json.dumps(data_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(data_str.encode()).hexdigest()


def fit_term_structure_cached(
    T_data: np.ndarray, 
    vol_data: np.ndarray,
    use_cache: bool = True,
    ttl_sec: int = 1800  # 30 minutes
) -> Dict[str, Any]:
    """
    Cached version of term structure fitting.
    
    Args:
        T_data: Time to expiry array
        vol_data: Implied volatility array
        use_cache: Whether to use caching
        ttl_sec: Time-to-live for cache entries
        
    Returns:
        Dict containing fit parameters and metadata
    """
    if not use_cache:
        return fit_term_structure(T_data, vol_data)
    
    # Create cache payload
    cache_key = _create_term_cache_key(T_data, vol_data)
    payload = {
        "cache_key": cache_key,
        "T_shape": T_data.shape,
        "vol_shape": vol_data.shape,
        "T_range": [float(T_data.min()), float(T_data.max())],
        "vol_range": [float(vol_data.min()), float(vol_data.max())]
    }
    
    def _builder() -> Dict[str, Any]:
        return fit_term_structure(T_data, vol_data)
    
    return compute_or_load("term_fit", payload, _builder, ttl_sec=ttl_sec)


def compute_term_fit_curve_cached(
    T_data: np.ndarray,
    vol_data: np.ndarray, 
    fit_points: int = 100,
    use_cache: bool = True,
    ttl_sec: int = 1800  # 30 minutes
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Cached computation of term structure fit curve.
    
    Args:
        T_data: Time to expiry array
        vol_data: Implied volatility array
        fit_points: Number of points for the fit curve
        use_cache: Whether to use caching
        ttl_sec: Time-to-live for cache entries
        
    Returns:
        Tuple of (fit_x, fit_y) arrays, or (None, None) if fit failed
    """
    if not use_cache:
        # Direct computation without caching
        try:
            params = fit_term_structure(T_data, vol_data)
            if params["coeff"].size > 0:
                fit_x = np.linspace(float(T_data.min()), float(T_data.max()), fit_points)
                fit_y = term_structure_iv(fit_x, params)
                return fit_x, fit_y
        except Exception:
            pass
        return None, None
    
    # Create cache payload
    cache_key = _create_term_cache_key(T_data, vol_data)
    payload = {
        "cache_key": cache_key,
        "fit_points": fit_points,
        "T_shape": T_data.shape,
        "vol_shape": vol_data.shape,
        "T_range": [float(T_data.min()), float(T_data.max())],
        "vol_range": [float(vol_data.min()), float(vol_data.max())]
    }
    
    def _builder() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            params = fit_term_structure(T_data, vol_data)
            if params["coeff"].size > 0:
                fit_x = np.linspace(float(T_data.min()), float(T_data.max()), fit_points)
                fit_y = term_structure_iv(fit_x, params)
                return fit_x, fit_y
        except Exception:
            pass
        return None, None
    
    return compute_or_load("term_curve", payload, _builder, ttl_sec=ttl_sec)


def get_cached_term_structure_data(
    atm_curve: pd.DataFrame,
    fit_points: int = 100,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Get cached term structure fit data for a given ATM curve.
    
    Args:
        atm_curve: DataFrame with 'T' and 'atm_vol' columns
        fit_points: Number of points for the fit curve
        use_cache: Whether to use caching
        
    Returns:
        Dict with 'params', 'fit_x', 'fit_y' keys
    """
    if atm_curve is None or atm_curve.empty:
        return {"params": None, "fit_x": None, "fit_y": None}
    
    try:
        T_data = atm_curve["T"].to_numpy(float)
        vol_data = atm_curve["atm_vol"].to_numpy(float)
        
        # Get cached parameters
        params = fit_term_structure_cached(T_data, vol_data, use_cache=use_cache)
        
        # Get cached fit curve
        fit_x, fit_y = compute_term_fit_curve_cached(
            T_data, vol_data, fit_points=fit_points, use_cache=use_cache
        )
        
        return {
            "params": params,
            "fit_x": fit_x,
            "fit_y": fit_y
        }
        
    except Exception:
        return {"params": None, "fit_x": None, "fit_y": None}
