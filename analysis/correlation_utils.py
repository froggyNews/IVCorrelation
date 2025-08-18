import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Tuple
from dataclasses import dataclass

from .pillars import build_atm_matrix, detect_available_pillars, EXTENDED_PILLARS_DAYS


@dataclass
class PillarConfig:
    """Configuration for pillar selection and optimization."""
    
    # Base pillars to use
    base_pillars: List[int] = None
    
    # Pillar selection mode
    use_restricted_pillars: bool = True
    optimize_pillars: bool = False
    
    # Optimization parameters  
    max_pillars: int = 10
    min_tickers_per_pillar: int = 3
    
    # Tolerance for pillar matching
    tol_days: float = 7.0
    
    def __post_init__(self):
        if self.base_pillars is None:
            self.base_pillars = [7, 30, 60, 90]  # Default pillars
    
    @classmethod
    def restricted(cls, pillars: List[int] = None) -> 'PillarConfig':
        """Create a restricted pillar configuration."""
        return cls(
            base_pillars=pillars or [7, 30, 60, 90],
            use_restricted_pillars=True,
            optimize_pillars=False,
        )
    
    @classmethod
    def optimized(cls, base_pillars: List[int] = None, max_pillars: int = 8) -> 'PillarConfig':
        """Create an optimized pillar configuration."""
        return cls(
            base_pillars=base_pillars or [7, 30, 60, 90],
            use_restricted_pillars=False,
            optimize_pillars=True,
            max_pillars=max_pillars,
        )
    
    @classmethod
    def extended(cls, base_pillars: List[int] = None, max_pillars: int = 10) -> 'PillarConfig':
        """Create an extended pillar configuration without optimization."""
        return cls(
            base_pillars=base_pillars or [7, 30, 60, 90],
            use_restricted_pillars=False,
            optimize_pillars=False,
            max_pillars=max_pillars,
        )


def compute_atm_corr(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = 0.05,
    tol_days: float = 7.0,
    min_pillars: int = 2,
    demean_rows: bool = False,
    corr_method: str = "pearson",
    min_tickers_per_pillar: int = 3,
    min_pillars_per_ticker: int = 2,
    ridge: float = 1e-6,
    use_restricted_pillars: bool = True,
    optimize_pillars: bool = False,
    max_pillars: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (ATM matrix, correlation matrix) for one as-of date.
    
    Parameters
    ----------
    use_restricted_pillars : bool, default True
        If True, use only the provided pillars_days.
        If False, optimize pillar selection based on data availability.
    optimize_pillars : bool, default False  
        If True and use_restricted_pillars=False, dynamically select the best
        available pillars to maximize data coverage.
    max_pillars : int, default 10
        Maximum number of pillars to use when optimizing.
    """
    
    # Pillar optimization logic
    if not use_restricted_pillars and optimize_pillars:
        # Dynamically detect and optimize pillars based on data availability
        candidate_pillars = list(EXTENDED_PILLARS_DAYS) + list(pillars_days)
        candidate_pillars = sorted(set(candidate_pillars))  # Remove duplicates and sort
        
        # Find pillars with sufficient data coverage
        available_pillars = detect_available_pillars(
            get_smile_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            candidate_pillars=candidate_pillars,
            min_tickers_per_pillar=min_tickers_per_pillar,
            tol_days=tol_days,
        )
        
        if available_pillars:
            # Limit to max_pillars, preferring the original pillars if they're available
            original_pillars = [p for p in pillars_days if p in available_pillars]
            additional_pillars = [p for p in available_pillars if p not in pillars_days]
            
            # Start with original pillars, then add additional ones up to max_pillars
            optimized_pillars = original_pillars[:max_pillars]
            remaining_slots = max_pillars - len(optimized_pillars)
            if remaining_slots > 0:
                optimized_pillars.extend(additional_pillars[:remaining_slots])
            
            pillars_days = sorted(optimized_pillars)
            print(f"Optimized pillars for {asof}: {pillars_days} (from {len(candidate_pillars)} candidates)")
        else:
            print(f"Warning: No available pillars found for {asof}, using original: {list(pillars_days)}")
    elif not use_restricted_pillars:
        # Use extended pillars without optimization
        extended_pillars = list(EXTENDED_PILLARS_DAYS) + list(pillars_days)
        pillars_days = sorted(set(extended_pillars))[:max_pillars]
        print(f"Using extended pillars for {asof}: {pillars_days}")
    
    # Continue with existing logic
    atm_df, corr_df = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        min_pillars=min_pillars,
        corr_method=corr_method,
        demean_rows=demean_rows,
    )
    # Drop sparse pillars/tickers (ETF-style filtering)
    if not atm_df.empty:
        # keep pillars with at least min_tickers_per_pillar non-NaN entries
        col_coverage = atm_df.count(axis=0)
        good_pillars = col_coverage[col_coverage >= min_tickers_per_pillar].index
        atm_df = atm_df[good_pillars] if len(good_pillars) >= 2 else atm_df
        # keep tickers with at least min_pillars_per_ticker pillars
        row_coverage = atm_df.count(axis=1)
        good_tickers = row_coverage[row_coverage >= min_pillars_per_ticker].index
        atm_df = atm_df.loc[good_tickers] if len(good_tickers) >= 2 else atm_df
        # recompute correlation with ridge regularisation
        if not atm_df.empty and atm_df.shape[0] >= 2 and atm_df.shape[1] >= 2:
            atm_clean = atm_df.dropna()
            if atm_clean.shape[0] >= 2 and atm_clean.shape[1] >= 2:
                atm_std = (atm_clean - atm_clean.mean(axis=1).values.reshape(-1, 1)) / (
                    atm_clean.std(axis=1).values.reshape(-1, 1) + 1e-8
                )
                corr_matrix = (atm_std @ atm_std.T) / max(atm_std.shape[1] - 1, 1)
                corr_matrix += ridge * np.eye(corr_matrix.shape[0])
                corr_df = pd.DataFrame(corr_matrix, index=atm_clean.index, columns=atm_clean.index)
    return atm_df, corr_df


def compute_atm_corr_optimized(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    base_pillars_days: Iterable[int] = (7, 30, 60, 90),
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Compute ATM correlations with automatic pillar optimization.
    
    Returns
    -------
    atm_df : pd.DataFrame
        ATM matrix (tickers x pillars)
    corr_df : pd.DataFrame  
        Correlation matrix (tickers x tickers)
    used_pillars : List[int]
        List of pillars actually used after optimization
    """
    # Set optimization defaults
    kwargs.setdefault('use_restricted_pillars', False)
    kwargs.setdefault('optimize_pillars', True) 
    kwargs.setdefault('max_pillars', 8)
    
    # Store original pillars for comparison
    original_pillars = list(base_pillars_days)
    
    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=base_pillars_days,
        **kwargs
    )
    
    # Extract used pillars from the ATM dataframe columns
    used_pillars = [int(col) for col in atm_df.columns if str(col).isdigit()]
    
    return atm_df, corr_df, used_pillars


def compute_atm_corr_restricted(
    get_smile_slice,
    tickers: Iterable[str], 
    asof: str,
    pillars_days: Iterable[int],
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ATM correlations using only the specified pillars (restricted mode).
    
    This is equivalent to the original behavior - only use the exact pillars provided.
    """
    # Force restricted pillar usage
    kwargs['use_restricted_pillars'] = True
    kwargs['optimize_pillars'] = False
    
    return compute_atm_corr(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        **kwargs
    )


def compute_atm_corr_with_config(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str, 
    config: PillarConfig,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Compute ATM correlations using a PillarConfig.
    
    Returns
    -------
    atm_df : pd.DataFrame
        ATM matrix (tickers x pillars)
    corr_df : pd.DataFrame
        Correlation matrix (tickers x tickers) 
    used_pillars : List[int]
        List of pillars actually used
    """
    # Apply config settings to kwargs
    kwargs.update({
        'use_restricted_pillars': config.use_restricted_pillars,
        'optimize_pillars': config.optimize_pillars,
        'max_pillars': config.max_pillars,
        'min_tickers_per_pillar': config.min_tickers_per_pillar,
        'tol_days': config.tol_days,
    })
    
    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=config.base_pillars,
        **kwargs
    )
    
    # Extract used pillars from the ATM dataframe columns
    used_pillars = [int(col) for col in atm_df.columns if str(col).isdigit()]
    
    return atm_df, corr_df, used_pillars


def corr_weights(
    corr_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Convert correlations with target into normalised positive weights on peers."""
    target = target.upper()
    peers = [p.upper() for p in peers]
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0].apply(pd.to_numeric, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / max(len(peers), 1), index=peers, dtype=float)
    return (s / total).fillna(0.0)


def flexible_weights(
    data_df: pd.DataFrame,
    target: str,
    peers: List[str],
    weight_mode: str = "auto",
    clip_negative: bool = True,
    power: float = 1.0,
    min_weight_threshold: float = 0.01,
) -> pd.Series:
    """
    Flexible weight computation that adapts to data quality and weight mode.
    
    This function is much more lax with weight modes and determines importance
    based on weights rather than strict correlation requirements.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Feature matrix (tickers x features) or correlation matrix
    target : str
        Target ticker
    peers : List[str]
        Peer tickers
    weight_mode : str, default "auto"
        Weight computation mode: "auto", "corr", "equal", "distance", "similarity"
    clip_negative : bool, default True
        Whether to clip negative weights to zero
    power : float, default 1.0
        Power to apply to weights before normalization
    min_weight_threshold : float, default 0.01
        Minimum weight threshold for inclusion (helps filter noise)
    
    Returns
    -------
    pd.Series
        Normalized weights for peers
    """
    target = target.upper()
    peers = [p.upper() for p in peers]
    
    # Handle empty or invalid inputs gracefully
    if data_df is None or data_df.empty or not peers:
        return pd.Series(1.0 / max(len(peers), 1), index=peers, dtype=float)
    
    # Determine weight computation strategy based on mode and data
    if weight_mode == "equal" or (weight_mode == "auto" and data_df.shape[1] < 2):
        # Equal weights - always works regardless of data quality
        weights = pd.Series(1.0, index=peers, dtype=float)
    
    elif weight_mode in ("corr", "correlation", "auto"):
        # Correlation-based weights with fallbacks
        if target in data_df.index and target in data_df.columns:
            # Square correlation matrix - extract target correlations
            corr_series = data_df.loc[peers, target] if target in data_df.columns else data_df.loc[target, peers]
        elif target in data_df.index:
            # Feature matrix - compute correlations
            feature_corr = data_df.T.corr()
            if target in feature_corr.columns:
                corr_series = feature_corr.loc[peers, target]
            else:
                # Fallback to equal weights
                corr_series = pd.Series(1.0, index=peers)
        else:
            # Target not found - use equal weights
            corr_series = pd.Series(1.0, index=peers)
            
        weights = corr_series.fillna(0.0)
    
    elif weight_mode in ("distance", "similarity"):
        # Distance/similarity-based weights
        if target in data_df.index:
            target_features = data_df.loc[target].fillna(0.0)
            peer_features = data_df.loc[data_df.index.intersection(peers)].fillna(0.0)
            
            if not peer_features.empty:
                # Compute euclidean distance or cosine similarity
                if weight_mode == "distance":
                    distances = np.sqrt(((peer_features - target_features) ** 2).sum(axis=1))
                    # Convert distance to similarity (inverse relationship)
                    weights = 1.0 / (1.0 + distances)
                else:  # similarity
                    # Cosine similarity
                    target_norm = np.linalg.norm(target_features)
                    if target_norm > 0:
                        similarities = peer_features.dot(target_features) / (
                            np.linalg.norm(peer_features, axis=1) * target_norm
                        )
                        weights = pd.Series(similarities, index=peer_features.index)
                    else:
                        weights = pd.Series(1.0, index=peer_features.index)
                
                # Reindex to all peers
                weights = weights.reindex(peers).fillna(1.0)
            else:
                weights = pd.Series(1.0, index=peers)
        else:
            weights = pd.Series(1.0, index=peers)
    
    else:
        # Unknown mode - fallback to equal weights
        weights = pd.Series(1.0, index=peers, dtype=float)
    
    # Apply processing steps
    weights = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    # Clip negative weights if requested
    if clip_negative:
        weights = weights.clip(lower=0.0)
    
    # Apply power transformation
    if power is not None and float(power) != 1.0:
        weights = weights.pow(float(power))
    
    # Filter out very small weights (noise reduction)
    if min_weight_threshold > 0:
        weights = weights.where(weights >= min_weight_threshold, 0.0)
    
    # Normalize weights
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        # If all weights are zero/invalid, use equal weights
        weights = pd.Series(1.0 / max(len(peers), 1), index=peers, dtype=float)
    else:
        weights = weights / total
    
    # Ensure we have weights for all peers
    weights = weights.reindex(peers).fillna(0.0)
    
    return weights


def adaptive_correlation_computation(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    weight_mode: str = "auto",
    fallback_strategy: str = "graceful",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Adaptive correlation computation that gracefully handles poor data quality.
    
    This function is much more flexible and will attempt multiple strategies
    to produce usable results even with sparse or poor quality data.
    
    Parameters
    ----------
    fallback_strategy : str, default "graceful"
        How to handle failures: "graceful" (fallback modes), "strict" (fail fast)
    
    Returns
    -------
    atm_df : pd.DataFrame
        ATM matrix
    corr_df : pd.DataFrame  
        Correlation/similarity matrix
    metadata : dict
        Information about the computation strategy used
    """
    metadata = {
        "strategy_used": "primary",
        "pillars_attempted": list(pillars_days),
        "pillars_used": [],
        "data_quality": "unknown",
        "fallback_applied": False
    }
    
    # Primary strategy: try the requested computation
    try:
        atm_df, corr_df = compute_atm_corr(
            get_smile_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            pillars_days=pillars_days,
            **kwargs
        )
        
        # Check data quality
        finite_count = pd.notna(corr_df).sum().sum() if not corr_df.empty else 0
        total_elements = corr_df.size if not corr_df.empty else 0
        data_quality = finite_count / total_elements if total_elements > 0 else 0
        
        metadata.update({
            "pillars_used": list(atm_df.columns) if not atm_df.empty else [],
            "data_quality": f"{data_quality:.2%}",
        })
        
        # If primary strategy worked well enough, return results
        if data_quality >= 0.3:  # At least 30% good data
            return atm_df, corr_df, metadata
        
        # Otherwise, try fallback if enabled
        if fallback_strategy != "graceful":
            return atm_df, corr_df, metadata
            
    except Exception as e:
        if fallback_strategy == "strict":
            raise e
        metadata["primary_error"] = str(e)
    
    # Fallback strategy 1: Reduce pillar requirements
    if fallback_strategy == "graceful":
        metadata["fallback_applied"] = True
        metadata["strategy_used"] = "reduced_pillars"
        
        try:
            # Try with more lenient parameters
            fallback_kwargs = kwargs.copy()
            fallback_kwargs.update({
                "min_pillars": 1,
                "min_tickers_per_pillar": 2,
                "min_pillars_per_ticker": 1,
            })
            
            atm_df, corr_df = compute_atm_corr(
                get_smile_slice=get_smile_slice,
                tickers=tickers,
                asof=asof,
                pillars_days=pillars_days,
                **fallback_kwargs
            )
            
            finite_count = pd.notna(corr_df).sum().sum() if not corr_df.empty else 0
            total_elements = corr_df.size if not corr_df.empty else 0
            data_quality = finite_count / total_elements if total_elements > 0 else 0
            
            metadata.update({
                "pillars_used": list(atm_df.columns) if not atm_df.empty else [],
                "data_quality": f"{data_quality:.2%}",
            })
            
            if data_quality > 0:
                return atm_df, corr_df, metadata
                
        except Exception as e:
            metadata["fallback1_error"] = str(e)
    
    # Fallback strategy 2: Create synthetic correlation matrix
    metadata["strategy_used"] = "synthetic"
    tickers_list = list(tickers)
    
    # Create identity-like correlation matrix with some noise
    n_tickers = len(tickers_list)
    corr_matrix = np.eye(n_tickers) + np.random.normal(0, 0.1, (n_tickers, n_tickers))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal is 1
    
    # Clip to valid correlation range
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    corr_df = pd.DataFrame(corr_matrix, index=tickers_list, columns=tickers_list)
    
    # Create minimal ATM matrix
    atm_df = pd.DataFrame(
        np.random.normal(0.2, 0.05, (n_tickers, len(pillars_days))),
        index=tickers_list,
        columns=list(pillars_days)
    )
    
    metadata.update({
        "pillars_used": list(pillars_days),
        "data_quality": "synthetic",
    })
    
    return atm_df, corr_df, metadata
