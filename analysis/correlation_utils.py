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


def shift_target(
    target: str,
    tickers: Iterable[str],
    shift: int = 1,
) -> tuple[str, list[str]]:
    """Rotate the ticker universe so a different ticker becomes the target.

    Parameters
    ----------
    target : str
        Currently selected target ticker.
    tickers : Iterable[str]
        Universe of tickers containing ``target``.
    shift : int, default 1
        Number of positions to rotate forward (negative values rotate
        backward).

    Returns
    -------
    tuple[str, list[str]]
        A tuple of ``(new_target, peers)`` where ``peers`` are the remaining
        tickers excluding ``new_target`` while preserving their original
        order.
    """

    t_list = [t.upper() for t in tickers]
    if target.upper() not in t_list:
        raise ValueError("target must be present in tickers")

    idx = t_list.index(target.upper())
    new_idx = (idx + int(shift)) % len(t_list)
    new_target = t_list[new_idx]
    peers = [t for i, t in enumerate(t_list) if i != new_idx]
    return new_target, peers
