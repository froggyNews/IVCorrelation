import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Tuple

from .pillars import build_atm_matrix


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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ATM matrix, correlation matrix) for one as-of date."""
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
