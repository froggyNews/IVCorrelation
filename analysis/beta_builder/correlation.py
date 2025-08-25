# analysis/correlation.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import logging
import numpy as np
import pandas as pd

# single source of ATM pillars
from analysis.pillars import build_atm_matrix

__all__ = [
    "corr_weights_from_matrix",      # weights from a ticker×feature matrix
    "corr_weights",                  # weights from a correlation matrix
    "compute_atm_corr",              # ATM pillars → (atm_df, corr_df)
    "compute_atm_corr_pillar_free",  # expiry-rank ATM → (atm_df, corr_df)
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _equal_weights(peers: List[str]) -> pd.Series:
    n = max(len(peers), 1)
    w = np.full(n, 1.0 / n, dtype=float)
    return pd.Series(w, index=[p.upper() for p in peers], dtype=float)

def _safe_normalize(s: pd.Series, peers: List[str], *, fallback_equal: bool = True) -> pd.Series:
    s = s.reindex([p.upper() for p in peers]).astype(float)
    total = float(np.nansum(s.values))
    if np.isfinite(total) and total > 0:
        return (s / total).fillna(0.0)
    if fallback_equal:
        log.debug("Normalization fell back to equal weights (sum<=0 or NaN).")
        return _equal_weights(peers)
    return s.fillna(0.0)

# ---------------------------------------------------------------------
# Method: correlation on any ticker×feature matrix
# ---------------------------------------------------------------------
def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
    fallback_equal_on_failure: bool = True,
) -> pd.Series:
    """
    Convert correlations (computed across feature columns) into positive,
    normalized weights for `peers` versus `target`.

    Lenient behavior:
      - Coerces non-numeric to NaN
      - Clips negatives if requested
      - If all weights are NaN/<=0 → equal weights (no raise)
    """
    target = target.upper()
    peers = [p.upper() for p in peers]
    if feature_df is None or feature_df.empty:
        log.debug("feature_df empty; returning equal weights.")
        return _equal_weights(peers)

    df = feature_df.apply(pd.to_numeric, errors="coerce")
    try:
        corr_df = df.T.corr(min_periods=1)
    except Exception as e:
        log.debug("corr() failed: %s; returning equal weights.", e)
        return _equal_weights(peers)

    s = corr_df.reindex(index=peers, columns=[target], fill_value=np.nan).iloc[:, 0]
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))

    return _safe_normalize(s, peers, fallback_equal=fallback_equal_on_failure)

# ---------------------------------------------------------------------
# Method: correlation → weights given a correlation matrix directly
# ---------------------------------------------------------------------
def corr_weights(
    corr_df: pd.DataFrame,
    target: str,
    peers: List[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
    fallback_equal_on_failure: bool = True,
) -> pd.Series:
    """
    Convert a correlation column (against `target`) into positive, normalized weights.

    Lenient behavior (no raises):
      - If target/peers missing → treat as 0
      - If sums to 0/NaN → equal weights
    """
    target = target.upper()
    peers = [p.upper() for p in peers]
    if corr_df is None or corr_df.empty:
        log.debug("corr_df empty; returning equal weights.")
        return _equal_weights(peers)

    s = corr_df.reindex(index=peers, columns=[target], fill_value=np.nan).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))

    return _safe_normalize(s, peers, fallback_equal=fallback_equal_on_failure)

# ---------------------------------------------------------------------
# Data type: ATM IVs (fixed pillars) → correlation
# ---------------------------------------------------------------------
def compute_atm_corr(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    *,
    atm_band: float = 0.05,
    tol_days: float = 7.0,
    min_pillars: int = 1,                 # was 2
    demean_rows: bool = False,
    corr_method: str = "pearson",
    min_tickers_per_pillar: int = 2,      # was 3
    min_pillars_per_ticker: int = 1,      # was 2
    ridge: float = 1e-6,
    fill_diag: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an ATM matrix (rows=tickers, cols=pillars) for `asof` via `build_atm_matrix`,
    apply *light* coverage filtering, and recompute a ridge-stabilized correlation.

    Lenient behavior:
      - Accepts single pillar / sparse coverage
      - Uses min_periods=1 where possible
      - Fills missing correlations with 0, diagonal with `fill_diag`
    """
    tickers = [t.upper() for t in tickers]

    try:
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
    except Exception as e:
        log.debug("build_atm_matrix failed: %s", e)
        atm_df = pd.DataFrame(index=tickers)
        corr_df = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    # Gentle coverage filtering
    if not atm_df.empty:
        # keep pillars with enough tickers
        keep_cols = atm_df.count(axis=0)
        keep_cols = keep_cols[keep_cols >= min_tickers_per_pillar].index
        if len(keep_cols) >= 1:
            atm_df = atm_df[keep_cols]

        # keep tickers with enough pillars
        keep_rows = atm_df.count(axis=1)
        keep_rows = keep_rows[keep_rows >= min_pillars_per_ticker].index
        if len(keep_rows) >= 1:
            atm_df = atm_df.loc[keep_rows]

        # recompute correlation (lenient)
        if atm_df.shape[0] >= 1 and atm_df.shape[1] >= 1:
            clean = atm_df.copy()
            if demean_rows and clean.shape[1] >= 1:
                clean = clean.sub(clean.mean(axis=1), axis=0)

            # If at least 2 cols, do standardized inner-product; else fall back to pandas corr
            if clean.shape[1] >= 2:
                std = (clean - clean.mean(axis=1).values[:, None]) / (clean.std(axis=1).values[:, None] + 1e-8)
                C = (std.fillna(0.0).to_numpy() @ std.fillna(0.0).to_numpy().T)
                denom = max(std.shape[1] - 1, 1)
                C = C / denom
                C = C + ridge * np.eye(C.shape[0])
                corr_df = pd.DataFrame(C, index=clean.index, columns=clean.index)
            else:
                corr_df = clean.T.corr(method=corr_method, min_periods=1)
                corr_df = corr_df.reindex(index=clean.index, columns=clean.index)

    # Final sanitation: fill diag, zeros elsewhere where missing
    if corr_df is None or corr_df.empty:
        corr_df = pd.DataFrame(0.0, index=tickers, columns=tickers, dtype=float)
    else:
        corr_df = corr_df.reindex(index=tickers, columns=tickers)
        corr_df = corr_df.astype(float).fillna(0.0)
    np.fill_diagonal(corr_df.values, float(fill_diag))

    return atm_df, corr_df

# ---------------------------------------------------------------------
# Data type: ATM IVs (expiry-rank / pillar-free) → correlation
# ---------------------------------------------------------------------
def compute_atm_corr_pillar_free(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    *,
    max_expiries: int = 6,
    atm_band: float = 0.05,
    min_tickers: int = 1,           # was 2
    corr_method: str = "pearson",
    min_periods: int = 1,           # was 2
    fill_diag: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pillar-free ATM correlation (lenient):
      1) Extract ATM per expiry,
      2) align by expiry rank (0=shortest,1=next,...),
      3) correlate across ranks with min_periods=1.
    """
    def _atm_curve_simple(df: pd.DataFrame, band: float) -> pd.DataFrame:
        need = {"T", "moneyness", "sigma"}
        if df is None or df.empty or not need.issubset(df.columns):
            return pd.DataFrame(columns=["T", "atm_vol"])
        d = df.copy()
        for c in ("T", "moneyness", "sigma"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["T", "moneyness", "sigma"])
        if d.empty:
            return pd.DataFrame(columns=["T", "atm_vol"])

        rows = []
        for Tval, g in d.groupby("T"):
            gg = g.dropna(subset=["moneyness", "sigma"])
            in_band = gg.loc[(gg["moneyness"] - 1.0).abs() <= band]
            if not in_band.empty:
                atm_vol = float(in_band["sigma"].median())
            else:
                idx = (gg["moneyness"] - 1.0).abs().astype(float).idxmin()
                atm_vol = float(gg.loc[idx, "sigma"])
            rows.append({"T": float(Tval), "atm_vol": atm_vol})
        return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    tickers = [t.upper() for t in tickers]
    rows = []
    for t in tickers:
        try:
            df = get_smile_slice(t, asof, T_target_years=None)
        except Exception as e:
            log.debug("get_smile_slice(%s) failed: %s", t, e)
            df = None
        if df is None or df.empty:
            rows.append(pd.Series({i: np.nan for i in range(max_expiries)}, name=t))
            continue
        atm = _atm_curve_simple(df, band=atm_band)
        values = {
            i: (float(atm.at[i, "atm_vol"]) if i < len(atm) and pd.notna(atm.at[i, "atm_vol"]) else np.nan)
            for i in range(max_expiries)
        }
        rows.append(pd.Series(values, name=t))

    atm_df = pd.DataFrame(rows)
    if atm_df.empty or len(atm_df.index) < min_tickers:
        corr_df = pd.DataFrame(0.0, index=tickers, columns=tickers, dtype=float)
    else:
        corr_df = atm_df.T.corr(method=corr_method, min_periods=min_periods)
        corr_df = corr_df.reindex(index=tickers, columns=tickers).astype(float).fillna(0.0)
    np.fill_diagonal(corr_df.values, float(fill_diag))
    return atm_df, corr_df
