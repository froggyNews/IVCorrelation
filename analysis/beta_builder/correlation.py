# analysis/correlation.py
from __future__ import annotations
from typing import Iterable, Tuple, List
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
) -> pd.Series:
    """
    Convert correlations (computed across feature columns) into positive,
    normalized weights for `peers` versus `target`.
    """
    target = target.upper()
    peers = [p.upper() for p in peers]
    corr_df = feature_df.T.corr()
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0].apply(pd.to_numeric, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("correlation weights sum to zero")
    return (s / total).reindex(peers).fillna(0.0)

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
) -> pd.Series:
    """
    Convert a correlation column (against `target`) into positive, normalized weights.
    """
    target = target.upper()
    peers = [p.upper() for p in peers]
    if target not in corr_df.columns:
        raise ValueError(f"Target {target} not in correlation matrix columns")
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0].apply(pd.to_numeric, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Correlation weights sum to zero or NaN")
    return (s / total).reindex(peers).fillna(0.0)

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
    min_pillars: int = 2,
    demean_rows: bool = False,
    corr_method: str = "pearson",
    min_tickers_per_pillar: int = 3,
    min_pillars_per_ticker: int = 2,
    ridge: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an ATM matrix (rows=tickers, cols=pillars) for `asof` via `build_atm_matrix`,
    apply light coverage filtering, and recompute a ridge‑stabilized correlation.
    Returns (atm_df, corr_df).
    """
    tickers = [t.upper() for t in tickers]

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

    # Coverage filtering + ridge recompute
    if not atm_df.empty:
        # keep pillars with enough tickers
        keep_cols = atm_df.count(axis=0)
        keep_cols = keep_cols[keep_cols >= min_tickers_per_pillar].index
        if len(keep_cols) >= 2:
            atm_df = atm_df[keep_cols]

        # keep tickers with enough pillars
        keep_rows = atm_df.count(axis=1)
        keep_rows = keep_rows[keep_rows >= min_pillars_per_ticker].index
        if len(keep_rows) >= 2:
            atm_df = atm_df.loc[keep_rows]

        # recompute correlation on row‑standardized values
        if atm_df.shape[0] >= 2 and atm_df.shape[1] >= 2:
            clean = atm_df.dropna()
            if clean.shape[0] >= 2 and clean.shape[1] >= 2:
                std = (clean - clean.mean(axis=1).values[:, None]) / (
                    clean.std(axis=1).values[:, None] + 1e-8
                )
                C = (std @ std.T) / max(std.shape[1] - 1, 1)
                C += ridge * np.eye(C.shape[0])
                corr_df = pd.DataFrame(C, index=clean.index, columns=clean.index)

    return atm_df, corr_df

# ---------------------------------------------------------------------
# Data type: ATM IVs (expiry‑rank / pillar‑free) → correlation
# ---------------------------------------------------------------------
def compute_atm_corr_pillar_free(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    *,
    max_expiries: int = 6,
    atm_band: float = 0.05,
    min_tickers: int = 2,
    corr_method: str = "pearson",
    min_periods: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pillar‑free ATM correlation:
      1) Extract ATM per expiry,
      2) align by expiry rank (0=shortest,1=next,...),
      3) correlate across ranks.
    Returns (atm_df with rank columns, corr_df).
    """
    def _atm_curve_simple(df: pd.DataFrame, band: float) -> pd.DataFrame:
        need = {"T", "moneyness", "sigma"}
        if df is None or df.empty or not need.issubset(df.columns):
            return pd.DataFrame(columns=["T", "atm_vol"])
        d = df.copy()
        d["T"] = pd.to_numeric(d["T"], errors="coerce")
        d["moneyness"] = pd.to_numeric(d["moneyness"], errors="coerce")
        d["sigma"] = pd.to_numeric(d["sigma"], errors="coerce")
        d = d.dropna(subset=["T", "moneyness", "sigma"])
        rows = []
        for Tval, g in d.groupby("T"):
            gg = g.dropna(subset=["moneyness", "sigma"])
            in_band = gg.loc[(gg["moneyness"] - 1.0).abs() <= band]
            if not in_band.empty:
                atm_vol = float(in_band["sigma"].median())
            else:
                idx = int((gg["moneyness"] - 1.0).abs().idxmin())
                atm_vol = float(gg.loc[idx, "sigma"])
            rows.append({"T": float(Tval), "atm_vol": atm_vol})
        return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    tickers = [t.upper() for t in tickers]
    rows = []
    for t in tickers:
        try:
            df = get_smile_slice(t, asof, T_target_years=None)
        except Exception:
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
        corr_df = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    else:
        corr_df = atm_df.T.corr(method=corr_method, min_periods=min_periods)
        corr_df = corr_df.reindex(index=tickers, columns=tickers)
    return atm_df, corr_df
