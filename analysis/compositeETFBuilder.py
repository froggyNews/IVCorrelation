"""
Builders for ETF-style composite volatility surfaces (grid & ATM-pillar).

Exports
-------
- build_surface_grids(...)  -> dict[ticker][date] -> IV grid DataFrame
- combine_surfaces(...)     -> dict[date] -> composite IV grid DataFrame
- build_composite_iv(...)   -> ATM composite by pillars (time series)
- build_composite_iv_series(...) -> convenience wrapper
- build_composite_iv_by_rank(...) -> ATM composite by expiry rank for a date

Notes
-----
- Surfaces are built with rows=moneyness bins (string labels like '0.95-1.05')
  and columns=tenor days (int). Values are IV.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sqlite3

from data.db_utils import get_conn
from analysis.pillars import load_atm, nearest_pillars

# If you already publish these elsewhere, import them instead:
DEFAULT_TENORS: Tuple[int, ...] = (7, 14, 28, 56, 94, 112, 182, 252)
DEFAULT_MNY_BINS: Tuple[Tuple[float, float], ...] = (
    (0.80, 0.90),
    (0.95, 1.05),
    (1.10, 1.25),
)


# ---------- helpers ----------

def _mny_labels(bins: Tuple[Tuple[float, float], ...]) -> tuple[list[str], list[float]]:
    edges: list[float] = [bins[0][0]] + [hi for (_, hi) in bins]
    labels: list[str] = [f"{lo:.2f}-{hi:.2f}" for (lo, hi) in bins]
    return labels, edges


def _nearest_tenor(days: float, tenors: Iterable[int]) -> int:
    arr = np.asarray(list(tenors), dtype=float)
    return int(arr[np.argmin(np.abs(arr - float(days)))])


# ---------- public builders ----------

def build_surface_grids(
    tickers: Iterable[str] | None = None,
    tenors: Iterable[int] = DEFAULT_TENORS,
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS,
    use_atm_only: bool = False,
    max_expiries: Optional[int] = None,
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Query options quotes and aggregate to IV grids per ticker/date.

    Returns:
        dict[ticker][pd.Timestamp] -> DataFrame(index=moneyness-bin label, columns=tenor days)
    """
    conn = get_conn()

    cols = "asof_date, ticker, ttm_years, moneyness, iv, is_atm"
    q = f"SELECT {cols} FROM options_quotes"
    params: list = []
    clauses: list[str] = []

    tickers_list = list(tickers) if tickers else []
    if tickers_list:
        placeholders = ",".join(["?"] * len(tickers_list))
        clauses.append(f"ticker IN ({placeholders})")
        params.extend(tickers_list)
    if use_atm_only:
        clauses.append("is_atm = ?")
        params.append(1)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)

    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return {}

    df = df.dropna(subset=["iv", "ttm_years", "moneyness"]).copy()
    df["ttm_days"] = df["ttm_years"].astype(float) * 365.25

    # Limit expiries per (ticker, asof_date) if requested
    if max_expiries and max_expiries > 0:
        limited = []
        for (ticker, asof_date), g in df.groupby(["ticker", "asof_date"], sort=False):
            uniq = g.groupby("ttm_years", sort=False)["ttm_years"].first().sort_values()
            keep_years = uniq.head(max_expiries).values
            limited.append(g[g["ttm_years"].isin(keep_years)])
        df = pd.concat(limited, ignore_index=True) if limited else pd.DataFrame(columns=df.columns)

    if df.empty:
        return {}

    # Bin to nearest tenor (vectorized)
    tenor_arr = np.asarray(list(tenors), dtype=float)
    idx = np.abs(df["ttm_days"].to_numpy(float)[:, None] - tenor_arr[None, :]).argmin(axis=1)
    df["tenor_bin"] = tenor_arr[idx].astype(int)

    # Bin moneyness
    labels, edges = _mny_labels(mny_bins)
    df["mny_bin"] = pd.cut(df["moneyness"].astype(float), bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=["mny_bin"])

    # Aggregate IV
    cell = (
        df.groupby(["asof_date", "ticker", "mny_bin", "tenor_bin"], observed=True)["iv"]
        .mean()
        .reset_index()
    )

    # Pivot per ticker/date
    out: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    for ticker, g in cell.groupby("ticker", sort=False):
        per_date: Dict[pd.Timestamp, pd.DataFrame] = {}
        for date, gd in g.groupby("asof_date", sort=False):
            grid = gd.pivot(index="mny_bin", columns="tenor_bin", values="iv").sort_index(axis=1)
            per_date[pd.to_datetime(date)] = grid
        out[str(ticker).upper()] = per_date
    return out


def combine_surfaces(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    rhos: Mapping[str, float],
    weight_grids: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Weighted grid combination into a composite surface (per date):
        sigma_synth(K,T) = sum_i rho_i * w_i(K,T) * sigma_i(K,T) / sum_i rho_i * w_i(K,T)
    """
    # Normalize top-level weights
    rhos = {k.upper(): float(v) for k, v in dict(rhos).items()}
    total = float(sum(rhos.values()))
    if total <= 0:
        # fallback to equal weights if nothing positive
        keys = list(surfaces.keys())
        rhos = {k: (1.0 / max(1, len(keys))) for k in keys}
    else:
        rhos = {k: v / total for k, v in rhos.items()}

    # All dates present across any ticker
    all_dates: set[pd.Timestamp] = set()
    for per_date in surfaces.values():
        all_dates.update(per_date.keys())

    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for date in sorted(all_dates):
        numerator = None
        denominator = None

        for ticker, per_date in surfaces.items():
            if date not in per_date:
                continue
            sigma = per_date[date]
            rho = rhos.get(ticker.upper(), 0.0)
            if rho == 0.0:
                continue

            wg = (weight_grids or {}).get(ticker, None)
            if wg is None:
                wg = pd.DataFrame(1.0, index=sigma.index, columns=sigma.columns)
            wg = wg.reindex_like(sigma).fillna(0.0)

            contrib_num = rho * wg * sigma
            contrib_den = rho * wg

            numerator = contrib_num if numerator is None else numerator.add(contrib_num, fill_value=0.0)
            denominator = contrib_den if denominator is None else denominator.add(contrib_den, fill_value=0.0)

        if numerator is not None and denominator is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                combined = numerator / denominator
            result[date] = combined

    return result


def build_composite_iv(
    weights: Mapping[str, float],
    pillar_days: Union[int, Iterable[int]] = (7, 30, 60, 90, 180, 365),
    tolerance_days: float = 7.0,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """
    Composite ATM IV series at requested pillars (per date).
    Columns: asof_date, pillar_days, iv, tickers_used, weighted_count, iv_constituents, weights
    """
    if isinstance(pillar_days, int):
        pillar_days = [pillar_days]

    # Normalize weights
    weights = {k.upper(): float(v) for k, v in dict(weights).items()}
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("All provided weights are zero; supply at least one positive weight.")
    w_norm = {k: v / total for k, v in weights.items()}

    conn = conn or get_conn()
    df_atm = load_atm(conn)  # expected columns include: asof_date,ticker,ttm_years,iv,...

    if df_atm.empty:
        raise RuntimeError("No ATM rows found. Did you run the historical loader?")

    df_atm = df_atm[df_atm["ticker"].isin(w_norm.keys())].copy()
    if df_atm.empty:
        raise RuntimeError("No ATM rows for the requested tickers in weights.")

    # nearest_pillars returns rows assigned to the closest expiry per (date,ticker,pillar)
    pillars = nearest_pillars(df_atm, pillars_days=list(pillar_days), tolerance_days=tolerance_days)
    if pillars.empty:
        raise RuntimeError("nearest_pillars returned no rows within tolerance.")

    out_rows = []
    for (asof, pday), g in pillars.groupby(["asof_date", "pillar_days"], sort=False):
        iv_map: dict[str, float] = {}
        iv_wsum = 0.0
        w_sum = 0.0
        n_used = 0

        for ticker, sub in g.groupby("ticker", sort=False):
            w_t = w_norm.get(str(ticker).upper(), 0.0)
            if w_t <= 0:
                continue
            # pick the absolute nearest within this (date,ticker,pillar)
            row = sub.iloc[(sub["pillar_diff_days"].abs()).to_numpy().argmin()]
            iv_t = float(row["iv"])
            iv_map[str(ticker).upper()] = iv_t
            iv_wsum += w_t * iv_t
            w_sum += w_t
            n_used += 1

        if n_used > 0 and w_sum > 0:
            out_rows.append(
                {
                    "asof_date": pd.to_datetime(asof),
                    "pillar_days": int(pday),
                    "iv": iv_wsum / w_sum,
                    "tickers_used": int(n_used),
                    "weighted_count": float(w_sum),
                    "iv_constituents": iv_map,
                    "weights": w_norm,
                }
            )

    return (
        pd.DataFrame(out_rows)
        .sort_values(["asof_date", "pillar_days"])
        .reset_index(drop=True)
    )


def build_composite_iv_series(
    weights: Mapping[str, float],
    pillar_days: Union[int, Iterable[int]] = (7, 30, 60, 90, 180, 365),
    tolerance_days: float = 7.0,
) -> pd.DataFrame:
    """Convenience wrapper for ATM pillar composites."""
    return build_composite_iv(weights, pillar_days=pillar_days, tolerance_days=tolerance_days)


def build_composite_iv_by_rank(
    weights: Mapping[str, float],
    asof: str,
    max_expiries: int = 6,
    atm_band: float = 0.05,
) -> pd.DataFrame:
    """
    Combine peer ATM vols by expiry order (rank) into a single curve for `asof`.
    Returns DataFrame with columns: ["rank", "synth_iv"].
    """
    from analysis.analysis_pipeline import get_smile_slice  # local import to avoid hard dep at import time
    from .beta_builder.correlation import compute_atm_corr_pillar_free

    weights = {k.upper(): float(v) for k, v in dict(weights).items()}
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        raise ValueError("All weights are zero.")
    w_norm = {k: v / total for k, v in weights.items()}
    tickers = list(w_norm.keys())

    atm_df, _ = compute_atm_corr_pillar_free(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        max_expiries=max_expiries,
        atm_band=atm_band,
    )
    if atm_df.empty:
        return pd.DataFrame(columns=["rank", "synth_iv"])

    rows = []
    for r in atm_df.columns:
        ivs, ws = [], []
        for t in tickers:
            if t in atm_df.index:
                v = atm_df.at[t, r]
                if pd.notna(v):
                    ivs.append(float(v))
                    ws.append(float(w_norm.get(t, 0.0)))
        if ws:
            rows.append({"rank": int(r), "synth_iv": float(np.dot(ws, ivs)) / float(np.sum(ws))})
    return pd.DataFrame(rows)
