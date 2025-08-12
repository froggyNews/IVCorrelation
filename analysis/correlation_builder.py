"""Vol betas builder focused on three modes:
  - 'ul'      : betas from underlying log-returns
  - 'iv_atm'  : betas from ATM pillar IVs (per pillar_days)
  - 'surface' : betas from standardized surface grid summaries

Provides:
  - build_vol_betas(...): returns betas (Series or dict[pillar_days->Series])
  - save_correlations(...): writes CSVs under data/
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Iterable, Dict, Tuple, Union

from data.db_utils import get_conn
from analysis.pillars import load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS

# ---------------- helpers ----------------

def _beta(df: pd.DataFrame, x: str, b: str) -> float:
    a = df[[x, b]].dropna()
    if len(a) < 5:
        return float("nan")
    var_b = a[b].var()
    return float(a[x].cov(a[b]) / var_b) if var_b and not math.isclose(var_b, 0.0) else float("nan")


def _spot_log_returns() -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT asof_date, ticker, spot FROM options_quotes"
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df
    px = df.groupby(["asof_date", "ticker"])['spot'].median().unstack("ticker").sort_index()
    ret = (px / px.shift(1)).map(lambda x: math.log(x) if pd.notna(x) and x>0 else float("nan"))
    return ret
def _corr_series_ul(benchmark: str, tickers: Iterable[str] | None = None) -> pd.Series:
    ret = _spot_log_returns()
    if ret is None or ret.empty or benchmark not in ret.columns:
        return pd.Series(dtype=float)
    cols = [c for c in ret.columns if c != benchmark]
    if tickers is not None:
        cols = [c for c in cols if c in set(tickers)]
    corrs = {}
    for t in cols:
        a = ret[[t, benchmark]].dropna()
        corrs[t] = a[t].corr(a[benchmark]) if len(a) >= 5 else np.nan
    return pd.Series(corrs, name='ul_corr')


def _corr_series_iv_atm(benchmark: str, pillar_days: Iterable[int]) -> pd.Series:
    atm = load_atm()
    if atm.empty:
        return pd.Series(dtype=float)
    piv = nearest_pillars(atm, pillars_days=pillar_days)
    # average correlation across requested pillars
    bucket = {}
    for d in sorted(set(piv['pillar_days'])):
        sub = piv[piv['pillar_days'] == d]
        wide = sub.pivot_table(index='asof_date', columns='ticker', values='iv', aggfunc='mean').sort_index()
        if benchmark not in wide.columns: 
            continue
        for t in wide.columns:
            if t == benchmark:
                continue
            a = wide[[t, benchmark]].dropna()
            if len(a) >= 5:
                bucket.setdefault(t, []).append(a[t].corr(a[benchmark]))
    # average (nanmean) across pillars
    out = {t: (float(np.nanmean(v)) if len(v) else np.nan) for t, v in bucket.items()}
    return pd.Series(out, name='iv_atm_corr')


def _corr_series_surface(benchmark: str,
                         tenor_days: Iterable[int],
                         mny_bins: Iterable[Tuple[float, float]]) -> pd.Series:
    # Reuse the same scalar surface series used in _surface_betas
    conn = get_conn()
    df = pd.read_sql_query("SELECT asof_date, ticker, ttm_years, moneyness, iv FROM options_quotes", conn)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.dropna(subset=['iv','ttm_years','moneyness']).copy()
    df['ttm_days'] = df['ttm_years'] * 365.25
    tarr = pd.Series(list(tenor_days))
    df['tenor_bin'] = df['ttm_days'].apply(lambda d: tarr.iloc[(tarr - d).abs().argmin()])
    labels = [f"{lo:.2f}-{hi:.2f}" for (lo,hi) in mny_bins]
    edges = [mny_bins[0][0]] + [hi for (_,hi) in mny_bins]
    df['mny_bin'] = pd.cut(df['moneyness'], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=['mny_bin'])
    cell = df.groupby(['asof_date','ticker','tenor_bin','mny_bin'])['iv'].mean().reset_index()
    grid = cell.pivot_table(index=['asof_date','ticker'], columns=['tenor_bin','mny_bin'], values='iv')
    scalar = grid.mean(axis=1).rename('iv_surface_level').reset_index()
    wide = scalar.pivot(index='asof_date', columns='ticker', values='iv_surface_level').sort_index()
    if benchmark not in wide.columns:
        return pd.Series(dtype=float)
    corrs = {}
    for t in wide.columns:
        if t == benchmark: 
            continue
        a = wide[[t, benchmark]].dropna()
        corrs[t] = a[t].corr(a[benchmark]) if len(a) >= 5 else np.nan
    return pd.Series(corrs, name='iv_surface_corr')


def peer_weights_from_correlations(
    benchmark: str,
    peers: Iterable[str] | None = None,
    mode: str = 'iv_atm',
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tenor_days: Iterable[int] = (7,30,60,90,180,365),
    mny_bins: Iterable[Tuple[float,float]] = ((0.8,0.9),(0.95,1.05),(1.1,1.25)),
    clip_negative: bool = True,
    power: float = 1.0,   # raise corr to a power for sharper weights
) -> pd.Series:
    """
    Return correlation-based weights vs `benchmark`, normalized to 1.0.
    - mode: 'ul' | 'iv_atm' | 'surface'
    - clip_negative: set negative corrs to 0 (avoid sign flips)
    - power: use >1 to emphasize higher corr names
    """
    mode = mode.lower()
    if mode == 'ul':
        corr = _corr_series_ul(benchmark, tickers=peers)
    elif mode == 'surface':
        corr = _corr_series_surface(benchmark, tenor_days=tenor_days, mny_bins=mny_bins)
        if peers is not None:
            corr = corr[corr.index.isin(set(peers))]
    else:  # 'iv_atm' default
        corr = _corr_series_iv_atm(benchmark, pillar_days=pillar_days)
        if peers is not None:
            corr = corr[corr.index.isin(set(peers))]

    if corr.empty:
        return pd.Series(dtype=float, name='weights')

    vals = corr.copy()
    if clip_negative:
        vals = vals.clip(lower=0.0)
    vals = vals.pow(power)
    s = float(vals.sum())
    if not np.isfinite(s) or s <= 0:
        # fallback equal weights across available peers
        if peers is None or len(vals.index) == 0:
            return pd.Series(dtype=float, name='weights')
        eq = pd.Series({p: 1.0 for p in vals.index}, name='weights')
        return eq / eq.sum()
    w = vals / s
    w.name = 'weights'
    return w

# -------------- modes --------------

def _ul_betas(benchmark: str, tickers: Iterable[str] | None = None) -> pd.Series:
    ret = _spot_log_returns()
    if ret is None or ret.empty or benchmark not in ret.columns:
        return pd.Series(dtype=float)
    if tickers is None:
        tickers = [c for c in ret.columns if c != benchmark]
    betas = {}
    for t in tickers:
        if t == benchmark or t not in ret.columns:
            continue
        betas[t] = _beta(ret.rename(columns={t: 'x', benchmark: 'b'}), 'x', 'b')
    return pd.Series(betas, name='ul_beta')


def _iv_atm_betas(benchmark: str, pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS) -> Dict[int, pd.Series]:
    atm = load_atm()
    if atm.empty:
        return {}
    piv = nearest_pillars(atm, pillars_days=pillar_days)
    out: Dict[int, pd.Series] = {}
    for d in sorted(set(piv['pillar_days'])):
        sub = piv[piv['pillar_days'] == d]
        wide = sub.pivot_table(index='asof_date', columns='ticker', values='iv', aggfunc='mean').sort_index()
        # optional de-meaning to reduce common moves
        wide = wide.sub(wide.mean(axis=1), axis=0)
        if benchmark not in wide.columns:
            continue
        betas = {}
        for t in wide.columns:
            if t == benchmark:
                continue
            betas[t] = _beta(wide.rename(columns={t: 'x', benchmark: 'b'}), 'x', 'b')
        out[int(d)] = pd.Series(betas, name=f'iv_atm_beta_{d}d')
    return out


def _surface_betas(benchmark: str,
                    tenor_days: Iterable[int] = (7,30,60,90,180,365),
                    mny_bins: Iterable[Tuple[float, float]] = ((0.8,0.9),(0.95,1.05),(1.1,1.25)),) -> pd.Series:
    conn = get_conn()
    df = pd.read_sql_query("SELECT asof_date, ticker, ttm_years, moneyness, iv FROM options_quotes", conn)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.dropna(subset=['iv','ttm_years','moneyness']).copy()
    df['ttm_days'] = df['ttm_years'] * 365.25
    # nearest tenor bin
    tarr = pd.Series(list(tenor_days))
    df['tenor_bin'] = df['ttm_days'].apply(lambda d: tarr.iloc[(tarr - d).abs().argmin()])
    # moneyness bins
    labels = [f"{lo:.2f}-{hi:.2f}" for (lo,hi) in mny_bins]
    edges = [mny_bins[0][0]] + [hi for (_,hi) in mny_bins]
    df['mny_bin'] = pd.cut(df['moneyness'], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=['mny_bin'])
    # mean per cell, then average across cells -> scalar per day/ticker
    cell = df.groupby(['asof_date','ticker','tenor_bin','mny_bin'])['iv'].mean().reset_index()
    grid = cell.pivot_table(index=['asof_date','ticker'], columns=['tenor_bin','mny_bin'], values='iv')
    scalar = grid.mean(axis=1).rename('iv_surface_level').reset_index()
    wide = scalar.pivot(index='asof_date', columns='ticker', values='iv_surface_level').sort_index()
    if benchmark not in wide.columns:
        return pd.Series(dtype=float)
    betas = {}
    for t in wide.columns:
        if t == benchmark:
            continue
        betas[t] = _beta(wide.rename(columns={t: 'x', benchmark: 'b'}), 'x', 'b')
    return pd.Series(betas, name='iv_surface_beta')

# -------------- public API --------------

def build_vol_betas(mode: str, benchmark: str,
                    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
                    tenor_days: Iterable[int] = (7,30,60,90,180,365),
                    mny_bins: Iterable[Tuple[float, float]] = ((0.8,0.9),(0.95,1.05),(1.1,1.25))
                   ) -> Union[pd.Series, Dict[int, pd.Series]]:
    mode = mode.lower()
    if mode == 'ul':
        return _ul_betas(benchmark)
    if mode == 'iv_atm':
        return _iv_atm_betas(benchmark, pillar_days=pillar_days)
    if mode == 'surface':
        return _surface_betas(benchmark, tenor_days=tenor_days, mny_bins=mny_bins)
    raise ValueError("mode must be one of: 'ul', 'iv_atm', 'surface'")


def save_correlations(mode: str, benchmark: str, base_path: str = 'data') -> list[str]:
    paths: list[str] = []
    res = build_vol_betas(mode=mode, benchmark=benchmark)
    if isinstance(res, dict):
        for d, s in res.items():
            p = f"{base_path}/betas_{mode}_{d}d_vs_{benchmark}.csv"
            s.sort_index().to_csv(p, header=True)
            paths.append(p)
    else:
        p = f"{base_path}/betas_{mode}_vs_{benchmark}.csv"
        res.sort_index().to_csv(p, header=True)
        paths.append(p)
    return paths


if __name__ == '__main__':
    print(save_correlations('ul', benchmark='SPY'))
    print(save_correlations('iv_atm', benchmark='SPY'))
    print(save_correlations('surface', benchmark='SPY'))