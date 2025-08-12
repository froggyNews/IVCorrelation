# analysis/analysis_pipeline.py
"""
GUI-ready analysis orchestrator.

This module wires together:
- ingest + enrich (data.historical_saver)
- surface grid building
- synthetic ETF surface construction (surface & ATM pillars)
- vol betas (UL, IV-ATM, Surface)
- lightweight snapshot/smile helpers for GUI

All public functions are fast to call from a GUI. Heavy work is cached
in-memory and can optionally be dumped to disk (parquet) for fast reloads.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple, List, Mapping, Union
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import pandas as pd
import numpy as np

# --- project modules ---
from data.historical_saver import save_for_tickers
from data.db_utils import get_conn
from analysis.syntheticETFBuilder import build_surface_grids, DEFAULT_TENORS, DEFAULT_MNY_BINS
from analysis.syntheticETFBuilder import combine_surfaces, build_synthetic_iv as build_synthetic_iv_pillars
from analysis.correlation_builder import build_vol_betas, save_correlations, peer_weights_from_correlations
from analysis.pillars import load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS



from volModel.volModel import VolModel


from data.data_pipeline import enrich_quotes

# =========================
# Config (GUI friendly)
# =========================
@dataclass(frozen=True)
class PipelineConfig:
    tenors: Tuple[int, ...] = DEFAULT_TENORS
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS
    pillar_days: Tuple[int, ...] = tuple(DEFAULT_PILLARS_DAYS)
    use_atm_only: bool = False
    cache_dir: str = "data/cache"  # optional disk cache for GUI speed

    def ensure_cache_dir(self) -> None:
        if self.cache_dir and not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


# =========================
# Ingest
# =========================
def ingest_and_process(
    tickers: Iterable[str],
    max_expiries: int = 6,
    r: float = 0.0,
    q: float = 0.0,
) -> int:
    """Download raw chains, enrich via pipeline, and persist to DB."""
    return save_for_tickers(tickers, max_expiries=max_expiries, r=r, q=q)


# =========================
# Surfaces (for GUI)
# =========================
@lru_cache(maxsize=16)
def get_surface_grids_cached(
    cfg: PipelineConfig,
    tickers_key: str,  # lru_cache needs hashables → we pass a joined string of tickers
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    tickers = tickers_key.split(",") if tickers_key else None
    return build_surface_grids(
        tickers=tickers,
        tenors=cfg.tenors,
        mny_bins=cfg.mny_bins,
        use_atm_only=cfg.use_atm_only,
    )


def build_surfaces(
    tickers: Iterable[str] | None = None,
    cfg: PipelineConfig = PipelineConfig(),
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Return dict[ticker][date] -> IV grid DataFrame."""
    key = ",".join(sorted(tickers)) if tickers else ""
    return get_surface_grids_cached(cfg, key)


def list_surface_dates(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]]
) -> List[pd.Timestamp]:
    """All unique dates available across tickers (sorted)."""
    dates = set()
    for dct in surfaces.values():
        dates.update(dct.keys())
    return sorted(dates)


def surface_to_frame_for_date(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Extract the grid for a single date across tickers."""
    out = {}
    for t, dct in surfaces.items():
        if date in dct:
            out[t] = dct[date]
    return out


# =========================
# Synthetic ETF (surface & ATM pillars)
# =========================
def build_synthetic_surface(
    weights: Mapping[str, float],
    cfg: PipelineConfig = PipelineConfig(),
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Create a synthetic ETF surface from ticker grids + weights."""
    surfaces = build_surfaces(tickers=list(weights.keys()), cfg=cfg)
    return combine_surfaces(surfaces, weights)


def build_synthetic_iv_series(
    weights: Mapping[str, float],
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = 7.0,
) -> pd.DataFrame:
    """Create a weighted ATM pillar IV time series."""
    return build_synthetic_iv_pillars(weights, pillar_days=pillar_days, tolerance_days=tolerance_days)


# =========================
# Betas
# =========================
def compute_betas(
    mode: str,  # 'ul' | 'iv_atm' | 'surface'
    benchmark: str,
    cfg: PipelineConfig = PipelineConfig(),
):
    """Compute vol betas per requested mode (GUI can show table/heatmap)."""
    if mode == "surface":
        # surface mode uses DB directly; cfg.tenors/mny_bins are respected in builder
        return build_vol_betas(
            mode=mode, benchmark=benchmark,
            tenor_days=cfg.tenors, mny_bins=cfg.mny_bins
        )
    if mode == "iv_atm":
        return build_vol_betas(mode=mode, benchmark=benchmark, pillar_days=cfg.pillar_days)
    return build_vol_betas(mode=mode, benchmark=benchmark)


def save_betas(
    mode: str,
    benchmark: str,
    base_path: str = "data",
    cfg: PipelineConfig = PipelineConfig(),
) -> list[str]:
    """Persist betas to CSVs (GUI may expose a 'Export' button)."""
    if mode == "surface":
        # plumb through config to keep in sync with GUI filters
        res = build_vol_betas(
            mode=mode, benchmark=benchmark,
            tenor_days=cfg.tenors, mny_bins=cfg.mny_bins
        )
        p = f"{base_path}/betas_{mode}_vs_{benchmark}.csv"
        res.sort_index().to_csv(p, header=True)
        return [p]
    return save_correlations(mode=mode, benchmark=benchmark, base_path=base_path)
# =========================
# Relative value (target vs synthetic peers by corr)
# =========================

def _fetch_target_atm(target: str, pillar_days, tolerance_days: float = 7.0) -> pd.DataFrame:
    atm = load_atm()
    atm = atm[atm["ticker"] == target].copy()
    if atm.empty:
        return pd.DataFrame(columns=["asof_date","pillar_days","iv"])
    piv = nearest_pillars(atm, pillars_days=list(pillar_days), tolerance_days=tolerance_days)
    out = (piv.groupby(["asof_date","pillar_days"])["iv"].mean().rename("iv").reset_index())
    return out[["asof_date","pillar_days","iv"]]


def _rv_metrics_join(target_iv: pd.DataFrame, synth_iv: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    tgt = target_iv.rename(columns={"iv":"iv_target"})
    syn = synth_iv.rename(columns={"iv":"iv_synth"})
    df = pd.merge(tgt, syn, on=["asof_date","pillar_days"], how="inner").sort_values(["pillar_days","asof_date"])
    if df.empty:
        return df
    def per_pillar(g):
        g = g.copy()
        g["spread"] = g["iv_target"] - g["iv_synth"]
        roll = max(5, int(lookback // 5))
        m = g["spread"].rolling(lookback, min_periods=roll).mean()
        s = g["spread"].rolling(lookback, min_periods=roll).std(ddof=1)
        g["z"] = (g["spread"] - m) / s
        def _pct_rank(x):
            r = x.rank(pct=True).iloc[-1]
            return r
        g["pct_rank"] = g["spread"].rolling(lookback, min_periods=roll).apply(_pct_rank, raw=False)
        return g
    return df.groupby("pillar_days", group_keys=False).apply(per_pillar).reset_index(drop=True)


def relative_value_atm_report_corrweighted(
    target: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",                # 'ul' | 'iv_atm' | 'surface'  (weights source)
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    lookback: int = 60,
    tolerance_days: float = 7.0,
    weight_power: float = 1.0,
    clip_negative: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute relative value (spread/z/pct) using correlation-based peer weights.
    Returns (rv_dataframe, weights_used).
    """
    # 1) compute weights from correlations vs target
    w = peer_weights_from_correlations(
        benchmark=target,
        peers=peers,
        mode=mode,
        pillar_days=pillar_days,
        tenor_days=DEFAULT_TENORS,
        mny_bins=DEFAULT_MNY_BINS,
        clip_negative=clip_negative,
        power=weight_power,
    )
    if w.empty:
        return pd.DataFrame(columns=["asof_date","pillar_days","iv_target","iv_synth","spread","z","pct_rank"]), w

    # 2) synthetic ATM series using those weights
    synth = build_synthetic_iv_series(weights=w.to_dict(), pillar_days=pillar_days, tolerance_days=tolerance_days)
    # 3) target ATM series
    tgt = _fetch_target_atm(target, pillar_days=pillar_days, tolerance_days=tolerance_days)
    # 4) join + metrics
    rv = _rv_metrics_join(tgt, synth, lookback=lookback)
    return rv, w


def latest_relative_snapshot_corrweighted(
    target: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",
    pillar_days: Iterable[int] = (7,30,60,90),
    lookback: int = 60,
    tolerance_days: float = 7.0,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience: last date per pillar with RV metrics and the weights used.
    """
    rv, w = relative_value_atm_report_corrweighted(
        target=target, peers=peers, mode=mode, pillar_days=pillar_days,
        lookback=lookback, tolerance_days=tolerance_days, **kwargs
    )
    if rv.empty:
        return rv, w
    last = rv.sort_values("asof_date").groupby("pillar_days").tail(1)
    cols = ["asof_date","pillar_days","iv_target","iv_synth","spread","z","pct_rank"]
    return last[cols].sort_values("pillar_days"), w


# =========================
# ATM pillars (for GUI lists & filters)
# =========================
@lru_cache(maxsize=8)
def get_atm_pillars_cached() -> pd.DataFrame:
    """Tidy ATM rows from DB (asof_date, ticker, expiry, ttm_years, iv, spot, moneyness, delta, pillar_days, ...)"""
    atm = load_atm()
    return atm


def available_tickers() -> List[str]:
    """Unique tickers present in DB (for GUI dropdowns)."""
    conn = get_conn()
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM options_quotes ORDER BY 1", conn)["ticker"].tolist()
    return tickers


def available_dates(ticker: Optional[str] = None) -> List[str]:
    """Unique asof_date strings (optionally filtered by ticker)."""
    conn = get_conn()
    base = "SELECT DISTINCT asof_date FROM options_quotes"
    if ticker:
        df = pd.read_sql_query(f"{base} WHERE ticker = ? ORDER BY 1", conn, params=[ticker])
    else:
        df = pd.read_sql_query(f"{base} ORDER BY 1", conn)
    return df["asof_date"].tolist()


# =========================
# Smile helpers (GUI plotting)
# =========================
def get_smile_slice(
    ticker: str,
    asof_date: str,
    T_target_years: float | None = None,
    call_put: Optional[str] = None,  # 'C' or 'P' or None for both
    nearest_by: str = "T",           # 'T' or 'moneyness'
) -> pd.DataFrame:
    """
    Return a slice of quotes for plotting a smile (one date, one ticker).
    If T_target_years is given, returns the nearest expiry; otherwise returns all expiries that day.
    Optionally filter by call_put ('C'/'P').
    """
    conn = get_conn()
    q = """
        SELECT asof_date, ticker, expiry, call_put, strike AS K, spot AS S, ttm_years AS T,
               moneyness, iv AS sigma, delta, is_atm
        FROM options_quotes
        WHERE asof_date = ? AND ticker = ?
    """
    df = pd.read_sql_query(q, conn, params=[asof_date, ticker])
    if df.empty:
        return df

    if call_put in ("C", "P"):
        df = df[df["call_put"] == call_put]

    if T_target_years is not None and not df.empty:
        # pick nearest expiry
        idx = (df["T"] - float(T_target_years)).abs().groupby(df["expiry"]).transform("min") == \
              (df["T"] - float(T_target_years)).abs()
        # keep only the nearest expiry rows
        df = df[idx]
        # in case of ties (multiple expiries equally near), keep the first expiry group
        if df["expiry"].nunique() > 1:
            first_expiry = df.groupby("expiry").size().sort_values(ascending=False).index[0]
            df = df[df["expiry"] == first_expiry]

    # Sort for nicer plots
    return df.sort_values(["call_put", "T", "moneyness", "K"]).reset_index(drop=True)

def fit_smile_for(
    ticker: str,
    asof_date: str,
    model: str = "svi",             # "svi" or "sabr"
    min_quotes_per_expiry: int = 3, # skip super sparse expiries
    beta: float = 0.5,              # SABR beta (fixed)
) -> VolModel:
    """
    Fit a volatility smile model (SVI or SABR) for one day/ticker using all expiries available that day.

    Returns a VolModel you can query/plot from the GUI:
      vm.available_expiries() -> list of T (years)
      vm.predict_iv(K, T)     -> IV at (K, T) using nearest fitted expiry
      vm.smile(Ks, T)         -> vectorized IVs across Ks at nearest fitted expiry
      vm.plot(T)              -> quick plot (if matplotlib installed)

    Notes:
      - Uses median spot across that day’s quotes for S.
      - Filters out expiries with fewer than `min_quotes_per_expiry` quotes.
    """
    import pandas as pd
    conn = get_conn()
    q = """
        SELECT spot AS S, strike AS K, ttm_years AS T, iv AS sigma
        FROM options_quotes
        WHERE asof_date = ? AND ticker = ?
    """
    df = pd.read_sql_query(q, conn, params=[asof_date, ticker])
    if df.empty:
        return VolModel(model=model)  # empty model

    # Median spot for the day
    S = float(df["S"].median())

    # Drop junk, enforce per-expiry density
    df = df.dropna(subset=["K", "T", "sigma"]).copy()
    if df.empty:
        return VolModel(model=model)

    # keep only expiries with at least min_quotes_per_expiry quotes
    counts = df.groupby("T").size()
    good_T = counts[counts >= max(1, int(min_quotes_per_expiry))].index
    df = df[df["T"].isin(good_T)]
    if df.empty:
        return VolModel(model=model)

    Ks = df["K"].to_numpy()
    Ts = df["T"].to_numpy()
    IVs = df["sigma"].to_numpy()

    vm = VolModel(model=model).fit(S, Ks, Ts, IVs, beta=beta)
    return vm


def sample_smile_curve(
    ticker: str,
    asof_date: str,
    T_target_years: float,
    model: str = "svi",
    moneyness_grid: tuple[float, float, int] = (0.6, 1.4, 81),  # (lo, hi, n)
    beta: float = 0.5,
) -> pd.DataFrame:
    """
    Convenience: fit a smile then return a tidy curve at the nearest expiry to T_target_years.

    Returns DataFrame with columns:
      ['asof_date','ticker','model','T_used','moneyness','K','IV']
    """
    import numpy as np
    import pandas as pd

    vm = fit_smile_for(ticker, asof_date, model=model, beta=beta)
    if not vm.available_expiries() or vm.S is None:
        return pd.DataFrame(columns=["asof_date","ticker","model","T_used","moneyness","K","IV"])

    # nearest fitted expiry to requested T
    Ts = np.array(vm.available_expiries(), dtype=float)
    Tq = float(T_target_years)
    T_used = float(Ts[np.argmin(np.abs(Ts - Tq))])

    lo, hi, n = moneyness_grid
    m_grid = np.linspace(float(lo), float(hi), int(n))
    K_grid = m_grid * vm.S
    iv = vm.smile(K_grid, T_used)

    out = pd.DataFrame({
        "asof_date": asof_date,
        "ticker": ticker,
        "model": model.upper(),
        "T_used": T_used,
        "moneyness": m_grid,
        "K": K_grid,
        "IV": iv,
    })
    return out

# =========================
# Lightweight disk cache (optional)
# =========================
def dump_surface_to_cache(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    cfg: PipelineConfig,
    tag: str = "default",
) -> str:
    """Store surfaces as parquet for fast GUI reloads."""
    cfg.ensure_cache_dir()
    rows = []
    for t, dct in surfaces.items():
        for date, grid in dct.items():
            # melt grid to tidy rows
            g = grid.copy()
            g.insert(0, "mny_bin", g.index.astype(str))
            tidy = g.melt(id_vars="mny_bin", var_name="tenor_days", value_name="iv")
            tidy["ticker"] = t
            tidy["asof_date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
            rows.append(tidy)
    if not rows:
        path = os.path.join(cfg.cache_dir, f"surfaces_{tag}.parquet")
        # create empty frame to write
        pd.DataFrame(columns=["ticker","asof_date","mny_bin","tenor_days","iv"]).to_parquet(path, index=False)
        return path

    out = pd.concat(rows, ignore_index=True)
    path = os.path.join(cfg.cache_dir, f"surfaces_{tag}.parquet")
    out.to_parquet(path, index=False)
    # save config next to it
    with open(os.path.join(cfg.cache_dir, f"surfaces_{tag}.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=list)
    return path


def load_surface_from_cache(path: str) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Reload cached surfaces (for GUI cold start)."""
    if not os.path.isfile(path):
        return {}
    df = pd.read_parquet(path)
    out: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    for (t, date), g in df.groupby(["ticker", "asof_date"]):
        grid = g.pivot(index="mny_bin", columns="tenor_days", values="iv").sort_index(axis=1)
        out.setdefault(t, {})[pd.to_datetime(date)] = grid
    return out


# =========================
# __main__ demo path
# =========================
if __name__ == "__main__":
    cfg = PipelineConfig()
    # 1) Ingest a couple of tickers (comment out if DB already populated)
    inserted = ingest_and_process(["SPY", "QQQ"], max_expiries=6, r=0.0, q=0.0)
    print(f"Inserted rows: {inserted}")

    # 2) Build and cache surfaces for the GUI
    surfaces = build_surfaces(["SPY", "QQQ"], cfg=cfg)
    cache_path = dump_surface_to_cache(surfaces, cfg, tag="spyqqq")
    print("Surface cache:", cache_path)

    # 3) Compute and save IV-ATM betas vs SPY
    paths = save_betas(mode="iv_atm", benchmark="SPY", base_path="data", cfg=cfg)
    print("Saved:", paths)

    # 4) Quick smile slice for GUI
    df_smile = get_smile_slice("SPY", asof_date=available_dates("SPY")[-1], T_target_years=30/365.25, call_put="C")
    print("Smile rows:", len(df_smile))

        # 1) Fit SVI for SPY on the latest date in DB, and list fitted expiries
    d = available_dates("SPY")[-1]
    vm = fit_smile_for("SPY", d, model="svi")
    print("Fitted expiries:", vm.available_expiries())

    # 2) Sample a smile curve around 30D (nearest fitted expiry), ready for plotting
    curve = sample_smile_curve("SPY", d, T_target_years=30/365.25, model="svi")
    print(curve.head())

    # 3) (Optional) quick plot from the model
    # vm.plot(30/365.25)
