# analysis/analysis_pipeline.py
"""
GUI-ready analysis orchestrator (modern, slim).

This module wires together:
- ingest (download + persist)
- surface grid building
- synthetic ETF surface & ATM-pillar curves
- unified peer-weight computation
- lightweight smile/term helpers for GUI

Design:
- No legacy modes / tuple dispatch
- Single weight entrypoint -> unified_weights.compute_unified_weights
- Small, dependency-light file
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List, Mapping, Union

import json
import logging
import os
import numpy as np
import pandas as pd

from data.data_downloader import save_for_tickers
from data.db_utils import get_conn
from data.interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD

from volModel.volModel import VolModel

from .syntheticETFBuilder import (
    build_surface_grids,
    DEFAULT_TENORS,
    DEFAULT_MNY_BINS,
    combine_surfaces,
    build_synthetic_iv as build_synthetic_iv_pillars,
)
from .pillars import (
    load_atm,
    DEFAULT_PILLARS_DAYS,
    _fit_smile_get_atm,
    compute_atm_by_expiry,
    atm_curve_for_ticker_on_date,
)
from .confidence_bands import (
    Bands,
    synthetic_etf_pillar_bands,
    svi_confidence_bands,
    sabr_confidence_bands,
    tps_confidence_bands,
)
from beta_builder.unified_weights import compute_unified_weights

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Global connection cache
# -----------------------------------------------------------------------------
_RO_CONN = None

# -----------------------------------------------------------------------------
# Config (GUI friendly)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for surface building and caching (GUI friendly)."""
    tenors: Tuple[int, ...] = DEFAULT_TENORS
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS
    pillar_days: Tuple[int, ...] = tuple(DEFAULT_PILLARS_DAYS)
    use_atm_only: bool = False
    max_expiries: Optional[int] = None  # Limit expiries in smiles/surfaces (perf)
    cache_dir: str = "data/cache"       # Optional disk cache for GUI speed

    def ensure_cache_dir(self) -> None:
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------
def ingest_and_process(
    tickers: Iterable[str],
    max_expiries: int = 6,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> int:
    """Download raw chains, enrich via pipeline, and persist to DB."""
    tickers = [t.upper() for t in tickers]
    logger.info("Ingesting: %s (max_expiries=%s)", ",".join(tickers), max_expiries)
    return save_for_tickers(tickers, max_expiries=max_expiries, r=r, q=q)

# -----------------------------------------------------------------------------
# Surfaces (for GUI)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def get_surface_grids_cached(
    cfg: PipelineConfig,
    tickers_key: str,  # joined string of tickers; lru_cache wants hashables
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    tickers = tickers_key.split(",") if tickers_key else None
    return build_surface_grids(
        tickers=tickers,
        tenors=cfg.tenors,
        mny_bins=cfg.mny_bins,
        use_atm_only=cfg.use_atm_only,
        max_expiries=cfg.max_expiries,
    )

def build_surfaces(
    tickers: Iterable[str] | None = None,
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = False,
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Return dict[ticker][date] -> IV grid DataFrame."""
    key = ",".join(sorted([t.upper() for t in tickers])) if tickers else ""
    all_surfaces = get_surface_grids_cached(cfg, key)

    if most_recent_only and all_surfaces:
        # Prefer global most recent; fallback to each ticker's latest available date
        most_recent = get_most_recent_date_global()
        if most_recent:
            most_recent_ts = pd.to_datetime(most_recent)
            filtered: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
            for ticker, date_dict in all_surfaces.items():
                if most_recent_ts in date_dict:
                    filtered[ticker] = {most_recent_ts: date_dict[most_recent_ts]}
                elif date_dict:
                    latest = max(date_dict.keys())
                    filtered[ticker] = {latest: date_dict[latest]}
            return filtered
    return all_surfaces

def list_surface_dates(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]]
) -> List[pd.Timestamp]:
    """All unique dates available across tickers (sorted)."""
    dates: set[pd.Timestamp] = set()
    for dct in surfaces.values():
        dates.update(dct.keys())
    return sorted(dates)

def surface_to_frame_for_date(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Extract the grid for a single date across tickers."""
    return {t: dct[date] for t, dct in surfaces.items() if date in dct}

# -----------------------------------------------------------------------------
# Weights (new unified only)
# -----------------------------------------------------------------------------
def compute_peer_weights(
    target: str,
    peers: Iterable[str],
    *,
    weight_mode: str = "corr_iv_atm",
    asof: str | None = None,
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tenor_days: Iterable[int] = DEFAULT_TENORS,
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS,
) -> pd.Series:
    """
    Compute normalized peer weights via the unified engine.

    weight_mode examples:
      "corr_iv_atm", "cosine_iv_atm", "pca_iv_atm",
      "corr_surface", "cosine_surface", "pca_surface_grid",
      "ul", "equal", "oi"
    """
    target = target.upper()
    peer_list = [p.upper() for p in peers]
    return compute_unified_weights(
        target=target,
        peers=peer_list,
        mode=weight_mode,
        asof=asof,
        pillars_days=tuple(pillar_days),
        tenors=tuple(tenor_days),
        mny_bins=tuple(mny_bins),
    )

# -----------------------------------------------------------------------------
# Synthetic ETF constructions
# -----------------------------------------------------------------------------
def build_synthetic_surface(
    weights: Mapping[str, float],
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = True,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Create a synthetic ETF surface from ticker grids + weights."""
    w = {k.upper(): float(v) for k, v in weights.items()}
    surfaces = build_surfaces(tickers=list(w.keys()), cfg=cfg, most_recent_only=most_recent_only)
    return combine_surfaces(surfaces, w)

def build_synthetic_iv_series_weighted(
    weights: Mapping[str, float],
    *,
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = 7.0,
) -> pd.DataFrame:
    """Build a weighted synthetic ATM pillar IV series (no weights computation here)."""
    return build_synthetic_iv_pillars({k.upper(): float(v) for k, v in weights.items()},
                                      pillar_days=pillar_days, tolerance_days=tolerance_days)

def build_synthetic_iv_series_corrweighted(
    target: str,
    peers: Iterable[str],
    *,
    weight_mode: str = "corr_iv_atm",
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = 7.0,
    asof: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build correlation/PCA/cosine/OI‑weighted synthetic ATM pillar IV series."""
    w = compute_peer_weights(
        target=target, peers=peers, weight_mode=weight_mode, asof=asof, pillar_days=pillar_days
    )
    df = build_synthetic_iv_pillars(w.to_dict(), pillar_days=pillar_days, tolerance_days=tolerance_days)
    return df, w

# -----------------------------------------------------------------------------
# Basic DB helpers + caches
# -----------------------------------------------------------------------------
def _get_ro_conn():
    global _RO_CONN
    if _RO_CONN is None:
        _RO_CONN = get_conn()
    return _RO_CONN

@lru_cache(maxsize=8)
def get_atm_pillars_cached() -> pd.DataFrame:
    """Tidy ATM rows from DB (asof_date, ticker, expiry, ttm_years, iv, spot, moneyness, delta, pillar_days, ...)"""
    return load_atm()

@lru_cache(maxsize=1)
def available_tickers() -> List[str]:
    """Unique tickers present in DB (for GUI dropdowns)."""
    conn = _get_ro_conn()
    return pd.read_sql_query(
        "SELECT DISTINCT ticker FROM options_quotes ORDER BY 1", conn
    )["ticker"].tolist()

@lru_cache(maxsize=None)
def available_dates(ticker: Optional[str] = None, most_recent_only: bool = False) -> List[str]:
    """Get available asof_date strings (globally or for a ticker)."""
    conn = _get_ro_conn()
    if most_recent_only:
        from data.db_utils import get_most_recent_date
        recent = get_most_recent_date(conn, ticker)
        return [recent] if recent else []
    base = "SELECT DISTINCT asof_date FROM options_quotes"
    if ticker:
        df = pd.read_sql_query(f"{base} WHERE ticker = ? ORDER BY 1", conn, params=[ticker])
    else:
        df = pd.read_sql_query(f"{base} ORDER BY 1", conn)
    return df["asof_date"].tolist()

def get_most_recent_date_global() -> Optional[str]:
    """Most recent asof_date across all tickers."""
    conn = _get_ro_conn()
    from data.db_utils import get_most_recent_date
    return get_most_recent_date(conn)

def invalidate_cache() -> None:
    """Clear all in-memory caches and reset shared connection."""
    get_surface_grids_cached.cache_clear()
    get_atm_pillars_cached.cache_clear()
    available_tickers.cache_clear()
    available_dates.cache_clear()
    global _RO_CONN
    if _RO_CONN is not None:
        try:
            _RO_CONN.close()
        except Exception:
            pass
        _RO_CONN = None

# -----------------------------------------------------------------------------
# Smile helpers (GUI plotting)
# -----------------------------------------------------------------------------
def get_smile_slice(
    ticker: str,
    asof_date: Optional[str] = None,
    T_target_years: float | None = None,
    call_put: Optional[str] = None,  # 'C' or 'P' or None for both
    max_expiries: Optional[int] = None,  # Limit number of expiries
) -> pd.DataFrame:
    """
    Return a slice of quotes for plotting a smile (one date, one ticker).
    If asof_date is None, uses the most recent date for the ticker.
    If T_target_years is given, returns only the nearest expiry; otherwise returns all expiries that day.
    Optionally filter by call_put ('C'/'P').
    """
    conn = get_conn()
    ticker = ticker.upper()

    if asof_date is None:
        from data.db_utils import get_most_recent_date
        asof_date = get_most_recent_date(conn, ticker)
        if asof_date is None:
            return pd.DataFrame()

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

    # Optionally limit number of expiries (smallest T first)
    if max_expiries is not None and max_expiries > 0 and not df.empty:
        unique_expiries = df.groupby("expiry")["T"].first().sort_values()
        limited_expiries = unique_expiries.head(max_expiries).index.tolist()
        df = df[df["expiry"].isin(limited_expiries)]

    if T_target_years is not None and not df.empty:
        # pick nearest expiry to T_target_years
        abs_diff = (df["T"] - float(T_target_years)).abs()
        nearest_mask = abs_diff.groupby(df["expiry"]).transform("min") == abs_diff
        df = df[nearest_mask]
        # if multiple expiries tie, keep the largest bucket (most rows)
        if df["expiry"].nunique() > 1:
            first_expiry = df.groupby("expiry").size().sort_values(ascending=False).index[0]
            df = df[df["expiry"] == first_expiry]

    return df.sort_values(["call_put", "T", "moneyness", "K"]).reset_index(drop=True)

def prepare_smile_data(
    target: str,
    asof: str,
    T_days: float,
    model: str = "svi",
    ci: float = 68.0,
    overlay_synth: bool = False,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    overlay_peers: bool = False,
    max_expiries: int = 6,
) -> Dict[str, Any]:
    """Precompute smile data and fitted parameters for plotting."""
    peers = list(peers or [])

    asof_ts = pd.to_datetime(asof).normalize()
    try:
        from .model_params_logger import append_params, load_model_params
        params_cache = load_model_params()
        params_cache = params_cache[
            (params_cache["ticker"] == target)
            & (params_cache["asof_date"] == asof_ts)
            & (params_cache["model"].isin(["svi", "sabr", "tps", "sens"]))
        ]
    except Exception:
        append_params = None  # type: ignore
        params_cache = pd.DataFrame()

    df = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df is None or df.empty:
        return {}

    T_arr = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
    K_arr = pd.to_numeric(df["K"], errors="coerce").to_numpy(float)
    sigma_arr = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(float)
    S_arr = pd.to_numeric(df["S"], errors="coerce").to_numpy(float)
    expiry_arr = pd.to_datetime(df.get("expiry"), errors="coerce").to_numpy()

    Ts = np.sort(np.unique(T_arr[np.isfinite(T_arr)]))
    if Ts.size == 0:
        return {}
    idx0 = int(np.argmin(np.abs(Ts * 365.25 - float(T_days))))
    T0 = float(Ts[idx0])

    # fit per-expiry
    fit_by_expiry: Dict[float, Dict[str, Any]] = {}
    for T_val in Ts:
        mask = np.isclose(T_arr, T_val)
        if not np.any(mask):
            tol = 1e-6
            mask = (T_arr >= T_val - tol) & (T_arr <= T_val + tol)
        if not np.any(mask):
            continue

        S = float(np.nanmedian(S_arr[mask])) if np.any(mask) else float("nan")
        K = K_arr[mask]
        IV = sigma_arr[mask]

        expiry_dt = None
        if expiry_arr.size and np.any(mask):
            try:
                expiry_dt = pd.to_datetime(expiry_arr[mask][0])
            except Exception:
                pass

        tenor_d = int((expiry_dt - asof_ts).days) if expiry_dt is not None else int(round(float(T_val) * 365.25))

        def _cached(model: str) -> Optional[Dict[str, float]]:
            if params_cache.empty:
                return None
            sub = params_cache[(params_cache["tenor_d"] == tenor_d) & (params_cache["model"] == model)]
            return None if sub.empty else sub.set_index("param")["value"].to_dict()

        # SVI
        svi_params = _cached("svi")
        if not svi_params:
            from volModel.sviFit import fit_svi_slice
            svi_params = fit_svi_slice(S, K, T_val, IV)
            if append_params:
                try:
                    append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None, "svi", svi_params,
                                  meta={"rmse": svi_params.get("rmse")})
                except Exception:
                    pass

        # SABR
        sabr_params = _cached("sabr")
        if not sabr_params:
            from volModel.sabrFit import fit_sabr_slice
            sabr_params = fit_sabr_slice(S, K, T_val, IV)
            if append_params:
                try:
                    append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None, "sabr", sabr_params,
                                  meta={"rmse": sabr_params.get("rmse")})
                except Exception:
                    pass

        # TPS (optional)
        tps_params = _cached("tps")
        if not tps_params:
            try:
                from volModel.polyFit import fit_tps_slice
                tps_params = fit_tps_slice(S, K, T_val, IV)
                if append_params:
                    append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None, "tps", tps_params,
                                  meta={"rmse": tps_params.get("rmse")})
            except Exception:
                tps_params = {}

        # simple sensitivities
        sens_params = _cached("sens")
        if not sens_params:
            dfe = df[mask].copy()
            try:
                dfe["moneyness"] = dfe["K"].astype(float) / float(S)
            except Exception:
                dfe["moneyness"] = np.nan
            sens = _fit_smile_get_atm(dfe, model="auto")
            sens_params = {k: sens[k] for k in ("atm_vol", "skew", "curv") if k in sens}
            if append_params:
                try:
                    append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None, "sens", sens_params)
                except Exception:
                    pass

        # optional CI bands at requested level
        bands_map: Dict[str, Bands] = {}
        if ci and ci > 0:
            m_grid = np.linspace(0.7, 1.3, 121)
            K_grid = m_grid * S
            try:
                bands_map["svi"] = svi_confidence_bands(S, K, T_val, IV, K_grid, level=float(ci))
            except Exception:
                pass
            try:
                bands_map["sabr"] = sabr_confidence_bands(S, K, T_val, IV, K_grid, level=float(ci))
            except Exception:
                pass
            try:
                bands_map["tps"] = tps_confidence_bands(S, K, T_val, IV, K_grid, level=float(ci))
            except Exception:
                pass

        entry = {
            "svi": svi_params,
            "sabr": sabr_params,
            "tps": tps_params,
            "sens": sens_params,
            "expiry": str(expiry_dt) if expiry_dt is not None else None,
        }
        if bands_map:
            entry["bands"] = bands_map
        fit_by_expiry[T_val] = entry

    fit_entry = fit_by_expiry.get(T0, {})
    fit_info = {
        "ticker": target,
        "asof": asof,
        "expiry": fit_entry.get("expiry"),
        "svi": fit_entry.get("svi", {}),
        "sabr": fit_entry.get("sabr", {}),
        "tps": fit_entry.get("tps", {}),
        "sens": fit_entry.get("sens", {}),
    }

    # optional overlays
    tgt_surface = None
    syn_surface = None
    if peers:
        try:
            tickers = list({target, *peers})
            surfaces = build_surface_grids(tickers=tickers, use_atm_only=False, max_expiries=max_expiries)
            if target in surfaces and asof in surfaces[target]:
                tgt_surface = surfaces[target][asof]
            peer_surfaces = {p: surfaces[p] for p in peers if p in surfaces and asof in surfaces[p]}
            if peer_surfaces:
                w = {p: float(weights.get(p, 1.0)) for p in peer_surfaces} if weights else {p: 1.0 for p in peer_surfaces}
                synth_by_date = combine_surfaces(peer_surfaces, w)
                syn_surface = synth_by_date.get(asof)
        except Exception:
            tgt_surface = None
            syn_surface = None

    peer_slices: Dict[str, Dict[str, np.ndarray]] = {}
    if overlay_peers and peers:
        for p in peers:
            df_p = get_smile_slice(p, asof, T_target_years=None, max_expiries=max_expiries)
            if df_p is None or df_p.empty:
                continue
            T_p = pd.to_numeric(df_p["T"], errors="coerce").to_numpy(float)
            K_p = pd.to_numeric(df_p["K"], errors="coerce").to_numpy(float)
            sigma_p = pd.to_numeric(df_p["sigma"], errors="coerce").to_numpy(float)
            S_p = pd.to_numeric(df_p["S"], errors="coerce").to_numpy(float)
            peer_slices[p.upper()] = {"T_arr": T_p, "K_arr": K_p, "sigma_arr": sigma_p, "S_arr": S_p}

    return {
        "T_arr": T_arr,
        "K_arr": K_arr,
        "sigma_arr": sigma_arr,
        "S_arr": S_arr,
        "Ts": Ts,
        "idx0": idx0,
        "tgt_surface": tgt_surface,
        "syn_surface": syn_surface,
        "peer_slices": peer_slices,
        "expiry_arr": expiry_arr,
        "fit_info": fit_info,
        "fit_by_expiry": fit_by_expiry,
    }

def prepare_term_data(
    target: str,
    asof: str,
    ci: float = 68.0,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    atm_band: float = 0.05,
    max_expiries: int = 6,
) -> Dict[str, Any]:
    """Precompute ATM term structure and (optional) synthetic overlay for plotting."""
    df_all = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df_all is None or df_all.empty:
        return {}

    min_boot = 64 if (ci and ci > 0) else 0
    atm_curve = compute_atm_by_expiry(
        df_all,
        atm_band=atm_band,
        method="fit",
        model="auto",
        vega_weighted=True,
        n_boot=min_boot,
        ci_level=ci,
    )

    synth_curve = None
    synth_bands = None

    if peers:
        w = pd.Series(weights if weights else {p: 1.0 for p in peers}, dtype=float)
        if w.sum() <= 0:
            w = pd.Series({p: 1.0 for p in peers}, dtype=float)
        w = (w / w.sum()).astype(float)
        peers = [p for p in w.index if p in (peers or [])]

        curves: Dict[str, pd.DataFrame] = {}
        for p in peers:
            c = atm_curve_for_ticker_on_date(
                get_smile_slice,
                p,
                asof,
                atm_band=atm_band,
                method="median",
                model="auto",
                vega_weighted=False,
            )
            if not c.empty:
                curves[p] = c

        if curves:
            # Align common expiries across target and peers (tolerance = 10d)
            tol_years = 10.0 / 365.25
            arrays = [atm_curve["T"].to_numpy(float)] + [c["T"].to_numpy(float) for c in curves.values()]
            common_T = arrays[0]
            for arr in arrays[1:]:
                common_T = np.array([t for t in common_T if np.any(np.abs(arr - t) <= tol_years)], float)
                if common_T.size == 0:
                    break

            if common_T.size > 0:
                common_T = np.sort(common_T)
                atm_curve = atm_curve[atm_curve["T"].apply(lambda x: np.any(np.abs(common_T - x) <= tol_years))]

                # Build per-peer ATM arrays aligned to common_T
                atm_data: Dict[str, np.ndarray] = {}
                for p, c in curves.items():
                    arr_T = c["T"].to_numpy(float)
                    arr_v = c["atm_vol"].to_numpy(float)
                    vals = []
                    for t in common_T:
                        j = int(np.argmin(np.abs(arr_T - t)))
                        if np.abs(arr_T[j] - t) <= tol_years:
                            vals.append(arr_v[j])
                    if len(vals) == len(common_T):
                        atm_data[p] = np.array(vals, float)

                if atm_data:
                    pillar_days = common_T * 365.25
                    level = float(ci) / 100.0 if ci and ci > 0 else 0.68
                    n_boot = max(min_boot, 1)
                    synth_bands = synthetic_etf_pillar_bands(
                        atm_data,
                        w.to_dict(),
                        pillar_days,
                        level=level,
                        n_boot=n_boot,
                    )
                    synth_curve = pd.DataFrame(
                        {
                            "T": common_T,
                            "atm_vol": synth_bands.mean,
                            "atm_lo": synth_bands.lo,
                            "atm_hi": synth_bands.hi,
                        }
                    )

    return {"atm_curve": atm_curve, "synth_curve": synth_curve, "synth_bands": synth_bands}

# -----------------------------------------------------------------------------
# Disk cache helpers (optional, tiny)
# -----------------------------------------------------------------------------
def dump_surface_to_cache(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    cfg: PipelineConfig,
    tag: str = "default",
) -> str:
    """Store surfaces as parquet for fast GUI reloads."""
    cfg.ensure_cache_dir()
    base = Path(cfg.cache_dir)
    rows: list[pd.DataFrame] = []

    for t, dct in surfaces.items():
        for date, grid in dct.items():
            g = grid.copy()
            g.insert(0, "mny_bin", g.index.astype(str))
            tidy = g.melt(id_vars="mny_bin", var_name="tenor_days", value_name="iv")
            tidy["ticker"] = t
            tidy["asof_date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
            rows.append(tidy)

    path = base / f"surfaces_{tag}.parquet"
    if not rows:
        pd.DataFrame(columns=["ticker", "asof_date", "mny_bin", "tenor_days", "iv"]).to_parquet(path, index=False)
    else:
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)

    (base / f"surfaces_{tag}.json").write_text(json.dumps(asdict(cfg), indent=2, default=list))
    return str(path)

def load_surface_from_cache(path: str) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Reload cached surfaces (for GUI cold start)."""
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_parquet(p)
    out: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    if df.empty:
        return out
    for (t, date), g in df.groupby(["ticker", "asof_date"]):
        grid = g.pivot(index="mny_bin", columns="tenor_days", values="iv").sort_index(axis=1)
        out.setdefault(t, {})[pd.to_datetime(date)] = grid
    return out

def is_cache_valid(cfg: PipelineConfig, tag: str = "default") -> bool:
    """Check if disk cache is valid for the given configuration."""
    base = Path(cfg.cache_dir)
    cache_path = base / f"surfaces_{tag}.parquet"
    config_path = base / f"surfaces_{tag}.json"
    if not (cache_path.is_file() and config_path.is_file()):
        return False
    try:
        cached_config = json.loads(config_path.read_text())
        current_config = asdict(cfg)
        def _normalize(v):
            if isinstance(v, tuple):
                return [_normalize(x) for x in v]
            if isinstance(v, list):
                return [_normalize(x) for x in v]
            return v
        fields = ["tenors", "mny_bins", "pillar_days", "use_atm_only", "max_expiries"]
        return all(_normalize(cached_config.get(f)) == _normalize(current_config.get(f)) for f in fields)
    except Exception:
        return False

def load_surface_from_cache_if_valid(cfg: PipelineConfig, tag: str = "default") -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Load cached surfaces only if the cache is valid for the current configuration."""
    if not is_cache_valid(cfg, tag):
        return {}
    return load_surface_from_cache(str(Path(cfg.cache_dir) / f"surfaces_{tag}.parquet"))

# -----------------------------------------------------------------------------
# __main__ demo (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = PipelineConfig()
    print("=== IVCorrelation Analysis Pipeline (modern) ===")
    try:
        surfaces = build_surfaces(["SPY", "QQQ"], cfg=cfg, most_recent_only=True)
        print("Built surfaces:", [k for k in surfaces.keys()])
        weights = compute_peer_weights("SPY", ["QQQ"], weight_mode="corr_iv_atm")
        print("Weights:\n", weights)
        synth, w = build_synthetic_iv_series_corrweighted("SPY", ["QQQ"], weight_mode="corr_iv_atm")
        print("Synthetic ATM pillars len:", len(synth))
    except Exception as e:
        print("Demo error:", e)
