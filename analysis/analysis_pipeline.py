# analysis/analysis_pipeline.py
"""
GUI-ready analysis orchestrator (modern, slim).

This module wires together:
- ingest (download + persist)
- surface grid building
- composite ETF surface & ATM-pillar curves
- unified peer-weight computation
- lightweight smile/term helpers for GUI

Design:
- No legacy modes / tuple dispatch
- Single weight entrypoint -> unified_weights.compute_unified_weights
- Small, dependency-light file
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List, Mapping, Union, Callable

import json
import logging
import os
import numpy as np
import pandas as pd

from data.data_downloader import save_for_tickers
from data.db_utils import get_conn
from data.interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD

from volModel.volModel import VolModel

from .compositeETFBuilder import (
    build_surface_grids,
    DEFAULT_TENORS,
    DEFAULT_MNY_BINS,
    combine_surfaces,
    build_composite_iv as build_composite_iv_pillars,
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
    composite_etf_pillar_bands,
    svi_confidence_bands,
    sabr_confidence_bands,
    tps_confidence_bands,
)
from .beta_builder.unified_weights import compute_unified_weights

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
    tenors: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_TENORS)
    mny_bins: Tuple[Tuple[float, float], ...] = field(default_factory=lambda: DEFAULT_MNY_BINS)
    pillar_days: Tuple[int, ...] = field(default_factory=lambda: tuple(DEFAULT_PILLARS_DAYS))
    use_atm_only: bool = False
    max_expiries: Optional[int] = None
    cache_dir: str = "data/cache"

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
# composite ETF constructions
# -----------------------------------------------------------------------------
def build_composite_surface(
    weights: Mapping[str, float],
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = True,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Create a composite ETF surface from ticker grids + weights."""
    w = {k.upper(): float(v) for k, v in weights.items()}
    surfaces = build_surfaces(tickers=list(w.keys()), cfg=cfg, most_recent_only=most_recent_only)
    return combine_surfaces(surfaces, w)

def build_composite_iv_series_weighted(
    weights: Mapping[str, float],
    *,
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = 7.0,
) -> pd.DataFrame:
    """Build a weighted composite ATM pillar IV series (no weights computation here)."""
    return build_composite_iv_pillars({k.upper(): float(v) for k, v in weights.items()},
                                      pillar_days=pillar_days, tolerance_days=tolerance_days)

def build_composite_iv_series_corrweighted(
    target: str,
    peers: Iterable[str],
    *,
    weight_mode: str = "corr_iv_atm",
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = 7.0,
    asof: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build correlation/PCA/cosine/OI‑weighted composite ATM pillar IV series."""
    w = compute_peer_weights(
        target=target, peers=peers, weight_mode=weight_mode, asof=asof, pillar_days=pillar_days
    )
    df = build_composite_iv_pillars(w.to_dict(), pillar_days=pillar_days, tolerance_days=tolerance_days)
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
    - If asof_date is None: use the most recent trading day for that ticker.
    - Accepts either date-only ('YYYY-MM-DD') or full timestamp inputs.
    - Matches rows by calendar day regardless of how asof_date is stored in DB.
    - If T_target_years is provided, keep only the nearest expiry for that T.
    """
    conn = get_conn()
    try:
        ticker = (ticker or "").upper()

        # Resolve the calendar day we should pull
        if not asof_date:
            from data.db_utils import get_most_recent_date
            asof_date = get_most_recent_date(conn, ticker)
            if not asof_date:
                return pd.DataFrame()

        asof_ts = pd.to_datetime(asof_date)
        day_str = asof_ts.strftime("%Y-%m-%d")

        # Robust day match: works whether DB stores TEXT dates with/without time
        q = """
            SELECT asof_date, ticker, expiry, call_put, strike AS K, spot AS S, ttm_years AS T,
                   moneyness, iv AS sigma, delta, is_atm
            FROM options_quotes
            WHERE ticker = ?
              AND substr(asof_date, 1, 10) = ?
        """
        df = pd.read_sql_query(q, conn, params=[ticker, day_str])

        if df.empty:
            return df

        # Optional call/put filter
        if call_put in ("C", "P"):
            df = df[df["call_put"] == call_put]
            if df.empty:
                return df

        # Optionally limit number of expiries (smallest T first)
        if max_expiries is not None and max_expiries > 0 and not df.empty:
            # group by expiry and keep the earliest (smallest T) expiries
            unique_exp = (
                df.groupby("expiry", as_index=True)["T"]
                  .first()
                  .sort_values(kind="stable")
            )
            keep = unique_exp.head(max_expiries).index.tolist()
            df = df[df["expiry"].isin(keep)]
            if df.empty:
                return df

        # If a target T is specified, keep only the nearest expiry
        if T_target_years is not None and not df.empty:
            # Find rows belonging to the expiry whose T is closest to target
            abs_diff = (df["T"].astype(float) - float(T_target_years)).abs()
            nearest_mask = abs_diff.groupby(df["expiry"]).transform("min") == abs_diff
            df = df[nearest_mask]
            if df["expiry"].nunique() > 1:
                # If multiple expiries tie, keep the one with the most rows
                top = df.groupby("expiry").size().sort_values(ascending=False).index[0]
                df = df[df["expiry"] == top]

        return df.sort_values(["call_put", "T", "moneyness", "K"], kind="stable").reset_index(drop=True)

    finally:
        try:
            conn.close()
        except Exception:
            pass



def prepare_smile_data(
    target: str,
    asof: str,
    T_days: float,
    model: str = "svi",
    ci: float = 0.68,
    overlay_composite: bool = False,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    overlay_peers: bool = False,
    max_expiries: int = 6,
) -> Dict[str, Any]:
    """
    Precompute smile data and fitted parameters for plotting (optimized).

    Perf notes:
    - Single pass conversion to NumPy, group by expiry once.
    - Optional subsample per-expiry to cap solver work while preserving shape.
    - Skip overlay surface building unless overlay_composite or overlay_peers is True.
    - Avoid re-reading params cache per expiry.
    """
    peers = list(peers or [])
    asof_ts = pd.to_datetime(asof).normalize()

    # ---- param cache (one filter) -----------------------------------------
    try:
        from .model_params_logger import append_params, load_model_params
        params_cache = load_model_params()
        if not params_cache.empty:
            params_cache = params_cache[
                (params_cache["ticker"] == target)
                & (params_cache["asof_date"] == asof_ts)
                & (params_cache["model"].isin(["svi", "sabr", "tps", "sens"]))
            ]
    except Exception:
        append_params = None  # type: ignore
        params_cache = pd.DataFrame()

    # ---- quotes for one day (at most max_expiries expiries) ---------------
    df = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df is None or df.empty:
        return {}

    # Light, one-time coercions
    # Keep a pre-sorted view so per-expiry slices are contiguous
    df = df.sort_values(["expiry", "moneyness", "K"], kind="stable").reset_index(drop=True)
    # Convert columns once
    T_col = pd.to_numeric(df["T"], errors="coerce").to_numpy(dtype=float, copy=False)
    K_col = pd.to_numeric(df["K"], errors="coerce").to_numpy(dtype=float, copy=False)
    IV_col = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(dtype=float, copy=False)
    S_col = pd.to_numeric(df["S"], errors="coerce").to_numpy(dtype=float, copy=False)
    expiry_col = pd.to_datetime(df.get("expiry"), errors="coerce").to_numpy(copy=False)

    # Unique expiries via stable grouping (vectorized)
    # Map each row to an expiry group id
    # (astype('datetime64[ns]') ensures exact equality within a day)
    exp_vals, inv = np.unique(expiry_col.astype("datetime64[ns]"), return_inverse=True)
    # Filter out NaT groups
    valid_grp = ~(exp_vals.astype("datetime64[ns]") == np.datetime64("NaT"))
    if not valid_grp.any():
        return {}

    # For target T, pick nearest expiry
    Ts_all = np.array([np.nanmedian(T_col[inv == g]) for g in range(len(exp_vals))], dtype=float)
    Ts = np.sort(Ts_all[np.isfinite(Ts_all)])
    if Ts.size == 0:
        return {}
    idx0 = int(np.argmin(np.abs(Ts * 365.25 - float(T_days))))
    T0 = float(Ts[idx0])

    # Helper: cached params by (tenor_d, model)
    def _cached(tenor_d: int, model_name: str) -> Optional[Dict[str, float]]:
        if params_cache is None or params_cache.empty:
            return None
        sub = params_cache[(params_cache["tenor_d"] == tenor_d) & (params_cache["model"] == model_name)]
        return None if sub.empty else sub.set_index("param")["value"].to_dict()

    # Optional uniform subsample per expiry to cap solver work
    # Set via env: PREPARE_SMILE_MAX_PER_EXP (e.g., 60). 0 disables.
    try:
        max_per_exp = int(os.environ.get("PREPARE_SMILE_MAX_PER_EXP", "0"))
    except Exception:
        max_per_exp = 0

    # Per-expiry fitting
    fit_by_expiry: Dict[float, Dict[str, Any]] = {}

    # Build an array of unique group ids sorted by their median T (so stable)
    grp_ids = np.argsort(Ts_all)
    for g in grp_ids:
        if not valid_grp[g]:
            continue
        row_mask = (inv == g)
        if not row_mask.any():
            continue

        # slice arrays once
        K = K_col[row_mask]
        IV = IV_col[row_mask]
        S_slice = S_col[row_mask]
        T_slice = T_col[row_mask]
        exp_dt = exp_vals[g]  # datetime64[ns] or NaT

        # basic sanity
        if K.size < 5 or not np.isfinite(IV).any():
            continue

        # robust S and single T for this expiry
        S = float(np.nanmedian(S_slice))
        T_val = float(np.nanmedian(T_slice))
        if not (np.isfinite(S) and np.isfinite(T_val)):
            continue

        # optional subsample uniformly in moneyness
        if max_per_exp and K.size > max_per_exp:
            with np.errstate(invalid="ignore", divide="ignore"):
                mny = K / S
            ord_idx = np.argsort(mny)
            sel = np.linspace(0, ord_idx.size - 1, num=max_per_exp, dtype=int)
            take = ord_idx[sel]
            K = K.take(take)
            IV = IV.take(take)

        expiry_dt = None
        try:
            # convert numpy datetime64 -> pandas Timestamp only once
            expiry_dt = pd.Timestamp(exp_dt) if str(exp_dt) != "NaT" else None
        except Exception:
            expiry_dt = None

        tenor_d = int((expiry_dt - asof_ts).days) if expiry_dt is not None else int(round(T_val * 365.25))

        # ---- fit models using unified multi-model cache -------------------
        # Always ensure a consistent set of model params comes from the same
        # cached computation to avoid mismatches (e.g., SABR missing).
        try:
            from volModel.multi_model_cache import fit_all_models_cached
            all_models = fit_all_models_cached(S, K, T_val, IV, beta=0.5, use_cache=True)
            svi_params = all_models.get("svi") or _cached(tenor_d, "svi")
            sabr_params = all_models.get("sabr") or _cached(tenor_d, "sabr")
            tps_params = all_models.get("tps") or _cached(tenor_d, "tps")
        except Exception:
            # Fallback to previous per-model path
            svi_params = _cached(tenor_d, "svi")
            sabr_params = _cached(tenor_d, "sabr")
            tps_params = _cached(tenor_d, "tps")
            # Lazy imports only when needed
            if svi_params is None:
                from volModel.sviFit import fit_svi_slice
                svi_params = fit_svi_slice(S, K, T_val, IV)
            if sabr_params is None:
                from volModel.sabrFit import fit_sabr_slice
                sabr_params = fit_sabr_slice(S, K, T_val, IV)
            if tps_params is None:
                try:
                    from volModel.polyFit import fit_tps_slice
                    tps_params = fit_tps_slice(S, K, T_val, IV)
                except Exception:
                    tps_params = {}

        # Persist params (best-effort)
        if append_params:
            try:
                append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None,
                              "svi", svi_params, meta={"rmse": (svi_params or {}).get("rmse")})
            except Exception:
                pass
            try:
                append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None,
                              "sabr", sabr_params, meta={"rmse": (sabr_params or {}).get("rmse")})
            except Exception:
                pass
            try:
                append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None,
                              "tps", tps_params, meta={"rmse": (tps_params or {}).get("rmse")})
            except Exception:
                pass

        # Sensitivities (cheap) ------------------------------------------------
        sens_params = _cached(tenor_d, "sens")
        if sens_params is None:
            # very cheap single-pass sensitivities - need S, T, K, sigma for _fit_smile_get_atm
            dfe = df.loc[row_mask, ["K", "S", "T", "sigma"]].copy()
            try:
                dfe["moneyness"] = dfe["K"].astype(float) / float(S)
            except Exception:
                dfe["moneyness"] = np.nan
            sens = _fit_smile_get_atm(dfe, model="auto")
            sens_params = {k: sens[k] for k in ("atm_vol", "skew", "curv") if k in sens}
            if append_params:
                try:
                    append_params(asof, target, str(expiry_dt) if expiry_dt is not None else None,
                                  "sens", sens_params)
                except Exception:
                    pass

        # ---- optional CI bands -------------------------------------------
        entry: Dict[str, Any] = {
            "svi": svi_params,
            "sabr": sabr_params,
            "tps": tps_params,
            "sens": sens_params,
            "expiry": str(expiry_dt) if expiry_dt is not None else None,
        }

        if ci and ci > 0:
            m_grid = np.linspace(0.7, 1.3, 121, dtype=float)
            K_grid = m_grid * S  # cheap scale
            bands_map: Dict[str, Bands] = {}
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
            if bands_map:
                entry["bands"] = bands_map

        fit_by_expiry[T_val] = entry

    # pick the nearest fitted expiry info
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

    # ---- optional overlays (do work only if requested) ---------------------
    tgt_surface = None
    syn_surface = None
    if (overlay_composite or overlay_peers) and peers:
        try:
            tickers = list({target, *peers})
            surfaces = build_surface_grids(
                tickers=tickers, use_atm_only=False, max_expiries=max_expiries
            )
            if target in surfaces and asof in surfaces[target]:
                tgt_surface = surfaces[target][asof]

            if overlay_composite:
                composite_surfaces = {p: surfaces[p] for p in peers if p in surfaces and asof in surfaces[p]}
                if composite_surfaces:
                    w = ({p: float(weights.get(p, 1.0)) for p in composite_surfaces}
                         if weights else {p: 1.0 for p in composite_surfaces})
                    composite_by_date = combine_surfaces(composite_surfaces, w)
                    syn_surface = composite_by_date.get(asof)
        except Exception:
            tgt_surface = None
            syn_surface = None

    # Optional peer smile overlays (uses same max_expiries cap)
    composite_slices: Dict[str, Dict[str, np.ndarray]] = {}
    if overlay_peers and peers:
        for p in peers:
            df_p = get_smile_slice(p, asof, T_target_years=None, max_expiries=max_expiries)
            if df_p is None or df_p.empty:
                continue
            T_p = pd.to_numeric(df_p["T"], errors="coerce").to_numpy(float, copy=False)
            K_p = pd.to_numeric(df_p["K"], errors="coerce").to_numpy(float, copy=False)
            sigma_p = pd.to_numeric(df_p["sigma"], errors="coerce").to_numpy(float, copy=False)
            S_p = pd.to_numeric(df_p["S"], errors="coerce").to_numpy(float, copy=False)
            composite_slices[p.upper()] = {"T_arr": T_p, "K_arr": K_p, "sigma_arr": sigma_p, "S_arr": S_p}

    # Return the raw arrays from the (already converted) columns
    return {
        "T_arr": T_col,
        "K_arr": K_col,
        "sigma_arr": IV_col,
        "S_arr": S_col,
        "Ts": Ts,
        "idx0": idx0,
        "tgt_surface": tgt_surface,
        "syn_surface": syn_surface,
        "composite_slices": composite_slices,
        "expiry_arr": expiry_col,
        "fit_info": fit_info,
        "fit_by_expiry": fit_by_expiry,
    }

def normalize_ci_level(ci, default: float = 0.68) -> Optional[float]:
    """Normalize user-provided CI into (0,1) or return None.

    Rules:
    - None or falsy -> None
    - >1 and <=100 treated as percentage (e.g. 95 -> 0.95)
    - >100 or <=0 -> fallback to default (0.95)
    - Returns float in (0,1)
    """
    DEFAULT_CI_LEVEL = default if (0.0 < default < 1.0) else 0.68
    if ci is None:
        return None
    try:
        c = float(ci)
    except Exception:
        return DEFAULT_CI_LEVEL
    if c > 1.0:
        if c <= 100.0:
            c = c / 100.0
        else:
            return DEFAULT_CI_LEVEL
    if not (0.0 < c < 1.0):
        return DEFAULT_CI_LEVEL
    return c

def prepare_term_data(
    target: str,
    asof: str,
    ci: float = 68.0,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    atm_band: float = 0.05,
    max_expiries: int = 6,
    overlay_composite: bool = True,
    get_slice: Callable[[str, str, float, int], pd.DataFrame] = get_smile_slice,
) -> Dict[str, Any]:
    """
    Precompute ATM term structure (+ optional composite overlay) (optimized).

    Perf notes:
    - One call to get_smile_slice; no repeated conversions.
    - When aligning peers, uses vectorized nearest-neighbor within tolerance.
    - Skips bootstraps entirely when ci==0.
    """
    ci = normalize_ci_level(ci)
    df_all = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df_all is None or df_all.empty:
        return {}

    # If CI==None -> no bootstrap; keep compute_atm_by_expiry cheap
    # Use fewer bootstrap iterations for GUI responsiveness
    if not (ci and ci > 0):
        n_boot = 0
    else:
        # Reduce bootstrap count significantly for GUI use
        # 12-16 iterations are usually sufficient for confidence bands in GUI
        n_boot = max(12, min(16, int(ci * 20))) if ci < 1.0 else max(12, min(16, ci // 4))
    atm_curve = compute_atm_by_expiry(
        df_all,
        atm_band=atm_band,
        method="fit",
        model="auto",
        vega_weighted=True,
        n_boot=n_boot,
        ci_level=ci if ci else 0.0,
    )

    composite_curve = None
    composite_bands = None

    # Early out if overlay not requested and no peers
    if not (overlay_composite or peers):
        return {"atm_curve": atm_curve, "composite_curve": composite_curve, "composite_bands": composite_bands}

    peers = list(peers or [])
    if not peers:
        return {"atm_curve": atm_curve, "composite_curve": None, "composite_bands": None}

    # Normalize weights
    w = pd.Series(weights if weights else {p: 1.0 for p in peers}, dtype=float)
    if w.sum() <= 0:
        w = pd.Series({p: 1.0 for p in peers}, dtype=float)
    w = (w / w.sum()).astype(float)
    peers = [p for p in w.index if p in (peers or [])]
    if not peers:
        return {"atm_curve": atm_curve, "composite_curve": None, "composite_bands": None}

    # Build each peer's ATM curve (median is faster + robust)
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

    if not curves:
        return {"atm_curve": atm_curve, "composite_curve": None, "composite_bands": None}

    # Align on common T within tolerance (vectorized)
    tol_years = 10.0 / 365.25
    tgt_T = atm_curve["T"].to_numpy(float, copy=False)
    # intersect all peer T's with target T within tolerance
    common_mask = np.ones_like(tgt_T, dtype=bool)
    for c in curves.values():
        Tp = c["T"].to_numpy(float, copy=False)
        # for each t in tgt_T, require at least one close Tp within tol
        # vectorized via broadcasting
        close_any = (np.abs(tgt_T[:, None] - Tp[None, :]) <= tol_years).any(axis=1)
        common_mask &= close_any
        if not common_mask.any():
            break

    if not common_mask.any():
        return {"atm_curve": atm_curve, "composite_curve": None, "composite_bands": None}

    # Filter to common T points
    T_common = tgt_T[common_mask]
    atm_curve = atm_curve.iloc[np.flatnonzero(common_mask)]

    # Build aligned arrays per peer by nearest match
    atm_data: Dict[str, np.ndarray] = {}
    for p, c in curves.items():
        Tp = c["T"].to_numpy(float, copy=False)
        Vp = c["atm_vol"].to_numpy(float, copy=False)
        # nearest neighbor indices for each t in T_common
        j = np.abs(Tp[None, :] - T_common[:, None]).argmin(axis=1)
        ok = np.abs(Tp[j] - T_common) <= tol_years
        if ok.all():
            atm_data[p] = Vp[j]

    if not atm_data:
        return {"atm_curve": atm_curve, "composite_curve": None, "composite_bands": None}

    pillar_days = T_common * 365.25
    if ci and ci > 0:
        composite_bands = composite_etf_pillar_bands(
            atm_data,
            w.to_dict(),
            pillar_days,
            level=ci,
            n_boot=max(n_boot, 1),
        )
        composite_curve = pd.DataFrame(
            {"T": T_common, "atm_vol": composite_bands.mean, "atm_lo": composite_bands.lo, "atm_hi": composite_bands.hi}
        )
    else:
        # fast deterministic mean (no bootstrap)
        weights_vec = np.array([w.get(p, 0.0) for p in atm_data.keys()], dtype=float)
        mat = np.column_stack([atm_data[p] for p in atm_data.keys()])
        mean_curve = (mat * weights_vec).sum(axis=1)  # weights already sum to 1
        composite_curve = pd.DataFrame({"T": T_common, "atm_vol": mean_curve})
        composite_bands = None

    return {"atm_curve": atm_curve, "composite_curve": composite_curve, "composite_bands": composite_bands}


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
        composite, w = build_composite_iv_series_corrweighted("SPY", ["QQQ"], weight_mode="corr_iv_atm")
        print("composite ATM pillars len:", len(composite))
    except Exception as e:
        print("Demo error:", e)
