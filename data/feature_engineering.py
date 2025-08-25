import os
import sqlite3
import json
import logging
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO)

@contextmanager
def suppress_runtime_warnings():
    """Context manager to suppress specific runtime warnings during SABR calculations."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")
        yield

# Config
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
ANNUAL_MINUTES = 252 * 390

# What to hide when predicting each target (preserved from original)
HIDE_COLUMNS = {
    "iv_ret_fwd": ["iv_ret_fwd_abs"],
    "iv_ret_fwd_abs": ["iv_ret_fwd"],
    "iv_clip": ["iv_ret_fwd", "iv_ret_fwd_abs"]
}

# Core features (preserved from original)
CORE_FEATURE_COLS = [
    "opt_volume", "time_to_expiry", "days_to_expiry", "strike_price",
    "option_type_enc", "delta", "gamma", "vega", "hour", "minute", "day_of_week"
]


def _hagan_implied_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """Approximate Black implied volatility under the SABR model."""
    with suppress_runtime_warnings():
        if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
            return np.nan

        # Check for invalid parameters
        if abs(rho) >= 1 or nu < 0 or not (0 <= beta <= 1):
            return np.nan

        if np.isclose(F, K):
            term1 = alpha / (F ** (1 - beta))
            term2 = 1 + (
                ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
                + (rho * beta * nu * alpha / (4 * F ** (1 - beta)))
                + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
            ) * T
            return term1 * term2

        FK_beta = (F * K) ** ((1 - beta) / 2)
        logFK = np.log(F / K)
        z = (nu / alpha) * FK_beta * logFK

        if np.isclose(z, 0, atol=1e-7):
            term2 = 1.0
        else:
            sqrt_term = np.sqrt(1 - 2 * rho * z + z * z)
            log_arg = (sqrt_term + z - rho) / (1 - rho)

            if log_arg <= 0 or (1 - rho) == 0:
                return np.nan

            x_z = np.log(log_arg)

            if np.isclose(x_z, 0, atol=1e-12):
                term2 = 1.0
            else:
                term2 = z / x_z

        term1 = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * (logFK ** 2) + ((1 - beta) ** 4 / 1920) * (logFK ** 4)))
        term3 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_beta ** 2))
            + (rho * beta * nu * alpha / (4 * FK_beta))
            + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
        ) * T

        result = term1 * term2 * term3

        if not np.isfinite(result) or result <= 0:
            return np.nan

        return result


def _solve_sabr_alpha(sigma: float, F: float, K: float, T: float, beta: float, rho: float, nu: float) -> float:
    """Calibrate alpha for a single observation using Hagan's formula."""
    if np.any(np.isnan([sigma, F, K, T])) or sigma <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    if abs(rho) >= 1 or nu < 0 or not (0 <= beta <= 1):
        return np.nan

    def objective(a: float) -> float:
        if a <= 0:
            return float('inf')
        vol = _hagan_implied_vol(F, K, T, a, beta, rho, nu)
        if np.isnan(vol):
            return float('inf')
        return vol - sigma

    try:
        obj_low = objective(1e-6)
        obj_high = objective(5.0)

        if not (np.isfinite(obj_low) and np.isfinite(obj_high)):
            return np.nan

        if obj_low * obj_high > 0:
            return np.nan

        return brentq(objective, 1e-6, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def _add_sabr_features(df: pd.DataFrame, beta: float = 0.5) -> pd.DataFrame:
    """Compute simple SABR parameter features and drop raw price/IV columns."""
    F_series = df.get("stock_close")
    K_series = df.get("strike_price")
    T_series = df.get("time_to_expiry")
    sigma_series = df.get("iv_clip")
    if F_series is None or K_series is None or T_series is None or sigma_series is None:
        return df

    F = F_series.astype(float).to_numpy()
    K = K_series.astype(float).to_numpy()
    T = np.maximum(T_series.astype(float).to_numpy(), 1e-9)
    sigma = sigma_series.astype(float).to_numpy()

    moneyness = (K / F) - 1.0
    rho = np.tanh(moneyness * 5.0)
    nu_series = (
        df["iv_clip"].astype(float).rolling(30).std() * np.sqrt(ANNUAL_MINUTES / 30)
    ).shift(1)
    nu = nu_series.to_numpy()

    alpha = np.array([
        _solve_sabr_alpha(sig, f, k, t, beta, r, n)
        for sig, f, k, t, r, n in zip(sigma, F, K, T, rho, nu)
    ])

    df["sabr_alpha"] = alpha
    df["sabr_beta"] = beta
    df["sabr_rho"] = rho
    df["sabr_nu"] = nu

    df = df.drop(columns=[c for c in ["stock_close", "iv_clip"] if c in df.columns])
    return df


def add_all_features(df: pd.DataFrame, forward_steps: int = 1, r: float = 0.045) -> pd.DataFrame:
    """Centralized feature engineering (preserves all original feature logic)."""
    df = df.copy()

    log_col = np.log(df["iv_clip"].astype(float))
    fwd = log_col.shift(-forward_steps) - log_col
    df["iv_ret_fwd"] = fwd
    df["iv_ret_fwd_abs"] = fwd.abs()

    S = df["stock_close"].astype(float).to_numpy()
    K = df["strike_price"].astype(float).to_numpy()
    T = np.maximum(df["time_to_expiry"].astype(float).to_numpy(), 1e-9)
    sig = df["iv_clip"].astype(float).to_numpy()
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    is_call = df["option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()
    df["delta"] = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    df["gamma"] = pdf / (S * sig * sqrtT)
    df["vega"] = S * pdf * sqrtT

    df["days_to_expiry"] = (df["time_to_expiry"] * 365.0).astype("float32")
    df["option_type_enc"] = (df["option_type"].astype(str).str.upper().str[0]
                            .map({"P": 0, "C": 1}).astype("float32"))

    if "stock_close" in df.columns:
        logS = np.log(df["stock_close"].astype(float))
        ret_1m = logS.diff()
        rv = ret_1m.rolling(30).std()
        df["rv_30m"] = (rv * np.sqrt(ANNUAL_MINUTES / 30)).shift(1)

    if "opt_volume" in df.columns:
        diff = df["opt_volume"].diff()
        df["opt_vol_change_1m"] = (diff.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        df["opt_vol_roll_15m"] = df["opt_volume"].rolling(15).mean().shift(1)

    df = _add_sabr_features(df)

    return df


def build_iv_panel(cores: Dict[str, pd.DataFrame], tolerance: str = "2s") -> pd.DataFrame:
    """Centralized IV panel building (preserves original merge logic)."""
    tol = pd.Timedelta(tolerance)
    iv_wide = None

    for ticker, df in cores.items():
        if df is None or df.empty or not {"ts_event", "iv_clip"}.issubset(df.columns):
            continue

        tmp = df[["ts_event", "iv_clip"]].rename(columns={"iv_clip": f"IV_{ticker}"}).copy()
        tmp["ts_event"] = pd.to_datetime(tmp["ts_event"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["ts_event", f"IV_{ticker}"]).sort_values("ts_event")
        tmp[f"IVRET_{ticker}"] = np.log(tmp[f"IV_{ticker}"]) - np.log(tmp[f"IV_{ticker}"].shift(1))
        tmp = tmp[["ts_event", f"IV_{ticker}", f"IVRET_{ticker}"]]

        if iv_wide is None:
            iv_wide = tmp
        else:
            iv_wide = pd.merge_asof(
                iv_wide.sort_values("ts_event"), tmp, on="ts_event",
                direction="backward", tolerance=tol
            )

    return iv_wide if iv_wide is not None else pd.DataFrame(columns=["ts_event"])


def finalize_dataset(df: pd.DataFrame, target_col: str, drop_symbol: bool = True, debug: bool = False) -> pd.DataFrame:
    """Centralized dataset finalization (preserves original cleanup logic)."""
    out = df.copy()

    if debug:
        debug_dir = Path("debug_snapshots")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_snapshot_path = debug_dir / f"raw_data_{target_col}_{timestamp}.csv"
        out.to_csv(raw_snapshot_path, index=False)
        print(f"DEBUG: Saved raw data snapshot to {raw_snapshot_path}")

    for c in out.columns:
        if c != "ts_event":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    initial_rows = len(out)
    out = out.dropna(subset=[target_col])
    dropped_target = initial_rows - len(out)

    if debug and dropped_target > 0:
        print(f"DEBUG: Dropped {dropped_target} rows with missing {target_col}")

    hidden_cols = []
    for col in HIDE_COLUMNS.get(target_col, []):
        if col in out.columns:
            out = out.drop(columns=col)
            hidden_cols.append(col)

    if debug and hidden_cols:
        print(f"DEBUG: Hidden leaky columns for {target_col}: {hidden_cols}")

    leak_cols = [c for c in out.columns if c in {"stock_close", "iv_clip"} or c.startswith("IV_") or c.startswith("IVRET_")]
    if leak_cols:
        out = out.drop(columns=leak_cols)
        if debug:
            print(f"DEBUG: Removed leak columns: {leak_cols}")

    if drop_symbol and "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
        if debug:
            print(f"DEBUG: Dropped symbol column")

    out = out.reset_index(drop=True)
    out = _normalize_numeric_features(out, target_col=target_col)

    if debug:
        print(f"DEBUG: Final dataset shape: {out.shape}")
        print(f"DEBUG: Final columns: {list(out.columns)}")
        final_snapshot_path = debug_dir / f"final_data_{target_col}_{timestamp}.csv"
        out.to_csv(final_snapshot_path, index=False)
        print(f"DEBUG: Saved final data snapshot to {final_snapshot_path}")
        info_path = debug_dir / f"column_info_{target_col}_{timestamp}.json"
        column_info = {
            "target_column": target_col,
            "final_columns": list(out.columns),
            "hidden_columns": hidden_cols,
            "leak_columns": leak_cols,
            "initial_rows": initial_rows,
            "final_rows": len(out),
            "dropped_rows": dropped_target,
        }
        with open(info_path, 'w') as f:
            json.dump(column_info, f, indent=2, default=str)
        print(f"DEBUG: Saved column info to {info_path}")

    logging.getLogger(__name__).info("Final dataset columns: %s", list(out.columns))

    return out


def build_pooled_iv_return_dataset_time_safe(
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    cores: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Build pooled dataset for forecasting forward IV return, keeping peer IV/IVRET columns."""

    if debug:
        print(f"DEBUG: Building pooled dataset for {len(tickers)} tickers")
        print(f"DEBUG: Parameters - forward_steps: {forward_steps}, tolerance: {tolerance}")

    if cores is None:
        from data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)

    if debug:
        print(f"DEBUG: Cores loaded for tickers: {list(cores.keys())}")
        for ticker, core in cores.items():
            print(f"DEBUG: {ticker} core shape: {core.shape if core is not None else 'None'}")

    panel = build_iv_panel(cores, tolerance=tolerance)

    if debug:
        print(f"DEBUG: IV panel shape: {panel.shape}")
        if not panel.empty:
            debug_dir = Path("debug_snapshots")
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            panel_path = debug_dir / f"iv_panel_{timestamp}.csv"
            panel.to_csv(panel_path, index=False)
            print(f"DEBUG: Saved IV panel to {panel_path}")

    frames = []
    for ticker in tickers:
        if ticker not in cores:
            if debug:
                print(f"DEBUG: Skipping {ticker} - not in cores")
            continue

        if debug:
            print(f"DEBUG: Processing {ticker}")

        feats = add_all_features(cores[ticker], forward_steps=forward_steps, r=r)

        if debug:
            print(f"DEBUG: {ticker} after add_all_features: {feats.shape}")

        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )

        if debug:
            print(f"DEBUG: {ticker} after panel merge: {feats.shape}")

        out = feats.copy()
        for c in out.columns:
            if c != "ts_event":
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["iv_ret_fwd"])
        for col in HIDE_COLUMNS.get("iv_ret_fwd", []):
            if col in out.columns:
                out = out.drop(columns=col)
        for c in ["stock_close", "iv_clip"]:
            if c in out.columns:
                out = out.drop(columns=c)
        out = out.reset_index(drop=True)
        out = _normalize_numeric_features(out, target_col="iv_ret_fwd")
        if debug:
            print(f"DEBUG: {ticker} after finalization: {out.shape}")
        frames.append(out)

    if not frames:
        if debug:
            print("DEBUG: No frames to concatenate - returning empty DataFrame")
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    if pooled.empty:
        if debug:
            print("DEBUG: Pooled dataset is empty after concatenation")
        return pooled

    if debug:
        print(f"DEBUG: Pooled dataset after concatenation: {pooled.shape}")

    pooled = pd.get_dummies(pooled, columns=["symbol"], prefix="sym", dtype=float)

    for ticker in tickers:
        col = f"sym_{ticker}"
        if col not in pooled.columns:
            pooled[col] = 0.0

    front = ["iv_ret_fwd"]
    if "iv_ret_fwd_abs" in pooled.columns:
        front.append("iv_ret_fwd_abs")
    if "iv_clip" in pooled.columns:
        front.append("iv_clip")
    onehots = [f"sym_{t}" for t in tickers]
    other = [c for c in pooled.columns if c not in front + onehots]

    final_pooled = pooled[front + other + onehots]

    if debug:
        print(f"DEBUG: Final pooled dataset shape: {final_pooled.shape}")
        debug_dir = Path("debug_snapshots")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pooled_path = debug_dir / f"pooled_final_{timestamp}.csv"
        final_pooled.to_csv(pooled_path, index=False)
        print(f"DEBUG: Saved final pooled dataset to {pooled_path}")

    return final_pooled


def build_iv_return_dataset_time_safe(
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 15,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    cores: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build per‑ticker datasets for forecasting forward IV return."""

    if debug:
        print(f"DEBUG: Building per-ticker datasets for {len(tickers)} tickers")

    if cores is None:
        from data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)

    panel = build_iv_panel(cores, tolerance=tolerance)

    datasets = {}
    for ticker in tickers:
        if ticker not in cores:
            if debug:
                print(f"DEBUG: Skipping {ticker} - not in cores")
            continue

        if debug:
            print(f"DEBUG: Processing per-ticker dataset for {ticker}")

        feats = add_all_features(cores[ticker], forward_steps=forward_steps, r=r)

        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )

        datasets[ticker] = finalize_dataset(feats, "iv_ret_fwd", drop_symbol=False, debug=debug)

    if debug:
        print(f"DEBUG: Built {len(datasets)} per-ticker datasets")

    return datasets


def build_target_peer_dataset(
    target: str,
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    target_kind: str = "iv_ret",
    cores: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Build dataset for single target vs peers."""

    if debug:
        print(f"DEBUG: Building target-peer dataset for {target} vs {len(tickers)} tickers")
        print(f"DEBUG: target_kind: {target_kind}")

    if target not in tickers:
        raise AssertionError("target must be included in tickers")

    if cores is None:
        from data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)

    if target not in cores:
        raise ValueError(f"Target {target} produced no valid core")

    panel = build_iv_panel(cores, tolerance=tolerance)
    feats = add_all_features(cores[target], forward_steps=forward_steps, r=r)

    feats = pd.merge_asof(
        feats.sort_values("ts_event"), panel.sort_values("ts_event"),
        on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
    )

    if target_kind in ("iv_ret", "iv_ret_fwd"):
        target_col = "iv_ret_fwd"
    elif target_kind == "iv_ret_fwd_abs":
        target_col = "iv_ret_fwd_abs"
    elif target_kind == "iv":
        target_col = "iv_clip"
    else:
        raise ValueError(
            "target_kind must be one of 'iv_ret', 'iv_ret_fwd', 'iv_ret_fwd_abs', or 'iv'"
        )

    if debug:
        print(f"DEBUG: Using target column: {target_col}")
        print(f"DEBUG: Dataset shape before finalization: {feats.shape}")

    feats = finalize_dataset(feats, target_col, drop_symbol=True, debug=debug)
    final_dataset = feats.rename(columns={target_col: "y"})

    if debug:
        print(f"DEBUG: Final target-peer dataset shape: {final_dataset.shape}")
        debug_dir = Path("debug_snapshots")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_peer_path = debug_dir / f"target_peer_{target}_{target_kind}_{timestamp}.csv"
        final_dataset.to_csv(target_peer_path, index=False)
        print(f"DEBUG: Saved target-peer dataset to {target_peer_path}")

    return final_dataset


def _normalize_numeric_features(out: pd.DataFrame, target_col: str) -> pd.DataFrame:
    keep = {"ts_event"}
    skip_prefixes = ("sym_",)
    skip_exact = {target_col}

    cols = []
    for c in out.columns:
        if c in keep or c in skip_exact:
            continue
        if any(c.startswith(p) for p in skip_prefixes):
            continue
        if is_numeric_dtype(out[c]):
            cols.append(c)

    if cols:
        mu = out[cols].mean()
        sd = out[cols].std().replace(0.0, 1.0).fillna(1.0)
        out[cols] = (out[cols] - mu) / sd
        out.attrs["norm_means"] = {k: float(mu[k]) for k in mu.index}
        out.attrs["norm_stds"] = {k: float(sd[k]) for k in sd.index}
    return out

__all__ = [
    "build_pooled_iv_return_dataset_time_safe",
    "build_iv_return_dataset_time_safe",
    "build_target_peer_dataset",
    "add_all_features",
    "build_iv_panel",
    "finalize_dataset",
]
