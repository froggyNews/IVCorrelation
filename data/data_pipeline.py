"""
Data pipeline:
- takes raw downloader output
- computes T, moneyness, log_moneyness
- computes BS Greeks (r,q configurable)
- flags ATM (closest to |Δ|-0.5 per date/ticker/expiry/CP)
- applies light sanity filters
- renames to DB schema fields ready for insert
"""
from __future__ import annotations
from datetime import timezone
import numpy as np
import pandas as pd

from .greeks import compute_all_greeks_df
from .rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD

def _compute_ttm_years(expiry_iso: str, asof_date_iso: str) -> float:
    expiry_dt = pd.to_datetime(expiry_iso, utc=True)
    asof_dt = pd.to_datetime(asof_date_iso, utc=True)
    return float((expiry_dt - asof_dt) / pd.Timedelta(days=365.25))

def enrich_quotes(
    raw_df: pd.DataFrame,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return raw_df

    df = raw_df.copy()

    # Compute T (years), rename fields for downstream
    df["T"] = df.apply(lambda r_: _compute_ttm_years(r_["expiry"], r_["asof_date"]), axis=1)
    df = df[df["T"] > 0]

    # Vendor IV -> sigma, spot
    df["sigma"] = df["iv_raw"]
    df["S"] = df["spot_raw"]
    df["K"] = df["strike"]

    # Moneyness and log-moneyness
    df["moneyness"] = df["K"] / df["S"]
    df["log_moneyness"] = np.log(df["moneyness"])

    # Compute Greeks in bulk (adds: price, delta, gamma, vega, theta, rho, d1, d2)
    # Requires columns: S, K, T, sigma, call_put
    df = df.rename(columns={"call_put": "call_put"})
    df = compute_all_greeks_df(df, r=r, q=q)
    df["r"] = r
    df["q"] = q

    # ATM flag per (date, ticker, expiry, call_put)
    def _mark_atm(g: pd.DataFrame) -> pd.DataFrame:
        if g["delta"].notna().any():
            idx = (g["delta"].abs() - 0.5).abs().idxmin()
        else:
            idx = (g["moneyness"] - 1.0).abs().idxmin()
        g = g.copy()
        g["is_atm"] = 0
        g.loc[idx, "is_atm"] = 1
        return g

    df = df.groupby(["asof_date", "ticker", "expiry", "call_put"], group_keys=False).apply(_mark_atm)

    # Light sanity filters (tune later)
    df = df[(df["sigma"] > 0.01) & (df["sigma"] < 2.0)]
    df = df[(df["moneyness"] > 0.1) & (df["moneyness"] < 10.0)]

    # Keep expiries with enough quotes per CP
    counts = df.groupby(["ticker", "expiry", "call_put"]).size()
    valid = counts[counts >= 3].index
    df = df.set_index(["ticker", "expiry", "call_put"]).loc[valid].reset_index()

    # Map to DB column names and keep only what DB expects
    out = df.rename(
        columns={
            "strike": "K",  # ensure K exists (already set above)
        }
    )

    # final selection in DB field order (db_utils.insert expects these keys)
    cols = [
        "asof_date", "ticker", "expiry", "K", "call_put",
        "sigma", "S", "T", "moneyness", "log_moneyness", "delta", "is_atm",
        "volume_raw", "bid_raw", "ask_raw", "last_raw",
        "r", "q", "price", "gamma", "vega", "theta", "rho", "d1", "d2",
        "vendor"
    ]
    # Some raw cols may be missing; add if needed
    for c in ["volume_raw", "bid_raw", "ask_raw", "last_raw"]:
        if c not in out.columns:
            out[c] = None

    return out[cols]
