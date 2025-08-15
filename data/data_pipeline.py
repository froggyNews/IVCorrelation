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
import numpy as np
import pandas as pd

from .greeks import compute_all_greeks_df
from .interest_rates import (
    STANDARD_RISK_FREE_RATE,
    STANDARD_DIVIDEND_YIELD,
    get_ticker_interest_rate,
)

def enrich_quotes(
    raw_df: pd.DataFrame,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return raw_df

    df = raw_df.copy()

    # Parse dates and compute time to maturity in years
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    df["asof_date"] = pd.to_datetime(df["asof_date"], utc=True)
    df["T"] = (df["expiry"] - df["asof_date"]).dt.days / 365.25
    df = df[df["T"] > 0]

    # Vendor IV -> sigma, spot
    df["sigma"] = df["iv_raw"]
    df["S"] = df["spot_raw"]
    df["K"] = df["strike"]

    # Moneyness and log-moneyness
    df["moneyness"] = df["K"] / df["S"]
    df["log_moneyness"] = np.log(df["moneyness"])

    # Compute Greeks in bulk (adds: price, delta, gamma, vega, theta, rho, d1, d2)
    # Uses ticker-specific rates if available, falls back to provided r
    df = df.rename(columns={"call_put": "call_put"})
    df = compute_all_greeks_df(df, r=r, q=q, use_ticker_rates=True)
    
    # Store the rates that were actually used in the calculation
    # The compute_all_greeks_df function handles ticker-specific rates internally
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
