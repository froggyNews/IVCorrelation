"""
Thin Yahoo Finance downloader.
- Fetches raw option chains only (no calculations, no filters).
- Returns a DataFrame with minimal raw fields + asof_date, spot.
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Iterable

from .db_utils import get_conn, ensure_initialized, insert_quotes, insert_features
from .data_pipeline import enrich_quotes
from .interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD
from .feature_engineering import add_all_features
import pandas as pd

def _get_spot(tk: yf.Ticker) -> float | None:
    spot = None
    try:
        spot = tk.info.get("regularMarketPrice")
    except Exception:
        pass
    if spot is None:
        try:
            hist = tk.history(period="1d")
            if not hist.empty:
                spot = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return float(spot) if spot is not None else None

def download_raw_option_data(ticker: str, max_expiries: int = 8) -> pd.DataFrame | None:
    tk = yf.Ticker(ticker)
    expiries = tk.options or []
    if not expiries:
        return None

    spot = _get_spot(tk)
    if spot is None:
        return None

    asof_iso = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict] = []

    for expiry in expiries[:max_expiries]:
        try:
            opt = tk.option_chain(expiry)
        except Exception:
            continue

        for df, cp in ((opt.calls, "C"), (opt.puts, "P")):
            if df is None or df.empty:
                continue
            # keep only raw vendor columns we need
            sub = df.loc[:, ["strike", "impliedVolatility", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "asof_date": asof_iso,
                        "ticker": ticker,
                        "expiry": pd.to_datetime(expiry).date().isoformat(),
                        "call_put": cp,
                        "strike": float(r["strike"]),
                        "iv_raw": None if pd.isna(r["impliedVolatility"]) else float(r["impliedVolatility"]),
                        "bid_raw": None if pd.isna(r["bid"]) else float(r["bid"]),
                        "ask_raw": None if pd.isna(r["ask"]) else float(r["ask"]),
                        "last_raw": None if pd.isna(r["lastPrice"]) else float(r["lastPrice"]),
                        "volume_raw": None if pd.isna(r["volume"]) else float(r["volume"]),
                        "open_interest_raw": None if pd.isna(r["openInterest"]) else float(r["openInterest"]),
                        "spot_raw": float(spot),
                        "vendor": "yfinance",
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows)


def save_for_tickers(
    tickers: Iterable[str],
    max_expiries: int = 8,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> int:
    conn = get_conn()
    ensure_initialized(conn)
    total = 0
    for t in tickers:
        raw = download_raw_option_data(t, max_expiries=max_expiries)
        if raw is None or raw.empty:
            print(f"No raw rows for {t}")
            continue
        enriched = enrich_quotes(raw, r=r, q=q)
        if enriched is None or enriched.empty:
            print(f"No enriched rows for {t}")
            continue
    
        if enriched.columns.duplicated().any():
            enriched = enriched.loc[:, ~enriched.columns.duplicated()].copy()

        total += insert_quotes(conn, enriched.to_dict(orient="records"))
        print(f"Inserted {t}: {len(enriched)} rows")

    # Compute feature rows and persist to feature_table
        feat_input = enriched.rename(
        columns={
            "asof_date": "ts_event",
            "ticker": "symbol",
            "K": "strike_price",
            "call_put": "option_type",
            "sigma": "iv_clip",
            "S": "stock_close",
            "volume_raw": "opt_volume",
            "T": "time_to_expiry",
        }
    )
        feat_input["ts_event"] = pd.to_datetime(feat_input["ts_event"], errors="coerce")
        features = add_all_features(feat_input)
        features["ts_event"] = features["ts_event"].astype(str)
        insert_features(conn, features.to_dict(orient="records"))
    return total