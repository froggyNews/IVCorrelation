"""Utilities for fetching and storing underlying price history."""

from __future__ import annotations

from typing import Iterable
import pandas as pd
import yfinance as yf

from .db_utils import get_conn, ensure_initialized


def _fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download historical daily close prices for ``ticker`` from yfinance."""
    tk = yf.Ticker(ticker)
    try:
        hist = tk.history(period=period)
    except Exception:
        return pd.DataFrame()
    if hist.empty or "Close" not in hist.columns:
        return pd.DataFrame()
    df = hist.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "asof_date", "Close": "close"}
    )
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date.astype(str)
    df["ticker"] = ticker.upper()
    return df[["asof_date", "ticker", "close"]]


def update_underlying_prices(tickers: Iterable[str], period: str = "1y") -> int:
    """Fetch and upsert historical prices for ``tickers`` into the DB.

    Returns the total number of rows inserted.
    """
    conn = get_conn()
    ensure_initialized(conn)

    total = 0
    for t in tickers:
        df = _fetch_history(t, period=period)
        if df.empty:
            continue
        rows = [tuple(x) for x in df.itertuples(index=False, name=None)]
        conn.executemany(
            """
            INSERT OR REPLACE INTO underlying_prices(asof_date, ticker, close)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        total += len(rows)
    return total


__all__ = ["update_underlying_prices"]

