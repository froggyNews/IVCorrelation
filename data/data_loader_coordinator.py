"""Minimal data loader coordinator to satisfy feature engineering needs.

Provides:
- load_cores_with_auto_fetch: returns per‑ticker core DataFrames with required columns
- validate_cores: light validation helper
- DataCoordinator: simple wrapper (currently thin) for potential future expansion

A "core" dataframe is expected to contain at least the columns used in
feature_engineering:
    ts_event, iv_clip, stock_close, strike_price, time_to_expiry,
    option_type, opt_volume (optional)

If additional columns exist they are preserved.
"""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd

DEFAULT_DB_PATH = Path(os.getenv("DB_PATH", "data/iv_data.db"))

REQUIRED_COLUMNS = [
    "ts_event",
    "iv_clip",
    "stock_close",
    "strike_price",
    "time_to_expiry",
    "option_type",
]

OPTIONAL_COLUMNS = ["opt_volume"]


def _open_conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    return sqlite3.connect(path)


def _load_single_ticker(ticker: str, start: Optional[str], end: Optional[str], conn: sqlite3.Connection) -> pd.DataFrame:
    clauses = ["ticker = ?"]
    params: list = [ticker]
    if start:
        clauses.append("asof_date >= ?")
        params.append(str(start))
    if end:
        clauses.append("asof_date <= ?")
        params.append(str(end))
    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            asof_date AS ts_event,
            iv        AS iv_clip,
            spot      AS stock_close,
            strike    AS strike_price,
            ttm_years AS time_to_expiry,
            call_put  AS option_type,
            volume    AS opt_volume
        FROM options_quotes
        WHERE {where_sql}
        ORDER BY asof_date
    """
    try:
        df = pd.read_sql_query(sql, conn, params=params, parse_dates=["ts_event"])
    except Exception:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    if df.empty:
        return df
    # Basic cleaning
    df = df.dropna(subset=["iv_clip"])  # ensure IV exists
    # Normalize option_type to single char C/P
    df["option_type"] = df["option_type"].astype(str).str.upper().str[0]
    return df


def load_cores_with_auto_fetch(
    tickers: Iterable[str],
    start=None,
    end=None,
    db_path: Path | str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load per‑ticker core datasets from SQLite.

    This minimal implementation DOES NOT auto‑download missing data; it simply
    returns what is present. If a ticker has no data an empty DataFrame is
    returned for that key so that callers can decide how to handle it.
    """
    tickers = list(dict.fromkeys(tickers))  # dedupe preserving order
    out: Dict[str, pd.DataFrame] = {}
    with _open_conn(db_path) as conn:
        for t in tickers:
            out[t] = _load_single_ticker(t, start, end, conn)
    return out


def validate_cores(cores: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Light validation: ensure required columns exist (add empty if missing)."""
    for t, df in cores.items():
        if df is None or df.empty:
            continue
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
    return cores


class DataCoordinator:
    """Thin wrapper mirroring (removed) richer implementation.

    Present for compatibility; may be expanded later. It simply stores the
    loaded cores.
    """

    def __init__(self, tickers, start=None, end=None, db_path: Path | str | None = None):
        self.cores = load_cores_with_auto_fetch(tickers, start=start, end=end, db_path=db_path)
        validate_cores(self.cores)

    def get_cores(self) -> Dict[str, pd.DataFrame]:  # compatibility method
        return self.cores

__all__ = [
    "load_cores_with_auto_fetch",
    "validate_cores",
    "DataCoordinator",
]
