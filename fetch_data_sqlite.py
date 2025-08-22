"""Compatibility layer for legacy data loading functions.

Provides ``fetch_and_save``, ``get_conn``, and ``init_schema`` so that
modules expecting the old ``fetch_data_sqlite`` module continue to work.

The implementation re-uses the modern modules under ``data/``.  When a new
SQLite database is created ``init_schema`` initialises the minimal set of
tables required by :func:`data.databento_historical_downloader.fetch_and_save`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Lazy imports avoid circular dependency via data.__init__

def get_conn(db_path: str | Path | None = None) -> sqlite3.Connection:
    from data.db_utils import get_conn as _get_conn
    return _get_conn(str(db_path) if db_path else None)


def fetch_and_save(*args, **kwargs):
    from data.databento_historical_downloader import fetch_and_save as _fetch
    return _fetch(*args, **kwargs)


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialise required tables for historical option data.

    The downloader expects a number of tables to exist.  Older code relied on a
    dedicated ``fetch_data_sqlite`` module to create them.  This function
    recreates that behaviour by creating the minimal schema used by
    :mod:`data.databento_historical_downloader`.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS opra_1m (
            ts_event    TEXT,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            symbol      TEXT,
            ticker      TEXT
        );

        CREATE TABLE IF NOT EXISTS equity_1m (
            ticker      TEXT,
            ts_event    TEXT,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            symbol      TEXT
        );

        CREATE TABLE IF NOT EXISTS equity_1h (
            ticker      TEXT,
            ts_event    TEXT,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL
        );

        CREATE TABLE IF NOT EXISTS merged_1m (
            ticker       TEXT,
            ts_event     TEXT,
            opt_symbol   TEXT,
            stock_symbol TEXT,
            opt_close    REAL,
            stock_close  REAL,
            opt_volume   REAL,
            stock_volume REAL
        );

        CREATE TABLE IF NOT EXISTS processed_merged_1m (
            ticker        TEXT,
            ts_event      TEXT,
            opt_symbol    TEXT,
            stock_symbol  TEXT,
            opt_close     REAL,
            stock_close   REAL,
            opt_volume    REAL,
            stock_volume  REAL,
            expiry_date   TEXT,
            option_type   TEXT,
            strike_price  REAL,
            time_to_expiry REAL,
            moneyness     REAL
        );

        CREATE TABLE IF NOT EXISTS atm_slices_1m (
            ticker        TEXT,
            ts_event      TEXT,
            expiry_date   TEXT,
            opt_symbol    TEXT,
            stock_symbol  TEXT,
            opt_close     REAL,
            stock_close   REAL,
            opt_volume    REAL,
            stock_volume  REAL,
            option_type   TEXT,
            strike_price  REAL,
            time_to_expiry REAL,
            moneyness     REAL
        );
        """
    )
    conn.commit()
