from __future__ import annotations
import sqlite3
from typing import Iterable, Optional
from .db_schema import init_db

DB_PATH = __file__.replace("db_utils.py", "iv_data.db")


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_initialized(conn: sqlite3.Connection) -> None:
    init_db(conn)


def insert_quotes(conn: sqlite3.Connection, quotes: Iterable[dict]) -> int:
    rows = []
    for q in quotes:
        rows.append((
            q["asof_date"], q["ticker"], q["expiry"], float(q["K"]), q["call_put"],
            q.get("sigma"), q.get("S"), q.get("T"), q.get("moneyness"), q.get("log_moneyness"), q.get("delta"),
            1 if q.get("is_atm") else 0,
            q.get("volume"), q.get("bid"), q.get("ask"), q.get("mid"),
            q.get("r"), q.get("q"), q.get("price"), q.get("gamma"), q.get("vega"), q.get("theta"), q.get("rho"), q.get("d1"), q.get("d2"),
            q.get("vendor", "yfinance"),
        ))

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO options_quotes (
                asof_date, ticker, expiry, strike, call_put,
                iv, spot, ttm_years, moneyness, log_moneyness, delta, is_atm,
                volume, bid, ask, mid,
                r, q, price, gamma, vega, theta, rho, d1, d2,
                vendor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def fetch_quotes(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    asof_date: Optional[str] = None,
):
    sql = "SELECT * FROM options_quotes WHERE 1=1"
    params: list = []
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker)
    if asof_date:
        sql += " AND asof_date = ?"
        params.append(asof_date)
    sql += " ORDER BY ticker, asof_date, expiry, strike, call_put"
    return conn.execute(sql, params).fetchall()
