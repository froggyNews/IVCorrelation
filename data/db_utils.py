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


def ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes used by common query patterns if they don't exist."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_options_quotes_ticker ON options_quotes(ticker)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_options_quotes_asof_date ON options_quotes(asof_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_options_quotes_is_atm ON options_quotes(is_atm)"
    )
    conn.commit()


def ensure_initialized(conn: sqlite3.Connection) -> None:
    init_db(conn)
    ensure_indexes(conn)


def insert_quotes(conn: sqlite3.Connection, quotes: Iterable[dict]) -> int:
    rows = []
    for q in quotes:
        # Ensure dates are strings, not Timestamp objects
        asof_date = q["asof_date"]
        if hasattr(asof_date, 'strftime'):  # pandas Timestamp or datetime
            asof_date = asof_date.strftime('%Y-%m-%d') if hasattr(asof_date, 'strftime') else str(asof_date)
        
        expiry = q["expiry"]
        if hasattr(expiry, 'strftime'):  # pandas Timestamp or datetime
            expiry = expiry.strftime('%Y-%m-%d') if hasattr(expiry, 'strftime') else str(expiry)
        
        rows.append((
            asof_date, q["ticker"], expiry, float(q["K"]), q["call_put"],
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


def get_most_recent_date(conn: sqlite3.Connection, ticker: Optional[str] = None) -> Optional[str]:
    """Get the most recent asof_date in the database, optionally for a specific ticker."""
    if ticker:
        sql = "SELECT MAX(asof_date) FROM options_quotes WHERE ticker = ?"
        params = [ticker]
    else:
        sql = "SELECT MAX(asof_date) FROM options_quotes"
        params = []
    
    result = conn.execute(sql, params).fetchone()
    return result[0] if result and result[0] else None


def fetch_quotes(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    asof_date: Optional[str] = None,
    use_most_recent: bool = True,
):
    """
    Fetch options quotes from database.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    ticker : Optional[str]
        Specific ticker to filter by
    asof_date : Optional[str]
        Specific date to filter by. If None and use_most_recent=True, uses most recent date
    use_most_recent : bool
        If True and asof_date is None, automatically use the most recent date
    """
    sql = "SELECT * FROM options_quotes WHERE 1=1"
    params: list = []
    
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker)
    
    if asof_date:
        sql += " AND asof_date = ?"
        params.append(asof_date)
    elif use_most_recent:
        # Use most recent date if no specific date provided
        recent_date = get_most_recent_date(conn, ticker)
        if recent_date:
            sql += " AND asof_date = ?"
            params.append(recent_date)
    
    sql += " ORDER BY ticker, asof_date, expiry, strike, call_put"
    return conn.execute(sql, params).fetchall()
