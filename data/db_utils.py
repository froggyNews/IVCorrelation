from __future__ import annotations

import sqlite3
from typing import Iterable, Optional
import json

import pandas as pd

from .db_schema import init_db

import os

# Allow override via environment variable; fallback to the old logic
import os

# Allow override via environment variable; fallback to the old logic
DB_PATH = os.getenv(
    "DB_PATH",
    __file__.replace("db_utils.py", "iv_data.db")
)


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection with sensible defaults.

    The GUI spins up background threads (for example when rendering animated
    plots) that may access the same database connection.  The default
    ``sqlite3`` behaviour restricts connections to the thread where they were
    created which triggered ``ProgrammingError`` exceptions on those workers.

    Setting ``check_same_thread=False`` relaxes this restriction so the
    connection can be safely shared across threads.  All current uses of this
    helper perform read-heavy operations, so the relaxed check is acceptable
    here.
    """

    path = db_path or DB_PATH
    conn = sqlite3.connect(
        path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
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


def check_db_health(conn: sqlite3.Connection) -> None:
    """Run a quick integrity check and raise if the database is corrupt."""
    status = conn.execute("PRAGMA quick_check").fetchone()
    if not status or status[0] != "ok":
        raise sqlite3.DatabaseError(
            f"Database health check failed: {status[0] if status else 'unknown'}"
        )


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
        
        volume = q.get("volume", q.get("volume_raw"))
        bid = q.get("bid", q.get("bid_raw"))
        ask = q.get("ask", q.get("ask_raw"))
        mid = q.get("mid")
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2
        if mid is None:
            mid = q.get("last_raw")
        open_interest = q.get("open_interest", q.get("open_interest_raw"))

        rows.append((
            asof_date, q["ticker"], expiry, float(q["K"]), q["call_put"],
            q.get("sigma"), q.get("S"), q.get("T"), q.get("moneyness"), q.get("log_moneyness"), q.get("delta"),
            1 if q.get("is_atm") else 0,
            volume, open_interest, bid, ask, mid,
            q.get("r"), q.get("q"), q.get("price"), q.get("gamma"), q.get("vega"), q.get("theta"), q.get("rho"), q.get("d1"), q.get("d2"),
            q.get("vendor", "yfinance"),
        ))

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO options_quotes (
                asof_date, ticker, expiry, strike, call_put,
                iv, spot, ttm_years, moneyness, log_moneyness, delta, is_atm,
                volume, open_interest, bid, ask, mid,
                r, q, price, gamma, vega, theta, rho, d1, d2,
                vendor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    check_db_health(conn)
    return len(rows)


def insert_features(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Insert engineered feature rows into feature_table.

    Each row should at minimum contain ``ts_event`` and ``symbol``. All other
    key/value pairs are stored as a JSON blob in the ``features`` column.
    """
    prepared = []
    for r in rows:
        ts = r.get("ts_event")
        if hasattr(ts, "strftime"):
            ts = ts.isoformat()
        symbol = r.get("symbol")
        payload = {k: v for k, v in r.items() if k not in ("ts_event", "symbol")}
        prepared.append((ts, symbol, json.dumps(payload, default=float)))

    if not prepared:
        return 0

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO feature_table (ts_event, symbol, features) VALUES (?, ?, ?)",
            prepared,
        )
    check_db_health(conn)
    return len(prepared)


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
def fetch_underlyings(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    asof_date: Optional[str] = None,
    use_most_recent: bool = True,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Return a time series DataFrame of underlying spots for the last ``lookback_days``.

    Priority is given to the ``underlying_prices`` table (daily closes). If no
    data is available there, falls back to aggregating ``options_quotes.spot``
    by day (using AVG) over the same window.

    Columns: [asof_date, ticker, spot]

    Args:
        conn: SQLite connection
        ticker: Optional ticker filter (single symbol)
        asof_date: End date (YYYY-MM-DD). If None and use_most_recent=True,
            uses the most recent available date (first from underlying_prices,
            else from options_quotes). If both are missing, uses today.
        use_most_recent: If True and asof_date is None, resolve to latest
            available date in DB before computing the lookback window.
        lookback_days: Number of days to include ending at ``asof_date``.
    """
    # Resolve end date
    if asof_date is None and use_most_recent:
        # Prefer latest date from underlying_prices, else from options_quotes
        try:
            row = conn.execute(
                "SELECT MAX(asof_date) FROM underlying_prices"
            ).fetchone()
            end_date = row[0] if row and row[0] else None
        except sqlite3.OperationalError:
            end_date = None
        if not end_date:
            end_date = get_most_recent_date(conn, ticker)
    else:
        end_date = asof_date

    # Default to today if still None
    if not end_date:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Compute start date window
    end_ts = pd.to_datetime(end_date)
    start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
    start_date = start_ts.strftime("%Y-%m-%d")

    # First try underlying_prices (preferred)
    params = {"start": start_date, "end": end_date}
    ul_sql = (
        "SELECT asof_date, ticker, close AS spot "
        "FROM underlying_prices "
        "WHERE asof_date BETWEEN :start AND :end"
    )
    if ticker:
        ul_sql += " AND ticker = :ticker"
        params["ticker"] = ticker
    ul_sql += " ORDER BY asof_date, ticker"

    df = pd.DataFrame()
    try:
        df = pd.read_sql_query(ul_sql, conn, params=params)
    except Exception:
        df = pd.DataFrame()

    # Fallback to options_quotes daily aggregation if no UL data
    if df.empty:
        oq_sql = (
            "SELECT asof_date, ticker, AVG(spot) AS spot "
            "FROM options_quotes "
            "WHERE spot IS NOT NULL AND asof_date BETWEEN :start AND :end"
        )
        if ticker:
            oq_sql += " AND ticker = :ticker"
        oq_sql += " GROUP BY asof_date, ticker ORDER BY asof_date, ticker"
        try:
            df = pd.read_sql_query(oq_sql, conn, params=params)
        except Exception:
            df = pd.DataFrame()

    # Ensure canonical column order/types
    if not df.empty:
        df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date.astype(str)
        df = df[["asof_date", "ticker", "spot"]]
    return df

def fetch_vol_shifts(
    conn: sqlite3.Connection,
    tickers: Optional[Iterable[str]] = None,
    threshold: float = 0.0,
):
    """Return implied volatility changes between the two most recent dates.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    tickers : Optional[Iterable[str]]
        Specific tickers to examine. If ``None`` all distinct tickers in the
        database are considered.
    threshold : float
        Minimum absolute change in implied volatility required for a row to be
        included in the result.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns
        ``[ticker, asof_date_new, asof_date_old, expiry, strike, call_put,
        iv_new, iv_old, iv_shift]``. The DataFrame will be empty if fewer than
        two distinct dates exist for a ticker or if no shifts exceed the
        threshold.
    """

    if tickers is None:
        tickers = [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM options_quotes")]  # type: ignore[assignment]

    results: list[pd.DataFrame] = []
    for t in tickers:
        rows = conn.execute(
            "SELECT DISTINCT asof_date FROM options_quotes WHERE ticker = ? ORDER BY asof_date DESC LIMIT 2",
            (t,),
        ).fetchall()
        if len(rows) < 2:
            continue

        date_new, date_old = rows[0][0], rows[1][0]
        df_new = pd.read_sql_query(
            "SELECT expiry, strike, call_put, iv FROM options_quotes WHERE ticker = ? AND asof_date = ?",
            conn,
            params=(t, date_new),
        )
        df_old = pd.read_sql_query(
            "SELECT expiry, strike, call_put, iv FROM options_quotes WHERE ticker = ? AND asof_date = ?",
            conn,
            params=(t, date_old),
        )
        merged = df_new.merge(df_old, on=["expiry", "strike", "call_put"], suffixes=("_new", "_old"))
        if merged.empty:
            continue
        merged["iv_shift"] = merged["iv_new"] - merged["iv_old"]
        shifted = merged.loc[merged["iv_shift"].abs() > threshold].copy()
        if shifted.empty:
            continue
        shifted.insert(0, "ticker", t)
        shifted.insert(1, "asof_date_new", date_new)
        shifted.insert(2, "asof_date_old", date_old)
        results.append(shifted)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "ticker",
            "asof_date_new",
            "asof_date_old",
            "expiry",
            "strike",
            "call_put",
            "iv_new",
            "iv_old",
            "iv_shift",
        ]
    )
