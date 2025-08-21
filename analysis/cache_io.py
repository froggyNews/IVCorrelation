from __future__ import annotations
import sqlite3
import json
import time
from typing import Any, Optional

# Increment this whenever the structure of cached artifacts changes
ARTIFACT_VERSION = 2

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS calc_cache (
    cache_key TEXT PRIMARY KEY,
    artifact_version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
"""

def ensure_table(conn: sqlite3.Connection) -> None:
    """Ensure the calc_cache table exists."""
    conn.execute(CREATE_SQL)
    conn.commit()


def get_latest_raw_timestamp(conn: sqlite3.Connection) -> int:
    """Return the latest timestamp from raw option tables.

    This checks both options_quotes and underlying_prices tables. If the
    tables are empty, 0 is returned.
    """
    q = """
    SELECT MAX(ts) FROM (
        SELECT MAX(strftime('%s', asof_date)) AS ts FROM options_quotes
        UNION ALL
        SELECT MAX(strftime('%s', asof_date)) AS ts FROM underlying_prices
    )
    """
    cur = conn.execute(q)
    val = cur.fetchone()[0]
    return int(val) if val is not None else 0


def load_calc_cache(
    conn: sqlite3.Connection,
    cache_key: str,
    latest_raw_ts: Optional[int] = None,
) -> Optional[Any]:
    """Load cached payload if still valid.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    cache_key : str
        Lookup key for the cached artifact.
    latest_raw_ts : int, optional
        Latest timestamp from raw option tables. If not provided it will be
        computed automatically.
    """
    ensure_table(conn)
    cur = conn.execute(
        "SELECT payload, created_at, expires_at, artifact_version FROM calc_cache WHERE cache_key=?",
        (cache_key,),
    )
    row = cur.fetchone()
    if not row:
        return None
    payload, created_at, expires_at, version = row
    now = int(time.time())
    if expires_at < now or version != ARTIFACT_VERSION:
        return None
    if latest_raw_ts is None:
        latest_raw_ts = get_latest_raw_timestamp(conn)
    if latest_raw_ts > created_at:
        # Raw data updated after this artifact was created
        return None
    return json.loads(payload)


def save_calc_cache(
    conn: sqlite3.Connection,
    cache_key: str,
    payload: Any,
    ttl_seconds: int = 86400,
) -> None:
    """Persist payload into calc_cache with a TTL."""
    ensure_table(conn)
    now = int(time.time())
    expires_at = now + ttl_seconds
    conn.execute(
        "REPLACE INTO calc_cache(cache_key, artifact_version, payload, created_at, expires_at) VALUES (?,?,?,?,?)",
        (cache_key, ARTIFACT_VERSION, json.dumps(payload), now, expires_at),
    )
    conn.commit()
