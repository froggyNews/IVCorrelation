from __future__ import annotations

import json
import sqlite3
import threading
import queue
import pickle
import hashlib
import time
from typing import Any, Callable, Dict, Optional

# Increment this whenever the structure of cached artifacts changes
ARTIFACT_VERSION = 2

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS calc_cache (
    type TEXT NOT NULL,
    hash TEXT PRIMARY KEY,
    data BLOB,
    created INTEGER NOT NULL,
    expires INTEGER NOT NULL,
    artifact_version INTEGER NOT NULL
);
"""

def _ensure_table(conn: sqlite3.Connection) -> None:
    """Ensure the unified calc_cache table exists."""
    conn.execute(CREATE_SQL)
    conn.commit()

def _hash_payload(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_latest_raw_timestamp(conn: sqlite3.Connection) -> int:
    """Return the latest timestamp from raw option tables."""
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

def compute_or_load(
    type_name: str,
    payload: Dict[str, Any],
    builder: Callable[[], Any],
    db_path: str = "data/calculations.db",
    ttl_seconds: int = 86400,
    latest_raw_ts: Optional[int] = None,
) -> Any:
    """Compute a value via `builder` or load a cached result from sqlite."""
    h = _hash_payload(payload)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_table(conn)
        cur = conn.execute(
            "SELECT data, created, expires, artifact_version FROM calc_cache WHERE hash=?",
            (h,),
        )
        row = cur.fetchone()
        now = int(time.time())

        if row:
            data, created, expires, version = row
            if version == ARTIFACT_VERSION and expires > now:
                if latest_raw_ts is None:
                    latest_raw_ts = get_latest_raw_timestamp(conn)
                if latest_raw_ts <= created:
                    try:
                        return pickle.loads(data)
                    except Exception:
                        pass  # Fall through to recompute

        # Recompute and store
        value = builder()
        try:
            blob = pickle.dumps(value)
            conn.execute(
                "REPLACE INTO calc_cache (type, hash, data, created, expires, artifact_version) VALUES (?,?,?,?,?,?)",
                (type_name, h, blob, now, now + ttl_seconds, ARTIFACT_VERSION),
            )
            conn.commit()
        except Exception:
            pass
        return value
    finally:
        conn.close()

class WarmupWorker:
    """Background worker to pre-warm the cache asynchronously."""

    def __init__(self, db_path: str = "data/calculations.db"):
        self.db_path = db_path
        self._q: queue.Queue[tuple[str, Dict[str, Any], Callable[[], Any]]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            type_name, payload, builder = self._q.get()
            try:
                compute_or_load(type_name, payload, builder, db_path=self.db_path)
            except Exception:
                pass
            finally:
                self._q.task_done()

    def enqueue(self, type_name: str, payload: Dict[str, Any], builder: Callable[[], Any]) -> None:
        """Schedule a computation for background warmup."""
        try:
            self._q.put_nowait((type_name, payload, builder))
        except Exception:
            pass
