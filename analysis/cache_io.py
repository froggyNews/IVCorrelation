# analysis/cache_io.py
from __future__ import annotations

import json
import sqlite3
import threading
import queue
import pickle
import hashlib
from typing import Any, Callable, Dict


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS calc_cache (
            type TEXT NOT NULL,
            hash TEXT NOT NULL,
            data BLOB,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(type, hash)
        )
        """
    )
    conn.commit()


def _hash_payload(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_or_load(
    type_name: str,
    payload: Dict[str, Any],
    builder: Callable[[], Any],
    db_path: str = "data/calculations.db",
) -> Any:
    """Compute a value via ``builder`` or load a cached result from sqlite."""
    h = _hash_payload(payload)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_table(conn)
        cur = conn.execute(
            "SELECT data FROM calc_cache WHERE type=? AND hash=?",
            (type_name, h),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            try:
                return pickle.loads(row[0])
            except Exception:
                pass
        value = builder()
        try:
            blob = pickle.dumps(value)
            conn.execute(
                "INSERT OR REPLACE INTO calc_cache(type, hash, data) VALUES (?,?,?)",
                (type_name, h, blob),
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
