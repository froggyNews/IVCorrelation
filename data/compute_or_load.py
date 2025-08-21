"""Utility to cache expensive computations in the project database.

The function :func:`compute_or_load` stores results keyed by a name and payload
inside the main SQLite database (``iv_data.db`` by default). Results are
pickled for storage. If a matching entry exists it is returned; otherwise the
provided ``builder`` callable is executed and its result cached.
"""
from __future__ import annotations

import json
import os
import pickle
import sqlite3
from typing import Any, Callable

from data.db_utils import DB_PATH, get_conn


def compute_or_load(
    name: str,
    payload: dict,
    builder: Callable[[], Any],
    *,
    db_path: str | None = None,
) -> Any:
    """Compute a value or load it from a SQLite-backed cache.

    Parameters
    ----------
    name: str
        Logical name for the computation. Combined with ``payload`` this forms
        a unique key in the cache.
    payload: dict
        Serializable dictionary describing the inputs to the computation.
    builder: Callable[[], Any]
        Function invoked to compute the value when a cached version is not
        available.
    db_path: str, optional
        Path to the SQLite database used for caching. Defaults to the project
        database from :mod:`data.db_utils`.
    """
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = f"{name}:{json.dumps(payload, sort_keys=True)}"

    conn: sqlite3.Connection | None = None
    try:
        conn = get_conn(path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS calc_cache ("  # type: ignore[assignment]
            "key TEXT PRIMARY KEY, value BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        row = conn.execute(
            "SELECT value FROM calc_cache WHERE key=?", (key,)
        ).fetchone()
        if row:
            try:
                return pickle.loads(row[0])
            except Exception:
                pass
    except Exception:
        # Fall back to computing directly if anything goes wrong with caching
        pass
    finally:
        if conn is not None and conn.in_transaction:
            conn.rollback()

    result = builder()

    try:
        if conn is None:
            conn = get_conn(path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS calc_cache ("
                "key TEXT PRIMARY KEY, value BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
        blob = pickle.dumps(result)
        conn.execute(
            "INSERT OR REPLACE INTO calc_cache (key, value, created_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, blob),
        )
        conn.commit()
    except Exception:
        if conn is not None and conn.in_transaction:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()

    return result
