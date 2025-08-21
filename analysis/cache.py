import os
import json
import pickle
import hashlib
import sqlite3
from typing import Any, Callable


def compute_or_load(kind: str, payload: dict, builder: Callable[[], Any], db_path: str) -> Any:
    """Compute a result or load it from an on-disk cache.

    Parameters
    ----------
    kind: str
        Logical name for the calculation (used to namespace cache keys).
    payload: dict
        Hashable parameters defining the calculation.
    builder: Callable[[], Any]
        Function that performs the computation when a cache miss occurs.
    db_path: str
        Path to the SQLite database used for storing cached results.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    payload_json = json.dumps(payload, sort_keys=True)
    key = hashlib.sha256(payload_json.encode()).hexdigest()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calculations (
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                payload TEXT,
                result BLOB,
                PRIMARY KEY(kind, key)
            )
            """
        )
        cur = conn.execute(
            "SELECT result FROM calculations WHERE kind=? AND key= ?",
            (kind, key),
        )
        row = cur.fetchone()
        if row is not None:
            return pickle.loads(row[0])
        result = builder()
        conn.execute(
            "INSERT INTO calculations(kind, key, payload, result) VALUES (?,?,?,?)",
            (kind, key, payload_json, pickle.dumps(result)),
        )
        conn.commit()
        return result
