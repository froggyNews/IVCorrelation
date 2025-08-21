import json
import pickle
import sqlite3
import time
import zlib
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict

import queue
import threading
import time
from typing import Any, Callable, Tuple

TTL_SEC = 900
ARTIFACT_VERSION: Dict[str, str] = {}

def _hash_inputs(kind: str, payload: dict) -> str:
    """Create a stable hash for the inputs.

    The hash combines the kind, the artifact version for that kind, and the payload
    dictionary serialized in a deterministic manner.
    """
    version = ARTIFACT_VERSION.get(kind, "1")
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    hasher = hashlib.sha256()
    hasher.update(kind.encode())
    hasher.update(b"|")
    hasher.update(version.encode())
    hasher.update(b"|")
    hasher.update(payload_json.encode())
    return hasher.hexdigest()

def compute_or_load(kind: str, payload: dict, builder_fn: Callable[[], Any], db_path: str) -> Any:
    """Compute an artifact or load it from cache.

    Parameters
    ----------
    kind : str
        Identifier for the artifact type.
    payload : dict
        Inputs used to build the artifact. Must be JSON serializable.
    builder_fn : Callable[[], Any]
        Function that builds the artifact when cache miss occurs.
    db_path : str
        Path to the SQLite database file used for caching.
    """
    key = _hash_inputs(kind, payload)
    now = int(time.time())
    db_file = Path(db_path)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS calc_cache(
  key TEXT PRIMARY KEY,
  artifact BLOB,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  kind TEXT NOT NULL,
  version TEXT NOT NULL,
  meta_json TEXT)"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS calc_cache_expires ON calc_cache(expires_at)"
        )
        row = conn.execute(
            "SELECT artifact, expires_at FROM calc_cache WHERE key=?", (key,)
        ).fetchone()
        if row and row[1] > now:
            try:
                return pickle.loads(zlib.decompress(row[0]))
            except Exception:
                pass
        artifact = builder_fn()
        blob = zlib.compress(pickle.dumps(artifact))
        expires_at = now + TTL_SEC
        version = ARTIFACT_VERSION.get(kind, "1")
        meta_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with conn:
            conn.execute(
                "REPLACE INTO calc_cache(key, artifact, created_at, expires_at, kind, version, meta_json) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (key, blob, now, expires_at, kind, version, meta_json),
            )
            conn.execute(
                "DELETE FROM calc_cache WHERE expires_at <= ?", (now,)
            )
        return artifact
    finally:
        conn.close()


class WarmupWorker:
    """Simple background worker to warm caches lazily.

    Jobs are enqueued via :meth:`enqueue` and processed sequentially by a
    dedicated daemon thread. Each job is described by a ``kind`` string, an
    arbitrary ``payload`` object and a ``builder_fn`` callable. The callable is
    executed through :func:`compute_or_load` which is responsible for building
    and populating the cache.
    """

    def __init__(self, db_path: str = "data/calculations.db"):
        self.db_path = db_path
        self._queue: "queue.Queue[Tuple[str, Any, Callable[[Any], Any]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, kind: str, payload: Any, builder_fn: Callable[[Any], Any]) -> None:
        """Add a warmup job to the queue."""
        self._queue.put((kind, payload, builder_fn))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            kind, payload, builder_fn = self._queue.get()
            try:
                compute_or_load(kind, payload, builder_fn)
            except Exception:
                # Never let cache warmup failures impact the GUI thread.
                pass
            finally:
                # Mark job done and briefly yield control to keep UI responsive.
                self._queue.task_done()
                time.sleep(0.01)

