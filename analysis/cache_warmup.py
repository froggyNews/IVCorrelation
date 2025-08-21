"""Background cache warmup utilities."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Tuple

# Placeholder compute_or_load implementation.
# Tests may monkeypatch this for custom behaviour.
def compute_or_load(kind: str, payload: Any, builder_fn: Callable[[Any], Any]) -> Any:
    """Compute or populate cache using ``builder_fn``.

    Parameters
    ----------
    kind : str
        A hint describing the type of cache to build.
    payload : Any
        Arbitrary data passed to ``builder_fn``.
    builder_fn : Callable[[Any], Any]
        Function that builds the cache entry when invoked.
    """
    return builder_fn(payload)


class WarmupWorker:
    """Simple background worker to warm caches lazily.

    Jobs are enqueued via :meth:`enqueue` and processed sequentially by a
    dedicated daemon thread. Each job is described by a ``kind`` string, an
    arbitrary ``payload`` object and a ``builder_fn`` callable. The callable is
    executed through :func:`compute_or_load` which is responsible for building
    and populating the cache.
    """

    def __init__(self) -> None:
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

