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
