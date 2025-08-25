from __future__ import annotations

import hashlib
import json
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STORE_PATH = DATA_DIR / "model_params.parquet"          # existing params log
CACHE_PATH = DATA_DIR / "calculation_cache.parquet"     # generic cache

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Parameter logging (audit/provenance)
# ---------------------------------------------------------------------
def append_params(
    asof_date: str,
    ticker: str,
    expiry: Optional[str],
    model: str,
    params: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Append fitted parameters to disk (deduped by asof,ticker,expiry,model,param)."""
    meta = meta or {}
    asof_ts = pd.to_datetime(asof_date).normalize()
    expiry_dt = pd.to_datetime(expiry) if expiry else None
    tenor_d = (expiry_dt - asof_ts).days if expiry_dt is not None else None
    rows = []
    for key, val in params.items():
        try:
            fval = float(val)
        except Exception:
            continue
        rows.append({
            "asof_date": asof_ts,
            "ticker": ticker.upper(),
            "expiry": expiry_dt,
            "tenor_d": tenor_d,
            "model": model.lower(),
            "param": key,
            "value": fval,
            "fit_meta": meta,
        })
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if STORE_PATH.exists():
        df_old = pd.read_parquet(STORE_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = (
            df.sort_values(["asof_date","ticker","expiry","model","param"])
              .drop_duplicates(["asof_date","ticker","expiry","model","param"], keep="last")
        )
    else:
        df = df_new
    df.to_parquet(STORE_PATH, index=False)


def load_model_params() -> pd.DataFrame:
    """Load logged parameters as DataFrame (empty frame if none)."""
    if not STORE_PATH.exists():
        return pd.DataFrame(columns=["asof_date","ticker","expiry","tenor_d","model","param","value","fit_meta"])
    df = pd.read_parquet(STORE_PATH)
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    if "expiry" in df:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    return df


# ---------------------------------------------------------------------
# 2) General-purpose calculation cache
# ---------------------------------------------------------------------
_ARTIFACT_VERSION: Dict[str, str] = {}   # e.g. {"atm_curve_v1": "3"}

@dataclass(frozen=True)
class CacheKey:
    kind: str
    key: str  # hashed payload + version

def set_artifact_version(kind: str, version: str) -> None:
    _ARTIFACT_VERSION[kind] = str(version)

def get_artifact_version(kind: str) -> str:
    return _ARTIFACT_VERSION.get(kind, "1")


def _hash_inputs(kind: str, payload: dict) -> CacheKey:
    version = get_artifact_version(kind)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    hasher = hashlib.sha256()
    hasher.update(kind.encode())
    hasher.update(b"|")
    hasher.update(version.encode())
    hasher.update(b"|")
    hasher.update(payload_json.encode())
    return CacheKey(kind=kind, key=hasher.hexdigest())


def _serialize(value: Any, fmt: str = "pickle") -> bytes:
    if fmt == "pickle":
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if fmt == "json":
        return json.dumps(value, default=str).encode("utf-8")
    raise ValueError(f"unknown serializer: {fmt}")

def _deserialize(blob: bytes, fmt: str = "pickle") -> Any:
    if fmt == "pickle":
        return pickle.loads(blob)
    if fmt == "json":
        return json.loads(blob.decode("utf-8"))
    raise ValueError(f"unknown serializer: {fmt}")


def _load_cache_df() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=[
            "kind", "key", "created_at", "expires_at", "serializer", "payload_json", "blob"
        ])
    df = pd.read_parquet(CACHE_PATH)
    if "created_at" in df:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if "expires_at" in df:
        df["expires_at"] = pd.to_datetime(df["expires_at"], errors="coerce")
    return df

def _save_cache_df(df: pd.DataFrame) -> None:
    df = df.sort_values("created_at").drop_duplicates(["kind", "key"], keep="last")
    df.to_parquet(CACHE_PATH, index=False)


def prune_expired(now: Optional[pd.Timestamp] = None) -> int:
    now = now or pd.Timestamp.utcnow()
    df = _load_cache_df()
    if df.empty:
        return 0
    before = len(df)
    df = df[(df["expires_at"].isna()) | (df["expires_at"] > now)]
    _save_cache_df(df)
    return before - len(df)

def clear_cache(kind: Optional[str] = None) -> int:
    df = _load_cache_df()
    if df.empty:
        return 0
    before = len(df)
    if kind is None:
        df = df.iloc[0:0]
    else:
        df = df[df["kind"] != kind]
    _save_cache_df(df)
    return before - len(_load_cache_df())

def cache_stats() -> Dict[str, Any]:
    df = _load_cache_df()
    if df.empty:
        return {"entries": 0, "kinds": [], "oldest": None, "newest": None}
    return {
        "entries": int(len(df)),
        "kinds": sorted(df["kind"].unique().tolist()),
        "oldest": pd.to_datetime(df["created_at"]).min(),
        "newest": pd.to_datetime(df["created_at"]).max(),
    }


def compute_or_load(
    kind: str,
    payload: Dict[str, Any],
    builder_fn: Callable[[], Any],
    *,
    ttl_sec: int = 900,
    serializer: str = "pickle",
) -> Any:
    """Compute artifact or load from cache (invalidated by version/payload)."""
    now = pd.Timestamp.utcnow()
    key = _hash_inputs(kind, payload)

    df = _load_cache_df()
    if not df.empty:
        hit = df[(df["kind"] == key.kind) & (df["key"] == key.key)]
        if not hit.empty:
            row = hit.iloc[-1]
            expires_at = row.get("expires_at")
            if pd.isna(expires_at) or pd.to_datetime(expires_at) > now:
                try:
                    return _deserialize(row["blob"], fmt=row.get("serializer", serializer))
                except Exception:
                    pass  # fall through

    value = builder_fn()
    try:
        blob = _serialize(value, fmt=serializer)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        row = pd.DataFrame([{
            "kind": key.kind,
            "key": key.key,
            "created_at": now,
            "expires_at": (now + pd.Timedelta(seconds=int(ttl_sec))) if ttl_sec > 0 else pd.NaT,
            "serializer": serializer,
            "payload_json": payload_json,
            "blob": blob,
        }])
        df = pd.concat([df, row], ignore_index=True)
        _save_cache_df(df)
    except Exception:
        pass
    return value


class WarmupWorker:
    """Background worker to pre-compute cache entries asynchronously."""

    def __init__(self):
        self._queue: "queue.Queue[tuple[str, dict, Callable[[], Any], int, str]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(
        self,
        kind: str,
        payload: Dict[str, Any],
        builder_fn: Callable[[], Any],
        *,
        ttl_sec: int = 900,
        serializer: str = "pickle",
    ) -> None:
        self._queue.put((kind, payload, builder_fn, ttl_sec, serializer))

    def _run(self) -> None:
        while True:
            kind, payload, builder_fn, ttl_sec, serializer = self._queue.get()
            try:
                compute_or_load(kind, payload, builder_fn, ttl_sec=ttl_sec, serializer=serializer)
            except Exception:
                pass
            finally:
                self._queue.task_done()
                time.sleep(0.01)


# ---------------------------------------------------------------------
# 3) Common artifact wrappers
# ---------------------------------------------------------------------
def cached_atm_curve(builder: Callable[[], Any], *, asof: str, ticker: str, pillar_days: Tuple[int, ...], ttl_sec: int = 900):
    payload = {"asof": str(pd.to_datetime(asof).date()), "ticker": ticker.upper(), "pillar_days": tuple(map(int, pillar_days))}
    return compute_or_load("atm_curve_v1", payload, builder, ttl_sec=ttl_sec, serializer="pickle")

def cached_surface_grid(builder: Callable[[], Any], *, asof: str, tickers: Tuple[str, ...], tenors: Tuple[int, ...], mny_bins: Tuple[Tuple[float,float], ...], ttl_sec: int = 900):
    payload = {
        "asof": str(pd.to_datetime(asof).date()),
        "tickers": tuple(t.upper() for t in tickers),
        "tenors": tuple(map(int, tenors)),
        "mny_bins": tuple((float(a), float(b)) for a,b in mny_bins),
    }
    return compute_or_load("surface_grid_v1", payload, builder, ttl_sec=ttl_sec, serializer="pickle")
