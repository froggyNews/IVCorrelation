from __future__ import annotations

import hashlib
import json
import pickle
import queue
import sqlite3
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
DB_PATH = DATA_DIR / "iv_data.db"                       # main database with cache table

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


def _get_cache_conn() -> sqlite3.Connection:
    """Get connection to main database with cache table."""
    from data.db_utils import get_conn
    return get_conn()

def _load_cache_df() -> pd.DataFrame:
    """Load cache data from SQLite database."""
    conn = _get_cache_conn()
    try:
        df = pd.read_sql_query("""
            SELECT kind, key, created_at, expires_at, serializer, payload_json, blob
            FROM calculation_cache
            ORDER BY created_at
        """, conn)
        
        if "created_at" in df:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        if "expires_at" in df:
            df["expires_at"] = pd.to_datetime(df["expires_at"], errors="coerce", utc=True)
        return df
    except Exception:
        # Table doesn't exist yet or other error - return empty DataFrame
        return pd.DataFrame(columns=[
            "kind", "key", "created_at", "expires_at", "serializer", "payload_json", "blob"
        ])
    finally:
        conn.close()

def _save_cache_df(df: pd.DataFrame) -> None:
    """Save cache data to SQLite database."""
    conn = _get_cache_conn()
    try:
        # Clear existing cache data
        conn.execute("DELETE FROM calculation_cache")
        
        if not df.empty:
            # Convert timestamps to ISO strings for SQLite and dedupe
            df_save = df.sort_values("created_at").drop_duplicates(["kind", "key"], keep="last").copy()
            for ts_col in ["created_at", "expires_at"]:
                if ts_col in df_save.columns:
                    df_save[ts_col] = df_save[ts_col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Insert new data
            df_save.to_sql("calculation_cache", conn, if_exists="append", index=False)
        
        conn.commit()
    finally:
        conn.close()


def prune_expired(now: Optional[pd.Timestamp] = None) -> int:
    """Remove expired cache entries from database."""
    now = now or pd.Timestamp.utcnow()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    conn = _get_cache_conn()
    try:
        # Count entries to be deleted
        cursor = conn.execute("""
            SELECT COUNT(*) FROM calculation_cache 
            WHERE expires_at IS NOT NULL AND expires_at <= ?
        """, (now_str,))
        count = cursor.fetchone()[0]
        
        # Delete expired entries
        conn.execute("""
            DELETE FROM calculation_cache 
            WHERE expires_at IS NOT NULL AND expires_at <= ?
        """, (now_str,))
        conn.commit()
        
        return count
    finally:
        conn.close()

def clear_cache(kind: Optional[str] = None) -> int:
    """Clear cache entries from database."""
    conn = _get_cache_conn()
    try:
        if kind is None:
            # Clear all cache entries
            cursor = conn.execute("SELECT COUNT(*) FROM calculation_cache")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM calculation_cache")
        else:
            # Clear specific kind
            cursor = conn.execute("SELECT COUNT(*) FROM calculation_cache WHERE kind = ?", (kind,))
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM calculation_cache WHERE kind = ?", (kind,))
        
        conn.commit()
        return count
    finally:
        conn.close()

def cache_stats() -> Dict[str, Any]:
    """Get cache statistics from database."""
    conn = _get_cache_conn()
    try:
        # Get count
        cursor = conn.execute("SELECT COUNT(*) FROM calculation_cache")
        count = cursor.fetchone()[0]
        
        if count == 0:
            return {"entries": 0, "kinds": [], "oldest": None, "newest": None}
        
        # Get kinds
        cursor = conn.execute("SELECT DISTINCT kind FROM calculation_cache ORDER BY kind")
        kinds = [row[0] for row in cursor.fetchall()]
        
        # Get date range
        cursor = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM calculation_cache")
        oldest_str, newest_str = cursor.fetchone()
        
        return {
            "entries": count,
            "kinds": kinds,
            "oldest": pd.to_datetime(oldest_str, utc=True) if oldest_str else None,
            "newest": pd.to_datetime(newest_str, utc=True) if newest_str else None,
        }
    finally:
        conn.close()


def get_data_snapshot() -> Dict[str, Any]:
    """Get a snapshot of current data state for cache freshness tracking."""
    conn = _get_cache_conn()
    try:
        # Get all tickers and their latest data
        cursor = conn.execute("""
            SELECT ticker, MAX(asof_date) as latest_date, COUNT(DISTINCT expiry) as expiry_count
            FROM options_quotes 
            GROUP BY ticker
            ORDER BY ticker
        """)
        tickers_data = {row[0]: {"latest_date": row[1], "expiry_count": row[2]} for row in cursor.fetchall()}
        
        # Get global latest date
        cursor = conn.execute("SELECT MAX(asof_date) FROM options_quotes")
        global_latest = cursor.fetchone()[0]
        
        return {
            "global_latest_date": global_latest,
            "tickers": tickers_data,
            "snapshot_time": pd.Timestamp.utcnow().isoformat()
        }
    finally:
        conn.close()


def detect_data_changes(baseline_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect changes in data since baseline snapshot."""
    if baseline_snapshot is None:
        # No baseline - consider everything new
        current = get_data_snapshot()
        return {
            "new_tickers": list(current["tickers"].keys()),
            "new_expiries": list(current["tickers"].keys()),  # All tickers have "new" expiries
            "updated_tickers": [],
            "global_date_changed": True,
            "current_snapshot": current
        }
    
    current = get_data_snapshot()
    baseline_tickers = set(baseline_snapshot.get("tickers", {}).keys())
    current_tickers = set(current["tickers"].keys())
    
    new_tickers = list(current_tickers - baseline_tickers)
    updated_tickers = []
    new_expiries = []
    
    # Check for updated expiry counts or dates
    for ticker in current_tickers & baseline_tickers:
        current_data = current["tickers"][ticker]
        baseline_data = baseline_snapshot["tickers"][ticker]
        
        if (current_data["latest_date"] != baseline_data["latest_date"] or 
            current_data["expiry_count"] != baseline_data["expiry_count"]):
            updated_tickers.append(ticker)
            
            # If expiry count increased, it's new expiries
            if current_data["expiry_count"] > baseline_data["expiry_count"]:
                new_expiries.append(ticker)
    
    global_date_changed = (current["global_latest_date"] != 
                          baseline_snapshot.get("global_latest_date"))
    
    return {
        "new_tickers": new_tickers,
        "new_expiries": new_expiries,
        "updated_tickers": updated_tickers,
        "global_date_changed": global_date_changed,
        "current_snapshot": current
    }


def invalidate_cache_for_tickers(tickers: Iterable[str], kinds: Optional[List[str]] = None) -> int:
    """Invalidate cache entries that depend on specific tickers."""
    if not tickers:
        return 0
        
    ticker_list = list(tickers)
    conn = _get_cache_conn()
    
    try:
        count = 0
        
        # Build WHERE clause for cache entries containing these tickers
        conditions = []
        params = []
        
        for ticker in ticker_list:
            # Match ticker in payload_json (covers smile, term, and other ticker-specific caches)
            conditions.append("payload_json LIKE ?")
            params.append(f'%"{ticker}"%')
        
        if kinds:
            # Limit to specific cache kinds
            kind_conditions = " AND kind IN ({})".format(','.join(['?' for _ in kinds]))
            params.extend(kinds)
        else:
            kind_conditions = ""
        
        where_clause = f"({' OR '.join(conditions)}){kind_conditions}"
        
        # Count entries to be deleted
        cursor = conn.execute(f"SELECT COUNT(*) FROM calculation_cache WHERE {where_clause}", params)
        count = cursor.fetchone()[0]
        
        if count > 0:
            # Delete matching entries
            conn.execute(f"DELETE FROM calculation_cache WHERE {where_clause}", params)
            conn.commit()
        
        return count
        
    finally:
        conn.close()


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
            if pd.isna(expires_at) or pd.to_datetime(expires_at, utc=True) > now:
                try:
                    return _deserialize(row["blob"], fmt=row.get("serializer", serializer))
                except Exception:
                    pass  # fall through

    value = builder_fn()
    try:
        blob = _serialize(value, fmt=serializer)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        
        # Use direct database insert for efficiency (avoid rewriting entire cache)
        conn = _get_cache_conn()
        try:
            expires_str = (now + pd.Timedelta(seconds=int(ttl_sec))).strftime('%Y-%m-%d %H:%M:%S.%f') if ttl_sec > 0 else None
            conn.execute("""
                INSERT OR REPLACE INTO calculation_cache 
                (kind, key, created_at, expires_at, serializer, payload_json, blob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                key.kind, key.key, 
                now.strftime('%Y-%m-%d %H:%M:%S.%f'), 
                expires_str,
                serializer, payload_json, blob
            ))
            conn.commit()
        finally:
            conn.close()
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
