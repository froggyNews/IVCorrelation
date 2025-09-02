"""Remove rows from the calc_cache table (expired by default).

Usage examples:
  - Remove only expired rows (default):
      python scripts/cleanup_calc_cache.py --db-path data/iv_data.db
  - Remove all rows (ignore expiry):
      python scripts/cleanup_calc_cache.py --db-path data/iv_data.db --all
  - Also VACUUM to reclaim file space:
      python scripts/cleanup_calc_cache.py --db-path data/iv_data.db --vacuum

This script can be scheduled via cron to keep the cache table small.
Example cron entry to run nightly at 2am:
    0 2 * * * /usr/bin/python path/to/cleanup_calc_cache.py --db-path=/path/db.db
"""
from __future__ import annotations
import argparse
import sqlite3
from typing import Optional, Sequence

# Default deletion predicate: remove expired rows using integer epoch seconds
SQL_ALL = "DELETE FROM calc_cache"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    return cur.fetchone() is not None


def _list_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [str(row[1]).lower() for row in cur.fetchall()]
        return cols
    except Exception:
        return []


def _build_expiry_sql(columns: Sequence[str]) -> Optional[str]:
    """Return a DELETE ... WHERE ... statement appropriate for the schema.

    Supports common layouts:
      - expires_at / expiry / expires / valid_until (epoch seconds)
      - created_at + ttl or created_at + ttl_seconds
    Returns None if no expiry predicate can be inferred.
    """
    cols = {c.lower() for c in columns}
    now_epoch = "CAST(strftime('%s','now') AS INTEGER)"

    # Direct expiry columns
    for name in ("expires_at", "expiry", "expires", "valid_until"):
        if name in cols:
            return f"DELETE FROM calc_cache WHERE COALESCE({name}, 0) < {now_epoch}"

    # Derived from created_at + ttl
    if "created_at" in cols:
        for ttl_col in ("ttl", "ttl_seconds", "ttl_sec", "expires_in"):
            if ttl_col in cols:
                return (
                    "DELETE FROM calc_cache "
                    f"WHERE (COALESCE(CAST(created_at AS INTEGER),0) + "
                    f"COALESCE(CAST({ttl_col} AS INTEGER),0)) < {now_epoch}"
                )

    return None


def cleanup(db_path: str, *, delete_all: bool = False, vacuum: bool = False) -> int:
    """Delete rows from calc_cache.

    Returns number of rows removed (0 if table does not exist).
    """
    with sqlite3.connect(db_path) as conn:
        try:
            if not _table_exists(conn, "calc_cache"):
                print("calc_cache table not found; nothing to do.")
                return 0
            if delete_all:
                sql = SQL_ALL
            else:
                cols = _list_columns(conn, "calc_cache")
                sql = _build_expiry_sql(cols)
                if sql is None:
                    print("No expiry columns detected; use --all to clear calc_cache.")
                    return 0
            cur = conn.execute(sql)
            removed = int(cur.rowcount or 0)
            if vacuum and removed >= 0:
                try:
                    conn.execute("VACUUM")
                except Exception:
                    # Non-fatal if VACUUM fails (e.g., not allowed on WAL journal)
                    pass
            conn.commit()
            return removed
        except Exception as e:
            # Fail softly for cron use; emit message and return -1
            print(f"Error cleaning calc_cache: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup calc_cache rows in the SQLite DB")
    parser.add_argument("--db-path", default="data/iv_data.db", help="Path to SQLite database")
    parser.add_argument("--all", action="store_true", help="Delete all rows (ignore expiry)")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after delete to reclaim space")
    args = parser.parse_args()

    removed = cleanup(args.db_path, delete_all=bool(args.all), vacuum=bool(args.vacuum))
    if removed >= 0:
        scope = "all" if args.all else "expired"
        print(f"Removed {removed} {scope} rows from calc_cache")
    else:
        # cleanup printed the error already
        pass


if __name__ == "__main__":
    main()
