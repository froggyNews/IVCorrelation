#!/usr/bin/env python3
from __future__ import annotations
import sqlite3, shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

DB = Path("data/iv_data.db")

def _backup_path(db: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return db.with_suffix(f".db.bak_{ts}")

def _integrity_report(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        return [f"integrity_check failed: {e}"]

def _quick_table_counts(conn: sqlite3.Connection) -> pd.DataFrame:
    counts = []
    try:
        tbls = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    except Exception:
        tbls = []
    for (t,) in tbls:
        try:
            c = conn.execute(f"SELECT COUNT(1) FROM {t}").fetchone()[0]
        except Exception as e:
            c = None
        counts.append({"table": t, "rows": c})
    return pd.DataFrame(counts).sort_values("table").reset_index(drop=True)

def diagnose_and_try_recover(db: Path, required_tickers: list[str] | None = None, max_expiries: int = 6) -> bool:
    """
    Returns True if DB is usable *now* (healthy or recovered or rebuilt),
    False if unrecoverable.
    """
    # 0) If file missing, just let caller rebuild with their own pipeline
    if not db.exists():
        print(f"[db_guard] {db} does not exist.")
        return False

    # 1) Try to open + integrity_check
    try:
        conn = sqlite3.connect(str(db))
    except Exception as e:
        print(f"[db_guard] Cannot open DB: {e}")
        # Backup raw file in case of total open failure
        try:
            shutil.copyfile(db, _backup_path(db))
        except Exception:
            pass
        return False

    try:
        report = _integrity_report(conn)
        ok = (len(report) == 1 and report[0].strip().lower() == "ok")
        print("[db_guard] integrity_check:", "; ".join(report[:3]))
        counts = _quick_table_counts(conn)
        print("[db_guard] table counts:\n", counts.to_string(index=False))
        conn.close()
        if ok:
            return True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass

    # 2) Backup the corrupted file
    try:
        bak = _backup_path(db)
        shutil.copyfile(db, bak)
        print(f"[db_guard] Backed up corrupted DB to: {bak}")
    except Exception as e:
        print(f"[db_guard] Backup failed: {e}")

    # 3) Attempt a logical backup/restore using Python's backup API (may work if corruption is localized)
    try:
        src = sqlite3.connect(str(db))
        dst_path = db.with_suffix(".db.recovered")
        dst = sqlite3.connect(str(dst_path))
        src.backup(dst)
        dst.close(); src.close()

        # Swap in recovered if it passes integrity_check
        c2 = sqlite3.connect(str(dst_path))
        rep2 = _integrity_report(c2)
        ok2 = (len(rep2) == 1 and rep2[0].strip().lower() == "ok")
        c2.close()
        if ok2:
            shutil.move(dst_path, db)  # replace
            print("[db_guard] Recovered via sqlite backup API.")
            return True
        else:
            print("[db_guard] Backup API recovery did not pass integrity_check:", rep2[:3])
            try: dst_path.unlink()
            except Exception: pass
    except Exception as e:
        print(f"[db_guard] Backup API recovery failed: {e}")

    # 4) Fallback: minimal rebuild path (schema + re-ingest small set through your existing pipeline)
    # We don’t import your project modules here to avoid import side effects.
    # Instead, we create a minimal empty DB with required tables so your pipeline can refill.
    try:
        print("[db_guard] Performing minimal schema reinit (options_quotes, underlying_prices).")
        new_conn = sqlite3.connect(str(db))
        cur = new_conn.cursor()
        cur.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS options_quotes(
            asof_date TEXT, ticker TEXT, expiry TEXT,
            strike REAL, spot REAL, ttm_years REAL,
            moneyness REAL, iv REAL, delta REAL, is_atm INTEGER
        );
        CREATE INDEX IF NOT EXISTS ix_optq_date_ticker ON options_quotes(asof_date, ticker);
        CREATE TABLE IF NOT EXISTS underlying_prices(
            asof_date TEXT, ticker TEXT, close REAL
        );
        CREATE INDEX IF NOT EXISTS ix_ul_date_ticker ON underlying_prices(asof_date, ticker);
        """)
        new_conn.commit()
        rep3 = _integrity_report(new_conn)
        print("[db_guard] Fresh schema integrity_check:", "; ".join(rep3[:3]))
        new_conn.close()
        print("[db_guard] DB ready for re-ingest (use your save_for_tickers pipeline).")
        # At this point, your downloader/ingest (save_for_tickers) should populate required tickers.
        return True
    except Exception as e:
        print(f"[db_guard] Minimal reinit failed: {e}")
        return False

if __name__ == "__main__":
    ok = diagnose_and_try_recover(DB, required_tickers=["RGTI","QUBT","QBTS","IONQ"], max_expiries=6)
    print("[db_guard] usable:", ok)
