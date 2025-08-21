from __future__ import annotations
import os
from pathlib import Path
import sqlite3
from typing import Tuple
import pandas as pd
import numpy as np
import databento as db
from dotenv import load_dotenv

# Default if caller doesn't pass a db_path
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))

# -----------------------------
# SQLite helpers
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def get_conn(db_path: Path) -> sqlite3.Connection:
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS opra_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            symbol TEXT,
            PRIMARY KEY (ticker, ts_event, symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_opra_1m_ts ON opra_1m(ticker, ts_event);

        CREATE TABLE IF NOT EXISTS equity_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            symbol TEXT,
            PRIMARY KEY (ticker, ts_event)
        );
        CREATE INDEX IF NOT EXISTS idx_equity_1m_ts ON equity_1m(ticker, ts_event);

        CREATE TABLE IF NOT EXISTS equity_1h (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            PRIMARY KEY (ticker, ts_event)
        );
        CREATE INDEX IF NOT EXISTS idx_equity_1h_ts ON equity_1h(ticker, ts_event);

        CREATE TABLE IF NOT EXISTS merged_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            opt_symbol TEXT,
            stock_symbol TEXT,
            opt_close REAL,
            stock_close REAL,
            opt_volume REAL,
            stock_volume REAL,
            PRIMARY KEY (ticker, ts_event, opt_symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_merged_1m_ts ON merged_1m(ticker, ts_event);

        CREATE TABLE IF NOT EXISTS processed_merged_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            opt_symbol TEXT,
            stock_symbol TEXT,
            opt_close REAL,
            stock_close REAL,
            opt_volume REAL,
            stock_volume REAL,
            expiry_date TEXT,
            option_type TEXT,
            strike_price REAL,
            time_to_expiry REAL,
            moneyness REAL,
            PRIMARY KEY (ticker, ts_event, opt_symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_processed_1m_ts  ON processed_merged_1m(ticker, ts_event);
        CREATE INDEX IF NOT EXISTS idx_processed_1m_exp ON processed_merged_1m(ticker, expiry_date);

        CREATE TABLE IF NOT EXISTS atm_slices_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            opt_symbol TEXT,
            stock_symbol TEXT,
            opt_close REAL,
            stock_close REAL,
            opt_volume REAL,
            stock_volume REAL,
            option_type TEXT,
            strike_price REAL,
            time_to_expiry REAL,
            moneyness REAL,
            PRIMARY KEY (ticker, ts_event, expiry_date)
        );
        CREATE INDEX IF NOT EXISTS idx_atm_1m_ts  ON atm_slices_1m(ticker, ts_event);
        CREATE INDEX IF NOT EXISTS idx_atm_1m_exp ON atm_slices_1m(ticker, expiry_date);
        """
    )
    conn.commit()

def populate_atm_slices(conn: sqlite3.Connection, ticker: str):
    q = """
    INSERT OR REPLACE INTO atm_slices_1m (
        ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
        opt_close, stock_close, opt_volume, stock_volume,
        option_type, strike_price, time_to_expiry, moneyness
    )
    SELECT
        ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
        opt_close, stock_close, opt_volume, stock_volume,
        option_type, strike_price, time_to_expiry, moneyness
    FROM (
        SELECT
            pm.*,
            ROW_NUMBER() OVER (
                PARTITION BY pm.ticker, pm.ts_event, pm.expiry_date
                ORDER BY ABS(pm.strike_price - pm.stock_close) ASC
            ) AS rn
        FROM processed_merged_1m pm
        WHERE pm.ticker = ?
    )
    WHERE rn = 1;
    """
    conn.execute(q, (ticker,))
    conn.commit()

def _iso_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _upsert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str):
    if df.empty:
        return
    tmp = f"tmp_{table}"
    df.to_sql(tmp, conn, if_exists="replace", index=False)
    cols = list(df.columns)
    col_list = ",".join(cols)
    conn.execute(f"INSERT OR IGNORE INTO {table} ({col_list}) SELECT {col_list} FROM {tmp};")
    conn.execute(f"DROP TABLE {tmp};")
    conn.commit()

# -----------------------------
# Databento fetch
# -----------------------------
def get_data_from_databento(API_KEY: str, start_date: pd.Timestamp, end_date: pd.Timestamp, ticker: str
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = db.Historical(API_KEY)
    opt_symbol = f"{ticker}.opt"

    opra_1m = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        stype_in="parent",
        symbols=[opt_symbol],
        schema="OHLCV-1m",
        start=start_date,
        end=end_date,
    ).to_df().reset_index()

    eq_1m = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        symbols=[ticker],
        schema="OHLCV-1m",
        start=start_date,
        end=end_date,
    ).to_df().reset_index()

    eq_1h = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        symbols=[ticker],
        schema="OHLCV-1H",
        start=start_date - pd.DateOffset(years=2),
        end=end_date,
    ).to_df().reset_index()

    for df in (opra_1m, eq_1m, eq_1h):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")

    return opra_1m, eq_1m, eq_1h

# -----------------------------
# Preprocess & persist to SQLite
# -----------------------------
def calculate_sigma(equity_1h: pd.DataFrame) -> float:
    eq = equity_1h.copy()
    eq["close"] = pd.to_numeric(eq["close"], errors="coerce")
    ret = eq["close"].pct_change()
    hourly_sigma = ret.std()
    ann_sigma = hourly_sigma * np.sqrt(252 * 6.5)  # rough scale
    return float(ann_sigma) if np.isfinite(ann_sigma) else np.nan

def preprocess_and_store(
    API_KEY: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker: str,
    conn: sqlite3.Connection,
    r: float = 0.045,
    force: bool = False,
):
    if not force:
        q = """
        SELECT COUNT(1) FROM processed_merged_1m
        WHERE ticker = ?
          AND ts_event >= ?
          AND ts_event <= ?;
        """
        cur = conn.execute(q, (ticker, start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                    end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
        cnt = cur.fetchone()[0]
        if cnt and cnt > 0:
            print(f"[SKIP] processed_merged_1m already has rows for {ticker} in window.")
            return

    opra_1m, eq_1m, eq_1h = get_data_from_databento(API_KEY, start_date, end_date, ticker)

    # persist raw
    opra_db = opra_1m[["ts_event","open","high","low","close","volume","symbol"]].copy()
    opra_db.insert(0, "ticker", ticker)
    opra_db["ts_event"] = _iso_utc(opra_db["ts_event"])
    _upsert_df(conn, opra_db, "opra_1m")

    equity_1m_db = eq_1m[["ts_event","open","high","low","close","volume","symbol"]].copy()
    equity_1m_db.insert(0, "ticker", ticker)
    equity_1m_db["ts_event"] = _iso_utc(equity_1m_db["ts_event"])
    _upsert_df(conn, equity_1m_db, "equity_1m")

    equity_1h_db = eq_1h[["ts_event","open","high","low","close","volume"]].copy()
    equity_1h_db.insert(0, "ticker", ticker)
    equity_1h_db["ts_event"] = _iso_utc(equity_1h_db["ts_event"])
    _upsert_df(conn, equity_1h_db, "equity_1h")

    # merge @ 1m
    merged = pd.merge(
        opra_1m.rename(columns={"close":"opt_close","volume":"opt_volume","symbol":"opt_symbol"}),
        eq_1m.rename(columns={"close":"stock_close","volume":"stock_volume","symbol":"stock_symbol"}),
        on="ts_event",
        how="inner",
        suffixes=("_opt","_eq"),
    )
    merged = merged.copy()
    merged["ts_event"] = pd.to_datetime(merged["ts_event"], utc=True, errors="coerce")
    merged = merged[(merged["ts_event"].dt.hour >= 14) & (merged["ts_event"].dt.hour <= 21)]

    # OCC parse
    extracted = merged["opt_symbol"].astype(str).str.extract(r"(\d{6})([CP])(\d{8})")
    merged["expiry_date"] = pd.to_datetime(extracted[0], format="%y%m%d", errors="coerce", utc=True)
    merged["option_type"] = extracted[1]
    merged["strike_price"] = pd.to_numeric(extracted[2], errors="coerce") / 1000.0
    merged["time_to_expiry"] = ((merged["expiry_date"] - merged["ts_event"]).dt.total_seconds() / (365*24*3600)).clip(lower=0.0)
    merged["moneyness"] = np.where(
        merged["option_type"].eq("C"),
        merged["stock_close"] - merged["strike_price"],
        np.where(merged["option_type"].eq("P"), merged["strike_price"] - merged["stock_close"], np.nan)
    )
    merged = merged.dropna(subset=["expiry_date","strike_price","option_type","opt_close","stock_close","time_to_expiry"])

    # persist merged_1m + processed_merged_1m
    merged_db = merged[["ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume"]].copy()
    merged_db.insert(0, "ticker", ticker)
    merged_db["ts_event"] = _iso_utc(merged_db["ts_event"])
    _upsert_df(conn, merged_db, "merged_1m")

    processed_db = merged[[
        "ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume",
        "expiry_date","option_type","strike_price","time_to_expiry","moneyness"
    ]].copy()
    processed_db.insert(0, "ticker", ticker)
    processed_db["ts_event"] = _iso_utc(processed_db["ts_event"])
    processed_db["expiry_date"] = _iso_utc(processed_db["expiry_date"])
    _upsert_df(conn, processed_db, "processed_merged_1m")

    # ATM slices
    populate_atm_slices(conn, ticker)

    sigma = calculate_sigma(eq_1h)
    sigma_str = f"{sigma:.4f}" if np.isfinite(sigma) else "nan"
    print(f"[DONE] {ticker}: processed_1m={len(processed_db)}  sigma_annual~{sigma_str}")

# -----------------------------
# Public entry
# -----------------------------
def fetch_and_save(API_KEY: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                   db_path: Path | str | None = None, force: bool = False) -> Path:
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = get_conn(dbp)
    init_schema(conn)
    preprocess_and_store(API_KEY, start, end, ticker, conn, force=force)
    conn.close()
    return dbp

# -----------------------------
# Demo main
# -----------------------------
def _calculate_iv(price: float, S: float, K: float, T: float, cp: str, r: float) -> float:
    """IV calculation (moved here to avoid circular imports)."""
    if not np.isfinite([price, S, K, T, r]).all() or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    intrinsic = max(S - K, 0.0) if cp.upper().startswith('C') else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6
        
    def bs_price(sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return intrinsic
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        if cp.upper().startswith('C'):
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    try:
        return brentq(lambda sig: bs_price(sig) - price, 1e-6, 5.0, maxiter=100, xtol=1e-8)
    except:
        return np.nan


def _safe_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Safely check if a table exists in the database."""
    try:
        result = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", 
            (table_name,)
        ).fetchone()
        return result is not None
    except Exception:
        return False


def _safe_data_exists(conn: sqlite3.Connection, table: str, ticker: str, start: str = None, end: str = None) -> bool:
    """Safely check if data exists for a ticker in the given time range."""
    try:
        where_clauses, params = ["ticker=?"], [ticker]
        if start:
            where_clauses.append("ts_event >= ?")
            params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end:
            where_clauses.append("ts_event <= ?")
            params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
        query = f"SELECT COUNT(*) FROM {table} WHERE {' AND '.join(where_clauses)}"
        result = conn.execute(query, params).fetchone()
        return result[0] > 0 if result else False
    except Exception:
        return False


def _populate_atm_slices(conn: sqlite3.Connection, ticker: str) -> None:
    """Populate ATM slices table from processed_merged_1m data."""
    try:
        # Get column schemas for both tables
        atm_cols = _get_table_columns(conn, "atm_slices_1m")
        processed_cols = _get_table_columns(conn, "processed_merged_1m")
        
        if not atm_cols or not processed_cols:
            print(f"  ✗ Could not get table schemas for {ticker}")
            return
        
        # Find common columns (excluding any extras)
        common_cols = [col for col in atm_cols if col in processed_cols]
        
        if len(common_cols) < 10:  # Sanity check
            print(f"  ✗ Too few common columns ({len(common_cols)}) for {ticker}")
            return
        
        # Build the query with exact column matching
        cols_str = ", ".join(common_cols)
        
        q = f"""
        INSERT OR REPLACE INTO atm_slices_1m ({cols_str})
        SELECT {cols_str}
        FROM (
            SELECT
                {cols_str},
                ROW_NUMBER() OVER (
                  PARTITION BY ticker, ts_event, expiry_date
                  ORDER BY ABS(strike_price - stock_close)
                ) rn
            FROM processed_merged_1m
            WHERE ticker = ?
        )
        WHERE rn = 1;
        """
        
        result = conn.execute(q, (ticker,))
        conn.commit()
        rows_affected = result.rowcount
        print(f"  ✓ Populated ATM slices for {ticker} ({rows_affected} rows)")
        
    except Exception as e:
        print(f"  ✗ Failed to populate ATM slices for {ticker}: {e}")


def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=None) -> pd.DataFrame:
    """Load ticker core data with IV calculation."""
    
    if db_path is None:
        db_path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    
    # Check if database file exists
    if not Path(db_path).exists():
        print(f"Database file does not exist: {db_path}")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Debug: Check table schemas
            # print(f"DEBUG: atm_slices_1m columns: {_get_table_columns(conn, 'atm_slices_1m')}")
            # print(f"DEBUG: processed_merged_1m columns: {_get_table_columns(conn, 'processed_merged_1m')}")
            
            # Try tables in order of preference
            table = None
            for candidate in ["atm_slices_1m", "processed_merged_1m", "processed_merged"]:
                if _safe_table_exists(conn, candidate):
                    if _safe_data_exists(conn, candidate, ticker, start, end):
                        table = candidate
                        break
            
            if table is None:
                print(f"No data found for {ticker} in any table")
                return pd.DataFrame()
            
            # If we have processed_merged_1m but not atm_slices_1m, populate ATM slices
            if table == "processed_merged_1m" and _safe_table_exists(conn, "atm_slices_1m"):
                # Only populate if ATM table is truly empty for this ticker
                if not _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                    _populate_atm_slices(conn, ticker)
                    # Check if ATM data is now available
                    if _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                        table = "atm_slices_1m"
            
            where_clauses, params = ["ticker=?"], [ticker]
            if start:
                where_clauses.append("ts_event >= ?")
                params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            if end:
                where_clauses.append("ts_event <= ?")
                params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
            # Get required columns that actually exist in the table
            required_cols = ["ts_event", "expiry_date", "opt_symbol", "stock_symbol",
                           "opt_close", "stock_close", "opt_volume", "stock_volume",
                           "option_type", "strike_price", "time_to_expiry", "moneyness"]
            
            available_cols = _get_table_columns(conn, table)
            select_cols = [col for col in required_cols if col in available_cols]
            
            if len(select_cols) < 8:  # Minimum required columns
                print(f"Insufficient columns in {table} for {ticker}")
                return pd.DataFrame()
            
            cols_str = ", ".join(select_cols)
            
            # Build query based on table
            if table == "atm_slices_1m":
                query = f"""
                SELECT {cols_str}
                FROM {table} WHERE {' AND '.join(where_clauses)}
                ORDER BY ts_event
                """
            else:
                # For processed_merged_1m, we need to select ATM options manually
                query = f"""
                SELECT {cols_str}
                FROM (
                    SELECT {cols_str},
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker, ts_event, expiry_date
                               ORDER BY ABS(strike_price - stock_close)
                           ) as rn
                    FROM {table} 
                    WHERE {' AND '.join(where_clauses)}
                ) ranked
                WHERE rn = 1
                ORDER BY ts_event
                """
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts_event", "expiry_date"])
            
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
        
    try:
        # IV calculation
        df["iv"] = df.apply(lambda row: _calculate_iv(
            row["opt_close"], row["stock_close"], row["strike_price"], 
            max(row["time_to_expiry"], 1e-6), row["option_type"], r
        ), axis=1)
        
        # Core cleanup (preserves original column selection and processing)
        available_keep = [col for col in ["ts_event", "expiry_date", "iv", "opt_volume", "stock_close", 
                                        "stock_volume", "time_to_expiry", "strike_price", "option_type"] 
                         if col in df.columns]
        df = df[available_keep].copy()
        df["symbol"] = ticker
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["iv"]).sort_values("ts_event").reset_index(drop=True)
        df["iv_clip"] = df["iv"].clip(lower=1e-6)
        
    except Exception as e:
        print(f"Error processing IV data for {ticker}: {e}")
        return pd.DataFrame()
    
    return df
