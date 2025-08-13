# data/interest_rates.py
from __future__ import annotations
import sqlite3
from typing import List, Optional, Tuple
from datetime import datetime
import json

from data.db_utils import get_conn

DEFAULT_INTEREST_RATE = 0.0408  # 4.08%

def create_default_interest_rates() -> None:
    """Create the default interest rate (4.08%) if no rates exist."""
    conn = get_conn()
    
    # Check if any rates exist
    existing = conn.execute("SELECT COUNT(*) FROM interest_rates").fetchone()[0]
    
    if existing == 0:
        # Create default rate
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO interest_rates (rate_id, rate_value, description, is_default, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("default", DEFAULT_INTEREST_RATE, "Default interest rate (4.08%)", 1, now, now))
        conn.commit()


def save_interest_rate(
    rate_id: str, 
    rate_value: float, 
    description: str = "", 
    is_default: bool = False
) -> None:
    """Save or update an interest rate."""
    conn = get_conn()
    now = datetime.now().isoformat()
    
    # If setting as default, first unset all other defaults
    if is_default:
        conn.execute("UPDATE interest_rates SET is_default = 0")
    
    # Insert or replace the rate
    conn.execute("""
        INSERT OR REPLACE INTO interest_rates 
        (rate_id, rate_value, description, is_default, created_at, updated_at)
        VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM interest_rates WHERE rate_id = ?), ?), ?)
    """, (rate_id, rate_value, description, int(is_default), rate_id, now, now))
    conn.commit()


def load_interest_rate(rate_id: str) -> Optional[Tuple[float, str, bool]]:
    """Load an interest rate by ID. Returns (rate_value, description, is_default) or None."""
    conn = get_conn()
    row = conn.execute("""
        SELECT rate_value, description, is_default 
        FROM interest_rates 
        WHERE rate_id = ?
    """, (rate_id,)).fetchone()
    
    if row:
        return (float(row[0]), str(row[1] or ""), bool(row[2]))
    return None


def get_default_interest_rate() -> float:
    """Get the default interest rate value."""
    conn = get_conn()
    row = conn.execute("""
        SELECT rate_value 
        FROM interest_rates 
        WHERE is_default = 1 
        ORDER BY updated_at DESC 
        LIMIT 1
    """).fetchone()
    
    if row:
        return float(row[0])
    
    # If no default found, create and return the hardcoded default
    create_default_interest_rates()
    return DEFAULT_INTEREST_RATE


def list_interest_rates() -> List[Tuple[str, float, str, bool]]:
    """List all interest rates. Returns [(rate_id, rate_value, description, is_default), ...]"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT rate_id, rate_value, description, is_default 
        FROM interest_rates 
        ORDER BY is_default DESC, rate_id ASC
    """).fetchall()
    
    return [(str(row[0]), float(row[1]), str(row[2] or ""), bool(row[3])) for row in rows]


def delete_interest_rate(rate_id: str) -> bool:
    """Delete an interest rate. Returns True if deleted, False if not found or is default."""
    conn = get_conn()
    
    # Check if it's the default rate
    row = conn.execute("SELECT is_default FROM interest_rates WHERE rate_id = ?", (rate_id,)).fetchone()
    if not row:
        return False
    
    if row[0]:  # is_default = 1
        # Don't allow deleting default rate
        return False
    
    conn.execute("DELETE FROM interest_rates WHERE rate_id = ?", (rate_id,))
    conn.commit()
    return True


def set_default_interest_rate(rate_id: str) -> bool:
    """Set a specific interest rate as the default. Returns True if successful."""
    conn = get_conn()
    
    # Check if the rate exists
    exists = conn.execute("SELECT 1 FROM interest_rates WHERE rate_id = ?", (rate_id,)).fetchone()
    if not exists:
        return False
    
    # Unset all defaults, then set this one
    conn.execute("UPDATE interest_rates SET is_default = 0")
    conn.execute("UPDATE interest_rates SET is_default = 1, updated_at = ? WHERE rate_id = ?", 
                (datetime.now().isoformat(), rate_id))
    conn.commit()
    return True


def get_interest_rate_names() -> List[str]:
    """Get list of all interest rate IDs for dropdown menus."""
    conn = get_conn()
    rows = conn.execute("SELECT rate_id FROM interest_rates ORDER BY is_default DESC, rate_id ASC").fetchall()
    return [str(row[0]) for row in rows]
