import sqlite3
from analysis.cache_io import save_calc_cache, load_calc_cache


def _setup_db():
    conn = sqlite3.connect(':memory:')
    conn.executescript(
        """
        CREATE TABLE options_quotes(asof_date TEXT);
        CREATE TABLE underlying_prices(asof_date TEXT);
        """
    )
    return conn


def test_cache_reuse_when_raw_unchanged():
    conn = _setup_db()
    conn.execute("INSERT INTO options_quotes(asof_date) VALUES('2000-01-01')")
    save_calc_cache(conn, 'k', {'v': 1})
    loaded = load_calc_cache(conn, 'k')
    assert loaded == {'v': 1}


def test_cache_invalidated_on_raw_update():
    conn = _setup_db()
    conn.execute("INSERT INTO options_quotes(asof_date) VALUES('1999-01-01')")
    save_calc_cache(conn, 'k', {'v': 1})
    # Force created_at to be old
    conn.execute("UPDATE calc_cache SET created_at = 0")
    # Later raw data arrives
    conn.execute("INSERT INTO options_quotes(asof_date) VALUES('2000-01-01')")
    conn.commit()
    assert load_calc_cache(conn, 'k') is None
