import sqlite3
from data.db_utils import ensure_initialized, insert_quotes, fetch_vol_shifts


def _make_quote(asof, iv):
    return {
        "asof_date": asof,
        "ticker": "TST",
        "expiry": "2024-06-01",
        "K": 100,
        "call_put": "C",
        "sigma": iv,
    }


def test_fetch_vol_shifts_detects_change():
    conn = sqlite3.connect(":memory:")
    ensure_initialized(conn)

    insert_quotes(conn, [_make_quote("2024-01-01", 0.2)])
    insert_quotes(conn, [_make_quote("2024-01-02", 0.25)])

    shifts = fetch_vol_shifts(conn, tickers=["TST"])
    assert len(shifts) == 1
    row = shifts.iloc[0]
    assert row["iv_old"] == 0.2
    assert row["iv_new"] == 0.25
    assert abs(row["iv_shift"] - 0.05) < 1e-9


def test_fetch_vol_shifts_no_prior_data():
    conn = sqlite3.connect(":memory:")
    ensure_initialized(conn)
    insert_quotes(conn, [_make_quote("2024-01-01", 0.2)])
    shifts = fetch_vol_shifts(conn, tickers=["TST"])
    assert shifts.empty
