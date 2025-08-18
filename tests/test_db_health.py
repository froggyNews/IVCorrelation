from data.db_utils import get_conn, ensure_initialized, insert_quotes


def _quote():
    return {
        "asof_date": "2024-01-01",
        "ticker": "SPY",
        "expiry": "2024-02-01",
        "K": 100,
        "call_put": "C",
    }


def test_insert_quotes_checks_db_health(monkeypatch):
    conn = get_conn(":memory:")
    ensure_initialized(conn)
    called = {"flag": False}

    def fake_health_check(c):
        called["flag"] = True

    monkeypatch.setattr("data.db_utils.check_db_health", fake_health_check)

    insert_quotes(conn, [_quote()])

    assert called["flag"]

