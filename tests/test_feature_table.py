import json
from data.db_utils import get_conn, ensure_initialized, insert_features

def test_insert_features_and_retrieve(tmp_path):
    db_file = tmp_path / "tmp.db"
    conn = get_conn(str(db_file))
    ensure_initialized(conn)
    rows = [{
        "ts_event": "2024-01-01T00:00:00",
        "symbol": "SPY",
        "foo": 1.0,
        "bar": 2.0,
    }]
    inserted = insert_features(conn, rows)
    assert inserted == 1
    cur = conn.execute("SELECT ts_event, symbol, features FROM feature_table")
    ts_event, symbol, feat_json = cur.fetchone()
    assert ts_event == "2024-01-01T00:00:00"
    assert symbol == "SPY"
    data = json.loads(feat_json)
    assert data == {"foo": 1.0, "bar": 2.0}
