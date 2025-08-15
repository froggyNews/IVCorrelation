import os, sys
import pandas as pd
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.historical_saver import save_for_tickers


def test_duplicate_columns_are_dropped():
    records_holder = {}

    def fake_download_raw_option_data(ticker, max_expiries=8):
        # Minimal non-empty frame so we exercise the rest of the pipeline
        return pd.DataFrame({"x": [1]})

    def fake_enrich_quotes(raw, r=0.0, q=0.0):
        # Create DataFrame with duplicate column names
        return pd.DataFrame([[1, 2]], columns=["a", "a"])

    def fake_insert_quotes(conn, records):
        records_holder["records"] = records
        return len(records)

    with patch("data.historical_saver.get_conn", lambda: object()), \
         patch("data.historical_saver.ensure_initialized", lambda conn: None), \
         patch("data.historical_saver.download_raw_option_data", fake_download_raw_option_data), \
         patch("data.historical_saver.enrich_quotes", fake_enrich_quotes), \
         patch("data.historical_saver.insert_quotes", fake_insert_quotes):
        total = save_for_tickers(["TST"])
        assert total == 1

    # After conversion to records only a single key should remain
    assert list(records_holder["records"][0].keys()) == ["a"]
