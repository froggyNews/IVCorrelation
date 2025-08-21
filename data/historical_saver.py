"""Download raw chains, enrich via pipeline, persist to DB."""
from __future__ import annotations
from typing import Iterable

from .db_utils import get_conn, ensure_initialized, insert_quotes, insert_features
from .data_downloader import download_raw_option_data
from .data_pipeline import enrich_quotes
from .interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD
from .feature_engineering import add_all_features
import pandas as pd

def save_for_tickers(
    tickers: Iterable[str],
    max_expiries: int = 8,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> int:
    conn = get_conn()
    ensure_initialized(conn)
    total = 0
    for t in tickers:
        raw = download_raw_option_data(t, max_expiries=max_expiries)
        if raw is None or raw.empty:
            print(f"No raw rows for {t}")
            continue
        enriched = enrich_quotes(raw, r=r, q=q)
        if enriched is None or enriched.empty:
            print(f"No enriched rows for {t}")
            continue
        
        if enriched.columns.duplicated().any():
            enriched = enriched.loc[:, ~enriched.columns.duplicated()].copy()

        total += insert_quotes(conn, enriched.to_dict(orient="records"))
        print(f"Inserted {t}: {len(enriched)} rows")

        # Compute feature rows and persist to feature_table
        feat_input = enriched.rename(
            columns={
                "asof_date": "ts_event",
                "ticker": "symbol",
                "K": "strike_price",
                "call_put": "option_type",
                "sigma": "iv_clip",
                "S": "stock_close",
                "volume_raw": "opt_volume",
                "T": "time_to_expiry",
            }
        )
        feat_input["ts_event"] = pd.to_datetime(feat_input["ts_event"], errors="coerce")
        features = add_all_features(feat_input)
        features["ts_event"] = features["ts_event"].astype(str)
        insert_features(conn, features.to_dict(orient="records"))
    return total

if __name__ == "__main__":
    inserted = save_for_tickers(["SPY", "QQQ"], max_expiries=6)
    print(f"Total inserted: {inserted}")
