#!/usr/bin/env python3
"""Update underlying prices with comprehensive historical data."""

from data.underlying_prices import update_underlying_prices
from data.db_utils import get_conn
import pandas as pd

def get_available_tickers():
    """Get all tickers that have options data."""
    conn = get_conn()
    df = pd.read_sql_query("SELECT DISTINCT ticker FROM options_quotes", conn)
    return sorted(df['ticker'].tolist())

def update_all_underlying_prices():
    """Update underlying prices for all available tickers."""
    print("Getting available tickers...")
    tickers = get_available_tickers()
    print(f"Found {len(tickers)} tickers: {tickers}")
    
    print(f"\nFetching 1 year of historical data for {len(tickers)} tickers...")
    print("This may take a few minutes due to API rate limits...")
    
    total_rows = update_underlying_prices(tickers, period="1y")
    print(f"\nSuccessfully updated {total_rows} price records")
    
    # Check the result
    conn = get_conn()
    result_df = pd.read_sql_query(
        "SELECT COUNT(*) as total_rows, COUNT(DISTINCT ticker) as unique_tickers, MIN(asof_date) as earliest, MAX(asof_date) as latest FROM underlying_prices", 
        conn
    )
    print(f"\nDatabase now contains:")
    print(f"  Total rows: {result_df['total_rows'].iloc[0]}")
    print(f"  Unique tickers: {result_df['unique_tickers'].iloc[0]}")
    print(f"  Date range: {result_df['earliest'].iloc[0]} to {result_df['latest'].iloc[0]}")

if __name__ == "__main__":
    update_all_underlying_prices()
