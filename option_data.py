import os
import pandas as pd
import yfinance as yf

def fetch_historical_option_chain(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch option chain data for each day in a date range.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol to download options for.
    start_date : str
        Start date in ``YYYY-MM-DD`` format.
    end_date : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame of option chains for the date range. Each row
        includes the strike, expiration, option type, and as-of date.
    """

    ticker_obj = yf.Ticker(ticker)
    save_dir = os.path.join("data", "raw_option_chains")
    os.makedirs(save_dir, exist_ok=True)

    all_frames = []
    for current_date in pd.date_range(start=start_date, end=end_date, freq="D"):
        date_str = current_date.strftime("%Y-%m-%d")
        day_frames = []
        for expiration in ticker_obj.options:
            chain = ticker_obj.option_chain(expiration)
            calls = chain.calls.copy()
            calls["optionType"] = "call"
            puts = chain.puts.copy()
            puts["optionType"] = "put"
            combined = pd.concat([calls, puts], ignore_index=True)
            combined["expiration"] = expiration
            combined["asOfDate"] = date_str
            day_frames.append(combined)
        if day_frames:
            day_chain = pd.concat(day_frames, ignore_index=True)
            fname = f"{ticker}_{date_str}.csv"
            day_chain.to_csv(os.path.join(save_dir, fname), index=False)
            all_frames.append(day_chain)

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    return pd.DataFrame()
