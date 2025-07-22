# Model utilities for volatility surface demo

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from compute_volatility import weighted_stats, fit_sabr_smile


def download_option_data(ticker: str = "AAPL", max_expiries: int = 8):
    """Download option chain data from Yahoo Finance with spot normalization."""
    print(f"Downloading option data for {ticker}...")

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        print(f"No option data available for {ticker}")
        return None

    # Get current spot price
    try:
        spot_price = tk.info.get("regularMarketPrice", None)
        if spot_price is None:
            hist = tk.history(period="1d")
            if not hist.empty:
                spot_price = hist["Close"].iloc[-1]
            else:
                print(f"Could not get spot price for {ticker}")
                return None
        print(f"Spot price for {ticker}: ${spot_price:.2f}")
    except Exception as e:  # pragma: no cover - network
        print(f"Error getting spot price for {ticker}: {e}")
        return None

    option_data = []
    for expiry in expiries[:max_expiries]:
        try:
            opt = tk.option_chain(expiry)
            for df, opt_type in [(opt.calls, "call"), (opt.puts, "put")]:
                expiry_date = pd.to_datetime(expiry)
                current_date = pd.to_datetime(datetime.now())
                ttm = (expiry_date - current_date).days / 365.25
                valid_iv = df["impliedVolatility"].notna() & (df["impliedVolatility"] > 0)
                if valid_iv.sum() > 0 and ttm > 0:
                    for _, row in df[valid_iv].iterrows():
                        moneyness = row["strike"] / spot_price
                        option_data.append(
                            {
                                "K": row["strike"],
                                "S": spot_price,
                                "moneyness": moneyness,
                                "T": ttm,
                                "sigma": row["impliedVolatility"],
                                "volume": row.get("volume", 0),
                                "type": opt_type,
                            }
                        )
        except Exception as e:  # pragma: no cover - network
            print(f"Error processing {expiry}: {e}")
            continue

    if not option_data:
        return None

    df = pd.DataFrame(option_data)
    df = df[df["sigma"] > 0.01]
    df = df[df["sigma"] < 2.0]
    df = df[df["T"] > 0.01]
    df = df[df["moneyness"] > 0.1]
    df = df[df["moneyness"] < 10.0]
    maturity_counts = df.groupby("T").size()
    valid_maturities = maturity_counts[maturity_counts >= 3].index
    df = df[df["T"].isin(valid_maturities)]
    return df


def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple correlation and liquidity weights."""
    df["corr_weight"] = 1.0
    df["liq_weight"] = df.groupby("T")["volume"].transform(
        lambda x: x / (x.sum() if x.sum() > 0 else 1)
    )
    return df


def download_multiple_tickers(tickers, max_expiries: int = 8):
    """Download option chains for a list of tickers."""
    all_data = {}
    for ticker in tickers:
        df = download_option_data(ticker, max_expiries)
        if df is not None and len(df) > 0:
            all_data[ticker] = df
    return all_data


def simple_sabr_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SABR parameters for each maturity."""
    results = []
    for T, group in df.groupby("T"):
        strikes = group["K"].values
        vols = group["sigma"].values
        f = strikes.mean() if len(strikes) > 0 else float("nan")
        sabr_params = fit_sabr_smile(strikes, vols, f, T)
        if sabr_params is not None:
            alpha, beta, rho, nu = sabr_params
        else:
            alpha = beta = rho = nu = float("nan")
        results.append(
            {
                "T": T,
                "sigma_med": np.median(vols),
                "sigma_std": np.std(vols),
                "sigma_low": np.min(vols),
                "sigma_high": np.max(vols),
                "sabr_alpha": alpha,
                "sabr_beta": beta,
                "sabr_rho": rho,
                "sabr_nu": nu,
            }
        )
    return pd.DataFrame(results)


def compute_correlation_weights(all_data, correlation_matrix=None):
    """Compute correlation-based weights for synthetic ETF."""
    tickers = list(all_data.keys())
    if correlation_matrix is None:
        correlation_matrix = compute_iv_correlation_matrix(all_data)
    weights = {}
    for i, ticker in enumerate(tickers):
        correlations = [correlation_matrix[i][j] for j in range(len(tickers)) if i != j]
        avg_correlation = np.mean(correlations) if correlations else 0.5
        weights[ticker] = avg_correlation
    total_weight = sum(weights.values())
    weights = {t: w / total_weight for t, w in weights.items()}
    return weights, correlation_matrix


def compute_iv_correlation_matrix(all_data):
    """Compute correlation matrix from IV data."""
    tickers = list(all_data.keys())
    n_tickers = len(tickers)
    correlation_matrix = np.zeros((n_tickers, n_tickers))
    iv_data = {}
    for ticker in tickers:
        df = all_data[ticker]
        iv_by_maturity = df.groupby("T")["sigma"].median()
        iv_data[ticker] = iv_by_maturity
    from scipy.stats import pearsonr

    for i in range(n_tickers):
        for j in range(n_tickers):
            if i == j:
                correlation_matrix[i][j] = 1.0
            else:
                iv1, iv2 = iv_data[tickers[i]], iv_data[tickers[j]]
                common = iv1.index.intersection(iv2.index)
                if len(common) >= 2:
                    corr, _ = pearsonr(iv1[common], iv2[common])
                    correlation_matrix[i][j] = corr
                else:
                    correlation_matrix[i][j] = 0.0
    return correlation_matrix


def construct_synthetic_etf(all_data, weights=None):
    """Create a basic synthetic ETF surface from multiple tickers."""
    if not all_data:
        return None
    tickers = list(all_data.keys())
    if weights is None:
        weights = {t: 1.0 / len(tickers) for t in tickers}
    total = sum(weights.values())
    weights = {t: w / total for t, w in weights.items()}

    all_maturities = set()
    for df in all_data.values():
        all_maturities.update(df["T"].unique())
    synthetic = []
    for T in sorted(all_maturities):
        maturity_data = {t: df[df["T"] == T] for t, df in all_data.items() if T in df["T"].values}
        calls = {t: d[d["type"] == "call"] for t, d in maturity_data.items() if len(d[d["type"] == "call"]) > 0}
        puts = {t: d[d["type"] == "put"] for t, d in maturity_data.items() if len(d[d["type"] == "put"]) > 0}
        if len(calls) >= 2:
            synthetic.extend(construct_synthetic_surface(calls, T, "call", weights))
        if len(puts) >= 2:
            synthetic.extend(construct_synthetic_surface(puts, T, "put", weights))
    if not synthetic:
        return None
    return pd.DataFrame(synthetic)


def construct_synthetic_surface(maturity_data, T, option_type, weights):
    """Helper for ``construct_synthetic_etf``."""
    synthetic_data = []
    all_moneyness = set()
    for ticker_data in maturity_data.values():
        all_moneyness.update(ticker_data["moneyness"].unique())
    for m in sorted(all_moneyness):
        weighted_iv = 0.0
        total_w = 0.0
        for ticker, data in maturity_data.items():
            sub = data[data["moneyness"] == m]
            if len(sub) > 0:
                iv = sub["sigma"].median()
                w = weights[ticker]
                weighted_iv += w * iv
                total_w += w
        if total_w > 0:
            avg_spot = np.mean([d["S"].iloc[0] for d in maturity_data.values()])
            synthetic_data.append(
                {
                    "K": m * avg_spot,
                    "S": avg_spot,
                    "moneyness": m,
                    "T": T,
                    "sigma": weighted_iv / total_w,
                    "type": option_type,
                    "volume": 0,
                }
            )
    return synthetic_data
