"""Utility functions to compute correlation and liquidity weights."""

import pandas as pd


def correlation_weights(stock_returns: pd.DataFrame, theme_proxy: pd.Series, *, absolute: bool = True) -> pd.Series:
    r"""Compute correlation weights :math:`\rho_i` for each stock.

    Parameters
    ----------
    stock_returns : pandas.DataFrame
        Return series for individual stocks, columns correspond to stocks.
    theme_proxy : pandas.Series
        Return series representing the thematic proxy.
    absolute : bool, default True
        If True, uses absolute correlations so weights are non-negative.

    Returns
    -------
    pandas.Series
        Normalized correlation-based weights that sum to 1.
    """
    corr = stock_returns.corrwith(theme_proxy)
    if absolute:
        corr = corr.abs()
    weights = corr / corr.sum()
    return weights


def liquidity_weights(options: pd.DataFrame, *, window: int = 3) -> pd.DataFrame:
    r"""Compute strike-level liquidity weights :math:`\omega_i(K, T)`.

    Liquidity is based on option trading volume smoothed across neighbouring
    strikes. The smoothed volume for each maturity is then normalized to
    produce weights.

    Parameters
    ----------
    options : pandas.DataFrame
        Input DataFrame must contain columns ``'maturity'``, ``'strike'`` and
        ``'volume'``. Rows typically represent the latest observed volume for
        each option.
    window : int, default 3
        Rolling window size used when smoothing volumes across strikes.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with two additional columns:
        ``'smoothed_volume'`` and ``'liquidity_weight'``.
    """

    df = options.sort_values(['maturity', 'strike']).copy()
    df['smoothed_volume'] = (
        df.groupby('maturity')['volume']
        .transform(lambda s: s.rolling(window=window, center=True, min_periods=1).mean())
    )
    df['liquidity_weight'] = (
        df['smoothed_volume'] / df.groupby('maturity')['smoothed_volume'].transform('sum')
    )
    return df
