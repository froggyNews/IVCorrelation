"""Utilities for analyzing historically calibrated model parameters."""
from __future__ import annotations

import pandas as pd
from typing import Optional

from .model_params_logger import load_model_params


def historical_param_timeseries(
    ticker: str,
    model: str,
    param: str,
    df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Return the time series of a specific parameter's fitted values.

    Parameters
    ----------
    ticker : str
        The underlying ticker symbol.
    model : str
        Name of the model (case-insensitive).
    param : str
        Parameter name (case-insensitive).
    df : Optional[pd.DataFrame]
        Optional pre-loaded DataFrame of historical parameters. If not
        provided, :func:`load_model_params` is used.

    Returns
    -------
    pandas.Series
        Series indexed by ``asof_date`` containing the parameter values.
    """
    if df is None:
        df = load_model_params()
    sel = (
        (df["ticker"].str.upper() == ticker.upper())
        & (df["model"].str.lower() == model.lower())
        & (df["param"].str.lower() == param.lower())
    )
    series = (
        df.loc[sel, ["asof_date", "value"]]
        .dropna(subset=["asof_date"])
        .set_index("asof_date")
        ["value"]
        .sort_index()
    )
    return series


def historical_param_summary(
    ticker: str | None = None,
    model: str | None = None,
    param: str | None = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute summary statistics for historically fitted parameters.

    Parameters
    ----------
    ticker : Optional[str]
        Filter for a specific ticker symbol.
    model : Optional[str]
        Filter for a specific model name.
    param : Optional[str]
        Filter for a specific parameter name.
    df : Optional[pd.DataFrame]
        Optional pre-loaded DataFrame. If not provided, the parameters are
        loaded from disk.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``ticker``, ``model``, ``param``, ``count``,
        ``mean``, ``std``, ``min`` and ``max``.
    """
    if df is None:
        df = load_model_params()

    if ticker is not None:
        df = df[df["ticker"].str.upper() == ticker.upper()]
    if model is not None:
        df = df[df["model"].str.lower() == model.lower()]
    if param is not None:
        df = df[df["param"].str.lower() == param.lower()]

    if df.empty:
        return pd.DataFrame(columns=["ticker", "model", "param", "count", "mean", "std", "min", "max"])

    grouped = df.groupby(["ticker", "model", "param"])["value"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    return summary
