import pandas as pd

from analysis.historical_params import (
    historical_param_timeseries,
    historical_param_summary,
)


def test_historical_param_timeseries_sorted_and_nonempty():
    ts = historical_param_timeseries('QQQ', 'sabr', 'alpha')
    assert not ts.empty
    assert ts.index.is_monotonic_increasing
    assert (ts >= 0).all()


def test_historical_param_summary_matches_timeseries():
    ts = historical_param_timeseries('QQQ', 'sabr', 'alpha')
    summary = historical_param_summary('QQQ', 'sabr', 'alpha')
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row['count'] == len(ts)
    assert row['mean'] == ts.mean()
    assert row['min'] == ts.min()
    assert row['max'] == ts.max()
