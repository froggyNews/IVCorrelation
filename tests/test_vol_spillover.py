import pandas as pd
import numpy as np

from analysis.spillover.vol_spillover import (
    compute_responses,
    compute_weights_and_regression,
)


def test_compute_responses_horizon_offsets():
    dates = pd.date_range('2023-01-01', periods=4)
    df = pd.DataFrame({
        'date': list(dates) * 2,
        'ticker': ['AAA'] * 4 + ['BBB'] * 4,
        'atm_iv': [100, 120, 110, 115, 50, 55, 60, 65],
    })
    events = pd.DataFrame({
        'ticker': ['AAA'],
        'date': [dates[1]],
        'rel_change': [0.2],
        'sign': [1],
    })
    peers = {'AAA': ['BBB']}
    responses = compute_responses(df, events, peers, horizons=[1, 2])
    result = responses.sort_values('h')['peer_pct'].tolist()
    assert np.allclose(result, [0.2, 0.3])


def test_weighting_and_regression_90d(monkeypatch):
    dates = pd.date_range('2023-01-01', periods=100)
    base = 100.0
    t_ret = np.sin(np.linspace(0, 10, 100)) * 0.01
    p1_ret = 0.6 * t_ret + 0.001  # high correlation
    p2_ret = 0.2 * t_ret + 0.001 * np.cos(np.linspace(0, 3, 100))

    def build_iv(r):
        return base * np.exp(np.cumsum(r))

    df = pd.DataFrame({
        'date': list(dates) * 3,
        'ticker': ['AAA'] * 100 + ['BBB'] * 100 + ['CCC'] * 100,
        'atm_iv': np.concatenate([
            build_iv(t_ret),
            build_iv(p1_ret),
            build_iv(p2_ret),
        ]),
    })

    def fake_get_groups(ticker, conn=None):
        return ['grp'] if ticker == 'AAA' else []

    def fake_load_group(name, conn=None):
        return {'peer_tickers': ['BBB', 'CCC']} if name == 'grp' else None

    monkeypatch.setattr(
        'analysis.spillover.vol_spillover.get_groups_for_target', fake_get_groups
    )
    monkeypatch.setattr(
        'analysis.spillover.vol_spillover.load_ticker_group', fake_load_group
    )

    weights, betas = compute_weights_and_regression(df, 'AAA', window=90)

    assert np.isclose(weights.sum(), 1.0)
    assert weights['BBB'] > weights['CCC']
    assert np.isclose(betas['BBB'], 0.6, atol=0.05)
    assert np.isclose(betas['CCC'], 0.2, atol=0.05)
