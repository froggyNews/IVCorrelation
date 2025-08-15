import pandas as pd
import numpy as np

from analysis.spillover.vol_spillover import compute_responses


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
