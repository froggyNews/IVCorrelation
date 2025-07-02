import pandas as pd
import datetime

from src.synthetic_etf import combine_surfaces


def test_combine_surfaces_single_day():
    K = [100, 110]
    T = [30, 60]
    idx = pd.Index(K, name="K")
    cols = pd.Index(T, name="T")

    date = pd.Timestamp("2021-01-01")

    surf_a = pd.DataFrame([[0.1, 0.11], [0.12, 0.13]], index=idx, columns=cols)
    surf_b = pd.DataFrame([[0.2, 0.21], [0.22, 0.23]], index=idx, columns=cols)

    surfaces = {
        "A": {date: surf_a},
        "B": {date: surf_b},
    }
    rhos = {"A": 1.0, "B": 2.0}
    result = combine_surfaces(surfaces, rhos)
    etf = result[date]

    expected = (1.0 * surf_a + 2.0 * surf_b) / (1.0 + 2.0)
    pd.testing.assert_frame_equal(etf, expected)
