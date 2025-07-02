import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from ivcorrelation.normalizer import normalize_surface

class TestNormalizeSurface(unittest.TestCase):
    def test_basic_normalization(self):
        data = [
            {'ticker': 'A', 'date': '2024-01-01', 'maturity': 30, 'strike': 90, 'implied_vol': 0.2},
            {'ticker': 'A', 'date': '2024-01-01', 'maturity': 30, 'strike': 100, 'implied_vol': 0.25},
            {'ticker': 'A', 'date': '2024-01-01', 'maturity': 30, 'strike': 110, 'implied_vol': 0.22},
        ]
        result = normalize_surface(data, atm_strike=100)
        key = ('A', '2024-01-01')
        self.assertIn(key, result)
        grid = result[key]
        self.assertAlmostEqual(grid[30][90], 0.2/0.25)
        self.assertAlmostEqual(grid[30][100], 1.0)
        self.assertAlmostEqual(grid[30][110], 0.22/0.25)

    def test_fallback_median(self):
        data = [
            {'ticker': 'B', 'date': '2024-01-01', 'maturity': 20, 'strike': 80, 'implied_vol': 0.15},
            {'ticker': 'B', 'date': '2024-01-01', 'maturity': 20, 'strike': 90, 'implied_vol': 0.16},
            {'ticker': 'B', 'date': '2024-01-01', 'maturity': 20, 'strike': 120, 'implied_vol': 0.18},
        ]
        result = normalize_surface(data, atm_strike=100)
        key = ('B', '2024-01-01')
        grid = result[key]
        self.assertAlmostEqual(grid[20][80], 0.15/0.16)
        self.assertAlmostEqual(grid[20][120], 0.18/0.16)

if __name__ == '__main__':
    unittest.main()

