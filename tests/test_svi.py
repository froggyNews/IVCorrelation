import math
import unittest

from svi import fit_svi_smile, svi_implied_vol


class TestSVIFitting(unittest.TestCase):
    def test_fit_svi_smile(self):
        strikes = [90, 100, 110]
        vols = [0.25, 0.2, 0.22]
        f = 100.0
        t = 0.5
        params = fit_svi_smile(strikes, vols, f, t)
        self.assertIsNotNone(params)
        a, b, rho, m, sigma = params
        fitted = [svi_implied_vol(math.log(k / f), t, a, b, rho, m, sigma) for k in strikes]
        for mv, ov in zip(fitted, vols):
            self.assertAlmostEqual(mv, ov, delta=0.05)


if __name__ == "__main__":
    unittest.main()
