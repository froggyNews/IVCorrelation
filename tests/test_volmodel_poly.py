"""Tests for integrating polynomial fits into VolModel."""

import os
import sys
import numpy as np


# Ensure project root on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volModel.volModel import VolModel


def _synthetic_data():
    """Create simple synthetic smile data."""
    S = 100.0
    Ks = np.linspace(80, 120, 15)
    T = 0.5
    Ts = np.full_like(Ks, T, dtype=float)
    k = np.log(Ks / S)
    IVs = 0.2 + 0.05 * k + 0.1 * k ** 2
    return S, Ks, Ts, IVs, T


def test_volmodel_poly_simple():
    S, Ks, Ts, IVs, T = _synthetic_data()
    vm = VolModel(model="poly", poly_method="simple").fit(S, Ks, Ts, IVs)
    iv_pred = vm.predict_iv(S, T)
    assert np.isfinite(iv_pred)
    assert abs(iv_pred - 0.2) < 1e-3


def test_volmodel_poly_tps():
    S, Ks, Ts, IVs, T = _synthetic_data()
    vm = VolModel(model="poly", poly_method="tps").fit(S, Ks, Ts, IVs)
    K_test = S * 1.05
    k_test = np.log(K_test / S)
    expected = 0.2 + 0.05 * k_test + 0.1 * k_test ** 2
    iv_pred = vm.predict_iv(K_test, T)
    assert np.isfinite(iv_pred)
    # TPS fit should approximate the true value closely
    assert abs(iv_pred - expected) < 5e-3

