import numpy as np


def test_termfit_exposes_sabr_and_svi():
    # Ensure top-level volModel exports include sabr, svi, and term fitting
    from volModel import (
        fit_term_structure,
        fit_sabr_slice,
        fit_svi_slice,
    )

    # simple polynomial term fit
    T = np.array([0.1, 0.2, 0.3])
    iv = np.array([0.2, 0.21, 0.22])
    params = fit_term_structure(T, iv)
    assert "coeff" in params

    # smoke-test sabr and svi fits on tiny synthetic slice
    S = 100.0
    Ks = np.array([95.0, 100.0, 105.0])
    slice_iv = np.array([0.21, 0.2, 0.22])

    sabr_params = fit_sabr_slice(S, Ks, 0.5, slice_iv)
    assert isinstance(sabr_params, dict)

    svi_params = fit_svi_slice(S, Ks, 0.5, slice_iv)
    assert isinstance(svi_params, dict)

