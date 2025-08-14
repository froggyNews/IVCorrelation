import numpy as np
import pandas as pd

from analysis.beta_builder import cosine_similarity_weights


def test_cosine_ul_weights(monkeypatch):
    def fake_returns(conn_fn):
        return pd.DataFrame({
            'TGT': [0.1, 0.2],
            'P1': [0.1, 0.2],
            'P2': [-0.1, 0.0],
        })
    monkeypatch.setattr('analysis.beta_builder._underlying_log_returns', fake_returns)
    w = cosine_similarity_weights(
        get_smile_slice=None,
        mode='cosine_ul',
        target='TGT',
        peers=['P1', 'P2'],
        asof='2024-01-01',
    )
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w.idxmax() == 'P1'


def test_cosine_surface_weights(monkeypatch):
    def fake_surface_feature_matrix(tickers, asof, tenors=None, mny_bins=None, standardize=True):
        ok = [t.upper() for t in tickers]
        grids = {t: None for t in ok}
        X = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        names = ['f1', 'f2']
        return grids, X, names
    monkeypatch.setattr('analysis.beta_builder.surface_feature_matrix', fake_surface_feature_matrix)
    w = cosine_similarity_weights(
        get_smile_slice=None,
        mode='cosine_surface',
        target='TGT',
        peers=['P1', 'P2'],
        asof='2024-01-01',
    )
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w.idxmax() == 'P1'
