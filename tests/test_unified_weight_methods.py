import numpy as np
import pandas as pd
import pytest


from analysis.unified_weights import (
    UnifiedWeightComputer,
    WeightConfig,
    WeightMethod,
    FeatureSet,
)


def _patch_feature_matrix(monkeypatch, feature_df):
    monkeypatch.setattr(
        UnifiedWeightComputer,
        "_build_feature_matrix",
        lambda self, target, peers, asof, config: feature_df,
    )


def test_correlation_weights_success(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 2], [1, 2], [1, 2]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(0.5)
    assert weights.loc["P2"] == pytest.approx(0.5)


def test_correlation_weights_zero_sum_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 1], [-1, -1], [-2, -2]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    with pytest.raises(ValueError):
        uwc.compute_weights("TGT", ["P1", "P2"], cfg)


def test_cosine_weights_success(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.COSINE,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 0], [1, 0], [1, 0]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(0.5)
    assert weights.loc["P2"] == pytest.approx(0.5)


def test_cosine_weights_zero_sum_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.COSINE,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 0], [0, 1], [0, -1]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    with pytest.raises(ValueError):
        uwc.compute_weights("TGT", ["P1", "P2"], cfg)


def test_pca_weights(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.PCA,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 2], [2, 1], [3, 0]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    monkeypatch.setattr(
        "analysis.beta_builder._impute_col_median", lambda arr: arr
    )
    monkeypatch.setattr(
        "analysis.beta_builder.pca_regress_weights",
        lambda Xp, y, k=None, nonneg=True: np.array([2.0, 1.0]),
    )
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(2.0 / 3.0)
    assert weights.loc["P2"] == pytest.approx(1.0 / 3.0)


def test_equal_weights():
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.EQUAL, feature_set=FeatureSet.ATM)
    weights = uwc.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert all(weight == pytest.approx(1.0 / 3.0) for weight in weights)


def test_surface_feature_set_dispatch(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.SURFACE,
        asof="2024-01-01",
    )

    feature_df = pd.DataFrame(
        [[1, 2], [1, 2], [1, 2]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )

    called = {"surface": False}

    def fake_surface(self, tickers, asof, config):
        called["surface"] = True
        return feature_df

    monkeypatch.setattr(UnifiedWeightComputer, "_build_surface_features", fake_surface)
    monkeypatch.setattr(
        UnifiedWeightComputer,
        "_compute_weights_from_features",
        lambda self, df, target, peers, config: pd.Series(0.5, index=peers),
    )

    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert called["surface"]
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(0.5)
    assert weights.loc["P2"] == pytest.approx(0.5)
