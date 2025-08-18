import pandas as pd
import pytest

from analysis.unified_weights import UnifiedWeightComputer, WeightConfig, WeightMethod, FeatureSet


def test_missing_target_data_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.ATM)
    monkeypatch.setattr(
        'analysis.unified_weights.available_dates',
        lambda ticker=None, most_recent_only=True: []
    )
    with pytest.raises(ValueError):
        uwc.compute_weights('TGT', ['P1', 'P2'], cfg)


def test_empty_feature_matrix_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.ATM)
    monkeypatch.setattr(
        'analysis.unified_weights.available_dates',
        lambda ticker=None, most_recent_only=True: ['2024-01-01']
    )
    monkeypatch.setattr(
        UnifiedWeightComputer,
        '_build_feature_matrix',
        lambda self, target, peers, asof, config: pd.DataFrame()
    )
    with pytest.raises(ValueError):
        uwc.compute_weights('TGT', ['P1', 'P2'], cfg)
