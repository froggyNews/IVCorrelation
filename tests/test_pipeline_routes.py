import pandas as pd
import pytest

from analysis.analysis_pipeline import compute_peer_weights


def test_compute_peer_weights_dispatch(monkeypatch):
    monkeypatch.setattr(
        "analysis.analysis_pipeline.available_dates",
        lambda ticker=None, most_recent_only=False: ["2024-01-01"],
    )

    calls = []

    def fake_build_peer_weights(method, feature, target, peers, **kwargs):
        calls.append((method, feature))
        return pd.Series({"PEER": 1.0})

    def fake_build_vol_betas(**kwargs):
        calls.append(("build_vol_betas", kwargs))
        return pd.DataFrame({"PEER": [1.0]})

    monkeypatch.setattr(
        "analysis.analysis_pipeline.build_peer_weights", fake_build_peer_weights
    )
    monkeypatch.setattr(
        "analysis.analysis_pipeline.build_vol_betas", fake_build_vol_betas
    )

    res = compute_peer_weights(
        target="SPY", peers=["QQQ"], weight_mode=("cosine", "ul_vol")
    )
    assert calls and calls[0] == ("cosine", "ul_vol")

    calls.clear()
    res = compute_peer_weights(
        target="SPY", peers=["QQQ"], weight_mode="surface_grid"
    )
    assert calls and calls[0][0] == "build_vol_betas"
