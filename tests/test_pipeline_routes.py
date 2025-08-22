import pandas as pd
import pytest

from analysis.analysis_pipeline import compute_peer_weights


def test_compute_peer_weights_dispatch(monkeypatch):
    called = {}

    def fake_compute_unified_weights(*, target, peers, mode, **kwargs):
        called["target"] = target
        called["peers"] = tuple(peers)
        called["mode"] = mode
        return pd.Series({"PEER": 1.0})

    monkeypatch.setattr(
        "analysis.analysis_pipeline.compute_unified_weights",
        fake_compute_unified_weights,
    )

    res = compute_peer_weights(target="SPY", peers=["QQQ"], weight_mode="cosine_ul")

    assert called == {"target": "SPY", "peers": ("QQQ",), "mode": "cosine_ul"}
    assert res.loc["PEER"] == pytest.approx(1.0)

