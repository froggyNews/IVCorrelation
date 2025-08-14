import os
import sys
import logging
import pandas as pd
import pytest

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.analysis_pipeline import compute_peer_weights


def test_compute_peer_weights_route_logging(monkeypatch, caplog):
    """All weight modes should dispatch to the correct helper with provided config."""
    # Avoid DB access for date lookup
    monkeypatch.setattr(
        "analysis.analysis_pipeline.available_dates",
        lambda ticker=None, most_recent_only=False: ["2024-01-01"],
    )

    calls = []

    # Stub helpers to record which path was taken
    def fake_pca_weights(**kwargs):
        calls.append(("pca_weights", kwargs))
        return pd.Series({"PEER": 1.0})

    def fake_cosine_similarity_weights(**kwargs):
        calls.append(("cosine_similarity_weights", kwargs))
        return pd.Series({"PEER": 1.0})

    def fake_build_vol_betas(**kwargs):
        calls.append(("build_vol_betas", kwargs))
        return pd.DataFrame({"PEER": [1.0]})

    def fake_peer_weights_from_correlations(**kwargs):
        calls.append(("peer_weights_from_correlations", kwargs))
        return pd.Series({"PEER": 1.0})

    monkeypatch.setattr("analysis.analysis_pipeline.pca_weights", fake_pca_weights)
    monkeypatch.setattr(
        "analysis.analysis_pipeline.cosine_similarity_weights",
        fake_cosine_similarity_weights,
    )
    monkeypatch.setattr("analysis.analysis_pipeline.build_vol_betas", fake_build_vol_betas)
    monkeypatch.setattr(
        "analysis.analysis_pipeline.peer_weights_from_correlations",
        fake_peer_weights_from_correlations,
    )

    pillar_opts = [(7,), (7, 30)]
    tenor_opts = [(30,), (30, 60)]
    mny_opts = [((0.8, 0.9),), ((0.9, 1.1),)]
    mode_map = {
        "pca_atm_market": "pca_weights",
        "cosine_surface": "cosine_similarity_weights",
        "surface_grid": "build_vol_betas",
        "iv_atm": "peer_weights_from_correlations",
    }

    caplog.set_level(logging.INFO, logger="route_logger")
    logger = logging.getLogger("route_logger")
    expected_logs = 0

    for pillars in pillar_opts:
        for tenors in tenor_opts:
            for mny in mny_opts:
                for mode, expected in mode_map.items():
                    calls.clear()
                    res = compute_peer_weights(
                        target="SPY",
                        peers=["QQQ"],
                        weight_mode=mode,
                        pillar_days=pillars,
                        tenor_days=tenors,
                        mny_bins=mny,
                    )
                    assert calls, "no helper was called"
                    name, kwargs = calls[0]
                    assert name == expected
                    if mode != "surface_grid":
                        assert (
                            kwargs.get("pillar_days") == pillars
                            or kwargs.get("pillars_days") == pillars
                        )
                    # tenors parameter name differs across helpers
                    assert (
                        kwargs.get("tenor_days") == tenors
                        or kwargs.get("tenors") == tenors
                    )
                    assert kwargs.get("mny_bins") == mny
                    logger.info(
                        "mode=%s pillars=%s tenors=%s mny=%s -> %s",
                        mode,
                        pillars,
                        tenors,
                        mny,
                        name,
                    )
                    expected_logs += 1

    # Ensure we logged each combination
    route_logs = [r for r in caplog.records if r.name == "route_logger"]
    assert len(route_logs) == expected_logs
