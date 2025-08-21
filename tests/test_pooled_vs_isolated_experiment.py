import pandas as pd

from analysis.experiments.pooled_vs_isolated import run_experiment


def test_experiment_uses_model_and_eval(monkeypatch):
    calls = {"build": False, "combine": False, "corr": False}

    def fake_build_surface_grids(tickers, **kwargs):
        calls["build"] = True
        return {
            t: {"2024-01-01": pd.DataFrame({"mny": [1.0], "iv": [0.2]})}
            for t in tickers
        }

    def fake_combine_surfaces(surfaces, weights):
        calls["combine"] = True
        return {"2024-01-01": pd.DataFrame({"mny": [1.0], "iv": [0.2]})}

    def fake_compute_atm_corr(get_smile_slice, tickers, asof, pillars_days, **kwargs):
        calls["corr"] = True
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        "analysis.experiments.pooled_vs_isolated.build_surface_grids",
        fake_build_surface_grids,
    )
    monkeypatch.setattr(
        "analysis.experiments.pooled_vs_isolated.combine_surfaces",
        fake_combine_surfaces,
    )
    monkeypatch.setattr(
        "analysis.experiments.pooled_vs_isolated.compute_atm_corr",
        fake_compute_atm_corr,
    )

    run_experiment(tickers=["A", "B"], weights={"A": 0.5, "B": 0.5}, asof="2024-01-01")

    assert all(calls.values())
