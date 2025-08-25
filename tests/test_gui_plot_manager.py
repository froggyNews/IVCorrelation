import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from display.gui.gui_plot_manager import PlotManager
import display.gui.gui_plot_manager as gpm


def _base_settings(plot_type):
    return {
        "plot_type": plot_type,
        "target": "AAPL",
        "asof": "2020-01-01",
        "model": "svi",
        "T_days": 30,
        "ci": 0,
        "x_units": "years",
        "atm_band": 0.05,
        "weight_method": "corr",
        "feature_mode": "iv_atm",
        "peers": [],
        "pillars": [],
        "overlay_synth": False,
        "overlay_peers": False,
        "max_expiries": 6,
    }


def test_plot_smile_no_data_sets_title(monkeypatch):
    pm = PlotManager()
    fig, ax = plt.subplots()

    monkeypatch.setattr(gpm, "compute_or_load", lambda *a, **k: {})

    settings = _base_settings("Smile (K/S vs IV)")
    pm.plot(ax, settings)

    assert ax.get_title() == "No data"


def test_plot_smile_with_data_calls_fit(monkeypatch):
    pm = PlotManager()
    fig, ax = plt.subplots()

    dataset = {
        "T_arr": np.array([0.1]),
        "K_arr": np.array([100.0]),
        "sigma_arr": np.array([0.2]),
        "S_arr": np.array([100.0]),
        "Ts": np.array([0.1]),
        "idx0": 0,
        "fit_by_expiry": {0.1: {"svi": np.array([1, 2, 3, 4, 5])}},
        "tgt_surface": None,
        "syn_surface": None,
        "peer_slices": {},
        "expiry_arr": np.array([0.1]),
    }

    called = {}

    def fake_compute_or_load(*a, **k):
        return dataset

    def fake_fit_and_plot_smile(*a, **k):
        called["done"] = True
        return {"params": dataset["fit_by_expiry"][0.1]["svi"], "rmse": 0.0}

    monkeypatch.setattr(gpm, "compute_or_load", fake_compute_or_load)
    monkeypatch.setattr(gpm, "fit_and_plot_smile", fake_fit_and_plot_smile)

    settings = _base_settings("Smile (K/S vs IV)")
    pm.plot(ax, settings)

    assert called.get("done") is True
    assert pm.last_fit_info["ticker"] == "AAPL"


def test_plot_term_no_data_sets_title(monkeypatch):
    pm = PlotManager()
    fig, ax = plt.subplots()

    def fake_compute_or_load(*a, **k):
        return {"atm_curve": pd.DataFrame()}

    monkeypatch.setattr(gpm, "compute_or_load", fake_compute_or_load)

    settings = _base_settings("Term (ATM vs T)")
    pm.plot(ax, settings)

    assert ax.get_title() == "No data"
