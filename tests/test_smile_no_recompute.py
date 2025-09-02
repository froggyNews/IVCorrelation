import json
import numpy as np
import matplotlib.pyplot as plt
from display.gui.gui_plot_manager import PlotManager


def test_smile_no_recompute_on_model_switch(monkeypatch):
    """Changing only the model should reuse cached smile data."""
    pm = PlotManager()
    fig, ax = plt.subplots()

    calls = {"n": 0}
    cache: dict[str, object] = {}

    def fake_compute_or_load(name, payload, builder):
        key = json.dumps(payload, sort_keys=True)
        if key not in cache:
            cache[key] = builder()
            calls["n"] += 1
        return cache[key]

    monkeypatch.setattr("display.gui.gui_plot_manager.compute_or_load", fake_compute_or_load)

    def fake_prepare_smile_data(**kwargs):
        return {
            "T_arr": np.array([0.1]),
            "K_arr": np.array([1.0]),
            "sigma_arr": np.array([0.2]),
            "S_arr": np.array([1.0]),
            "Ts": np.array([0.1]),
            "idx0": 0,
            "tgt_surface": None,
            "syn_surface": None,
            "composite_slices": {},
            "expiry_arr": np.array([0.1]),
            "fit_by_expiry": {0.1: {"svi": {"a": 1}, "sabr": {"alpha": 1}, "tps": {}, "sens": {}}},
        }

    monkeypatch.setattr(
        "display.gui.gui_plot_manager.prepare_smile_data", fake_prepare_smile_data
    )

    monkeypatch.setattr(PlotManager, "_render_smile_at_index", lambda self: None)

    settings = {
        "plot_type": "Smile",
        "target": "T",
        "asof": "2024-01-01",
        "model": "svi",
        "T_days": 30,
        "ci": 68.0,
        "x_units": "mny",
        "atm_band": 0.05,
        "weight_method": "corr",
        "feature_mode": "iv_atm",
        "overlay_composite": False,
        "overlay_peers": False,
        "peers": [],
        "pillars": [],
        "max_expiries": 6,
    }

    pm.plot(ax, settings)
    settings["model"] = "sabr"
    pm.plot(ax, settings)

    assert calls["n"] == 1
