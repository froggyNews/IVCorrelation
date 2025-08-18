import pandas as pd
import matplotlib
matplotlib.use('Agg')

from display.gui.gui_plot_manager import PlotManager


def test_weights_recomputed_on_weight_mode_change(monkeypatch):
    pm = PlotManager()
    pm.last_corr_df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["PEER", "TARGET"],
        columns=["PEER", "TARGET"],
    )
    pm.last_corr_meta = {"weight_mode": "iv_atm"}

    calls = {"corr": 0, "compute": 0}

    def fake_corr_weights(df, target, peers, clip_negative=True, power=1.0):
        calls["corr"] += 1
        return pd.Series({peers[0]: 1.0})

    def fake_compute_peer_weights(target, peers, weight_mode, asof=None, pillar_days=None, tenor_days=None, mny_bins=None):
        calls["compute"] += 1
        return pd.Series({peers[0]: 1.0})

    monkeypatch.setattr(
        "display.gui.gui_plot_manager.corr_weights", fake_corr_weights
    )
    monkeypatch.setattr(
        "analysis.analysis_pipeline.compute_peer_weights", fake_compute_peer_weights
    )

    pm._weights_from_ui_or_matrix("TARGET", ["PEER"], "iv_atm")
    assert calls["corr"] == 1 and calls["compute"] == 0

    pm._weights_from_ui_or_matrix("TARGET", ["PEER"], "ul")
    assert calls["corr"] == 1
    assert calls["compute"] == 1
