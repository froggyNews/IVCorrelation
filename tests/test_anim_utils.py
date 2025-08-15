import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from display.plotting.anim_utils import (
    animate_smile_over_time,
    animate_surface_timesweep,
    animate_spillover,
    add_checkboxes,
    add_keyboard_toggles,
    add_legend_toggles,
    apply_profile_visibility,
    set_scatter_group_visibility,
)


T, N_k = 5, 7
k = np.linspace(0.8, 1.2, N_k)
dates = [f"d{i}" for i in range(T)]


def test_animate_smile_and_toggles():
    iv_syn = 0.2 + 0.05 * (k - 1) ** 2 + 0.01 * np.arange(T)[:, None]
    iv_raw = iv_syn + 0.01
    ci_lo = iv_syn - 0.005
    ci_hi = iv_syn + 0.005

    fig, ani, series = animate_smile_over_time(k, iv_syn, dates, iv_raw_tk=iv_raw, ci_lo_tk=ci_lo, ci_hi_tk=ci_hi)
    assert "Synthetic" in series and "Raw" in series and "CI" in series

    # Update once to build polygon
    ani._func(1)
    band = series["CI"][0]
    assert band.get_paths()[0].vertices.shape[0] == 2 * N_k

    apply_profile_visibility(series, {"Raw": False})
    assert series["Raw"][0].get_visible() is False

    check = add_checkboxes(fig, series)
    check.set_active(list(series.keys()).index("Raw"))
    assert series["Raw"][0].get_visible() is True

    add_keyboard_toggles(fig, series, {"r": "Raw"})
    event = type(
        "E",
        (),
        {"key": "r", "name": "key_press_event", "canvas": fig.canvas, "inaxes": None},
    )()
    fig.canvas.callbacks.process("key_press_event", event)
    assert series["Raw"][0].get_visible() is False

    ax = fig.axes[0]
    leg = add_legend_toggles(ax, series)
    handle = leg.legend_handles[0] if hasattr(leg, "legend_handles") else leg.legendHandles[0]
    event = type("E", (), {"artist": handle, "canvas": fig.canvas, "name": "pick_event"})()
    fig.canvas.callbacks.process("pick_event", event)
    assert series["Synthetic"][0].get_visible() is False
    plt.close(fig)


def test_surface_and_spillover():
    tau = np.linspace(0.1, 1.0, 3)
    iv_tktau = np.random.rand(T, N_k, tau.size)
    fig1, ani1, series1 = animate_surface_timesweep(k, tau, iv_tktau, dates)
    assert "Surface" in series1
    apply_profile_visibility(series1, {"Surface": False})
    assert series1["Surface"][0].get_visible() is False
    plt.close(fig1)

    peers_xy = {"A": (0, 0), "B": (1, 1)}
    responses = {p: np.random.rand(T) for p in peers_xy}
    fig2, ani2, series2, state = animate_spillover(dates, peers_xy, responses)
    assert "Peers" in series2
    groups = {"G": np.array([True, False])}
    scat = series2["Peers"][0]
    set_scatter_group_visibility(scat, groups, "G", False)
    assert scat.get_sizes()[0] < 1e-5
    set_scatter_group_visibility(scat, groups, "G", True)
    assert scat.get_sizes()[0] > 1e-5
    plt.close(fig2)


def test_legend_toggle_single_connection():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], label="Series")
    series = {"Series": [line]}

    # Call twice to ensure previous handler is replaced
    leg1 = add_legend_toggles(ax, series)
    add_legend_toggles(ax, series)

    handle = leg1.legend_handles[0] if hasattr(leg1, "legend_handles") else leg1.legendHandles[0]
    event = type("E", (), {"artist": handle, "canvas": fig.canvas, "name": "pick_event"})()
    fig.canvas.callbacks.process("pick_event", event)

    # If multiple callbacks were registered, visibility would toggle twice and stay True
    assert series["Series"][0].get_visible() is False
    plt.close(fig)
