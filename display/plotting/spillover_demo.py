import numpy as np
import matplotlib.pyplot as plt

from display.plotting.anim_utils import (
    animate_spillover,
    add_checkboxes,
    add_legend_toggles,
    set_scatter_group_visibility,
)


def main():
    times = [f"t{i}" for i in range(20)]
    peers_xy = {
        "AAPL": (0, 0),
        "MSFT": (1, 0.5),
        "GOOGL": (0.5, 1.0),
        "XOM": (-0.5, -0.2),
        "CVX": (-1.0, 0.3),
    }
    responses = {k: np.random.randn(len(times)) for k in peers_xy}

    fig, ani, series_map, state = animate_spillover(times, peers_xy, responses)

    add_checkboxes(fig, series_map)
    ax = fig.axes[0]
    add_legend_toggles(ax, series_map)

    labels = state["labels"]
    groups = {
        "Tech": np.isin(labels, ["AAPL", "MSFT", "GOOGL"]),
        "Energy": np.isin(labels, ["XOM", "CVX"]),
    }
    visible = {name: True for name in groups}

    def on_key(event):
        if event.key == "t":
            visible["Tech"] = not visible["Tech"]
            set_scatter_group_visibility(series_map["Peers"][0], groups, "Tech", visible["Tech"])
            fig.canvas.draw_idle()
        elif event.key == "e":
            visible["Energy"] = not visible["Energy"]
            set_scatter_group_visibility(series_map["Peers"][0], groups, "Energy", visible["Energy"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
