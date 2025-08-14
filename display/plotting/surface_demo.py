import numpy as np
import matplotlib.pyplot as plt

from src.viz.anim_utils import (
    animate_surface_timesweep,
    add_checkboxes,
    add_keyboard_toggles,
    add_legend_toggles,
)


def main():
    T, N_k, N_tau = 30, 25, 20
    k = np.linspace(-0.3, 0.3, N_k)
    tau = np.linspace(0.1, 2.0, N_tau)
    base = 0.2 + 0.05 * k[:, None] ** 2 + 0.02 * np.sqrt(tau)[None, :]
    iv_tktau = base[None, :, :] + 0.01 * np.sin(np.linspace(0, 3 * np.pi, T))[:, None, None]
    dates = [f"t{i}" for i in range(T)]

    fig, ani, series_map = animate_surface_timesweep(k, tau, iv_tktau, dates)

    add_checkboxes(fig, series_map)
    add_keyboard_toggles(fig, series_map, keymap={"u": "Surface"})
    ax = fig.axes[0]
    add_legend_toggles(ax, series_map)

    plt.show()


if __name__ == "__main__":
    main()
