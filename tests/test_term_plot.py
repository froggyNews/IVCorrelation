import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from display.plotting.term_plot import plot_atm_term_structure


def test_plot_atm_term_structure_handles_string_vols():
    atm_curve = pd.DataFrame({
        'T': [0.5, 1.0, 1.5],
        'atm_vol': ['0.2', '0.3', '0.4'],
    })
    fig, ax = plt.subplots()
    plot_atm_term_structure(ax, atm_curve)
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 1], [0.2, 0.3, 0.4])
    plt.close(fig)
