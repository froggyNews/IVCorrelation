import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from display.plotting.smile_plot import plot_synthetic_etf_smile
from display.plotting.term_plot import plot_synthetic_etf_term_structure

def test_plot_synthetic_etf_smile_runs():
    surfaces = {
        'A': np.array([0.2, 0.21, 0.22]),
        'B': np.array([0.25, 0.24, 0.23]),
    }
    weights = {'A': 0.5, 'B': 0.5}
    grid = np.array([0.9, 1.0, 1.1])

    fig, ax = plt.subplots()
    bands = plot_synthetic_etf_smile(ax, surfaces, weights, grid, n_boot=5)
    assert bands.mean.shape == grid.shape
    plt.close(fig)

def test_plot_synthetic_etf_term_structure_runs():
    atm_data = {
        'A': np.array([0.2, 0.21, 0.22]),
        'B': np.array([0.25, 0.24, 0.23]),
    }
    weights = {'A': 0.5, 'B': 0.5}
    pillar_days = np.array([30, 60, 90])

    fig, ax = plt.subplots()
    bands = plot_synthetic_etf_term_structure(ax, atm_data, weights, pillar_days, n_boot=5)
    assert bands.mean.shape == pillar_days.shape
    plt.close(fig)
