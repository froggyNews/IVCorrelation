import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from analysis.analysis_synthetic_etf import SyntheticETFArtifacts
from display.plotting.display_viewers_synthetic_etf_viewer import (
    show_synthetic_etf,
    _extract_latest,
)

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

def test_plot_synthetic_etf_smile_adds_to_legend():
    surfaces = {
        'A': np.array([0.2, 0.21, 0.22]),
    }
    weights = {'A': 1.0}
    grid = np.array([0.9, 1.0, 1.1])

    fig, ax = plt.subplots()
    ax.plot(grid, surfaces['A'], label='Target')
    ax.legend()

    plot_synthetic_etf_smile(ax, surfaces, weights, grid, n_boot=5)
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert 'Synthetic ETF' in labels
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


def test_plot_synthetic_etf_term_structure_adds_to_legend():
    atm_data = {
        'A': np.array([0.2, 0.21, 0.22]),
    }
    weights = {'A': 1.0}
    pillar_days = np.array([30, 60, 90])

    fig, ax = plt.subplots()
    ax.plot(pillar_days, atm_data['A'], label='Target')
    ax.legend()

    plot_synthetic_etf_term_structure(ax, atm_data, weights, pillar_days, n_boot=5)
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert 'Synthetic ATM' in labels
    plt.close(fig)


def test_show_synthetic_etf_handles_disjoint_dates(monkeypatch):
    df_tgt = pd.DataFrame(
        [[0.2, 0.21], [0.22, 0.23]],
        index=["0.9", "1.1"],
        columns=["30", "60"],
    )
    df_syn = pd.DataFrame(
        [[0.25, 0.24], [0.26, 0.23]],
        index=["0.9", "1.1"],
        columns=["30", "60"],
    )
    artifacts = SyntheticETFArtifacts(
        weights=pd.Series(dtype=float),
        surfaces={"A": {"2024-01-01": df_tgt}},
        synthetic_surfaces={"2024-01-02": df_syn},
        rv_metrics=pd.DataFrame(),
        meta={"target": "A"},
    )
    tgt_df, syn_df, d_tgt, d_syn = _extract_latest(artifacts, "A")
    assert d_tgt == "2024-01-01"
    assert d_syn == "2024-01-02"
    assert tgt_df is not None and syn_df is not None
    monkeypatch.setattr(plt, "show", lambda: None)
    show_synthetic_etf(artifacts, target="A")
