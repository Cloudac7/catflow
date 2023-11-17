from pathlib import Path

import pandas as pd
import pytest
from catalyner.tesla.dpgen.exploration import DPExplorationAnalyzer
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@pytest.fixture
def analyzer(shared_datadir):
    ana = DPExplorationAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    return ana


def test_make_set(analyzer):
    results = analyzer.make_set(iteration=0)
    assert results[0]['iteration'] == 'iter.000000'
    assert results[0]['temps'] == 100


def test_make_set_dataframe(analyzer):
    result_df = analyzer.make_set_dataframe(iteration=0)
    assert type(result_df) == pd.DataFrame


def test_make_set_pickle(analyzer, shared_datadir):
    analyzer.make_set_pickle(0)
    assert (shared_datadir /
            'dpgen_run/model_devi_each_iter/data_000000.pkl').is_file()


def test_data_prepareation(analyzer):
    import math

    steps, mdf = analyzer._data_prepareation(plot_item=100, iteration=0)
    assert len(steps) == 31
    assert len(mdf) == 31
    assert math.isclose(mdf.to_list()[0], 2.015300e-03)


@image_comparison(baseline_images=['single_iteration'], remove_text=True,
                  extensions=['png'], style='mpl20')
def test_plot_single_iteration(analyzer):
    fig = analyzer.plot_single_iteration(
        group_by='temps', ylimit=1.0, temps=100, label_unit='K')
    fig.show()


@image_comparison(baseline_images=['multiple_iteration'], remove_text=True,
                  extensions=['png'], style='mpl20')
def test_plot_multiple_iteration(analyzer):
    fig = analyzer.plot_multiple_iterations(
        group_by='temps', iterations=[0], ylimit=1.0, temps=100, label_unit='K'
    )
    fig.show()


@image_comparison(baseline_images=['multi_iter_distribution'], remove_text=True, 
                  extensions=['png'], style='mpl20')
def test_plot_multi_iter_distribution(analyzer):
    fig = analyzer.plot_multi_iter_distribution(
        group_by='temps', iterations=[0], ylimit=1.0, temps=100, label_unit='K'
    )
    fig.show()

@image_comparison(baseline_images=['multiple_ratio_bars'], remove_text=True, 
                  extensions=['png'], style='mpl20')
def test_plot_ensemble_bar(analyzer):
    fig = analyzer.plot_ensemble_ratio_bar(
        iterations=[0], label_unit='K'
    )
    fig.show()
