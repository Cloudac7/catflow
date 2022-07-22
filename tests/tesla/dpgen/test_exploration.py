from pathlib import Path

import pandas as pd
import pytest
from miko.tesla.dpgen.exploration import DPExplorationAnalyzer

@pytest.fixture
def analyzer(shared_datadir):
    ana = DPExplorationAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    return ana

def test_make_set(analyzer, shared_datadir):
    results = analyzer.make_set()
    assert results[0]['iteration'] == 'iter.000000'
    assert results[0]['temps'] == 100

def test_make_set_dataframe(analyzer, shared_datadir):
    result_df = analyzer.make_set_dataframe()
    assert type(result_df) == pd.DataFrame

def test_make_set_pickle(analyzer, shared_datadir):
    analyzer.make_set_pickle()
    assert (shared_datadir / 'dpgen_run/model_devi_each_iter/data_000000.pkl').is_file()

def test_data_prepareation(analyzer, shared_datadir):
    steps, mdf = analyzer._data_prepareation(plot_item=100)
    assert len(steps) == 31
    assert len(mdf) == 31
    assert mdf[0] == 2.015300e-03

def test_plot_single_iteration(analyzer):
    #TODO: separate each part into real units.
    fig = analyzer.plot_single_iteration(group_by='temps', temps=100, label_unit='K')
