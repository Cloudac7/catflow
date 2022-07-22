from pathlib import Path

import pandas as pd
import pytest
from miko.tesla.dpgen.exploration import DPExplorationAnalyzer

@pytest.fixture
def analyzer(shared_datadir):
    ana = DPExplorationAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    return ana


def test_make_set(shared_datadir):
    ana = DPExplorationAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    results = ana.make_set()
    assert results[0]['iteration'] == 'iter.000000'
    assert results[0]['temps'] == 100


def test_make_set_dataframe(shared_datadir):
    ana = DPExplorationAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    result_df = ana.make_set_dataframe()
    assert type(result_df) == pd.DataFrame

def test_make_set_pickle(analyzer, shared_datadir):
    analyzer.make_set_pickle()
    assert (shared_datadir / 'dpgen_run/model_devi_each_iter/data_000000.pkl').is_file()


def test_plot_single_iteration(analyzer):
    #TODO: separate each part into real units.
    fig = analyzer.plot_single_iteration(group_by='temps', temps=100, label_unit='K')
