from miko.tesla.dpgen.training import DPTrainingAnalyzer


def test_load_curve(shared_datadir):
    ana = DPTrainingAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    results = ana.load_lcurve(iteration=0)
    assert len(results) == 3
    assert results['step'][0] == 0
    assert results['energy_train'][1] == 5.82e-04
    assert results['force_train'][2] == 6.13e-02


def test_plot_curve(shared_datadir):
    ana = DPTrainingAnalyzer.setup_task(path=shared_datadir / 'dpgen_run')
    fig = ana.plot_lcurve()
