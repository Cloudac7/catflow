from miko.tesla.dpgen.task import DPTask


def _inner_test_dp_task(t: DPTask):
    assert t.step_code == 4
    assert t.state == 'Parsing'
    assert t.step == 'Exploring'
    assert t.param_data.get("fp_task_max") == 100
    assert t.machine_data.get("model_devi_group_size") == 10


def test_dp_task_from_params(shared_datadir):
    t = DPTask(
        path=shared_datadir / "dpgen_task",
        param_file="param.json",
        machine_file="machine.json",
        record_file="record.dpgen"
    )
    _inner_test_dp_task(t)


def test_dp_task_from_dict(shared_datadir):
    t = DPTask.from_dict({
        "path": shared_datadir / "dpgen_task",
        "param_file": "param.json",
        "machine_file": "machine.json",
        "record_file": "record.dpgen"
    })
    _inner_test_dp_task(t)
