import pytest

import os
import math
import numpy as np
from pathlib import Path

from catflow.utils.config import load_yaml_configs
from catflow.utils.cp2k import Cp2kInputToDict
from catflow.tasker.tasks.pmf import PMFTask, DPPMFTask
from dpdispatcher.submission import Submission, Task, Resources
from dpdispatcher.machine import Machine


@pytest.fixture
def pmf_datadir(shared_datadir):
    return shared_datadir / "pmf"


@pytest.fixture
def pmf_task(pmf_datadir, tmp_path):
    config = load_yaml_configs(pmf_datadir / "workflow_settings.yml")
    config["work_path"] = str(tmp_path)
    config["init_structure_path"] = str(pmf_datadir / "test_init.xyz")
    return PMFTask(**config)

class MockJobFactory(object):
    def __init__(self, task_list, submission_dict, machine_name, resource_dict, **kwargs):
        work_base = submission_dict.get('work_base', os.getcwd())
        if 'work_base' in submission_dict:
            submission_dict.pop('work_base', None)
        if isinstance(task_list, list):
            task_list = [Task(**task_dict) for task_dict in task_list]
        elif isinstance(task_list, Task):
            task_list = task_list
        elif isinstance(task_list, dict):
            task_list = Task(**task_list)
        else:
            raise ValueError("Task list must be a Task, task_dict or list of task_dict")

        machine_dict = {
            "batch_type": "Slurm",
            "context_type": "LocalContext",
            "local_root": ".",
            "remote_root": "."
        }
        machine = Machine.load_from_dict(machine_dict)
        resources = Resources.load_from_dict(resource_dict)

        self.submission = Submission(
            work_base=work_base,
            machine=machine,
            resources=resources,
            task_list=task_list,
            **submission_dict
        )

@pytest.fixture(autouse=True, scope="class")
def randint_mock(class_mocker):
    return class_mocker.patch("random.randint", lambda x, y: 5)


def test_pmf_task_generation(mocker, tmp_path, pmf_task: PMFTask):
    from dpdata.unit import LengthConversion

    mocker.patch(
        "catflow.tasker.tasks.pmf.JobFactory",
        MockJobFactory
    )

    submission = pmf_task.generate_submission()
    task_path = tmp_path / "task.1.4_700.0"
    assert task_path.exists()
    assert (task_path / "init.xyz").exists()
    new_input_dict = Cp2kInputToDict(task_path / "input.inp").get_tree()
    new_coord = float(
        new_input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"]["TARGET"]
    )
    conv = LengthConversion("angstrom", "bohr")
    assert math.isclose(
        new_coord, pmf_task.coordinate * conv.value(), rel_tol=1e-5
    )

    assert math.isclose(
        float(new_input_dict["MOTION"]["MD"]["TEMPERATURE"]), 
        pmf_task.temperature, 
        rel_tol=1e-5
    )

def task_pmf_restart():
    pass
