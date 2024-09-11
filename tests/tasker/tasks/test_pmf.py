import pytest

import os
import math
import numpy as np
from pathlib import Path

from ai2_kit.core.executor import HpcExecutor

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
    config["command"] = "sleep 1"
    config["work_path"] = str(tmp_path)
    config["init_structure_path"] = str(pmf_datadir / "test_init.xyz")
    config["executor"] = {
            'queue_system': {
                'slurm': {}  # Specify queue system
            },
            'work_dir': str(tmp_path),  # Specify working directory
            'python_cmd': 'python',  # Specify python command
        }
    return PMFTask(**config)


@pytest.fixture(autouse=True, scope="class")
def randint_mock(class_mocker):
    return class_mocker.patch("random.randint", lambda x, y: 5)

def process_input(self, base_dir: Path, inputs):
    from catflow.utils.cp2k import Cp2kInput

    restart_count = inputs.get('restart_count', 0)
    task_dir = base_dir / \
        f"task-{restart_count}-{inputs['coordinate']}-{inputs['temperature']}"
    task_dir.mkdir(exist_ok=True)

    if restart_count == 0:
        # create new task
        # write init.xyz
        structure = inputs["structure"]
        structure.write(task_dir / 'init.xyz')

        # write input.inp
        with open(task_dir / 'input.inp', 'w') as f:
            output = Cp2kInput(params=inputs['input_dict']).render()
            f.write(output)


@pytest.mark.asyncio
async def test_pmf_task_generation(mocker, tmp_path, pmf_task: PMFTask):
    from dpdata.unit import LengthConversion
    mocker.patch("catflow.tasker.resources.submit.HpcExecutor.mkdir", return_value=None)
    mocker.patch(
        "catflow.tasker.resources.submit.HpcExecutor.run_python_fn", 
        process_input
    )

    submission = await pmf_task.task_generate()
    task_path = tmp_path / "pmf/task-0-1.4-700.0"
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
