import json
from abc import ABC
from pathlib import Path

class DPTask(object):
    """DPTask is a class reading a DP-GEN directory, where the DP-GEN task run.
    """

    def __init__(
        self,
        path: str,
        param_file: str = 'param.json',
        machine_file: str = 'machine.json',
        record_file: str = 'record.dpgen',
        deepmd_version: str = '2.0'
    ):
        """Generate a class of tesla task.

        Args:
            path (str): The path of the tesla task.
            param_file (str): The param json file name.
            machine_file (str): The machine json file name.
            record_file (str): The record file name.
            deepmd_version (str): DeepMD-kit version used. Default: 2.0.
        """
        self.path = Path(path).resolve()
        self.param_file = param_file
        self.machine_file = machine_file
        self.record_file = record_file
        self.deepmd_version = deepmd_version
        self._load_task()

    def _load_task(self):
        """set properties for instance.
        """
        self._read_record()
        self._read_param_data()
        self._read_machine_data()
        if self.step_code in [0, 3, 6]:
            self.state = 'Waiting'
        elif self.step_code in [1, 4, 7]:
            self.state = 'Parsing'
        else:
            self.state = 'Stopped'
        if self.step_code < 3:
            self.step = 'Training'
        elif self.step_code < 6:
            self.step = 'Exploring'
        else:
            self.step = 'Labeling'

    def _read_record(self):
        import numpy as np
        _record_path = self.path / self.record_file
        steps = np.loadtxt(_record_path)
        if steps.shape == (2,):
            # support miko-tasker like record
            self.iteration = int(steps[0])
            self.step_code = int(steps[1])
        else:
            self.iteration = int(steps[-1][0])
            self.step_code = int(steps[-1][1])

    def _read_param_data(self):
        _param_path = self.path / self.param_file
        with open(_param_path) as f:
            self.param_data = json.load(f)

    def _read_machine_data(self):
        _param_path = self.path / self.machine_file
        with open(_param_path) as f:
            self.machine_data = json.load(f)

    @classmethod
    def from_dict(cls, dp_task_dict: dict):
        return cls(**dp_task_dict)


class DPAnalyzer(ABC):
    """Base class to be implemented as analyzer for `DPTask`
    """
    def __init__(self, dp_task: DPTask) -> None:
        self.dp_task = dp_task

    def _iteration_control_code(self, control_step, iteration=None):
        if iteration is None:
            if self.dp_task.step_code < control_step:
                iteration = self.dp_task.iteration - 1
            else:
                iteration = self.dp_task.iteration
        return iteration

    def _iteration_dir(self, **kwargs):
        iteration = self._iteration_control_code(**kwargs)
        return 'iter.' + str(iteration).zfill(6)

    @classmethod
    def setup_task(cls, **kwargs):
        task = DPTask(**kwargs)
        return cls(task)
