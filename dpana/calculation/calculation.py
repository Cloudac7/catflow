import json
import os
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from glob import glob
from dpdispatcher.submission import Submission, Task, Resources
from dpdispatcher.machine import Machine


class CalculationTask(object):
    """
    Calculation tasks.
    """
    def __init__(
            self,
            machine: Machine,
            resources: Resources,
            work_base: str,
            task_list: list,
            forward_common_files=None,
            backward_common_files=None,
            forward_files=None,
            backward_files=None,
    ):
        """Generate calculation tasks.

        Parameters
        ----------
        machine : Machine object. Set up the config of machine to run the task on.
        resources: Resources object. Set up the config of machine resources.
        work_base:
        task_list:
        forward_common_files:
        backward_common_files:
        forward_files:
        backward_files : object

        """
        self.machine = machine
        self.resources = resources
        self.work_base = work_base
        if forward_common_files is None:
            forward_common_files = []
        if backward_common_files is None:
            backward_common_files = []
        if forward_files is None:
            forward_files = []
        if backward_files is None:
            backward_files = []
        self.submission = Submission(
            work_base=work_base,
            machine=machine,
            resources=resources,
            forward_common_files=list(forward_common_files),
            backward_common_files=list(backward_common_files),
        )
        self.task_list = task_list
        self.forward_files = forward_files
        self.backward_files = backward_files

    def run(self, clean=True):
        tasks = self._generate_tasks(
            self.task_list,
            self.resources.kwargs['command'],
            forward_files=self.forward_files,
            backward_files=self.backward_files
        )
        self.submission.register_task_list(tasks)
        self.submission.run_submission(clean=clean)

    def _generate_tasks(self, command, task_list, **kwargs):
        _real_task_list = []
        for item in task_list:
            os.path.abspath(os.path.join(self.work_base, item))
            _add = glob(item)
            _add.sort()
            _real_task_list += _add
        tasks = [Task(
            command=command,
            task_work_path=i,
            forward_files=list(kwargs.get('forward_files', None)),
            backward_files=list(kwargs.get('backward_files', None))
        ) for i in _real_task_list]
        return tasks


def from_json(path, **kwargs):
    with open(path, 'r') as f:
        dict_data = json.load(f)
    machine = Machine.load_from_dict(dict_data['machine'])
    resources = Resources.load_from_dict(dict_data['resources'])
    calculation = CalculationTask(machine, resources, **kwargs)
    return calculation


def from_yaml(path, **kwargs):
    with open(path, 'r') as f:
        dict_data = load(f, Loader=Loader)
    machine = Machine.load_from_dict(dict_data['machine'])
    resources = Resources.load_from_dict(dict_data['resources'])
    calculation = CalculationTask(machine, resources, **kwargs)
    return calculation
