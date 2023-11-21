import os
from typing import List, Dict
from catflow.tasker.resources.config import settings
from dpdispatcher.submission import Submission, Task, Resources
from dpdispatcher.machine import Machine


class JobFactory(object):
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

        machine_dict = settings['POOL'][machine_name]
        machine = Machine.load_from_dict(machine_dict['machine'])
        resources = Resources.load_from_dict(resource_dict)

        self.submission = Submission(
            work_base=work_base,
            machine=machine,
            resources=resources,
            task_list=task_list,
            **submission_dict
        )


class Queues(object):
    def __init__(self, machine_name):
        machine_dict = settings['POOL'][machine_name]
        queues_dict = machine_dict['queues']
        self.queues_list = queues_dict.keys()
