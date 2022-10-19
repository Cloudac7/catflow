import os
from miko_tasker.resources.config import settings
from dpdispatcher.submission import Submission, Task, Resources
from dpdispatcher.machine import Machine


class JobFactory(object):
    def __init__(self, task_dict_list, submission_dict, machine_name, resource_dict, **kwargs):

        task_list = [Task(**task_dict) for task_dict in task_dict_list]

        machine_dict = settings['POOL'][machine_name]
        machine = Machine.load_from_dict(machine_dict['machine'])
        resources = Resources.load_from_dict(resource_dict)

        self.submission = Submission(
            machine=machine,
            resources=resources,
            task_list=task_list,
            **submission_dict
        )

    def run_submission(self):
        _origin = os.getcwd()
        self.submission.run_submission()
        os.chdir(_origin)


class Queues(object):
    def __init__(self, machine_name):
        machine_dict = settings['POOL'][machine_name]
        queues_dict = machine_dict['queues']
        self.queues_list = queues_dict.keys()
