from miko.resources.config import settings
from dpdispatcher.submission import Submission, Task, Resources
from dpdispatcher.machine import Machine


class JobFactory(object):
    def __init__(self, task_dict_list, submission_dict, machine_name, resource_name, **kwargs):

        task_list = [Task(**task_dict) for task_dict in task_dict_list]

        machine_dict = settings[machine_name]
        machine = Machine.load_from_dict(**machine_dict)

        resources = Resources.load_from_dict(**machine_dict[resource_name])
        group_size = kwargs.get('group_size', 1)
        resources.group_size = group_size

        self.submission = Submission(
            machine=machine,
            resources=resources,
            task_list=task_list,
            **submission_dict
        )

    def run_submission(self):
        self.submission.run_submission()
