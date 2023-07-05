import json
from abc import ABC
from pathlib import Path


class BaseTask(object):
    """BaseTask is a base class reading a directory, where task run.
    """

    def __init__(self, path: str):
        """Generate a class of tesla task.

        Args:
            path (str): The path of the tesla task.
        """
        self.path = Path(path).resolve()

    @classmethod
    def from_dict(cls, dp_task_dict: dict):
        return cls(**dp_task_dict)


class BaseAnalyzer(ABC):
    """Base class to be implemented as analyzer for `DPTask`
    """

    def __init__(self, dp_task: BaseTask) -> None:
        self.dp_task = dp_task

    def _iteration_dir(self, iteration: int, **kwargs) -> str:
        return ""

    @classmethod
    def setup_task(cls, **kwargs):
        task = BaseTask(**kwargs)
        return cls(task)
