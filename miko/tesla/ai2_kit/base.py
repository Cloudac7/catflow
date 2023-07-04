import json
import cloudpickle
from abc import ABC
from pathlib import Path
from typing import Union, List, Dict, Optional

from ai2_kit.core.util import load_yaml_files
from ai2_kit.core.checkpoint import set_checkpoint_file
from ai2_kit.workflow.cll_mlp import CllWorkflowConfig


class CllTask(object):
    """CllTask is a class reading a ai2-kit directory, where the Cll-Workflow run.
    """

    def __init__(
        self,
        *config_files,
        path: str,
        deepmd_version: str = '2.0',
        **kwargs
    ):
        """Generate a class of tesla task.

        Args:
            path (str): The path of the tesla task.
            deepmd_version (str): DeepMD-kit version used. Default: 2.0.
        """

        config_data = load_yaml_files(*config_files)
        self.config = CllWorkflowConfig.parse_obj(config_data)
        self.path = Path(path).resolve()
        self.deepmd_version = deepmd_version        

    @classmethod
    def from_dict(cls, dp_task_dict: dict):
        return cls(**dp_task_dict)


class CllAnalyzer(ABC):
    """Base class to be implemented as analyzer for `DPTask`
    """
    def __init__(
        self, 
        dp_task: CllTask,
        iteration: int = 0
    ) -> None:
        self.dp_task = dp_task
        self.iteration = iteration

    def _iteration_dir(self, iteration: Optional[int] = None):
        if iteration is None:
            iteration = self.iteration
        return 'iters-' + str(iteration).zfill(3)

    @classmethod
    def setup_task(cls, **kwargs):
        task = CllTask(**kwargs)
        return cls(task, kwargs.get('iteration', 0))
