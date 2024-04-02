import json
import cloudpickle
from abc import ABC
from pathlib import Path
from typing import Union, List, Dict, Optional

from ai2_kit.core.util import load_yaml_files
from ai2_kit.core.checkpoint import set_checkpoint_file
from ai2_kit.workflow.cll_mlp import CllWorkflowConfig

from catflow.analyzer.tesla.base.task import BaseTask, BaseAnalyzer


class CllTask(BaseTask):
    """CllTask is a class reading a ai2-kit directory, where the Cll-Workflow run.
    """

    def __init__(
        self,
        *config_files,
        path: str
    ):
        """Generate a class of tesla task.

        Args:
            path (str): The path of the tesla task.
            deepmd_version (str): DeepMD-kit version used. Default: 2.0.
        """
        super().__init__(path)
        config_data = load_yaml_files(*config_files)
        self.config = CllWorkflowConfig.model_validate(config_data)   

    @classmethod
    def from_dict(cls, dp_task_dict: dict):
        return cls(**dp_task_dict)


class CllAnalyzer(BaseAnalyzer):
    """Base class to be implemented as analyzer for `DPTask`
    """
    def __init__(
        self, 
        dp_task: CllTask,
        iteration: int = 0,
        **kwargs
    ) -> None:
        self.dp_task = dp_task
        self.iteration = iteration
        if type(self.dp_task) is not CllTask:
            self.dp_task = CllTask.from_dict(**self.dp_task.__dict__, **kwargs)

    def _iteration_dir(self, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration is None:
            iteration = self.iteration
        return 'iters-' + str(iteration).zfill(3)

    @classmethod
    def setup_task(cls, **kwargs):
        task = CllTask(**kwargs)
        return cls(task, kwargs.get('iteration', 0))
