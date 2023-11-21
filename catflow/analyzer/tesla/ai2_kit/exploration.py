import re
from pathlib import Path
from typing import List

import numpy as np

from catflow.utils.log_factory import logger
from catflow.utils.lammps import lammps_variable_parser
from catflow.analyzer.tesla.base.exploration import ExplorationAnalyzer
from catflow.analyzer.tesla.ai2_kit.task import CllAnalyzer


def read_model_deviation(model_devi_path: Path):
    model_devi_path = model_devi_path.resolve()
    try:
        steps = np.loadtxt(model_devi_path, usecols=0)
        max_devi_f = np.loadtxt(model_devi_path, usecols=4)
        max_devi_v = np.loadtxt(model_devi_path, usecols=3)
    except FileNotFoundError as err:
        logger.error('Please select an existing model_devi.out')
        raise err
    return steps, max_devi_f, max_devi_v


class CllExplorationAnalyzer(ExplorationAnalyzer, CllAnalyzer):
    """Analyzer for exploration tasks.
    """

    def _iteration_tasks(self, iteration) -> List[Path]:
        n_iter = self._iteration_dir(iteration=iteration)
        stage_path = self.dp_task.path / n_iter / 'explore-lammps/tasks'
        task_files = [
            item for item in stage_path.iterdir() if re.search(r'^\d+$', str(item.name))
        ]
        return task_files

    def load_task_job_dict(self, task: Path):
        try:
            job_dict = lammps_variable_parser(task / 'lammps.input')
        except Exception as err:
            logger.error(err)
            job_dict = {}
        return job_dict
