from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from catflow.utils import logger
from catflow.analyzer.graph.plotting import canvas_style
from catflow.analyzer.tesla.base.training import TrainingAnalyzer
from catflow.analyzer.tesla.ai2_kit.task import CllAnalyzer, CllTask


class CllTrainingAnalyzer(TrainingAnalyzer, CllAnalyzer):
    """Analyzer for training tasks.
    """

    def get_lcurve_path(self, iteration: int, model=0) -> Path:
        _iteration_dir = self._iteration_dir(iteration=iteration)
        lcurve_path = self.dp_task.path / _iteration_dir / \
            f'train-deepmd/tasks/{str(model).zfill(3)}/lcurve.out'
        return lcurve_path

    def load_lcurve(self, iteration=None, model=0):
        lcurve_path = self.get_lcurve_path(iteration=iteration, model=model)

        if self.validation is True:
            return {
                'step': np.loadtxt(lcurve_path, usecols=0),
                'energy_train': np.loadtxt(lcurve_path, usecols=4),
                'energy_test': np.loadtxt(lcurve_path, usecols=3),
                'force_train': np.loadtxt(lcurve_path, usecols=6),
                'force_test': np.loadtxt(lcurve_path, usecols=5),
            }
        else:
            return {
                'step': np.loadtxt(lcurve_path, usecols=0),
                'energy_train': np.loadtxt(lcurve_path, usecols=2),
                'force_train': np.loadtxt(lcurve_path, usecols=3)
            }
