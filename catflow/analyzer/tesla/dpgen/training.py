from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from catflow.utils import logger
from catflow.analyzer.graph.plotting import canvas_style
from catflow.analyzer.tesla.base.training import TrainingAnalyzer
from catflow.analyzer.tesla.dpgen.task import DPAnalyzer

class DPTrainingAnalyzer(TrainingAnalyzer, DPAnalyzer):
    """Analyzer for training tasks.
    """

    def get_lcurve_path(self, iteration: int, model=0) -> Path:
        _iteration_dir = self._iteration_dir(iteration=iteration)
        lcurve_path = self.dp_task.path / _iteration_dir / \
            f'00.train/{str(model).zfill(3)}/lcurve.out'
        return lcurve_path
    
    def load_lcurve(self, iteration: int, model=0):
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

    def plot_lcurve(self, iteration: Optional[int] = None, model=0, **kwargs):
        """plot learning curve of the training task

        Args:
            iteration (int): Iteration of the training task
            model (int, optional): Index of trained model. Defaults to 0.

        Returns:
            fig: plt.figure
        """
        if iteration is None:
            iteration = self._iteration_control_code(
                control_step=2, iteration=self.dp_task.iteration
            )

        fig = super().plot_lcurve(iteration=iteration, model=model, **kwargs)

        return fig
