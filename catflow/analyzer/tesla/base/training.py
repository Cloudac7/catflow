from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from catflow.utils import logger
from catflow.analyzer.graph.plotting import canvas_style
from catflow.analyzer.tesla.base.task import BaseAnalyzer, BaseTask


class TrainingAnalyzer(BaseAnalyzer):
    """Analyzer for training tasks.
    """

    def __init__(
        self,
        dp_task: BaseTask,
        validation: bool = False
    ) -> None:
        super().__init__(dp_task)
        self.validation = validation

    def get_lcurve_path(self, iteration: int, model=0) -> Path:
        ...
        return Path()

    def load_lcurve(self, iteration: int, model=0):
        lcurve_path = self.get_lcurve_path(iteration=iteration, model=model)

        if self.validation is True:
            return {
                'step': np.loadtxt(lcurve_path, usecols=0),
                'energy_train': np.loadtxt(lcurve_path, usecols=4),
                'energy_validation': np.loadtxt(lcurve_path, usecols=3),
                'force_train': np.loadtxt(lcurve_path, usecols=6),
                'force_validation': np.loadtxt(lcurve_path, usecols=5),
            }
        else:
            return {
                'step': np.loadtxt(lcurve_path, usecols=0),
                'energy_train': np.loadtxt(lcurve_path, usecols=2),
                'force_train': np.loadtxt(lcurve_path, usecols=3)
            }

    def plot_lcurve(self, iteration: int, model=0, **kwargs):
        lcurve_data = self.load_lcurve(iteration=iteration, model=model)

        canvas_style(**kwargs)
        fig, axs = plt.subplots(2, 1)

        # energy figure
        step = lcurve_data['step']
        for key in lcurve_data.keys():
            if key.startswith('energy_'):
                axs[0].plot(step[10:], lcurve_data[key][10:],
                            alpha=0.4, label=key.replace('energy_', ''))
        axs[0].axhline(0.005, linestyle='dashed', color='red', label='5 meV')
        axs[0].axhline(0.01, linestyle='dashed', color='blue', label='10 meV')
        axs[0].axhline(0.05, linestyle='dashed', label='50 meV')
        axs[0].set_xlabel('Number of training batch')
        axs[0].set_ylabel('$E$(eV)')
        axs[0].legend()

        # force figure
        for key in lcurve_data.keys():
            if key.startswith('force_'):
                axs[1].plot(
                    step[10:], lcurve_data[key][10:], label=key.replace('force_', '')
                )
        axs[1].axhline(0.05, linestyle='dashed', color='red', label='50 meV/Å')
        axs[1].axhline(0.1, linestyle='dashed', color='blue', label='100 meV/Å')
        axs[1].axhline(0.2, linestyle='dashed', label='200 meV/Å')
        axs[1].set_xlabel('Number of training batch')
        axs[1].set_ylabel('$F$(eV/Å)')
        axs[1].legend()

        return fig
