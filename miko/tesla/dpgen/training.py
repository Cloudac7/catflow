import numpy as np
from matplotlib import pyplot as plt

from miko.utils import logger
from miko.graph.plotting import canvas_style
from miko.tesla.dpgen.base import DPAnalyzer

class DPTrainingAnalyzer(DPAnalyzer):
    """Analyzer for training tasks.
    """

    def load_lcurve(self, iteration=None, model=0):
        n_iter = self._iteration_dir(control_step=2, iteration=iteration)
        lcurve_path = self.dp_task.path / n_iter / \
            f'00.train/{str(model).zfill(3)}/lcurve.out'

        from distutils.version import LooseVersion

        if LooseVersion(self.dp_task.deepmd_version) < LooseVersion('2.0'):
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

    def plot_lcurve(self, iteration=None, model=0, **kwargs):
        lcurve_data = self.load_lcurve(iteration=iteration, model=model)

        canvas_style(**kwargs)
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("DeepMD training and tests error")

        # energy figure
        step = lcurve_data['step']
        for key in lcurve_data.keys():
            if key.startswith('energy_'):
                axs[0].scatter(step[10:], lcurve_data[key][10:],
                               alpha=0.4, label=key.replace('energy_', ''))
        axs[0].hlines(0.005, step[0], step[-1], linestyles='--',
                      colors='red', label='5 meV')
        axs[0].hlines(0.01, step[0], step[-1], linestyles='--',
                      colors='blue', label='10 meV')
        axs[0].hlines(0.05, step[0], step[-1], linestyles='--', label='50 meV')
        axs[0].set_xlabel('Number of training batch')
        axs[0].set_ylabel('$E$(eV)')
        axs[0].legend()

        # force figure
        for key in lcurve_data.keys():
            if key.startswith('force_'):
                axs[1].scatter(step[10:], lcurve_data[key][10:],
                               alpha=0.4, label=key.replace('force_', ''))
        axs[1].hlines(0.05, step[0], step[-1], linestyles='--',
                      colors='red', label='50 meV/Å')
        axs[1].hlines(0.1, step[0], step[-1], linestyles='--',
                      colors='blue', label='100 meV/Å')
        axs[1].hlines(0.2, step[0], step[-1],
                      linestyles='--', label='200 meV/Å')
        axs[1].set_xlabel('Number of training batch')
        axs[1].set_ylabel('$F$(eV/Å)')
        axs[1].legend()

        return fig
