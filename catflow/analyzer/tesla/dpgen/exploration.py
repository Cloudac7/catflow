import json
from collections.abc import Iterable
from pathlib import Path

from typing import Optional, Union, List
from matplotlib.figure import Figure

import numpy as np
from matplotlib import pyplot as plt

from catflow.utils.log_factory import logger
from catflow.analyzer.graph.plotting import canvas_style, square_grid
from catflow.analyzer.tesla.base.exploration import ExplorationAnalyzer, PlottingExploartion
from catflow.analyzer.tesla.dpgen.task import DPAnalyzer


class DPExplorationAnalyzer(ExplorationAnalyzer, DPAnalyzer):
    """Analyzer for DP exploration tasks."""

    def _iteration_tasks(self, iteration) -> List[Path]:
        n_iter = self._iteration_dir(control_step=2, iteration=iteration)
        return list((self.dp_task.path / n_iter).glob('01.model_devi/task*'))

    def load_task_job_dict(self, task: Path):
        # load job config
        try:
            with open(task / 'job.json', 'r') as f:
                job_dict = json.load(f)
        except Exception as err:
            logger.error(err)
            job_dict = {}
        return job_dict

    def get_cur_job(self, iteration: Optional[int] = None) -> dict:
        """Get `cur_job.json` for the selected iteration

        Args:
            iteration (int, optional): the iteration to get

        Returns:
            dict: current job parameters
        """
        n_iter = self._iteration_dir(control_step=2, iteration=iteration)
        try:
            with open(
                self.dp_task.path / n_iter / '01.model_devi' / 'cur_job.json',
                'r'
            ) as f:
                job_dict = json.load(f)
        except Exception as err:
            logger.warning(err)
            job_dict = {}
        return job_dict

    def _read_model_devi_trust_level(
        self,
        trust_level_key,
        iteration: Optional[int] = None
    ):
        cur_job = self.get_cur_job(iteration)
        trust_level = cur_job.get(trust_level_key)
        if trust_level is None:
            trust_level = self.dp_task.param_data[trust_level_key]
        # Is average OK for different systems?
        if isinstance(trust_level, Iterable):
            trust_level = np.mean(trust_level)  # type: ignore
        return trust_level

    def plot_single_iteration(
        self,
        iteration: Optional[int] = None,
        *,
        x_limit: Optional[Union[float, List[float]]] = None,
        y_limit: Optional[Union[float, List[float]]] = None,
        use_log: bool = False,
        group_by: Optional[str] = None,
        f_trust_lo: Optional[float] = None,
        f_trust_hi: Optional[float] = None,
        select: Optional[str] = None,
        select_value: Optional[str] = None,
        **kwargs
    ) -> Figure:
        """Generate a plot of model deviation in each iteration.

        Args:
            iteration (int, optional): The iteration. Defaults to current iteration.
            x_limit (float, List[float], optional): Choose the limit of x axis. Defaults to None.
            y_limit (float, List[float], optional): Choose the limit of y axis. Defaults to None.
            use_log (bool, optional): Choose whether log scale used. Defaults to False.
            group_by (str, optional): Choose which the plots are grouped by, which should be included. 
                Should be corresponding to keys in model_devi_job. Defaults to 'temps'.
            select (str, optional): Choose which param selected as plot zone. Defaults to None.
            select_value (str, optional): The dependence of `select`. 
                Different from `group_by`, please pass only one number. Defaults to None.
            kwargs (_type_, optional): Additional keyword arguments. Include other params, such as:
                `temps`: please use the value of `group_by`, whose default input is `"temps"`.
                `label_unit`: the unit of `select_value`, such as 'Å'.
                Parameters of `canvas_style`: please refer to `catflow.analyzer.graph.plotting.canvas_style`.

        Returns:
            Figure: A plot for different desired values.
        """
        if f_trust_hi is None:
            f_trust_hi = self._read_model_devi_trust_level(
                "model_devi_f_trust_hi", iteration)
        if f_trust_lo is None:
            f_trust_lo = self._read_model_devi_trust_level(
                "model_devi_f_trust_lo", iteration)
        if iteration is None:
            iteration = self.dp_task.iteration

        fig = super().plot_single_iteration(
            iteration,
            x_limit=x_limit,
            y_limit=y_limit,
            use_log=use_log,
            group_by=group_by,
            f_trust_lo=f_trust_lo,
            f_trust_hi=f_trust_hi,
            select=select,
            select_value=select_value,
            **kwargs
        )
        return fig

    def plot_multi_iter_distribution(
            self,
            iterations: Iterable,
            group_by: Optional[str] = None,
            f_trust_lo: Optional[float] = None,
            f_trust_hi: Optional[float] = None,
            select: Optional[str] = None,
            select_value: Optional[str] = None,
            x_limit: Optional[Union[float, List[float]]] = None,
            y_limit: Optional[Union[float, List[float]]] = None,
            **kwargs
    ) -> Figure:
        """Draw distribution in histogram of model deviation for multiple iterations.

        Args:
            iterations (Iterable): _description_
            group_by (str, optional): _description_. Defaults to None.
            select (str, optional): _description_. Defaults to None.
            select_value (str, optional): _description_. Defaults to None.
            f_trust_lo (Optional[float], optional): The lower limit of max_deviation_force. Defaults to None.
            f_trust_hi (Optional[float], optional): The higher limit of max_deviation_force. Defaults to None.
            x_limit (Union[float, List[float]], optional): _description_. Defaults to None.
            y_limit (Union[float, List[float]], optional): _description_. Defaults to None.

        Returns:
            Figure: A figure containing distribution of model deviation for multiple iterations.
        """
        num_item, plot_items = self._convert_group_by(group_by, **kwargs)
        label_unit = kwargs.get('label_unit', 'K')

        canvas_style(**kwargs)

        nrows = square_grid(num_item)
        fig, axs = plt.subplots(nrows, nrows, figsize=[
                                12, 12], constrained_layout=True)

        colors = plt.colormaps['viridis_r'](  # type: ignore
            np.linspace(0.15, 0.85, len(iterations))  # type: ignore
        )
        for i, plot_item in enumerate(plot_items):
            try:
                ax = axs.flatten()[i]
            except AttributeError:
                ax = axs
            for j, iteration in enumerate(iterations):
                step, mdf = self._data_prepareation(
                    plot_item, iteration, group_by, select, select_value, **kwargs)
                if f_trust_lo is None:
                    f_trust_lo = self._read_model_devi_trust_level(
                        "model_devi_f_trust_lo", iteration)
                if f_trust_hi is None:
                    f_trust_hi = self._read_model_devi_trust_level(
                        "model_devi_f_trust_hi", iteration)
                ax_args = {
                    'data': mdf,
                    'plot_item': plot_item,
                    'label_unit': kwargs.get('label_unit'),
                    'f_trust_lo': f_trust_lo,
                    'f_trust_hi': f_trust_hi,
                    'iteration': iteration,
                    'x_limit': x_limit,
                    'y_limit': y_limit,
                    'color': colors[j],
                    'label': f'Iter {iteration}'
                }
                PlottingExploartion.plot_mdf_distribution(
                    ax, ax_args, orientation='vertical')  # type: ignore
            ax.set_ylabel('Distribution')  # type: ignore
            ax.set_xlabel(r'$\sigma_{f}^{max}$ (ev/Å)')  # type: ignore
            ax.legend()  # type: ignore
        for i in range(num_item, nrows * nrows):
            try:
                fig.delaxes(axs.flatten()[i])
            except AttributeError:
                pass
        return fig

    def plot_ensemble_ratio_bar(
            self,
            iterations: Iterable,
            group_by: Optional[str] = None,
            select: Optional[str] = None,
            select_value: Optional[str] = None,
            f_trust_lo: Optional[float] = None,
            f_trust_hi: Optional[float] = None,
            **kwargs
    ) -> Figure:
        """Draw ensemble ratio bar for multiple iterations.

        Args:
            iterations (Iterable): _description_
            group_by (Optional[str], optional): _description_. Defaults to None.
            select (Optional[str], optional): _description_. Defaults to None.
            select_value (Optional[str], optional): _description_. Defaults to None.
            f_trust_lo (Optional[float], optional): _description_. Defaults to None.
            f_trust_hi (Optional[float], optional): _description_. Defaults to None.

        Returns:
            Figure: _description_
        """

        num_item, plot_items = self._convert_group_by(group_by, **kwargs)
        nrows = square_grid(num_item)

        canvas_style(**kwargs)

        fig, axs = plt.subplots(nrows, nrows)
        for i, plot_item in enumerate(plot_items):
            if type(axs) is plt.Axes:
                ax = axs
            else:
                ax = axs.flatten()[i]  # type: ignore

            ratios = np.zeros((len(iterations), 3))  # type: ignore

            for j, iteration in enumerate(iterations):
                df = self._load_model_devi_dataframe(
                    plot_item, iteration, group_by, select, select_value, **kwargs)

                if f_trust_lo is None:
                    f_trust_lo = self._read_model_devi_trust_level(
                        "model_devi_f_trust_lo", iteration)
                if f_trust_hi is None:
                    f_trust_hi = self._read_model_devi_trust_level(
                        "model_devi_f_trust_hi", iteration)

                accu_count = (df['max_devi_f'] <= f_trust_lo).sum()
                failed_count = (df['max_devi_f'] > f_trust_hi).sum()
                candidate_count = (
                    (df['max_devi_f'] > f_trust_lo) &
                    (df['max_devi_f'] <= f_trust_hi)
                ).sum()

                all_count = accu_count + failed_count + candidate_count
                count_array = np.array(
                    [accu_count, candidate_count, failed_count])
                ratios[j] = count_array / all_count

            ax_args = {
                'category_names': ['Accurate', 'Candidate', 'Failed'],
                'ratios': ratios,
                'iterations': iterations,
            }
            PlottingExploartion.plot_ensemble_ratio_bar(ax, ax_args)
            handles, labels = ax.get_legend_handles_labels()
        fig.supxlabel('Iteration')
        fig.supylabel('Ratio')
        fig.legend(handles, labels, loc='upper center',  # type: ignore
                   ncol=3, bbox_to_anchor=(0.5, 1.0))
        return fig
