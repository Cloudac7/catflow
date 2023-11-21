import os

from collections.abc import Iterable, Collection, Sized
from pathlib import Path

from typing import Optional, Union, List, Tuple
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from catflow.utils.log_factory import logger
from catflow.analyzer.graph.plotting import canvas_style, square_grid
from catflow.analyzer.tesla.base.task import BaseAnalyzer


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


class ExplorationAnalyzer(BaseAnalyzer):
    """Analyzer for exploration tasks.
    """

    def _iteration_tasks(self, iteration: int) -> List[Path]:
        ...
        return []

    def load_task_job_dict(self, task: Path) -> dict:
        ...
        return {}

    def make_set(self, iteration: int, **kwargs) -> List[dict]:
        """Dump dataset for easy analysis as list of dict

        Args:
            iteration (int, optional): iteration to be dumped. Defaults to None, dumping the latest iteration.

        Returns:
            List[dict]: all model deviation results
        """

        all_data = []
        for task in self._iteration_tasks(iteration):
            # read model_devi.out
            steps, max_devi_f, max_devi_v = \
                read_model_deviation(task / 'model_devi.out')

            # load job config
            job_dict = self.load_task_job_dict(task)

            # gather result_dict
            result_dict = {
                'iteration': self._iteration_dir(iteration=iteration, **kwargs),
                'max_devi_v': max_devi_v,
                'max_devi_f': max_devi_f,
                'task_path': task,
                'steps': steps
            }
            all_dict = {**result_dict, **job_dict}
            all_data.append(all_dict)
        return all_data

    def make_set_dataframe(self, iteration: int) -> pd.DataFrame:
        """Dump dataset for easy analysis as `pandas.Dataframe`

        Args:
            iteration (int, optional): iteration to be dumped. Defaults to None, dumping the latest iteration.

        Returns:
            pd.DataFrame: Dataframe containing all model deviation logs.
        """
        all_data = self.make_set(iteration=iteration)
        df = pd.DataFrame(all_data)
        df = df.explode(["max_devi_v", "max_devi_f", "steps"])
        return df

    def make_set_pickle(self, iteration: int) -> pd.DataFrame:
        """Dump pickle from `self.make_set_dataframe` for quick load.
           Default to `<dpgen_task_path>/model_devi_each_iter/data_<iter>.pkl`

        Args:
            iteration (int, optional): iteration to be dumped. Defaults to None, dumping the latest iteration.

        Returns:
            pd.DataFrame: DataFrame containing all model deviation logs.
        """
        df = self.make_set_dataframe(iteration=iteration)
        save_path = self.dp_task.path / 'model_devi_each_iter'
        os.makedirs(name=save_path, exist_ok=True)
        df.to_pickle(save_path / f'data_{str(iteration).zfill(6)}.pkl')
        return df

    def load_from_pickle(self, iteration: int) -> pd.DataFrame:
        """Load DataFrame from pickle file.

        Args:
            iteration (int): the iteration to get

        Returns:
            pd.DataFrame: DataFrame containing all model deviation logs.
        """
        pkl_path = self.dp_task.path / \
            f'model_devi_each_iter/data_{str(iteration).zfill(6)}.pkl'
        df = pd.read_pickle(pkl_path)
        return df

    @staticmethod
    def _convert_group_by(
        group_by: Optional[str] = None,
        **kwargs
    ) -> Tuple[int, Union[List, Collection]]:
        if group_by is None:
            num_item = 1
            plot_items = [None]
        else:
            plot_items = kwargs.get(group_by)
            if isinstance(plot_items, str):
                num_item = 1
                plot_items = [int(plot_items)]
            elif isinstance(plot_items, (int, float)):
                num_item = 1
                plot_items = [plot_items]
            elif isinstance(plot_items, Collection):
                num_item = len(plot_items)
            else:
                num_item = 1
                plot_items = [plot_items]
        return num_item, plot_items

    @staticmethod
    def select_dataset(dataset, select, select_value=None):
        try:
            df = dataset[dataset[select] == select_value]
        except KeyError as err:
            logger.error(f'Please choose existing parameter for `select`')
            raise err
        return df

    @staticmethod
    def extract_group_dataset(dataset, group_item, group_by='temps'):
        try:
            part_data = dataset[dataset[group_by] == group_item]
        except KeyError as err:
            logger.error(
                f'Please choose existing parameter for `group_by`')
            raise err
        return part_data

    @staticmethod
    def extract_iteration_dataset(dataset, iteration_dir=None):
        try:
            parts = dataset[dataset['iteration'] == iteration_dir]
        except KeyError as err:
            logger.error(f'Please choose existing iteration as input.')
            raise err
        return parts

    def _load_model_devi_dataframe(
            self,
            plot_item,
            iteration: int,
            group_by=None,
            select=None,
            select_value=None,
            **kwargs
    ) -> pd.DataFrame:
        """Load model deviation DataFrame from tasks."""
        try:
            df = self.load_from_pickle(iteration=iteration)
        except FileNotFoundError:
            df = self.make_set_pickle(iteration=iteration)

        # select data frame of given select
        if all([select, select_value]):
            df = self.select_dataset(df, select, select_value)

        # extract data frame of given group
        if group_by:
            partdata = self.extract_group_dataset(df, plot_item, group_by)
        else:
            partdata = df

        # export data frame of given iteration
        parts = self.extract_iteration_dataset(
            partdata, self._iteration_dir(iteration=iteration, **kwargs)
        )
        return parts

    def _data_prepareation(
            self,
            plot_item,
            iteration: int,
            group_by=None,
            select=None,
            select_value=None,
            **kwargs
    ):
        parts = self._load_model_devi_dataframe(
            plot_item, iteration, group_by, select, select_value, **kwargs
        )

        steps = parts['steps']
        mdf = parts['max_devi_f']
        label_unit = kwargs.get('label_unit')
        logger.info(
            f"f_max = {mdf.max()} ev/Å at {group_by}={plot_item} {label_unit} at iter {iteration}.")
        return steps, mdf

    def _read_model_devi_trust_level(self, trust_level_key, iteration=None):
        pass

    def plot_single_iteration(
            self,
            iteration: int,
            *,
            x_limit: Optional[Union[float, List[float]]] = None,
            y_limit: Optional[Union[float, List[float]]] = None,
            use_log: bool = False,
            group_by: Optional[str] = None,
            f_trust_lo: float = 0.1,
            f_trust_hi: float = 0.3,
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
            f_trust_lo (float, optional): The lower limit of max_deviation_force. Defaults to 0.1.
            f_trust_hi (float, optional): The higher limit of max_deviation_force. Defaults to 0.3.
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

        num_item, plot_items = self._convert_group_by(group_by, **kwargs)

        canvas_style(**kwargs)
        fig = plt.figure(figsize=[12, 4 * num_item],
                         constrained_layout=True)
        gs = fig.add_gridspec(num_item, 3)

        for i, plot_item in enumerate(plot_items):
            steps, mdf = self._data_prepareation(
                plot_item, iteration, group_by, select, select_value, **kwargs)

            # left part
            fig_left = fig.add_subplot(gs[i, :-1])  # type: ignore
            fig_left_args = {
                'x': steps,
                'y': mdf,
                'plot_item': plot_item,
                'label_unit': kwargs.get('label_unit'),
                'x_limit': x_limit,
                'y_limit': y_limit,
                'use_log': use_log,
                'f_trust_lo': f_trust_lo,
                'f_trust_hi': f_trust_hi,
                'color': 'red',
                'iteration': iteration,
            }
            PlottingExploartion.plot_mdf_time_curve(fig_left, fig_left_args)
            global_ylim = fig_left.get_ylim()

            # right part
            fig_right = fig.add_subplot(gs[i, -1])  # type: ignore
            fig_right_args = {
                'data': mdf,
                'plot_item': plot_item,
                'label_unit': kwargs.get('label_unit'),
                'y_limit': global_ylim,
                'use_log': use_log,
                'f_trust_lo': f_trust_lo,
                'f_trust_hi': f_trust_hi,
                'color': 'red',
                'iteration': iteration,
            }
            PlottingExploartion.plot_mdf_distribution(
                fig_right, fig_right_args, orientation='horizontal')
            fig_right.set_xticklabels([])
            fig_right.set_yticklabels([])
        return fig

    def plot_multiple_iterations(
            self,
            iterations: Iterable,
            group_by: Optional[str] = None,
            f_trust_lo: float = 0.1,
            f_trust_hi: float = 0.3,
            x_limit: Optional[Union[float, List[float]]] = None,
            y_limit: Optional[Union[float, List[float]]] = None,
            select: Optional[str] = None,
            select_value: Optional[str] = None,
            **kwargs
    ):
        """Analyse trajectories for different temperatures.

        Args:
            iterations (Iterabke): Iterations selected, which should be iterable.
            group_by (str, optional): Choose which the plots are grouped by, which should be included.
            For value of group_by, a list, int or str containing desired value(s) should be included as kwargs.
            For example, if `group_by='temps'`, then `temps=[100., 200., 300.]` should also be passed to this function.
            Default: "temps".
            f_trust_lo (float, optional): The lower limit of max_deviation_force. Defaults to 0.1.
            f_trust_hi (float, optional): The higher limit of max_deviation_force. Defaults to 0.3.
            x_limit (_type_, optional): The limit of x scale. Defaults to None.
            y_limit (_type_, optional): The limit of y scale. Defaults to None.
            select (_type_, optional): Choose which param selected as plot zone.. Defaults to None.
            select_value (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: A plot for different iterations.

        """
        num_item, plot_items = self._convert_group_by(group_by, **kwargs)
        label_unit = kwargs.get('label_unit', 'K')

        canvas_style(**kwargs)
        nrows = square_grid(num_item)
        fig, axs = plt.subplots(nrows, nrows, figsize=[
                                12, 12], constrained_layout=True)
        for i, plot_item in enumerate(plot_items):
            try:
                ax = axs.flatten()[i]
            except AttributeError:
                ax = axs
            for iteration in iterations:
                step, mdf = self._data_prepareation(
                    plot_item, iteration, group_by, select, select_value, **kwargs)
                ax.scatter(step, mdf, s=80, alpha=0.3,  # type: ignore
                           label=f'iter {int(iteration)}', marker='o')
            ax.axhline(f_trust_lo, linestyle='dashed')  # type: ignore
            ax.axhline(f_trust_hi, linestyle='dashed')  # type: ignore
            ax.set_ylabel(r"$\sigma_{f}^{max}$ (ev/Å)")  # type: ignore
            ax.set_xlabel('Simulation time (fs)')  # type: ignore
            ax.legend()  # type: ignore
            if x_limit is not None:
                PlottingExploartion._plot_set_axis_limits(
                    ax, x_limit, 'x_limit')  # type: ignore
            if kwargs.get('use_log', False) == True:
                ax.set_yscale('log')  # type: ignore
            else:
                if y_limit is not None:
                    PlottingExploartion._plot_set_axis_limits(
                        ax, y_limit, 'y_limit')  # type: ignore
        for i in range(num_item, nrows * nrows):
            try:
                fig.delaxes(axs.flatten()[i])
            except AttributeError:
                pass
        return fig

    def plot_multi_iter_distribution(
            self,
            iterations: Iterable,
            group_by: Optional[str] = None,
            f_trust_lo: float = 0.1,
            f_trust_hi: float = 0.3,
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
            f_trust_lo (float, optional): The lower limit of max_deviation_force. Defaults to 0.1.
            f_trust_hi (float, optional): The higher limit of max_deviation_force. Defaults to 0.3.
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
            f_trust_lo: float = 0.1,
            f_trust_hi: float = 0.3,
            **kwargs
    ) -> Figure:
        """Draw ensemble ratio bar for multiple iterations.

        Args:
            iterations (Iterable): _description_
            group_by (Optional[str], optional): _description_. Defaults to None.
            select (Optional[str], optional): _description_. Defaults to None.
            select_value (Optional[str], optional): _description_. Defaults to None.
            f_trust_lo (float, optional): _description_. Defaults to 0.1.
            f_trust_hi (float, optional): _description_. Defaults to 0.3.

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


class PlottingExploartion:
    @staticmethod
    def plot_mdf_time_curve(ax: plt.Axes, args):
        plot_item = args.get('plot_item')
        label_unit = args.get('label_unit')
        x = args.get('x')
        y = args.get('y')
        x_limit = args.get('x_limit')
        y_limit = args.get('y_limit')
        f_trust_lo = args.get('f_trust_lo')
        f_trust_hi = args.get('f_trust_hi')
        iteration = args.get('iteration')
        color = args.get('color')

        sns.scatterplot(data=args, x='x', y='y', color=color,
                        alpha=0.5, ax=ax, label=f'{plot_item} {label_unit}')

        PlottingExploartion._plot_set_axis_limits(ax, x_limit, 'x_limit')
        if args.get('use_log', False) == True:
            ax.set_yscale('log')
        else:
            PlottingExploartion._plot_set_axis_limits(ax, y_limit, 'y_limit')

        if f_trust_lo is not None:
            ax.axhline(f_trust_lo, linestyle='dashed')
        if f_trust_hi is not None:
            ax.axhline(f_trust_hi, linestyle='dashed')
        if ax.get_subplotspec().is_last_row():  # type: ignore
            ax.set_xlabel('Simulation Steps')
        if ax.get_subplotspec().is_first_col():  # type: ignore
            ax.set_ylabel(r'$\sigma_{f}^{max}$ (ev/Å)')
        ax.legend()
        return ax

    @staticmethod
    def plot_mdf_distribution(ax: plt.Axes, args, orientation='vertical'):
        # data = args.get('data')
        x_limit = args.get('x_limit')
        y_limit = args.get('y_limit')
        f_trust_lo = args.get('f_trust_lo')
        f_trust_hi = args.get('f_trust_hi')
        color = args.get('color')
        label = args.get('label')

        # draw the kernel density estimate plot
        if orientation == 'vertical':
            sns.kdeplot(
                data=args, x="data", label=label, fill=True,
                color=color, alpha=0.5, ax=ax,
            )
            if f_trust_lo:
                ax.axvline(f_trust_lo, linestyle='dashed')
            if f_trust_hi:
                ax.axvline(f_trust_hi, linestyle='dashed')
        elif orientation == 'horizontal':
            sns.kdeplot(
                data=args, y="data", label=label, fill=True,
                color=color, alpha=0.5, ax=ax
            )
            if f_trust_lo:
                ax.axhline(f_trust_lo, linestyle='dashed')
            if f_trust_hi:
                ax.axhline(f_trust_hi, linestyle='dashed')
        else:
            raise ValueError('Invalid orientation')

        PlottingExploartion._plot_set_axis_limits(ax, x_limit, 'x_limit')
        if args.get('use_log', False) == True:
            ax.set_yscale('log')
        else:
            PlottingExploartion._plot_set_axis_limits(ax, y_limit, 'y_limit')
        return ax

    @staticmethod
    def plot_ensemble_ratio_bar(ax: plt.Axes, args):
        data = args.get('ratios')
        labels = args.get('iterations')
        category_names = args.get('category_names')

        category_colors = plt.colormaps['YlGnBu_r'](  # type: ignore
            np.linspace(0.15, 0.85, data.shape[1])
        )
        data_cum = data.cumsum(axis=1)
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.bar(labels, widths, width=0.5, bottom=starts,
                   label=colname, color=color)
        ax.plot(labels, data[:, 0], "o-", c=category_colors[0], mec="white")
        ax.set_ylim(0.0, 1.0)
        return ax

    @staticmethod
    def _plot_set_axis_limits(ax: plt.Axes, limitation, limitation_type):
        limitation_dict = {
            'x_limit': ax.set_xlim,
            'y_limit': ax.set_ylim,
        }
        _func = limitation_dict[limitation_type]
        if limitation is not None:
            if isinstance(limitation, (int, float)):
                _func(0, limitation)
            elif isinstance(limitation, Sized):
                if len(limitation) > 2:
                    raise ValueError("Limitation should be a value \
                        or a set with no more than 2 elements.")
                elif len(limitation) == 1:
                    _func(0, limitation[0])  # type: ignore
                else:
                    _func(limitation)
            else:
                raise ValueError("Limitation should be a value \
                    or a set with no more than 2 elements.")
