import json
import os
from re import A
import shutil
from collections.abc import Iterable, Sized
from glob import glob

from typing import Union, List
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import seaborn as sns
from ase.io import read, write
from matplotlib import pyplot as plt

from miko.utils import logger
from miko.utils.lammps import *
from miko.graph.plotting import canvas_style, AxesInit, square_grid
from miko.resources.submit import JobFactory
from miko.tesla.dpgen.base import DPAnalyzer


class DPExplorationAnalyzer(DPAnalyzer):
    """Analyzer for exploration tasks.
    """

    def make_set(self, iteration: int = None) -> dict:
        """Dump dataset for easy analysis as list of dict

        Args:
            iteration (int, optional): iteration to be dumped. Defaults to None, dumping the latest iteration.

        Returns:
            List[dict]: all model deviation results
        """
        n_iter = self._iteration_dir(control_step=2, iteration=iteration)
        all_data = []
        for task in (self.path / n_iter).glob('01.model_devi/task*'):
            # read model_devi.out
            steps, max_devi_f, max_devi_e = \
                read_model_deviation(task / 'model_devi.out')

            # load job config
            try:
                with open(task / 'job.json', 'r') as f:
                    job_dict = json.load(f)
            except Exception as err:
                logger.error(err)
                job_dict = {}

            # gather result_dict
            result_dict = {
                'iteration': n_iter,
                'max_devi_e': max_devi_e,
                'max_devi_f': max_devi_f,
                'task_path': task,
                'steps': steps
            }
            all_dict = {**result_dict, **job_dict}
            all_data.append(all_dict)
        return all_data

    def get_cur_job(self, iteration: int) -> dict:
        """Get `cur_job.json` for the selected iteration

        Args:
            iteration (int): the iteration to get

        Returns:
            dict: current job parameters
        """
        n_iter = self._iteration_dir(control_step=2, iteration=iteration)
        try:
            with open(self.path / n_iter / '01.model_devi' / 'cur_job.json', 'r') as f:
                job_dict = json.load(f)
        except Exception as err:
            logger.warning(err)
            job_dict = {}
        return job_dict

    def make_set_dataframe(self, iteration: int = None) -> pd.DataFrame:
        """Dump dataset for easy analysis as `pandas.Dataframe`

        Args:
            iteration (int, optional): iteration to be dumped. Defaults to None, dumping the latest iteration.

        Returns:
            pd.DataFrame: Dataframe containing all model deviation logs.
        """
        all_data = self.make_set(iteration=iteration)
        df = pd.DataFrame(all_data)
        return df

    def make_set_pickle(self, iteration: int = None) -> pd.DataFrame:
        """Dump pickle from `self.make_set_dataframe` for quick load.
           Default to `<dpgen_task_path>/model_devi_each_iter/data_<iter>.pkl`

        Args:
            iteration (int, optional): iteration to be dumped. Defaults to None, dumping the latest iteration.

        Returns:
            pd.DataFrame: DataFrame containing all model deviation logs.
        """
        df = self.make_set_dataframe(iteration=iteration)
        save_path = self.path / 'model_devi_each_iter'
        os.makedirs(name=save_path, exist_ok=True)
        if iteration is None:
            iteration = self._iteration_control_code(
                control_step=2, iteration=iteration)
        df.to_pickle(save_path / f'data_{str(iteration).zfill(6)}.pkl')
        return df

    def load_from_pickle(self, iteration: int) -> pd.DataFrame:
        """Load DataFrame from pickle file.

        Args:
            iteration (int): the iteration to get

        Returns:
            pd.DataFrame: DataFrame containing all model deviation logs.
        """
        pkl_path = self.path / \
            f'model_devi_each_iter/data_{str(iteration).zfill(6)}.pkl'
        df = pd.read_pickle(pkl_path)
        return df

    @staticmethod
    def _convert_group_by(group_by: str, **kwargs):
        plot_items = kwargs.get(group_by)
        if isinstance(plot_items, str):
            num_item = 1
            plot_items = [int(plot_items)]
        elif isinstance(plot_items, (int, float)):
            num_item = 1
            plot_items = [plot_items]
        elif isinstance(plot_items, Sized):
            num_item = len(plot_items)
        else:
            logger.error('The values chosen for plotting not exists.')
            raise TypeError(
                'Please pass values to be plotted with `group_by` value as variable name')
        return num_item, plot_items

    @staticmethod
    def select_dataset(dataset, select, select_value):
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

    def _data_prepareation(self, plot_item, iteration=None, group_by="temps", select=None, select_value=None, **kwargs):
        iteration_code = self._iteration_control_code(
            control_step=2, iteration=iteration)
        try:
            df = self.load_from_pickle(iteration=iteration_code)
        except FileNotFoundError:
            df = self.make_set_pickle(iteration=iteration_code)

        label_unit = kwargs.get('label_unit')

        if all([select, select_value]):
            df = self.select_dataset(df, select, select_value)
        partdata = self.extract_group_dataset(df, plot_item, group_by)
        parts = self.extract_iteration_dataset(
            partdata, self._iteration_dir(control_step=2, iteration=iteration)
        )
        steps = np.array(list(parts['steps'])).flatten()
        mdf = np.array(list(parts['max_devi_f'])).flatten()
        logger.info(
            f"f_max = {mdf.max()} ev/Å at {group_by}={plot_item} {label_unit} at iter {iteration_code}.")
        return steps, mdf

    def _read_model_devi_trust_level(self, trust_level_key, iteration=None):
        cur_job = self.get_cur_job(iteration)
        trust_level = cur_job.get(trust_level_key)
        if trust_level is None:
            trust_level = self.param_data[trust_level_key]
        # Is average OK for different systems?
        if isinstance(trust_level, Iterable):
            trust_level = np.mean(trust_level)
        return trust_level

    def plot_single_iteration(
            self,
            iteration: int = None,
            x_limit: Union[float, List[float]] = None,
            y_limit: Union[float, List[float]] = None,
            use_log: bool = False,
            group_by: str = 'temps',
            select: str = None,
            select_value: str = None,
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
                Parameters of `canvas_style`: please refer to `miko.graph.plotting.canvas_style`.

        Returns:
            Figure: A plot for different desired values.
        """

        num_item, plot_items = self._convert_group_by(group_by, **kwargs)

        canvas_style(**kwargs)
        fig = plt.figure(figsize=[16, 6 * num_item],
                         constrained_layout=True)
        gs = fig.add_gridspec(num_item, 3)

        for i, plot_item in enumerate(plot_items):
            steps, mdf = self._data_prepareation(
                plot_item, iteration, group_by, select, select_value, **kwargs)

            # left part
            fig_left = fig.add_subplot(gs[i, :-1])
            fig_left_args = {
                'x': steps,
                'y': mdf,
                'plot_item': plot_item,
                'label_unit': kwargs.get('label_unit'),
                'x_limit': x_limit,
                'y_limit': y_limit,
                'use_log': use_log,
                'f_trust_lo': self._read_model_devi_trust_level("model_devi_f_trust_lo", iteration),
                'f_trust_hi': self._read_model_devi_trust_level("model_devi_f_trust_hi", iteration),
                'iteration': iteration,
            }
            PlottingExploartion.plot_mdf_time_curve(fig_left, fig_left_args)
            global_ylim = fig_left.get_ylim()

            # right part
            fig_right = fig.add_subplot(gs[i, -1])
            fig_right_args = {
                'data': mdf,
                'plot_item': plot_item,
                'label_unit': kwargs.get('label_unit'),
                'y_limit': global_ylim,
                'use_log': use_log,
                'f_trust_lo': self._read_model_devi_trust_level("model_devi_f_trust_lo", iteration),
                'f_trust_hi': self._read_model_devi_trust_level("model_devi_f_trust_hi", iteration),
                'iteration': iteration,
            }
            PlottingExploartion.plot_mdf_distribution(
                fig_right, fig_right_args, orientation='horizontal')
        return fig

    def plot_multiple_iterations(
            self,
            iterations,
            group_by='temps',
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            select=None,
            select_value=None,
            **kwargs
    ):
        """
        Analyse trajectories for different temperatures.
        :param iterations: Iterations selected, which should be iterable.
        :param group_by: Choose which the plots are grouped by, which should be included.
            For value of group_by, a list, int or str containing desired value(s) should be included as kwargs.
            For example, if `group_by='temps'`, then `temps=[100., 200., 300.]` should also be passed to this function.
            Default: "temps".
        :param select: Choose which param selected as plot zone.
        :param f_trust_lo: The lower limit of max_deviation_force.
        :param f_trust_hi: The higher limit of max_deviation_force.
        :param x_lower_limit: The lower limit of x scale.
        :param x_higher_limit: The higher limit of x scale.
        :param y_limit: The limit of y.
        :param x_log: Choose whether use log scale for x axis.
        :param y_log: Choose whether use log scale for y axis.
        :return: A plot for different iterations.
        """

        num_item, plot_items = self._convert_group_by(group_by, **kwargs)
        label_unit = kwargs.get('label_unit', 'K')

        canvas_style(**kwargs)
        nrows = square_grid(num_item)
        fig, axs = plt.subplots(nrows, nrows, figsize=[12, 12], constrained_layout=True)
        for i, plot_item in enumerate(plot_items):
            try:
                ax = axs[i]
            except TypeError:
                ax = axs
            for iteration in iterations:
                step, mdf = self._data_prepareation(
                    plot_item, iteration, group_by, select, select_value, **kwargs)
                ax.scatter(step, mdf, s=80, alpha=0.3,
                           label=f'iter {int(iteration)}', marker='o')
            ax.axhline(f_trust_lo, linestyle='dashed')
            ax.axhline(f_trust_hi, linestyle='dashed')
            ax.set_ylabel(r'$\sigma_{f}^{max}$ (ev/Å)', fontsize=24)
            ax.set_xlabel('Simulation time (fs)', fontsize=24)
            ax.legend()
        try:
            plot_title = f'Iteration {",".join(iterations)}'
        except TypeError:
            plot_title = f'Iteration {iterations}'
        fig.suptitle(plot_title)
        return fig

    def plot_multi_iter_distribution(
            self,
            iterations,
            group_by='temps',
            select=None,
            select_value=None,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            x_lower_limit=1,
            x_higher_limit=None,
            y_limit=0.6,
            **kwargs
    ):
        num_item, plot_items = self._convert_group_by(group_by, **kwargs)
        label_unit = kwargs.get('label_unit', 'K')

        canvas_style(**kwargs)
        
        nrows = square_grid(num_item)
        fig, axs = plt.subplots(nrows, nrows, figsize=[12, 12], constrained_layout=True)
        for i, plot_item in enumerate(plot_items):
            try:
                ax = axs[i]
            except TypeError:
                ax = axs
            for iteration in iterations:
                step, mdf = self._data_prepareation(
                    plot_item, iteration, group_by, select, select_value, **kwargs)
                ax_args = {
                    'data': mdf,
                    'plot_item': plot_item,
                    'label_unit': kwargs.get('label_unit'),
                    'f_trust_lo': self._read_model_devi_trust_level("model_devi_f_trust_lo", iteration),
                    'f_trust_hi': self._read_model_devi_trust_level("model_devi_f_trust_hi", iteration),
                    'iteration': iteration,
                }
                PlottingExploartion.plot_mdf_distribution(
                    ax, ax_args, orientation='vertical')
            ax.axhline(f_trust_lo, linestyle='dashed')
            ax.axhline(f_trust_hi, linestyle='dashed')
            ax.set_ylabel('Distribution', fontsize=24)
            ax.set_xlabel(r'$\sigma_{f}^{max}$ (ev/Å)', fontsize=24)
            ax.legend()
        try:
            plot_title = f'Iteration {",".join(iterations)}'
        except TypeError:
            plot_title = f'Iteration {iterations}'
        fig.suptitle(plot_title)
        return fig

        frames = []
        for it in iterations:
            location = os.path.join(
                self.path, f'data_pkl/data_{str(it).zfill(2)}.pkl')
            if os.path.exists(location):
                frames.append(self.md_set_load_pkl(iteration=it))
            else:
                frames.append(self.md_set_pd(iteration=it))
        items = kwargs.get(group_by, None)
        if isinstance(items, (list, tuple)):
            num_items = len(items)
        elif isinstance(items, (int, float)):
            num_items = 0
            items = [items]
        elif isinstance(items, str):
            num_items = 0
            items = [str(items)]
        else:
            raise TypeError("temps should be a value or a list of value.")
        label_unit = kwargs.get('label_unit', 'K')
        df = pd.concat(frames)
        plt.figure(figsize=[12, 12])
        for i, item in enumerate(items):
            ax = plt.subplot(num_items, 1, i + 1)
            for k in iterations:
                if select is not None:
                    select_value = kwargs.get('select_value', None)
                    if select_value is not None:
                        df = df[df[select] == select_value]
                part_data = df[df[group_by] == item]
                parts = part_data[part_data['iteration']
                                  == 'iter.' + str(k).zfill(6)]
                for j, [temp, part] in enumerate(parts.groupby(group_by)):
                    mdf = np.array(list(part['max_devi_f']))
                    t_freq = np.average(part['t_freq'])
                    flatmdf = np.ravel(mdf)
                    plt.hist(flatmdf, bins=100, density=True,
                             label=f'iter {int(k)}', alpha=0.5)
            if x_higher_limit is None:
                x_higher_limit = ax.get_xlim()[1]
            ax.set_xlim(x_lower_limit, x_higher_limit)
            if y_limit is None:
                y_limit = ax.get_ylim()[1]
            ax.set_ylim(0, y_limit)
            plt.axvline(f_trust_lo, linestyle='dashed')
            plt.axvline(f_trust_hi, linestyle='dashed')
            plt.xlabel(r'$\sigma_{f}^{max}$ (eV/Å)', fontsize=16)
            plt.ylabel('Distribution', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16)
            plt.title(f'{item} {label_unit}', fontsize=16)
        plt.tight_layout()
        return plt


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

        sns.scatterplot(data=args, x='x', y='y', color='red',
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
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Simulation Steps')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r'$\sigma_{f}^{max}$ (ev/Å)')
        ax.legend()
        if iteration is None:
            if ax.get_subplotspec().is_first_row():
                ax.set_title(f'Iteration {iteration}')
        return ax

    @staticmethod
    def plot_mdf_distribution(ax: plt.Axes, args, orientation='vertical'):
        #data = args.get('data')
        x_limit = args.get('x_limit')
        y_limit = args.get('y_limit')
        f_trust_lo = args.get('f_trust_lo')
        f_trust_hi = args.get('f_trust_hi')

        if orientation == 'vertical':
            sns.histplot(
                data=args, x="data", bins=50,
                kde=True, stat='density', color='red', ec=None, alpha=0.5, ax=ax
            )
            ax.axvline(f_trust_lo, linestyle='dashed')
            ax.axvline(f_trust_hi, linestyle='dashed')
        elif orientation == 'horizontal':
            sns.histplot(
                data=args, y="data", bins=50,
                kde=True, stat='density', color='red', ec=None, alpha=0.5, ax=ax
            )
            ax.axhline(f_trust_lo, linestyle='dashed')
            ax.axhline(f_trust_hi, linestyle='dashed')
        else:
            raise ValueError('Invalid orientation')

        if ax.get_subplotspec().is_first_row():
            ax.set_title('Distribution of Deviation')

        PlottingExploartion._plot_set_axis_limits(ax, x_limit, 'x_limit')
        if args.get('use_log', False) == True:
            ax.set_yscale('log')
        else:
            PlottingExploartion._plot_set_axis_limits(ax, y_limit, 'y_limit')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
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
            elif isinstance(limitation, Iterable):
                if len(limitation) > 2:
                    raise ValueError("Limitation should be a value \
                        or a set with no more than 2 elements.")
                elif len(limitation) == 1:
                    _func(0, limitation[0])
                else:
                    _func(limitation)
            else:
                raise ValueError("Limitation should be a value \
                    or a set with no more than 2 elements.")
