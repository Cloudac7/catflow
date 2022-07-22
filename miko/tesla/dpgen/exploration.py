import json
import os
from re import A
import shutil
from collections.abc import Iterable, Sized
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
from ase.io import read, write
from matplotlib import pyplot as plt

from miko.utils import logger
from miko.utils.lammps import *
from miko.graph.plotting import canvas_style, AxesInit
from miko.resources.submit import JobFactory
from miko.tesla.dpgen.base import DPAnalyzer


class DPExplorationAnalyzer(DPAnalyzer):
    """Analyzer for exploration tasks.
    """

    def make_set(self, iteration=None):
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

    def get_cur_job(self, iteration):
        n_iter = self._iteration_dir(control_step=2, iteration=iteration)
        try:
            with open(self.path / n_iter / 'cur_job.json', 'r') as f:
                job_dict = json.load(f)
        except Exception as err:
            logger.warning(err)
            job_dict = {}
        return job_dict

    def make_set_dataframe(self, iteration=None):
        all_data = self.make_set(iteration=iteration)
        df = pd.DataFrame(all_data)
        return df

    def make_set_pickle(self, iteration=None):
        if iteration is None:
            iteration = self._iteration_control_code(control_step=2, iteration=iteration)
        df = self.make_set_dataframe(iteration=iteration)
        save_path = self.path / 'model_devi_each_iter'
        os.makedirs(name=save_path, exist_ok=True)
        df.to_pickle(save_path / f'data_{str(iteration).zfill(6)}.pkl')
        return df

    def load_from_pickle(self, iteration):
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
        iteration_code = self._iteration_control_code(control_step=2, iteration=iteration)
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
            f"max devi of F is :{mdf.max()} ev/Å at {group_by}={plot_item} {label_unit}.")
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
            iteration=None,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            x_limit=None,
            y_limit=None,
            use_log=False,
            group_by='temps',
            select=None,
            select_value=None,
            **kwargs):
        """Generate a plot of model deviation in each iteration.

        Args:
            iteration (_type_): The iteration. Defaults to current iteration.
            f_trust_lo (float, optional): The lower limit of max_deviation_force. Defaults to 0.10.
            f_trust_hi (float, optional): The higher limit of max_deviation_force. Defaults to 0.30.
            x_limit (float, List[float], optional): Choose the limit of x axis. Defaults to None.
            y_limit (float, List[float], optional): Choose the limit of y axis. Defaults to None.
            use_log (bool, optional): Choose whether log scale used. Defaults to False.
            group_by (str, optional): Choose which the plots are grouped by, which should be included. 
                Should be corresponding to keys in model_devi_job. Defaults to 'temps'.
            select (_type_, optional): Choose which param selected as plot zone. Defaults to None.
            select_value (_type_, optional): The dependence of `select`. 
                Different from `group_by`, please pass only one number. Defaults to None.
            kwargs (_type_, optional): Additional keyword arguments. Include other params, such as:
                `temps`: please use the value of `group_by`, whose default input is `"temps"`.
                `label_unit`: the unit of `select_value`, such as 'Å'.
                Parameters of `canvas_style`: please refer to `miko.graph.plotting.canvas_style`.

        Raises:
            TypeError: _description_

        Returns:
            _type_: A plot for different desired values.
        """

        iteration_code = self._iteration_control_code(control_step=2, iteration=iteration)

        try:
            df = self.load_from_pickle(iteration=iteration_code)
        except FileNotFoundError:
            df = self.make_set_pickle(iteration=iteration_code)

        num_item, plot_items = self._convert_group_by(group_by, **kwargs)

        label_unit = kwargs.get('label_unit')
        canvas_style(**kwargs)
        fig = plt.figure(figsize=[16, 6 * num_item],
                         constrained_layout=True)
        gs = fig.add_gridspec(num_item, 3)

        for i, plot_item in enumerate(plot_items):
            steps, mdf = self._data_prepareation(plot_item, iteration, group_by, select, select_value, **kwargs)

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
            
            # right part
            fig_right = fig.add_subplot(gs[i, -1])
            fig_right_args = {
                'y': mdf, 
                'plot_item': plot_item,
                'label_unit': kwargs.get('label_unit'),
                'x_limit': x_limit,
                'y_limit': fig_left.get_ylim(),
                'use_log': use_log,
                'f_trust_lo': self._read_model_devi_trust_level("model_devi_f_trust_lo", iteration),
                'f_trust_hi': self._read_model_devi_trust_level("model_devi_f_trust_hi", iteration),
                'iteration': iteration,
            }
            PlottingExploartion.plot_mdf_distribution(fig_right, fig_right_args, orientation='horizontal')
        return plt

    def md_multi_iter(
            self,
            iterations,
            group_by='temps',
            select=None,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            x_lower_limit=0,
            x_higher_limit=1e3,
            y_limit=None,
            x_log=False,
            y_log=False,
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
            num_items = 1
            items = [items]
        elif isinstance(items, str):
            num_items = 1
            items = [int(items)]
        else:
            raise TypeError("temps should be a value or a list of value.")
        label_unit = kwargs.get('label_unit', 'K')
        df = pd.concat(frames)
        plt.figure(figsize=[24, 8 * num_items])
        for i, item in enumerate(items):
            ax = plt.subplot(num_items, 1, i + 1)
            for k in iterations:
                if select is not None:
                    select_value = kwargs.get('select_value', None)
                    if select_value is not None:
                        df = df[df[select] == select_value]
                partdata = df[df[group_by] == item]
                parts = partdata[partdata['iteration']
                                 == 'iter.' + str(k).zfill(6)]
                for j, [item, part] in enumerate(parts.groupby(group_by)):
                    mdf = np.array(list(part['max_devi_f']))
                    t_freq = np.average(part['t_freq'])
                    dupt = np.tile(range(mdf.shape[1]) * t_freq, mdf.shape[0])
                    flatmdf = np.ravel(mdf)
                    plt.scatter(dupt, flatmdf, s=80, alpha=0.3,
                                label=f'iter {int(k)}', marker='o')
            if x_higher_limit is None:
                x_higher_limit = ax.get_xlim()[1]
            plt.xlim(x_lower_limit, x_higher_limit)
            if y_limit is None:
                y_limit = ax.get_ylim()[1]
            plt.ylim(0, y_limit)
            if x_log:
                plt.xscale('log')
            if y_log:
                plt.yscale('log')
            plt.axhline(f_trust_lo, linestyle='dashed')
            plt.axhline(f_trust_hi, linestyle='dashed')
            plt.xlabel('Simulation time (fs)', fontsize=24)
            plt.ylabel(r'$\sigma_{f}^{max}$ (ev/Å)', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.legend(fontsize=24)
            plt.title(f'{item} {label_unit}', fontsize=24)
        plt.tight_layout()
        return plt

    def multi_iter_distribution(
            self,
            iterations,
            group_by='temps',
            select=None,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            x_lower_limit=1,
            x_higher_limit=None,
            y_limit=0.6,
            **kwargs
    ):
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

    def md_single_task(
            self,
            work_path,
            model_path,
            machine_name,
            resource_dict,
            numb_models=4,
            **kwargs
    ):
        """Submit your own md task with the help of DPDispatcher.

        Parameters
        ----------
        work_path : str
            The dir contains your md tasks.
        model_path : str
            The path of models contained for calculation.
        machine_name : str
            machine name to use
        resource_dict : dict
            resource dict
        numb_models : int, optional
            The number of models selected., by default 4

        Returns
        -------
        JobFactory
            To generate job to be submitted.
        """
        mdata = self.machine_data['model_devi'][0]
        folder_list = kwargs.get('folder_list', ["task.*"])
        all_task = []
        for i in folder_list:
            _task = glob(os.path.join(work_path, i))
            _task.sort()
            all_task += _task
        lmp_exec = mdata['command']
        command = lmp_exec + " -i input.lammps"
        run_tasks_ = all_task
        run_tasks = [os.path.basename(ii) for ii in run_tasks_]

        model_names = kwargs.get(
            'model_names', [f'graph.{str(i).zfill(3)}.pb' for i in range(numb_models)])
        for ii in model_names:
            if not os.path.exists(os.path.join(work_path, ii)):
                os.symlink(os.path.join(model_path, ii),
                           os.path.join(work_path, ii))
        forward_files = kwargs.get(
            'forward_files', ['conf.lmp', 'input.lammps', 'traj'])
        backward_files = kwargs.get(
            'backward_files', ['model_devi.out', 'model_devi.log', 'traj'])

        task_dict_list = [
            {
                "command": command,
                "task_work_path": task,
                "forward_files": forward_files,
                "backward_files": backward_files,
                "outlog": kwargs.get('outlog', 'model_devi.log'),
                "errlog": kwargs.get('errlog', 'model_devi.log'),
            } for task in run_tasks
        ]

        submission_dict = {
            "work_base": work_path,
            "forward_common_files": model_names,
            "backward_common_files": kwargs.get('backward_common_files', [])
        }
        return JobFactory(task_dict_list, submission_dict, machine_name, resource_dict)

    def train_model_test(
            self,
            machine_name,
            resource_dict,
            iteration=None,
            params=None,
            restart=False,
            **kwargs
    ):
        """Run lammps MD tests from trained models.

        Parameters
        ----------
        machine_name : str
            machine name
        resource_dict : dict
            resource name
        iteration : str, optional
            Select the iteration for training. 
            If not selected, the last iteration where training has been finished would be chosen.
        params : str, optional
            Necessary params for MD tests.
        kwargs : str, optional
            Other optional parameters.

        Returns
        -------

        """
        location = os.path.abspath(self.path)
        logger.info(f"Task path: {location}")

        if iteration is None:
            if self.step_code < 2:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        n_iter = 'iter.' + str(iteration).zfill(6)
        model_path = os.path.join(location, n_iter, '00.train')
        test_path = os.path.join(location, n_iter, '04.model_test')

        if params is None:
            params = self.param_data

        if restart == True:
            logger.info("Restarting from old checkpoint")
        else:
            logger.info("Preparing MD input")
            template_base = kwargs.get('template_base', None)
            self._train_generate_md_test(
                params=params,
                work_path=test_path,
                model_path=model_path,
                template_base=template_base
            )

        if self.step_code < 6:
            md_iter = self.iteration - 1
        else:
            md_iter = self.iteration
        md_iter = 'iter.' + str(md_iter).zfill(6)

        logger.info("Task submitting")
        job = self.md_single_task(
            work_path=test_path,
            model_path=model_path,
            machine_name=machine_name,
            resource_dict=resource_dict,
            numb_models=self.param_data['numb_models'],
            forward_files=kwargs.get(
                "forward_files", ['conf.lmp', 'input.lammps']),
            backward_files=kwargs.get("backward_files",
                                      ['model_devi.out', 'md_test.log', 'md_test.err', 'dump.lammpstrj']),
            outlog=kwargs.get("outlog", 'md_test.log'),
            errlog=kwargs.get("errlog", 'md_test.err')
        )
        job.run_submission()
        logger.info("MD Test finished.")

    def _train_generate_md_test(self, params, work_path, model_path, template_base=None):
        cur_job = params['md_test']
        use_plm = params.get('model_devi_plumed', False)
        trj_freq = cur_job.get('traj_freq', False)

        if template_base is None:
            template_base = self.path

        lmp_templ = cur_job['template']['lmp']
        lmp_templ = os.path.abspath(os.path.join(template_base, lmp_templ))
        plm_templ = None

        if use_plm:
            plm_templ = cur_job['template']['plm']
            plm_templ = os.path.abspath(os.path.join(template_base, plm_templ))

        sys_idxs = cur_job.get('sys_idx')
        for sys_idx in sys_idxs:
            sys_init_stc = convert_init_structures(params, sys_idx)
            templ, rev_mat = parse_template(cur_job, sys_idx)
            for task_counter, task in enumerate(rev_mat):
                task_name = "task.%03d.%06d" % (sys_idx, task_counter)
                task_path = os.path.join(work_path, task_name)

                # create task path
                os.makedirs(task_path, exist_ok=True)
                shutil.copyfile(lmp_templ, os.path.join(
                    task_path, 'input.lammps'))
                model_list = glob(os.path.join(model_path, 'graph*pb'))
                model_list.sort()
                model_names = [os.path.basename(i) for i in model_list]
                task_model_list = []
                for jj in model_names:
                    task_model_list.append(
                        os.path.join('..', os.path.basename(jj)))

                # revise input of lammps
                with open(os.path.join(task_path, 'input.lammps')) as fp:
                    lmp_lines = fp.readlines()
                _dpmd_idx = check_keywords(lmp_lines, ['pair_style', 'deepmd'])
                graph_list = ' '.join(task_model_list)
                lmp_lines[_dpmd_idx] = \
                    f"pair_style      deepmd {graph_list} out_freq {trj_freq} out_file model_devi.out\n "
                _dump_idx = check_keywords(lmp_lines, ['dump', 'dpgen_dump'])
                lmp_lines[_dump_idx] = \
                    f"dump            dpgen_dump all custom {trj_freq} dump.lammpstrj id type x y z fx fy fz\n"
                lmp_lines = substitute_keywords(lmp_lines, task)

                # revise input of plumed
                if use_plm:
                    _plm_idx = check_keywords(lmp_lines, ['fix', 'dpgen_plm'])
                    lmp_lines[_plm_idx] = \
                        "fix            dpgen_plm all plumed plumedfile input.plumed outfile output.plumed\n "
                    with open(plm_templ) as fp:
                        plm_lines = fp.readlines()
                    plm_lines = substitute_keywords(plm_lines, task)
                    with open(os.path.join(task_path, 'input.plumed'), 'w') as fp:
                        fp.write(''.join(plm_lines))

                # dump input of lammps
                with open(os.path.join(task_path, 'input.lammps'), 'w') as fp:
                    fp.write(''.join(lmp_lines))
                with open(os.path.join(task_path, 'job.json'), 'w') as fp:
                    job = task
                    json.dump(job, fp, indent=4)

                # dump init structure
                write(os.path.join(task_path, 'conf.lmp'),
                      sys_init_stc, format='lammps-data')


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

        sns.scatterplot(x, y, color='red', alpha=0.5, ax=ax, label=f'{plot_item} {label_unit}')
            
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
        data = args.get('data')
        x_limit = args.get('x_limit')
        y_limit = args.get('y_limit')
        f_trust_lo = args.get('f_trust_lo')
        f_trust_hi = args.get('f_trust_hi')

        if orientation == 'vertical':
            sns.histplot(
                x=data, bins=50, kde=True, stat='density', color='red', ec=None, alpha=0.5, ax=ax
            )
            ax.axvline(f_trust_lo, linestyle='dashed')
            ax.axvline(f_trust_hi, linestyle='dashed')
        elif orientation == 'horizontal':
            sns.histplot(
                y=data, bins=50, kde=True, stat='density', color='red', ec=None, alpha=0.5, ax=ax
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
