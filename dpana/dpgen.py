import os
import json
import daemon

import dpdata
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob

import shutil
from ase.io import read, write
from dpgen.generator.run import parse_cur_job_revmat, find_only_one_key, \
    revise_by_keys, revise_lmp_input_plm
from matplotlib import pyplot as plt
from dpana.util import canvas_style
from dpgen.dispatcher.Dispatcher import make_dispatcher


class DPTask(object):
    """
    DPTask is a class reading a dpgen directory, where the dpgen task run.
    """

    def __init__(self, path, param_file, machine_file, record_file):
        """
        Generate a class of dpgen task.
        :param path: The path of the dpgen task.
        :param param_file: The param json file name.
        :param machine_file: The machine json file name.
        :param record_file: The record file name.
        """
        self.path = os.path.abspath(path)
        self.param_file = param_file
        self.machine_file = machine_file
        self.record_file = record_file
        self._load_task()

    def train_lcurve(self, iteration=None, model=0, **kwargs):
        if iteration is None:
            if self.step_code < 2:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        n_iter = 'iter.' + str(iteration).zfill(6)
        lcurve_path = os.path.join(self.path, n_iter, f'00.train/{str(model).zfill(3)}/lcurve.out')

        step = np.loadtxt(lcurve_path, usecols=0)
        energy_train = np.loadtxt(lcurve_path, usecols=4)
        energy_test = np.loadtxt(lcurve_path, usecols=3)
        force_train = np.loadtxt(lcurve_path, usecols=6)
        force_test = np.loadtxt(lcurve_path, usecols=5)

        canvas_style(**kwargs)
        fig = plt.figure()
        plt.title("DeepMD training and test error")
        plt.subplot(2, 1, 1)
        plt.scatter(step[10:], energy_train[10:], alpha=0.4, label='train')
        plt.scatter(step[10:], energy_test[10:], alpha=0.4, label='test')
        plt.hlines(0.005, step[0], step[-1], linestyles='--', colors='red', label='5 meV')
        plt.hlines(0.01, step[0], step[-1], linestyles='--', colors='blue', label='10 meV')
        plt.hlines(0.05, step[0], step[-1], linestyles='--', label='50 meV')
        plt.legend()
        plt.xlabel('Number of training batch')
        plt.ylabel('$E$(eV)')
        plt.subplot(2, 1, 2)
        plt.scatter(step[10:], force_train[10:], alpha=0.4, label='train')
        plt.scatter(step[10:], force_test[10:], alpha=0.4, label='test')
        plt.hlines(0.05, step[0], step[-1], linestyles='--', colors='red', label='50 meV/Å')
        plt.hlines(0.1, step[0], step[-1], linestyles='--', colors='blue', label='100 meV/Å')
        plt.hlines(0.2, step[0], step[-1], linestyles='--', label='200 meV/Å')
        plt.xlabel('Number of training batch')
        plt.ylabel('$F$(eV/Å)')
        plt.legend()
        return fig

    def md_make_set(self, iteration=None):
        location = self.path
        if iteration is None:
            if self.step_code < 6:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        n_iter = 'iter.' + str(iteration).zfill(6)
        all_data = []
        for task in glob(f'{location}/{n_iter}/01.model_devi/task*'):
            step = np.loadtxt(f'{task}/model_devi.out', usecols=0)
            dump_freq = step[1] - step[0]
            max_devi_f = np.loadtxt(f'{task}/model_devi.out', usecols=4)
            max_devi_e = np.loadtxt(f'{task}/model_devi.out', usecols=3)
            with open(f'{task}/input.lammps', 'r') as f:
                lines = f.readlines()
            temp = float(lines[3].split()[3])
            start, final = 0, 0
            with open(f'{task}/model_devi.log', 'r') as f:
                for i, line in enumerate(f):
                    key_line = line.strip()
                    if 'Step Temp' in key_line:
                        start = i + 1
                    elif 'Loop time of' in key_line:
                        final = i
            with open(f'{task}/model_devi.log', 'r') as f:
                if dump_freq > 10:
                    step = int(dump_freq / 10)
                    lines = f.readlines()[start:final:step]
                else:
                    lines = f.readlines()[start:final]
            pot_energy = np.array([p.split()[2] for p in lines if 'WARNING' not in p]).astype('float')
            try:
                with open(f'{task}/job.json', 'r') as f:
                    job_dict = json.load(f)
            except Exception as e:
                print(e)
                job_dict = {}
            result_dict = {
                'iter': n_iter,
                'temps': temp,
                'max_devi_e': max_devi_e,
                'max_devi_f': max_devi_f,
                'task': task,
                't_freq': dump_freq,
                'pot_energy': pot_energy
            }
            all_dict = {**result_dict, **job_dict}
            all_data.append(all_dict)
        return all_data

    def md_set_pd(self, iteration=None):
        if iteration is None:
            if self.step_code < 6:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        all_data = self.md_make_set(iteration=iteration)
        df = pd.DataFrame(all_data)
        return df

    def md_set_pkl(self, iteration=None):
        df = self.md_set_pd(iteration=iteration)
        save_path = self.path
        os.makedirs(name=f'{save_path}/data_pkl', exist_ok=True)
        df.to_pickle(f'{save_path}/data_pkl/data_{str(iteration).zfill(2)}.pkl')

    def md_set_load_pkl(self, iteration):
        pkl_path = os.path.join(self.path, f'data_pkl/data_{str(iteration).zfill(2)}.pkl')
        df = pd.read_pickle(pkl_path)
        return df

    def md_single_iter(
            self,
            iteration,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            xlimit=None,
            ylimit=None,
            log=False,
            group_by='temps',
            select=None,
            **kwargs):
        """
        Generate a plot of model deviation in each iteration.
        :param iteration: The iteration
        :param f_trust_lo: The lower limit of max_deviation_force.
        :param f_trust_hi: The higher limit of max_deviation_force.
        :param xlimit: Choose the limit of x axis.
        :param ylimit: Choose the limit of y axis.
        :param log: Choose whether log scale used. Default: False.
        :param group_by: Choose which the plots are grouped by, which should be included.
            For value of group_by, a list, int or str containing desired value(s) should be included as kwargs.
            For example, if `group_by='temps'`, then `temps=[100., 200., 300.]` should also be passed to this function.
            Default: "temps".
        :param select: Choose which param selected as plot zone.
        :param kwargs: Include other params, such as:
            `temps`: please use the value of `group_by`, whose default input is `"temps"`.
            `select_value`: the dependence of `select`. Different from `group_by`, please pass only one number.
            `label_unit`: the unit of `select_value`, such as 'Å'.
            `step`: control the step of each point along x axis, in prevention of overlap.
            Parameters of `canvas_style`: please refer to `dpana.util.canvas_style`.
        :return: A plot for different desired values.
        """
        flatmdf = None
        location = os.path.join(self.path, f'data_pkl/data_{str(iteration).zfill(2)}.pkl')
        if os.path.exists(location):
            df = self.md_set_load_pkl(iteration=iteration)
        else:
            df = self.md_set_pd(iteration=iteration)
        try:
            plot_items = kwargs.get(group_by, None)
            if isinstance(plot_items, (list, tuple)):
                num_item = len(plot_items)
            elif isinstance(plot_items, (int, float)):
                num_item = 1
                plot_items = [plot_items]
            elif isinstance(plot_items, str):
                num_item = 1
                plot_items = [int(plot_items)]
            else:
                raise TypeError("The value of `group_by` dependence should exist.")
            label_unit = kwargs.get('label_unit', None)
            canvas_style(**kwargs)
            fig = plt.figure(figsize=[16, 6 * num_item], constrained_layout=True)
            gs = fig.add_gridspec(num_item, 3)

            for i, item in enumerate(plot_items):
                if select is not None:
                    select_value = kwargs.get('select_value', None)
                    if select_value is not None:
                        df = df[df[select] == select_value]
                partdata = df[df[group_by] == item]
                # left part
                fig_left = fig.add_subplot(gs[i, :-1])
                parts = partdata[partdata['iter'] == 'iter.' + str(iteration).zfill(6)]
                for j, [item, part] in enumerate(parts.groupby(group_by)):
                    mdf = np.array(list(part['max_devi_f']))[:, ::kwargs.get('step', None)]
                    t_freq = np.average(part['t_freq']) * kwargs.get('step', 1)
                    dupt = np.tile(np.arange(mdf.shape[1]) * t_freq, mdf.shape[0])
                    flatmdf = np.ravel(mdf)
                    print(f"max devi of F is :{max(flatmdf)} ev/Å at {group_by}={item} {label_unit}.")
                    sns.scatterplot(
                        x=dupt,
                        y=flatmdf,
                        color='red',
                        alpha=0.5,
                        ax=fig_left,
                        label=f'{item} {label_unit}'
                    )
                if xlimit is not None:
                    fig_left.set_xlim(0, xlimit)
                else:
                    xlimit = fig_left.get_xlim()[1]
                    fig_left.set_xlim(0, xlimit)
                if ylimit is not None:
                    if not log:
                        fig_left.set_ylim(0, ylimit)
                    else:
                        fig_left.set_yscale('log')
                else:
                    ylimit = fig_left.get_ylim()[1]
                    if not log:
                        fig_left.set_ylim(0, ylimit)
                    else:
                        fig_left.set_yscale('log')
                fig_left.axhline(f_trust_lo, linestyle='dashed')
                fig_left.axhline(f_trust_hi, linestyle='dashed')
                if fig_left.is_last_row():
                    fig_left.set_xlabel('Simulation Steps')
                if fig_left.is_first_col():
                    fig_left.set_ylabel('$\sigma_{f}^{max}$ (ev/Å)')
                fig_left.legend()
                if fig_left.is_first_row():
                    fig_left.set_title(f'Iteration {iteration}')

                # right part
                fig_right = fig.add_subplot(gs[i, -1])
                sns.histplot(
                    y=flatmdf,
                    bins=50,
                    kde=True,
                    stat='density',
                    color='red',
                    ec=None,
                    alpha=0.5,
                    ax=fig_right
                )
                if fig_right.is_first_row():
                    fig_right.set_title('Distribution of Deviation')
                if not log:
                    fig_right.set_ylim(0, ylimit)
                else:
                    fig_right.set_yscale('log')
                fig_right.axhline(f_trust_lo, linestyle='dashed')
                fig_right.axhline(f_trust_hi, linestyle='dashed')
                fig_right.set_xticklabels([])
                fig_right.set_yticklabels([])
            return plt
        except Exception as e:
            print(e)
            print('Please choose proper `group_by` with in dict.')
            return None

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
            location = os.path.join(self.path, f'data_pkl/data_{str(it).zfill(2)}.pkl')
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
                parts = partdata[partdata['iter'] == 'iter.' + str(k).zfill(6)]
                for j, [item, part] in enumerate(parts.groupby(group_by)):
                    mdf = np.array(list(part['max_devi_f']))
                    t_freq = np.average(part['t_freq'])
                    dupt = np.tile(range(mdf.shape[1]) * t_freq, mdf.shape[0])
                    flatmdf = np.ravel(mdf)
                    plt.scatter(dupt, flatmdf, s=80, alpha=0.3, label=f'iter {int(k)}', marker='o')
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
            plt.axhline(f_trust_lo, linestyles='dashed')
            plt.axhline(f_trust_hi, linestyles='dashed')
            plt.xlabel('Simulation time (fs)', fontsize=24)
            plt.ylabel('$\sigma_{f}^{max}$ (ev/Å)', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.legend(fontsize=24)
            plt.title(f'{item} {label_unit}', fontsize=24)
        plt.tight_layout()
        return plt

    def md_single_task(self, work_path, model_path, numb_models=4, **kwargs):
        """
        Submit your own md task with the help of dpgen.
        :param work_path: The dir contains your md tasks.
        :param model_path: The path of models contained for calculation.
        :param numb_models: The number of models selected.
        :return:
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
        commands = [command]
        run_tasks_ = all_task
        run_tasks = [os.path.basename(ii) for ii in run_tasks_]

        model_names = kwargs.get('model_names', [f'graph.{str(i).zfill(3)}.pb' for i in range(numb_models)])
        for ii in model_names:
            if not os.path.exists(os.path.join(work_path, ii)):
                os.symlink(os.path.join(model_path, ii), os.path.join(work_path, ii))
        forward_files = kwargs.get('forward_files', ['conf.lmp', 'input.lammps', 'traj'])
        backward_files = kwargs.get('backward_files', ['model_devi.out', 'model_devi.log', 'traj'])

        dispatcher = make_dispatcher(mdata['machine'], mdata['resources'], work_path, run_tasks, 1)
        dispatcher.run_jobs(
            resources=mdata['resources'],
            command=commands,
            work_path=work_path,
            tasks=run_tasks,
            group_size=1,
            forward_common_files=model_names,
            forward_task_files=forward_files,
            backward_task_files=backward_files,
            outlog=kwargs.get('outlog', 'model_devi.log'),
            errlog=kwargs.get('errlog', 'model_devi.log'))
        return dispatcher

    def train_model_test(self, iteration=None, params=None, files=None, **kwargs):
        """
        Run MD tests from trained models.
        :param iteration: Select the iteration for training. If not selected, the last iteration where training has been finished would be chosen.
        :param params: Necessary params for MD test.
        :param files: Extra files attached with the MD runs. Not necessary.
        :param kwargs: Other optional parameters.
        :return:
        """
        location = os.path.abspath(self.path)
        print(location)
        if iteration is None:
            if self.step_code < 2:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        n_iter = 'iter.' + str(iteration).zfill(6)
        model_path = os.path.join(location, n_iter, '00.train')
        test_path = os.path.join(location, n_iter, '04.model_test')
        print("Preparing MD input......")
        self._train_generate_md_test(params=params, work_path=test_path, model_path=model_path)
        if self.step_code < 6:
            md_iter = self.iteration - 1
        else:
            md_iter = self.iteration
        md_iter = 'iter.' + str(md_iter).zfill(6)
        for p in glob(os.path.join(test_path, 'task.*')):
            if not os.path.exists(os.path.join(p, 'conf.lmp')):
                _lmp_data = glob(os.path.join(location, md_iter, '01.model_devi', 'task*', 'conf.lmp'))[0]
                os.symlink(_lmp_data, os.path.join(p, 'conf.lmp'))
            if files is not None:
                for file in files:
                    _file_abs = os.path.abspath(file)
                    _file_base = os.path.basename(file)
                    if not os.path.exists(os.path.join(p, _file_base)):
                        os.symlink(_file_abs, os.path.join(p, _file_base))
        print("Submitting")
        with daemon.DaemonContext():
            self.md_single_task(
                work_path=test_path,
                model_path=model_path,
                numb_models=self.param_data['numb_models'],
                forward_files=kwargs.get("forward_files", ['conf.lmp', 'input.lammps']),
                backward_files=kwargs.get("backward_files",
                                          ['model_devi.out', 'md_test.log', 'md_test.err', 'dump.lammpstrj']),
                outlog=kwargs.get("outlog", 'md_test.log'),
                errlog=kwargs.get("errlog", 'md_test.err')
            )
        # print("MD Test finished.")

    def _train_generate_md_test(self, params, work_path, model_path):
        cur_job = params['model_devi_jobs']
        use_plm = params.get('model_devi_plumed', False)
        use_plm_path = params.get('model_devi_plumed_path', False)
        trj_freq = cur_job.get('traj_freq', False)

        rev_keys, rev_mat, num_lmp = parse_cur_job_revmat(cur_job, use_plm=use_plm)
        lmp_templ = cur_job['template']['lmp']
        lmp_templ = os.path.abspath(lmp_templ)
        plm_templ = None
        plm_path_templ = None
        if use_plm:
            plm_templ = cur_job['template']['plm']
            plm_templ = os.path.abspath(plm_templ)
            if use_plm_path:
                plm_path_templ = cur_job['template']['plm_path']
                plm_path_templ = os.path.abspath(plm_path_templ)
        task_counter = 0
        for ii in range(len(rev_mat)):
            rev_item = rev_mat[ii]
            task_name = "task.%06d" % task_counter
            task_path = os.path.join(work_path, task_name)
            # create task path
            os.makedirs(task_path, exist_ok=True)
            # chdir to task path
            os.chdir(task_path)
            shutil.copyfile(lmp_templ, 'input.lammps')
            model_list = glob(os.path.join(model_path, 'graph*pb'))
            model_list.sort()
            model_names = [os.path.basename(i) for i in model_list]
            task_model_list = []
            for jj in model_names:
                task_model_list.append(os.path.join('..', os.path.basename(jj)))
            # revise input of lammps
            with open('input.lammps') as fp:
                lmp_lines = fp.readlines()
            _dpmd_idx = find_only_one_key(lmp_lines, ['pair_style', 'deepmd'])
            graph_list = ' '.join(task_model_list)
            lmp_lines[_dpmd_idx] = "pair_style      deepmd %s out_freq %d out_file model_devi.out\n" % (
                graph_list, trj_freq)
            _dump_idx = find_only_one_key(lmp_lines, ['dump', 'dpgen_dump'])
            lmp_lines[_dump_idx] = "dump            dpgen_dump all custom %d dump.lammpstrj id type x y z\n" % trj_freq
            lmp_lines = revise_by_keys(lmp_lines, rev_keys[:num_lmp], rev_item[:num_lmp])
            # revise input of plumed
            if use_plm:
                lmp_lines = revise_lmp_input_plm(lmp_lines, 'input.plumed')
                shutil.copyfile(plm_templ, 'input.plumed')
                with open('input.plumed') as fp:
                    plm_lines = fp.readlines()
                plm_lines = revise_by_keys(plm_lines, rev_keys[num_lmp:], rev_item[num_lmp:])
                with open('input.plumed', 'w') as fp:
                    fp.write(''.join(plm_lines))
                if use_plm_path:
                    shutil.copyfile(plm_path_templ, 'plmpath.pdb')
            # dump input of lammps
            with open('input.lammps', 'w') as fp:
                fp.write(''.join(lmp_lines))
            with open('job.json', 'w') as fp:
                job = {}
                for mm, nn in zip(rev_keys, rev_item):
                    job[mm] = nn
                json.dump(job, fp, indent=4)
            task_counter += 1

    def fp_group_distance(self, iteration, atom_group):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param atom_group:A tuple contains the index number of two selected atoms.
        :return: A plot of distance distribution.
        """
        dis_loc = []
        dis = []
        place = os.path.join(self.path, 'iter.' + str(iteration).zfill(6), '02.fp')
        _stc_name = self._fp_style()
        for i in os.listdir(place):
            if os.path.exists(os.path.join(place, i, _stc_name)):
                dis_loc.append(i)
                stc = read(os.path.join(place, i, _stc_name))
                dis.append(stc.get_distance(atom_group[0], atom_group[1], mic=True))
        diss = np.array(dis)
        plt.figure()
        plt.hist(diss, bins=np.arange(diss.min(), diss.max(), 0.01), label=f'iter {int(iteration)}', density=True)
        plt.legend(fontsize=16)
        plt.xlabel("d(Å)", fontsize=16)
        plt.xticks(np.arange(diss.min(), diss.max(), step=1.0), fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("Distibution of distance", fontsize=16)
        return plt

    def fp_element_distance(self, iteration, ele_group):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param ele_group:A tuple contains the index number of two selected elements.
        :return: A plot of distance distribution.
        """
        dis = []
        dis_loc = []
        place = os.path.join(self.path, 'iter.' + str(iteration).zfill(6), '02.fp')
        _output_name = self._fp_style()
        for i in os.listdir(place):
            if os.path.exists(os.path.join(place, i, _output_name)):
                dis_loc.append(i)
                stc = read(os.path.join(place, i, _output_name))
                symbol_list = stc.get_chemical_symbols()
                ele_list_1 = [i for i in range(len(symbol_list)) if symbol_list[i] == ele_group[0]]
                ele_list_2 = [i for i in range(len(symbol_list)) if symbol_list[i] == ele_group[0]]
                min_dis = min([stc.get_distance(ii, jj, mic=True) for ii in ele_list_1 for jj in ele_list_2])
                dis.append(min_dis)
        diss = np.array(dis)
        plt.figure(figsize=[16, 8], dpi=144)
        plt.hist(diss, bins=np.arange(1, 6, 0.01), label=f'iter {int(iteration)}')
        plt.legend(fontsize=16)
        plt.xlabel("d(Å)", fontsize=16)
        plt.xticks(np.arange(0, 6, step=0.5), fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f"Distibution of {ele_group[0]}-{ele_group[1]} distance", fontsize=16)
        return plt

    def fp_error_test(self, iteration=None, test_model=None):
        """
        Test your model quickly with the data generated from dpgen.
        :param iteration: Select the iteration of data for testing. Default: the latest one.
        :param test_model: Select the iteration of model for testing. Default: the latest one.
        :return:
        """
        location = self.path
        if iteration is None:
            if self.step_code < 7:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        n_iter = 'iter.' + str(iteration).zfill(6)
        quick_test_dir = os.path.join(location, n_iter, '03.quick_test')
        os.makedirs(os.path.join(quick_test_dir, 'task.md'), exist_ok=True)
        task_list = glob(os.path.join(location, n_iter, '02.fp', 'task*'))
        task_list.sort()
        _stc_file = self._fp_output_style()
        _dpgen_output = self._fp_output_dpgen()
        _dpdata_format = self._fp_output_format()
        all_sys = None
        stcs = []
        for idx, oo in enumerate(task_list):
            sys = dpdata.LabeledSystem(os.path.join(oo, _dpgen_output), fmt=_dpdata_format)
            stc = read(os.path.join(oo, _stc_file))
            if len(sys) > 0:
                sys.check_type_map(type_map=self.param_data['type_map'])
            if idx == 0:
                all_sys = sys
                stcs.append(stc)
            else:
                try:
                    all_sys.append(sys)
                    stcs.append(stc)
                except (RuntimeError, TypeError, NameError):
                    pass
        write(os.path.join(quick_test_dir, 'task.md/validate.xyz'), stcs, format='extxyz')
        atom_numb = np.sum(all_sys['atom_numbs'])
        dft_energy = all_sys['energies']
        dft_force = all_sys['forces']
        if test_model is None:
            if self.step_code < 2:
                test_model = self.iteration - 1
            else:
                test_model = self.iteration
        model_iter = 'iter.' + str(test_model).zfill(6)
        model_dir = os.path.join(location, model_iter, '00.train')
        self._fp_generate_error_test(work_path=quick_test_dir, model_dir=model_dir)
        if not os.path.exists(os.path.join(quick_test_dir, 'task.md/conf.lmp')):
            _lmp_data = glob(os.path.join(location, n_iter, '01.model_devi', 'task*', 'conf.lmp'))[0]
            os.symlink(_lmp_data, os.path.join(quick_test_dir, 'task.md/conf.lmp'))
        print("Quick test task submitting...")
        self.md_single_task(
            work_path=quick_test_dir,
            model_path=model_dir,
            numb_models=self.param_data['numb_models'],
            forward_files=['conf.lmp', 'input.lammps', 'validate.xyz'],
            backward_files=['model_devi.out', 'quick_test.log', 'quick_test.err', 'dump.lammpstrj'],
            outlog='quick_test.log',
            errlog='quick_test.err'
        )
        print("Finished")
        start, final = 0, 0
        with open(os.path.join(quick_test_dir, 'task.md/quick_test.log'), 'r') as f:
            for i, line in enumerate(f):
                key_line = line.strip()
                if 'Step ' in key_line:
                    start = i + 1
                elif 'Loop time of' in key_line:
                    final = i
        with open(os.path.join(quick_test_dir, 'task.md/quick_test.log'), 'r') as f:
            lines = f.readlines()[start:final]
        md_energy = np.array([p.split()[1] for p in lines]).astype('float')
        _md_stc = read(os.path.join(quick_test_dir, 'task.md/dump.lammpstrj'), index=':', format='lammps-dump-text')
        md_force = np.array([ss.get_forces() for ss in _md_stc])
        energy_rmse = np.sqrt(np.mean((md_energy - dft_energy) ** 2)) / atom_numb
        md_force_r = np.ravel(md_force)
        dft_force_r = np.ravel(dft_force)
        force_rmse = np.sqrt(np.mean((md_force_r - dft_force_r) ** 2))
        fig = plt.figure(figsize=[16, 8], dpi=96)
        # Plot of energy error
        plt.subplot(1, 2, 1)
        plt.scatter(dft_energy / atom_numb, md_energy / atom_numb, s=5, label=f'Iter. {iteration}')
        _x = np.linspace(np.min(dft_energy / atom_numb) - 0.05, np.max(dft_energy / atom_numb) + 0.05, 10)
        plt.plot(_x, _x, 'r--')
        plt.text(np.min(dft_energy / atom_numb) - 0.05, np.max(dft_energy / atom_numb) + 0.05,
                 f'RMSE={energy_rmse} (eV/atom)',
                 fontsize=14)
        plt.title(f'Energy error', fontsize=14)
        plt.xlabel(r'$e_{DFT}$ (eV/atom)', fontsize=14)
        plt.ylabel(r'$e_{DPMD}$ (eV/atom)', fontsize=14)
        # Plot of force error
        plt.subplot(1, 2, 2)
        plt.scatter(md_force_r, dft_force_r, s=5, label=f'Iter. {iteration}')
        _y = np.linspace(np.min(dft_force_r) - 0.05, np.max(dft_force_r) + 0.05, 10)
        plt.plot(_y, _y, 'r--')
        plt.text(np.min(dft_force_r) - 0.05, np.max(dft_force_r) + 0.05, f'RMSE={force_rmse} (eV/Å)', fontsize=14)
        plt.title(f'Force error', fontsize=14)
        plt.xlabel(r'$f_{DFT}$ (eV/Å)', fontsize=14)
        plt.ylabel(r'$f_{DPMD}$ (eV/Å)', fontsize=14)
        f, ax = plt.subplots()
        ax.set_aspect('equal')
        return fig

    def _fp_generate_error_test(self, work_path, model_dir):
        model_list = glob(os.path.join(model_dir, 'graph*pb'))
        model_list.sort()
        model_names = [os.path.basename(i) for i in model_list]
        input_file = "units           metal\n"
        input_file += "boundary        p p p\n"
        input_file += "atom_style      atomic\n"
        input_file += "\n"
        input_file += "neighbor        2.0 bin\n"
        input_file += "neigh_modify    every 10 delay 0 check no\n"
        input_file += "read_data       conf.lmp\n"
        _masses = self.param_data['mass_map']
        # change the masses of atoms
        for ii, jj in enumerate(_masses):
            input_file += f"mass            {ii + 1} {jj}\n"
        # change the file name of graphs
        input_file += "pair_style deepmd "
        for kk in model_names:
            input_file += f"../{kk} "
        input_file += "out_freq 1 out_file model_devi.out\n"
        input_file += "pair_coeff\n"
        input_file += "velocity        all create 330.0 23456789\n"
        input_file += "fix             1 all nvt temp 330.0 330.0 0.05\n"
        input_file += "timestep        0.0005\n"
        input_file += "thermo_style    custom step pe ke etotal temp press vol\n"
        input_file += "thermo          1\n"
        input_file += "dump            1 all custom 1 dump.lammpstrj id type x y z fx fy fz\n"
        input_file += "rerun validate.xyz dump x y z box no format xyz\n"
        with open(os.path.join(work_path, 'task.md/input.lammps'), 'w') as f:
            f.write(input_file)
        return input_file

    def _fp_style(self):
        styles = {
            "vasp": "POSCAR",
            "cp2k": "coord.xyz",
        }
        return styles.get(self.param_data['fp_style'], None)

    def _fp_output_style(self):
        styles = {
            "vasp": "OUTCAR",
            "cp2k": "coord.xyz",
        }
        return styles.get(self.param_data['fp_style'], None)

    def _fp_output_dpgen(self):
        styles = {
            "vasp": "OUTCAR",
            "cp2k": "output",
            "qe": "output",
            "siesta": "output",
            "gaussian": "output",
            "pwmat": "REPORT",
        }
        return styles.get(self.param_data['fp_style'], None)

    def _fp_output_format(self):
        styles = {
            "vasp": "vasp/outcar",
            "cp2k": "cp2k/output",
        }
        return styles.get(self.param_data['fp_style'], None)

    def _load_task(self):
        self._read_record()
        self._read_param_data()
        self._read_machine_data()
        if self.step_code in [0, 3, 6]:
            self.state = 'Waiting'
        elif self.step_code in [1, 4, 7]:
            self.state = 'Parsing'
        else:
            self.state = 'Stopped'
        if self.step_code < 3:
            self.step = 'Training'
        elif self.step_code < 6:
            self.step = 'Exploring'
        else:
            self.step = 'Labeling'

    def _read_record(self):
        _record_path = os.path.join(self.path, self.record_file)
        with open(_record_path) as f:
            _final_step = f.readlines()[-1]
        self.iteration = int(_final_step.split()[0])
        self.step_code = int(_final_step.split()[1])

    def _read_param_data(self):
        _param_path = os.path.join(self.path, self.param_file)
        with open(_param_path) as f:
            self.param_data = json.load(f)

    def _read_machine_data(self):
        _param_path = os.path.join(self.path, self.machine_file)
        with open(_param_path) as f:
            self.machine_data = json.load(f)
