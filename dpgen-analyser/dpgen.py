import os
import json
import numpy as np
import pandas as pd
from glob import glob
from ase.io import read, write
from matplotlib import pyplot as plt
from dpgen.dispatcher.Dispatcher import Dispatcher
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
        self.path = path
        self.param_file = param_file
        self.machine_file = machine_file
        self.record_file = record_file
        self._load_task()

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
            with open(f'{task}/model_devi.out', 'r') as f:
                lines = f.readlines()[1:]
            dump_freq = int(lines[2].split()[0]) - int(lines[1].split()[0])
            max_devi_f = np.array([p.split()[4] for p in lines]).astype('float')
            max_devi_e = np.array([p.split()[3] for p in lines]).astype('float')
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
            pot_energy = np.array([p.split()[2] for p in lines]).astype('float')
            all_data.append({
                'iter': n_iter,
                'temp': temp,
                'max_devi_e': max_devi_e,
                'max_devi_f': max_devi_f,
                'task': task,
                't_freq': dump_freq,
                'pot_energy': pot_energy
            })
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
        try:
            df = pd.read_pickle(pkl_path)
        except:
            raise RuntimeError(f"Data of iteration {str(iteration).zfill(2)} does not exist.")
        return df

    def md_single_iter(
            self,
            iteration,
            temps,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            xlimit=1e3,
            ylimit=0.50,
            log=False):
        """
        Generate a plot of model deviation in each iteration
        :param iteration: The iteration
        :param temps: A list of temperatures plotted.
        :param f_trust_lo: The lower limit of max_deviation_force.
        :param f_trust_hi: The higher limit of max_deviation_force.
        :param xlimit: Choose the limit of x axis.
        :param ylimit: Choose the limit of y axis.
        :param log: Choose whether log scale used. Default: False.
        :return: A plot for different temperatures
        """
        location = os.path.join(self.path, f'data_pkl/data_{str(iteration).zfill(2)}.pkl')
        if os.path.exists(location):
            df = self.md_set_load_pkl(iteration=iteration)
        else:
            df = self.md_set_pd(iteration=iteration)
        if isinstance(temps, (list, tuple)):
            num_temp = len(temps)
        elif isinstance(temps, (int, float)):
            num_temp = 0
            temps = [temps]
        elif isinstance(temps, str):
            num_temp = 0
            temps = [str(temps)]
        else:
            raise TypeError("temps should be a value or a list of value.")
        fig = plt.figure(figsize=[12, 4 * num_temp], constrained_layout=True)
        gs = fig.add_gridspec(num_temp, 3)
        for i, temp in enumerate(temps):
            partdata = df[df['temp'] == temp]
            # left part
            fig_left = fig.add_subplot(gs[i, :-1])
            parts = partdata[partdata['iter'] == 'iter.' + str(iteration).zfill(6)]
            for j, [temp, part] in enumerate(parts.groupby('temp')):
                mdf = np.array(list(part['max_devi_f']))
                t_freq = np.average(part['t_freq'])
                dupt = np.tile(range(mdf.shape[1]) * t_freq, mdf.shape[0])
                flatmdf = np.ravel(mdf)
                print(f"max devi of F is :{max(flatmdf)} ev/Å on {temp} K")
                fig_left.scatter(dupt, flatmdf, s=80, alpha=0.3, color='red', label=f'{int(temp)} K', marker='o')
            fig_left.set_xlim(0, xlimit)
            if not log:
                fig_left.set_ylim(0, ylimit)
            else:
                fig_left.set_yscale('log')
            fig_left.hlines(f_trust_lo, 0, xlimit, linestyles='dashed')
            fig_left.hlines(f_trust_hi, 0, xlimit, linestyles='dashed')
            fig_left.set_xlabel('Simulation time (fs)')
            fig_left.set_ylabel('$\sigma_{f}^{max}$ (ev/Å)')
            fig_left.legend()
            fig_left.set_title(f'Iteration {iteration}')
            # right part
            fig_right = fig.add_subplot(gs[i, -1])
            fig_right.hist(
                flatmdf,
                bins=np.linspace(0, ylimit, 1001),
                orientation='horizontal',
                density=True,
                color='red')
            fig_right.set_title('Distribution of Deviation')
            fig_right.set_xlim(0, 150)
            fig_right.set_ylim(0, ylimit)
            fig_right.hlines(f_trust_lo, 1, 150, linestyles='dashed')
            fig_right.hlines(f_trust_hi, 1, 150, linestyles='dashed')
            fig_right.set_xticklabels([])
            fig_right.set_yticklabels([])
        return plt

    def md_multi_iter(
            self,
            iterations,
            temps,
            f_trust_lo=0.10,
            f_trust_hi=0.30,
            x_lower_limit=0,
            x_higher_limit=1e3,
            y_limit=None,
            x_log=False,
            y_log=False
    ):
        """
        Analyse trajectories for different temperatures.
        :param iterations: Iterations selected, which should be iterable.
        :param temps: Temp(s) selected.
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
        if isinstance(temps, (list, tuple)):
            num_temp = len(temps)
        elif isinstance(temps, (int, float)):
            num_temp = 0
            temps = [temps]
        elif isinstance(temps, str):
            num_temp = 0
            temps = [str(temps)]
        else:
            raise TypeError("temps should be a value or a list of value.")
        df = pd.concat(frames)
        plt.figure(figsize=[24, 8 * num_temp])
        for i, temp in enumerate(temps):
            plt.subplot(num_temp, 1, i + 1)
            for k in iterations:
                partdata = df[df['temp'] == temp]
                parts = partdata[partdata['iter'] == 'iter.' + str(k).zfill(6)]
                for j, [temp, part] in enumerate(parts.groupby('temp')):
                    mdf = np.array(list(part['max_devi_f']))
                    t_freq = np.average(part['t_freq'])
                    dupt = np.tile(range(mdf.shape[1]) * t_freq, mdf.shape[0])
                    flatmdf = np.ravel(mdf)
                    plt.scatter(dupt, flatmdf, s=80, alpha=0.3, label=f'iter {int(k)}', marker='o')
            plt.xlim(x_lower_limit, x_higher_limit)
            if y_limit is not None:
                plt.ylim(0, y_limit)
            if x_log:
                plt.xscale('log')
            if y_log:
                plt.yscale('log')
            plt.hlines(f_trust_lo, x_lower_limit, x_higher_limit, linestyles='dashed')
            plt.hlines(f_trust_hi, x_lower_limit, x_higher_limit, linestyles='dashed')
            plt.xlabel('Simulation time (fs)', fontsize=24)
            plt.ylabel('$\sigma_{f}^{max}$ (ev/Å)', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.legend(fontsize=24)
            plt.title(f'{temp} K', fontsize=24)
        plt.tight_layout()
        return plt

    def md_single_task(self, work_path, model_path, numb_models=4, **kwargs):
        """
        Submit your own md task with the help of dpgen.
        :param work_path: The dir contains your md tasks. The tasks must start from "task.".
        :param model_path: The path of models contained for calculation.
        :param numb_models: The number of models selected.
        :return:
        """
        mdata = self.machine_data['model_devi'][0]
        all_task = glob(os.path.join(work_path, "task.*"))
        all_task.sort()
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

    def fp_group_distance(self, iteration, atom_group):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param atom_group:A tuple contains the index number of two selcted atoms.
        :return: A plot of distance distribution.
        """
        dis_loc = []
        dis = []
        place = os.path.join(self.path, 'iter.'+str(iteration).zfill(6), '02.fp')
        _stc_name = self._fp_style()
        for i in os.listdir(place):
            if os.path.exists(os.path.join(place, i, _stc_name)):
                dis_loc.append(i)
                stc = read(os.path.join(place, i, _stc_name))
                dis.append(stc.get_distance(atom_group[0], atom_group[1], mic=True))
        diss = np.array(dis)
        plt.figure()
        plt.hist(diss, bins=np.arange(1, 1.5, 0.01), label=f'iter {int(it)}', density=True)
        plt.legend(fontsize=16)
        plt.xlabel("d(Å)", fontsize=16)
        plt.xticks(np.arange(1, 1.51, step=0.1), fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("Distibution of distance", fontsize=16)
        return plt

    def fp_element_distance(self, iteration, ele_group):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param ele_group:A tuple contains the index number of two selcted elements.
        :return: A plot of distance distribution.
        """
        dis = []
        dis_loc = []
        place = os.path.join(self.path, 'iter.'+str(iteration).zfill(6), '02.fp')
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
        stcs = [read(os.path.join(tt, _stc_file)) for tt in task_list ]
        write(os.path.join(quick_test_dir, 'task.md/validate.xyz'), stcs, format='extxyz')
        if test_model is None:
            test_model = iteration
        model_iter = 'iter.' + str(test_model).zfill(6)
        model_dir = os.path.join(location, model_iter, '00.train')
        self._fp_generate_error_test(work_path=quick_test_dir, model_dir=model_dir)
        if not os.path.exists(os.path.join(quick_test_dir, 'task.md/conf.lmp')):
            _lmp_data = glob(os.path.join(location, n_iter, '01.model_devi', 'task*', 'conf.lmp'))[0]
            os.symlink(_lmp_data, os.path.join(quick_test_dir, 'task.md/conf.lmp'))
        self.md_single_task(
            work_path=quick_test_dir,
            model_path=model_dir,
            numb_models=self.param_data['numb_models'],
            forward_files=['conf.lmp', 'input.lammps', 'validate.xyz'],
            backward_files=['model_devi.out', 'quick_test.log', 'quick_test.err', 'dump.lammpstrj'],
            outlog='quick_test.log',
            errlog='quick_test.err'
        )

    def _fp_generate_error_test(self, work_path, model_dir):
        model_list = glob(os.path.join(model_dir, 'graph*pb'))
        model_list.sort()
        model_names = [os.path.basename(i) for i in model_list]
        input = "units           metal\n"
        input += "boundary        p p p\n"
        input += "atom_style      atomic\n"
        input += "\n"
        input += "neighbor        2.0 bin\n"
        input += "neigh_modify    every 10 delay 0 check no\n"
        input += "read_data       conf.lmp\n"
        _masses = self.param_data['mass_map']
        # change the masses of atoms
        for ii, jj in enumerate(_masses):
            input += f"mass            {ii + 1} {jj}\n"
        # change the file name of graphs
        input += "pair_style deepmd "
        for kk in model_names:
            input += f"../{kk} "
        input += "out_freq 1 out_file model_devi.out\n"
        input += "pair_coeff\n"
        input += "velocity        all create 330.0 23456789\n"
        input += "fix             1 all nvt temp 330.0 330.0 0.05\n"
        input += "timestep        0.0005\n"
        input += "thermo_style    custom step pe ke etotal temp press vol\n"
        input += "thermo          1\n"
        input += "dump            1 all custom 1 dump.lammpstrj id type x y z fx fy fz\n"
        input += "rerun validate.xyz dump x y z box no format xyz\n"
        with open(os.path.join(work_path, 'task.md/input.lammps'), 'w') as f:
            f.write(input)
        return input

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
        try:
            _record_path = os.path.join(self.path, self.record_file)
            with open(_record_path) as f:
                _final_step = f.readlines()[-1]
            self.iteration = int(_final_step.split()[0])
            self.step_code = int(_final_step.split()[1])
        except:
            raise FileNotFoundError('Record file record.dpgen not found')

    def _read_param_data(self):
        _param_path = os.path.join(self.path, self.param_file)
        with open(_param_path) as f:
            self.param_data = json.load(f)

    def _read_machine_data(self):
        _param_path = os.path.join(self.path, self.machine_file)
        with open(_param_path) as f:
            self.machine_data = json.load(f)