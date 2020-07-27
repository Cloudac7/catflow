import os
import json
import numpy as np
import pandas as pd
from ase.io import read
from matplotlib import pyplot as plt
from glob import glob


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
        _output_name = self._fp_style()
        for i in os.listdir(place):
            if os.path.exists(os.path.join(place, i, _output_name)):
                dis_loc.append(i)
                stc = read(os.path.join(place, i, _output_name))
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

    def _fp_style(self):
        styles = {
            "vasp": "OUTCAR",
            "cp2k": "output",
            "qe": "output",
            "siesta": "output",
            "gaussian": "output",
            "pwmat": "REPORT",
        }
        return styles.get(self.jdata['fp_style'], None)

    def _load_task(self):
        self._read_record()
        self._read_jdata()
        if self.step_code in [0, 3, 6]:
            self.state = 'Waiting'
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

    def _read_jdata(self):
        try:
            _param_path = os.path.join(self.path, self.param_file)
            with open(_param_path) as f:
                self.jdata = json.load(f)
        except:
            raise FileNotFoundError('Record file param.json not found')
