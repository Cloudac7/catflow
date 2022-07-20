import os

import dpdata
import numpy as np
from pathlib import Path
from glob import glob

from ase.io import read, write
from matplotlib import pyplot as plt

from miko.utils import logger
from miko.resources.submit import JobFactory
from miko.utils import LogFactory
from miko.utils.lammps import *
from miko.tesla.dpgen.base import DPAnalyzer



class DPLabelingAnalyzer(DPAnalyzer):

    def fp_group_distance(self, iteration, atom_group):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param atom_group:A tuple contains the index number of two selected atoms.
        :return: A plot of distance distribution.
        """
        dis_loc = []
        dis = []
        place = os.path.join(self.path, 'iter.' +
                             str(iteration).zfill(6), '02.fp')
        _stc_name = self._fp_style()
        for i in os.listdir(place):
            if os.path.exists(os.path.join(place, i, _stc_name)):
                dis_loc.append(i)
                stc = read(os.path.join(place, i, _stc_name))
                dis.append(stc.get_distance(
                    atom_group[0], atom_group[1], mic=True))
        diss = np.array(dis)
        plt.figure()
        plt.hist(diss, bins=np.arange(diss.min(), diss.max(), 0.01),
                 label=f'iter {int(iteration)}', density=True)
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
        place = os.path.join(self.path, 'iter.' +
                             str(iteration).zfill(6), '02.fp')
        _output_name = self._fp_style()
        for i in os.listdir(place):
            if os.path.exists(os.path.join(place, i, _output_name)):
                dis_loc.append(i)
                stc = read(os.path.join(place, i, _output_name))
                symbol_list = stc.get_chemical_symbols()
                ele_list_1 = [i for i in range(
                    len(symbol_list)) if symbol_list[i] == ele_group[0]]
                ele_list_2 = [i for i in range(
                    len(symbol_list)) if symbol_list[i] == ele_group[0]]
                min_dis = min([stc.get_distance(ii, jj, mic=True)
                               for ii in ele_list_1 for jj in ele_list_2])
                dis.append(min_dis)
        diss = np.array(dis)
        plt.figure(figsize=[16, 8], dpi=144)
        plt.hist(diss, bins=np.arange(1, 6, 0.01),
                 label=f'iter {int(iteration)}')
        plt.legend(fontsize=16)
        plt.xlabel("d(Å)", fontsize=16)
        plt.xticks(np.arange(0, 6, step=0.5), fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(
            f"Distibution of {ele_group[0]}-{ele_group[1]} distance", fontsize=16)
        return plt

    def fp_error_test(
            self,
            machine_name,
            resource_dict,
            iteration=None,
            test_model=None
    ):
        """Test your model quickly with the data generated from tesla.

        Parameters
        ----------
        resource_dict : dict
        machine_name : str
        iteration : str, optional
            Select the iteration of data for testing. Default: the latest one.
        test_model : str, optional
            Select the iteration of model for testing. Default: the latest one.

        Return
        ------

        """
        logger = LogFactory(__name__).get_log()

        location = self.path
        if iteration is None:
            if self.step_code < 7:
                iteration = self.iteration - 1
            else:
                iteration = self.iteration
        logger.info("Preparing structures from FP runs.")
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
            logger.debug(f"Task: {oo}")
            sys = dpdata.LabeledSystem(os.path.join(
                oo, _dpgen_output), fmt=_dpdata_format)
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
        write(os.path.join(quick_test_dir, 'task.md/validate.xyz'),
              stcs, format='extxyz')
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
        self._fp_generate_error_test(
            work_path=quick_test_dir, model_dir=model_dir)
        if not os.path.exists(os.path.join(quick_test_dir, 'task.md/conf.lmp')):
            _lmp_data = glob(os.path.join(location, n_iter,
                                          '01.model_devi', 'task*', 'conf.lmp'))[0]
            os.symlink(_lmp_data, os.path.join(
                quick_test_dir, 'task.md/conf.lmp'))
        logger.info("Quick tests task submitting.")
        job = self.md_single_task(
            work_path=quick_test_dir,
            model_path=model_dir,
            numb_models=self.param_data['numb_models'],
            forward_files=['conf.lmp', 'input.lammps', 'validate.xyz'],
            backward_files=['model_devi.out', 'energy.log',
                            'quick_test.log', 'quick_test.err', 'dump.lammpstrj'],
            outlog='quick_test.log',
            errlog='quick_test.err',
            machine_name=machine_name,
            resource_dict=resource_dict
        )
        job.run_submission()
        logger.info("Quick tests finished.")
        quick_test_result_dict = self._fp_error_test_result(
            quick_test_dir, atom_numb, dft_energy, dft_force)
        fig = self._fp_error_test_plot(iteration, **quick_test_result_dict)
        return quick_test_result_dict, fig

    @staticmethod
    def _fp_error_test_result(quick_test_dir, atom_numb, dft_energy, dft_force):
        md_energy = np.loadtxt(os.path.join(
            quick_test_dir, 'task.md/energy.log'), usecols=3)
        _md_stc = read(os.path.join(
            quick_test_dir, 'task.md/dump.lammpstrj'), index=':', format='lammps-dump-text')
        md_force = np.array([ss.get_forces() for ss in _md_stc])
        dft_energy_per_atom = dft_energy / atom_numb
        md_energy_per_atom = md_energy / atom_numb
        energy_per_atom_rmse = np.sqrt(
            np.mean((md_energy - dft_energy) ** 2)) / atom_numb
        md_force_r = np.ravel(md_force)
        dft_force_r = np.ravel(dft_force)
        force_rmse = np.sqrt(np.mean((md_force_r - dft_force_r) ** 2))
        result_dict = {
            "dft_energy_per_atom": dft_energy_per_atom,
            "md_energy_per_atom": md_energy_per_atom,
            "energy_per_atom_rmse": energy_per_atom_rmse,
            "dft_force": dft_force_r,
            "md_force": md_force_r,
            "force_rmse": force_rmse
        }
        return result_dict

    @staticmethod
    def _fp_error_test_plot(iteration, dft_energy_per_atom, md_energy_per_atom, energy_per_atom_rmse, dft_force,
                            md_force, force_rmse):
        from matplotlib.offsetbox import AnchoredText

        fig, axs = plt.subplots(1, 2)
        # Plot of energy error
        axs[0].scatter(dft_energy_per_atom, md_energy_per_atom,
                       s=5, label=f'Iter. {iteration}')
        _x = np.linspace(np.min(dft_energy_per_atom) - 0.05,
                         np.max(dft_energy_per_atom) + 0.05, 10)
        axs[0].plot(_x, _x, 'r--')

        box = AnchoredText(
            s=f'RMSE={energy_per_atom_rmse} (eV/atom)', prop=dict(fontsize=14),
            loc="upper left", frameon=False)
        box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axs[0].add_artist(box)
        axs[0].set_title(f'Energy error', fontsize=14)
        axs[0].set_xlabel(r'$e_{DFT}$ (eV/atom)', fontsize=14)
        axs[0].set_ylabel(r'$e_{DPMD}$ (eV/atom)', fontsize=14)
        axs[0].set_aspect('equal')
        # Plot of force error
        axs[1].scatter(md_force, dft_force, s=5, label=f'Iter. {iteration}')
        _y = np.linspace(np.min(dft_force) - 0.05,
                         np.max(dft_force) + 0.05, 10)
        axs[1].plot(_y, _y, 'r--')
        axs[1].text(np.min(dft_force) - 0.05, np.max(dft_force) +
                    0.05, f'RMSE={force_rmse} (eV/Å)', fontsize=14)
        axs[1].set_title(f'Force error', fontsize=14)
        axs[1].set_xlabel(r'$f_{DFT}$ (eV/Å)', fontsize=14)
        axs[1].set_ylabel(r'$f_{DPMD}$ (eV/Å)', fontsize=14)
        axs[1].set_aspect('equal')
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

        input_file += \
            """
            pair_coeff
            velocity        all create 330.0 23456789
            fix             1 all nvt temp 330.0 330.0 0.05
            timestep        0.0005
            thermo_style    custom step pe ke etotal temp press vol
            thermo          1
            dump            1 all custom 1 dump.lammpstrj id type x y z fx fy fz
            variable temp equal temp
            variable etotal equal etotal
            variable pe equal pe
            variable ke equal ke
            variable step equal step
            fix sys_info all print 1 "${step} ${temp} ${etotal} ${pe} ${ke}" title "#step temp etotal pe ke" file energy.log
            """
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
            "vasp": "vasprun.xml",
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
