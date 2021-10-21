from datetime import datetime
import logging
import numpy as np
import os
from glob import glob
from ase.io import read, write
from dscribe.descriptors import SOAP
from ase import Atoms
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

from dpana.dpgen import DPTask
from dpgen.generator.run import make_vasp_incar


def structure_collection(path, symbols):
    """
    generate structure from data_set
    """
    if type(path) is str:
        ls = glob(path)
    elif type(path) is list:
        ls = []
        for i in path:
            ls += glob(i)
    else:
        ls = []
    stcs = []
    for item in ls:
        _coord = np.load(os.path.join(item, 'coord.npy'))
        _box = np.load(os.path.join(item, 'box.npy'))
        for i, j in enumerate(_coord):
            j = np.reshape(j, (-1, 3))
            s = Atoms(symbols=symbols, positions=j)
            _cell = np.reshape(_box[i], (-1, 3))
            s.set_cell(_cell)
            s.set_pbc([1, 1, 1])
            stcs.append(s)
    return stcs


def lammps_collection(path, symbols, log_path=None):
    """
    collect lammps structures as list of ase.Atoms
    """
    _add = []
    _idx = []
    for i in glob(os.path.join(path, 'traj', '[!0]*.lammpstrj')):
        try:
            _s = read(i, format='lammps-dump-text')
            _add.append(_s)
            _idx.append(i)
        except FileNotFoundError:
            continue
    for s in _add:
        s.set_chemical_symbols(symbols)
    if log_path is not None:
        with open(os.path.join(log_path, 'lmp_col.out'), 'w') as f:
            for i in range(len(_idx)):
                line = _idx[i] + '\n'
                f.write(line)
    return _add


class SOAPScreening(DPTask):
    """
    Create a screening task with SOAP descriptor.
    """

    def __init__(self, path, param_file, machine_file, record_file, symbols, rcut, nmax, lmax, sigma=1.0,
                 periodic=True, **kwargs):
        super().__init__(path, param_file, machine_file, record_file)
        self.soap_descriptor = SOAP(
            species=self.species,
            rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma,
            average="inner", crossover=True,
            periodic=periodic
        )
        self.symbols = symbols
        self.soap_dcut = kwargs.get('dcut', 0.08)

    @property
    def species(self):
        return self.param_data['type_map']

    def soap_from_npy(self, path):
        """
        soap from data set
        """
        ori = structure_collection(path=path, symbols=self.symbols)
        soap_ori = np.zeros((len(ori), self.soap_descriptor.get_number_of_features()))
        for i, s in enumerate(ori):
            _soap_item = self.soap_descriptor.create(s)
            soap_ori[i] = _soap_item
        return soap_ori

    def soap_from_md(self, iteration=0, log_path=None):
        """
        generate new soap descriptor within structures added.
        """
        path = os.path.join(self.path, f'iter.{str(iteration).zfill(6)}/01.model_devi/task.*')
        stc_md = lammps_collection(path, self.symbols, log_path)
        soap_md = np.zeros((len(stc_md), self.soap_descriptor.get_number_of_features()))
        for i, s in enumerate(stc_md):
            _soap_item = self.soap_descriptor.create(s)
            soap_md[i] = _soap_item
        return soap_md, stc_md

    def get_inner_diss_matrix(self, soap_add):
        """
        generate distance matrix (based on soap) between the new structures 

        return:
            inner_diss_matrix[i][j]: distance between i-th and j-th structure in soap_add
        """
        inner_diss_matrix = np.zeros((len(soap_add),len(soap_add))) 
        for i, item in enumerate(soap_add):
            for j in range(i+1, len(soap_add)):
                inner_diss_matrix[i][j] = np.linalg.norm(item - soap_add[j])
                inner_diss_matrix[j][i] = inner_diss_matrix[i][j]
            inner_diss_matrix[i][i] = 0
        return inner_diss_matrix

    def get_inter_diss_matrix(self, soap_add, soap_ori):
        """
        generate distance matrix (based on soap) between the new structures and the original structrues

        return:
            inter_diss_matrix[i][j]: distance between i-th structure in soap_add and j-th structure in soap_add
        """
        inner_diss_matrix = np.zeros((len(soap_add),len(soap_ori))) 
        for i, i_item in enumerate(soap_add):
            for j, j_item in enumerate(soap_ori): 
                inter_diss_matrix[i][j] = np.linalg.norm(i_item - j_item)
        return inner_diss_matrix

    def soap_compare(self, soap_ori, soap_add, plot=False, **kwargs):
        """compare added soap with original one

        Args:
            soap_ori: SOAP descriptor of original structures
            soap_add: SOAP descriptor of structures to be added
            plot: choose whether make the plot of distribution
        """
        dis_all = []
        for i in tqdm(
                iterable=soap_add,
                desc='Structure adding'
        ):
            diss = []
            for j in soap_ori:
                dis = np.linalg.norm(i - j)
                diss.append(dis)
            dis_all.append(min(diss))
        if plot is True:
            plt.figure()
            plt.hist(dis_all, bins=100)
            _time = datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S')
            _plot_path = kwargs.get(
                'plot_path',
                os.path.join(self.path, f'soap_compare_{_time}.png')
            )
            plt.savefig(os.path.abspath(_plot_path))
        dis_all = np.array(dis_all)
        dcut = self.soap_dcut
        _select = np.where(dis_all >= dcut)[0]
        select_ratio = len(_select) / len(dis_all)
        logging.info('Ratio of available structures: {}'.format(select_ratio))
        return dis_all

    @staticmethod
    def soap_pick_idx(inner_diss_matrix, inter_diss_matrix, fp_task_max=100, fp_task_min=5):
        """
        find (fp_task_max) structures with farthest distance to other stuctures (both original and added structure)
        used when the number of candidate structures is larger than the fp_task_max

        Args:
            inner_diss_matrix: (len(soap_add),len(soap_add))
            inter_diss_matrix: (len(soap_add),len(soap_ori))

        return:
            top_k_idx: list of index of selected structures 
        """
        top_k_idx = np.zeros(fp_task_max)
        for i in range(fp_task_max):
            diss = np.min(inter_diss_matrix, axis=1)

        return top_k_idx 

        frc = np.delete(_frc, dummy_ids, axis=1)

    def soap_pick_idx(soap_dis, fp_task_max=100, fp_task_min=5):
        """
        pick farthest structures from soap_dis
        """
        top_k_idx = soap_dis.argsort()[(0 - fp_task_max):]
        logging.info('Structures decided.')
        if len(top_k_idx) >= fp_task_min:
            return top_k_idx
        else:
            return None

    def _fp_gen_from_soap(self, stc, stc_idx, incar_path, potcars, sys_idx=0, iteration=0, log_file=None):
        """
        generate vasp fp from idx
        """
        path = os.path.join(self.path, f'iter.{str(iteration).zfill(6)}/02.fp')
        os.makedirs(path, exist_ok=True)

        # make INCAR
        if incar_path is None:
            incar_path = os.path.join(path, 'INCAR')
            make_vasp_incar(jdata=self.param_data, filename=incar_path)
        else:
            incar_path = os.path.abspath(incar_path)

        # make POTCAR
        fp_pp_path = self.param_data['fp_pp_path']
        pot_path = os.path.join(path, 'POTCAR')
        with open(pot_path, 'w') as fp_pot:
            for jj in potcars:
                with open(os.path.join(fp_pp_path, jj)) as fp:
                    fp_pot.write(fp.read())

        # read index of lammps
        if log_file is not None:
            with open(os.path.join(log_file, 'lmp_col.out')) as f:
                dir_list = f.readlines()[:]
        for i, j in enumerate(stc_idx):
            td = os.path.join(path, f'task.{str(sys_idx).zfill(3)}.{str(i).zfill(6)}')
            os.makedirs(td, exist_ok=True)
            _s = stc[j]
            write(os.path.join(td, 'POSCAR'), _s)
            shutil.copyfile(incar_path, os.path.join(td, 'INCAR'))
            shutil.copyfile(pot_path, os.path.join(td, 'POTCAR'))
            job_path = os.path.abspath(os.path.join(dir_list[j], "../.."))
            job_path = os.path.join(job_path, 'job.json')
            os.symlink(job_path, os.path.join(td, 'job.json'))

    def fp_make_screen(self, iteration=None, **kwargs):
        """

        Make fp tasks from screening steps.

        Args:
            iteration: The iteration of screening step
        """
        path = self.path
        if iteration is None:
            if self.step_code > 4:
                iteration = self.iteration
            else:
                iteration = self.iteration - 1

        init_data_prefix = self.param_data['init_data_prefix']
        init_data_sys = self.param_data['init_data_sys']
        init_list = [os.path.join(init_data_prefix, k, 'set.*') for k in init_data_sys]
        add_list = [os.path.join(path, f'iter.{str(k).zfill(6)}/02.fp/data.*/set.*') for k in range(iteration - 1)]
        path_list = init_list + add_list

        soap_ori = self.soap_from_npy(
            path=path_list
        )
        soap_add, stc_md = self.soap_from_md(
            iteration=iteration,
            log_path=os.path.join(path, f'iter.{str(iteration).zfill(6)}/01.model_devi')
        )
        dis_all = self.soap_compare(
            soap_ori,
            soap_add,
            plot=kwargs.get('plot', False),
            plot_path=os.path.join(self.path, f'iter.{str(iteration).zfill(6)}/01.model_devi/soap_compare.png')
        )

        # read json to get the parameters
        idx = self.soap_pick_idx(
            dis_all,
            fp_task_max=self.param_data['fp_task_max'],
            fp_task_min=self.param_data['fp_task_min']
        )
        self._fp_gen_from_soap(
            stc=stc_md,
            stc_idx=idx,
            incar_path=self.param_data.get('fp_incar', None),
            potcars=self.param_data.get('fp_pp_files'),
            sys_idx=self.param_data['model_devi_jobs'][iteration]['sys_idx'][0],
            iteration=iteration,
            log_file=os.path.join(path, f'iter.{str(iteration).zfill(6)}/01.model_devi')
        )
