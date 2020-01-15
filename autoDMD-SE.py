#!/usr/bin/env python3
import json
import os
import numpy as np

from ase.io import read, write
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor

from collections import Counter
from itertools import product

from sklearn.model_selection import train_test_split

class DeepMD(object):
    def __init__(self, energy_xyz, force_xyz, cell):
        # todo system

        self.atoms = read(energy_xyz, index=':-1')  # drop the last one
        self.forces = read(force_xyz, index=':-1') # drop the last one
        self.n_atom = len(self.atoms[0].numbers)
        self.n_frame = len(self.atoms)

        self.box = np.empty([self.n_frame, 9], dtype='float32')
        self.energy = np.empty(self.n_frame, dtype='float32')
        self.coord = np.empty([self.n_frame, self.n_atom*3], dtype='float32')
        self.force = np.empty([self.n_frame, self.n_atom*3], dtype='float32')

        for i, atom in enumerate(self.atoms):
            self.energy[i] = atom.info['E'] * 27.2114  # energy
            self.force[i] = self.forces[i].get_positions().reshape(self.n_atom*3) * 27.2114 / 0.52918 # force
            atom.set_cell(cell)
            struct = AseAtomsAdaptor.get_structure(atom)
            struct = struct.get_reduced_structure()
            self.coord[i] = struct.cart_coords.reshape(self.n_atom*3)  # coord
            self.box[i] = cell.reshape(9)

        self.struct = AseAtomsAdaptor.get_structure(self.atoms[0])
        self.struct = self.struct.get_reduced_structure()
        self.symbol_set = self.struct.symbol_set
        self.sym_dict = dict(zip(self.symbol_set, range(len(self.symbol_set))))

        self.train_json = {
            "_comment": " model parameters",
            "use_smooth": True,
            "sel_a": [46, 92],
            "rcut_smth": 2.00,
            "rcut": 6.00, #6.00
            "filter_neuron": [25, 50, 100],
            "filter_resnet_dt": False,
            "axis_neuron": 16,
            "fitting_neuron": [240, 240, 240],
            "fitting_resnet_dt": True,
            "coord_norm": True,
            "type_fitting_net": False,

            "systems": ["../data"],  # different system: water, ice, ...
            "set_prefix": "set",
            "stop_batch": 400000,
            "batch_size": 1,
            "start_lr": 0.005,
            "decay_steps": 5000,
            "decay_rate": 0.95,

            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,

            "seed": 1,

            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "numb_test": 100,
            "save_freq": 100,
            "save_ckpt": "model.ckpt",
            "load_ckpt": "model.ckpt",
            "disp_training": True,
            "time_training": True,
            "profiling": False,
            "profiling_file": "timeline.json",
        }

    def analyze_neighbours(self, Rcut=6.1, n_nearest=2):
        self.sel_a = np.ones([len(self.symbol_set)]).astype('int')
        self.sel_max = np.ones([len(self.symbol_set)]).astype('int')
        # self.train_json['rcut'] = float("%.1f" % np.amax(self.struct.distance_matrix)) + 0.2
        # self.train_json['rcut_smth'] = self.train_json['rcut'] - 0.2
        Rcut = self.train_json['rcut'] + 0.1
        modified_dm = self.struct.distance_matrix + \
            np.identity(self.struct.num_sites) * Rcut
        near_idx = list(map(lambda d: np.argsort(
            d)[np.where(np.sort(d) < Rcut)], modified_dm))
        num_neighbors = list(map(lambda x: len(x), near_idx))
        for i, sym in enumerate(self.struct.symbol_set):
            sym_idx = self.struct.indices_from_symbol(sym)
            self.sel_max[i] = np.array(num_neighbors)[np.array(sym_idx)].max()
            self.sel_a[i] = self.sel_max[i]+1

        self.train_json['sel_a'] = list([int(i) for i in self.sel_a])

    def write_input(self, fitting_neuron=[240, 240, 240], set_size=128, *kargs):
        # if self.force:
        # set
        Etrain, Etest, Coordtrain, Coordtest, Forcetrain, Forcetest = train_test_split(
            self.energy, self.coord, self.force, test_size=0.2)
        set_size = set_size
        n_set = len(Etrain)//set_size
        ## train
        for i in range(n_set):
            set_path = 'data/set.' + str(i).zfill(3)
            os.makedirs(set_path, exist_ok=True)
            np.save(set_path+'/energy.npy', Etrain[set_size*i:set_size*(i+1)])
            np.save(set_path+'/coord.npy', Coordtrain[set_size*i:set_size*(i+1)])
            np.save(set_path+'/force.npy', Forcetrain[set_size*i:set_size*(i+1)])
            np.save(set_path+'/box.npy', self.box[set_size*i:set_size*(i+1)])
        ## test
        test_path = 'data/set.' + str(n_set).zfill(3)
        os.makedirs(test_path, exist_ok=True)
        np.save(test_path+'/energy.npy', Etest)
        np.save(test_path+'/coord.npy', Coordtest)
        np.save(test_path+'/force.npy', Forcetest)
        np.save(test_path+'/box.npy', self.box[:len(Etest)])

        # train.json
        self.train_json['fitting_neuron'] = fitting_neuron
        os.makedirs('train', exist_ok=True)
        with open('train/train.json', 'w+') as f:
            json.dump(self.train_json, f, indent=4)

        # type.raw
        type_raw = [str(self.sym_dict[specie.name])
                    for specie in self.struct.species]
        with open('data/type.raw', 'w+') as f:
            f.write(' '.join(type_raw))
            
lattice = Lattice.from_lengths_and_angles((15, 15, 15), (90, 90, 90))
cell = lattice.matrix
dm = DeepMD(energy_xyz='Au-O2-pos-1.xyz', force_xyz='Au-O2-frc-1.xyz', cell=cell)

dm.analyze_neighbours()
#  train.json
train_json = {
    "stop_batch": 1000000,
    "batch_size": 32,
    "decay_steps": 2000,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "numb_test": 100,
    "save_freq": 2000,
    "restart": False,
}
for k, v in train_json.items():
    dm.train_json[k] = v
dm.write_input(fitting_neuron=[240, 240, 240], set_size=128)
