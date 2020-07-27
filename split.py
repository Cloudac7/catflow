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

class DeepMD(object):
    def __init__(self, energy_xyz, cell):
        self.atoms = read(energy_xyz, index=':-1')  # drop the last one
        self.n_atom = len(self.atoms[0].numbers)
        self.n_frame = len(self.atoms)

        self.box = np.empty([self.n_frame, 9], dtype='float32')
        self.energy = np.empty(self.n_frame, dtype='float32')
        self.coord = np.empty([self.n_frame, self.n_atom*3], dtype='float32')
        self.struct = []

        for i, atom in enumerate(self.atoms):
            self.energy[i] = atom.info['E'] * 27.2114  # energy
            atom.set_cell(cell)
            struct = AseAtomsAdaptor.get_structure(atom)
            struct = struct.get_reduced_structure()
            self.coord[i] = struct.cart_coords.reshape(self.n_atom*3)  # coord
            self.box[i] = cell.reshape(9)
            self.struct.append(AseAtomsAdaptor.get_structure(self.atoms[i]))
            self.struct[i] = self.struct[i].get_reduced_structure()
        self.symbol_set = self.struct[0].symbol_set
        self.sym_dict = dict(zip(self.symbol_set, range(len(self.symbol_set))))

    def write_input(self, *kargs):
        set_path = 'data/set.000'
        os.makedirs(set_path, exist_ok=True)
        np.save(set_path+'/energy.npy', self.energy)
        np.save(set_path+'/coord.npy', self.coord)
        np.save(set_path+'/box.npy', self.box)
        type_raw = [str(self.sym_dict[specie.name])
                    for specie in self.struct[0].species]
        with open('data/type.raw', 'w+') as f:
            f.write(' '.join(type_raw))
            
lattice = Lattice.from_lengths_and_angles((15, 15, 15), (90, 90, 90))
cell = lattice.matrix
dm = DeepMD(energy_xyz='Cu13-pos-1.xyz', cell=cell)
dm.write_input()
