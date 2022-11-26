import os

import numpy as np
from pathlib import Path

from ase.io import read
from matplotlib import pyplot as plt

from miko.utils import logger
from miko.utils import LogFactory
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
