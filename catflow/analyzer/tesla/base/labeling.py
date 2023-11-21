import os

import numpy as np
from pathlib import Path
from typing import List, Literal

from ase.io import read
from matplotlib import pyplot as plt

from catflow.utils.log_factory import logger
from catflow.utils.log_factory import LogFactory
from catflow.analyzer.tesla.base.task import BaseTask, BaseAnalyzer
from catflow.analyzer.graph.plotting import canvas_style



class LabelingAnalyzer(BaseAnalyzer):

    def __init__(
        self, dp_task: BaseTask, 
        fp_style: Literal['vasp', 'cp2k'] = 'vasp'
    ) -> None:
        super().__init__(dp_task)
        self.fp_style = fp_style
    
    def get_fp_tasks(self, iteration: int = 0) -> List[Path]:
        ...
        return []

    def fp_group_distance(self, iteration, atom_group, **kwargs):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param atom_group:A tuple contains the index number of two selected atoms.
        :return: A plot of distance distribution.
        """
        task_files = self.get_fp_tasks(iteration=iteration)

        dis_loc = []
        dis = []

        _stc_name = self._fp_style()
        for i in task_files:
            stc_file_path = i / _stc_name # type: ignore
            if stc_file_path.exists():
                dis_loc.append(i)
                stc = read(stc_file_path) 
                dis.append(stc.get_distance( # type: ignore
                    atom_group[0], atom_group[1], mic=True))
        diss = np.array(dis)

        fig, ax = plt.subplots()
        canvas_style(**kwargs)

        ax.hist(diss, bins=np.arange(diss.min(), diss.max(), 0.01), # type: ignore
                 label=f'iter {int(iteration)}', density=True)
        ax.legend(fontsize=16)
        ax.set_xlabel("d(Å)", fontsize=16)
        ax.set_xticks(np.arange(diss.min(), diss.max(), step=1.0), fontsize=16)
        ax.set_yticks(fontsize=16)
        ax.set_title("Distibution of distance", fontsize=16)
        return fig

    def fp_element_distance(self, iteration, ele_group, **kwargs):
        """
        Analyse the distance of selected structures.
        :param iteration: The iteration selected.
        :param ele_group:A tuple contains the index number of two selected elements.
        :return: A plot of distance distribution.
        """
        dis = []
        dis_loc = []
        task_files = self.get_fp_tasks(iteration=iteration)
        _stc_name = self._fp_style()

        for i in task_files:
            stc_file_path = i / _stc_name # type: ignore
            if stc_file_path.exists():
                dis_loc.append(i)
                stc = read(stc_file_path)
                symbol_list = stc.get_chemical_symbols() # type: ignore
                ele_list_1 = [i for i in range(
                    len(symbol_list)) if symbol_list[i] == ele_group[0]]
                ele_list_2 = [i for i in range(
                    len(symbol_list)) if symbol_list[i] == ele_group[0]]
                min_dis = min([stc.get_distance(ii, jj, mic=True) # type: ignore
                               for ii in ele_list_1 for jj in ele_list_2])
                dis.append(min_dis)
        diss = np.array(dis)

        fig, ax = plt.subplots()
        canvas_style(**kwargs)

        ax.hist(diss, bins=np.arange(1, 6, 0.01),
                 label=f'iter {int(iteration)}')
        ax.legend(fontsize=16)
        ax.set_xlabel("d(Å)", fontsize=16)
        ax.set_xticks(np.arange(0, 6, step=0.5), fontsize=16)
        ax.set_yticks(fontsize=16)
        ax.set_title(
            f"Distibution of {ele_group[0]}-{ele_group[1]} distance", fontsize=16)
        return fig
    
    def _fp_style(self):
        styles = {
            "vasp": "POSCAR",
            "cp2k": "coord.xyz",
        }
        return styles.get(self.fp_style, None)

    def _fp_output_style(self):
        styles = {
            "vasp": "vasprun.xml",
            "cp2k": "coord.xyz",
        }
        return styles.get(self.fp_style, None)

    def _fp_output_dpgen(self):
        styles = {
            "vasp": "OUTCAR",
            "cp2k": "output",
            "qe": "output",
            "siesta": "output",
            "gaussian": "output",
            "pwmat": "REPORT",
        }
        return styles.get(self.fp_style, None)

    def _fp_output_format(self):
        styles = {
            "vasp": "vasp/outcar",
            "cp2k": "cp2k/output",
        }
        return styles.get(self.fp_style, None)
