import numpy as np
import MDAnalysis as mda

from MDAnalysis.analysis.base import AnalysisBase

from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.log import ProgressBar


class LindemannIndex(AnalysisBase):

    def __init__(self,
                 atomgroup: mda.AtomGroup,
                 box=None,
                 **kwargs):
        self.ag = atomgroup
        self.ag_natoms = len(self.ag)
        self.box = box
        super(LindemannIndex, self).__init__(
            atomgroup.universe.trajectory, 
            **kwargs
        )

    def _prepare(self):

        natoms = self.ag_natoms

        self.array_mean = np.zeros((natoms, natoms))
        self.array_var = np.zeros((natoms, natoms))

        self.results.lindemann_index = []

    def _single_frame(self):

        n_atoms = len(self.ag)

        array_distance = distance_array(self.ag, self.ag, box=self.box)
        delta = array_distance - self.array_mean
        self.array_mean += delta / (self._frame_index + 1)
        self.array_var += delta * (array_distance - self.array_mean)

        if self._frame_index == 0:
            lindemann_indices = np.zeros((n_atoms,))
        else:
            lindemann_indices = np.nanmean(
                np.sqrt(self.array_var / self._frame_index) / self.array_mean, axis=1)
        self.results.lindemann_index.append(lindemann_indices)
