import numpy as np
import MDAnalysis as mda

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import distance_array


class EqualAroundIndex(AnalysisBase):
    """Get indices of atoms from `AtomGroup` closest to given `AtomGroup`.

    Args:
        reference (mda.AtomGroup): AtomGroup. Reference group, should be static.
        environment (mda.AtomGroup): AtomGroup. Environment group. `updating` should be `True` for dynamic selection.
        box (np.ndarray, optional): The box dimensions. Defaults to None.
        size (int, optional): The number of atoms to select. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the parent class.

    Attributes:
        ag1 (mda.AtomGroup): The first AtomGroup.
        ag2 (mda.AtomGroup): The second AtomGroup.
        universe (mda.Universe): The universe containing the AtomGroups.
        size (int): The number of atoms to select.
        size_initial (bool): Whether the size was initially None.
        box (np.ndarray): The box dimensions.
        results (AnalysisResults): The results of the analysis.

    Methods:
        _prepare(): Prepare the analysis.
        _single_frame(): Perform the analysis on a single frame.
    """

    def __init__(self,
                 reference: mda.AtomGroup,
                 environment: mda.AtomGroup,
                 box=None,
                 size=None,
                 **kwargs):
        self.reference = reference
        self.environment = environment
        self.universe = reference.universe
        self.atom_groups = []

        if size:
            self.size = size
        else:
            self.size = None
        if box:
            self.universe.dimensions = box
            self.box = self.universe.dimensions
        elif not box and self.universe.dimensions:
            self.box = self.universe.dimensions
        super(EqualAroundIndex, self).__init__(
            reference.universe.trajectory,
            **kwargs
        )

    def _prepare(self):
        self.results.indices = []
        ref_size = len(self.reference)

        if not self.size:
            # loop to find minimum size
            for ts in self.universe.trajectory:
                if len(self.environment) == 0:
                    raise ValueError("Environment should not be empty through the trajectory.")
                if not self.size:
                    self.size = len(self.environment) + ref_size
                elif len(self.environment) + ref_size < self.size:
                    self.size = len(self.environment) + ref_size

    def _single_frame(self):
        array_distance = distance_array(self.reference, self.environment, box=self.box)
        min_distances = np.min(array_distance, axis=0)
        if self.size is not None:
            clip = self.size - len(self.reference)
        else:
            clip = None
        new_ag = self.reference | self.environment[np.argsort(min_distances)[:clip]]
        self.atom_groups.append(new_ag)
        self.results.indices.append(new_ag.indices)
    
    def _conclude(self):
        self.results.universe = mda.Merge(self.atom_groups)


class EqualAxisIndex(AnalysisBase):

    def __init__(self,
                 ag1: mda.AtomGroup,
                 ag2: mda.AtomGroup,
                 box=None,
                 size=None,
                 **kwargs):
        self.ag1 = ag1
        self.ag2 = ag2
        self.universe = ag1.universe
        self.ag_union = self.ag1 | self.ag2
        if size:
            self.size = size
        else:
            self.size = None
        self.box = box
        if box and self.universe.dimensions is None:
            self.universe.dimensions = box
        super(EqualAxisIndex, self).__init__(
            ag1.universe.trajectory, 
            **kwargs
        )

    def _prepare(self):
        self.results.new_ag = []
        self.results.indexes = []
        if self.size is None:
            for ts in self.universe.trajectory:
                if self.size is None:
                    self.size = len(self.ag2)
                if len(self.ag2) < self.size:
                    self.size = len(self.ag2)

    def _single_frame(self):
        reference = self.ag1.center_of_geometry()
        reference[0] = 0.
        reference[1] = 0.
        temp_distances = self.ag2.positions[:]
        temp_distances[:, 0] = 0.
        temp_distances[:, 1] = 0.
        self_distances = distance_array(
            temp_distances, reference, box=self.box
        )
        new_ag = self.ag1 | self.ag2[np.argsort(self_distances.flatten())[:self.size]]
        self.results.new_ag.append(new_ag)
        self.results.indexes.append(new_ag.indices)
