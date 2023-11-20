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
        self.results.atom_groups = []
        self.results.indices = []
        ref_size = len(self.reference)

        if not self.size:
            # loop to find minimum size
            for _ in self.universe.trajectory:
                if len(self.environment) == 0:
                    raise ValueError(
                        "Environment should not be empty through the trajectory.")
                if not self.size:
                    self.size = len(self.environment) + ref_size
                elif len(self.environment) + ref_size < self.size:
                    self.size = len(self.environment) + ref_size

    def _single_frame(self):
        array_distance = distance_array(
            self.reference, self.environment, box=self.box)
        min_distances = np.min(array_distance, axis=0)
        if self.size is not None:
            clip = self.size - len(self.reference)
        else:
            clip = None
        new_ag = self.reference | self.environment[np.argsort(min_distances)[
            :clip]]
        self.atom_groups.append(new_ag)
        self.results.atom_groups.append(new_ag)
        self.results.indices.append(new_ag.indices)


class AxisMaxDistance(AnalysisBase):
    """Calculates the maximum distance between two groups of atoms along a given axis.

    Args:
        ag1 (mda.AtomGroup): The first group of atoms.
        ag2 (mda.AtomGroup): The second group of atoms.
        universe (MDAnalysis.core.universe.Universe): The MDAnalysis universe containing the atoms.
        axis (str): The axis along which to calculate the maximum distance.
        box (numpy.ndarray or None): The dimensions of the simulation box, 
            or None if periodic boundary conditions are not used.
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Return:
        results (MDAnalysis.analysis.base.AnalysisResults): The results of the analysis.

    Raises:
        ValueError: If `axis` is not "x", "y", or "z".

    Notes:
        This class is a subclass of `MDAnalysis.analysis.base.AnalysisBase`.

    Examples:
        >>> import MDAnalysis as mda
        >>> from catflow.analyzer.structure.dynamic_selection import AxisMaxDistance
        >>> u = mda.Universe("system.gro", "system.trr")
        >>> ag1 = u.select_atoms("protein")
        >>> ag2 = u.select_atoms("resname LIG")
        >>> analysis = AxisMaxDistance(ag1, ag2, axis="x", box=u.dimensions)
        >>> analysis.run()
        >>> max_distances = analysis.results.distances
    """

    def __init__(self,
                 ag1: mda.AtomGroup,
                 ag2: mda.AtomGroup,
                 axis: str = "z",
                 box=None,
                 **kwargs):
        self.ag1 = ag1
        self.ag2 = ag2
        self.universe = ag1.universe
        self.axis = axis
        self.box = box
        if box and self.universe.dimensions is None:
            self.universe.dimensions = box
        super(AxisMaxDistance, self).__init__(
            ag1.universe.trajectory,
            **kwargs
        )

    def _prepare(self):
        self.results.distances = []

    def _single_frame(self):
        reference = self.ag1.center_of_geometry()
        temp_distances = self.ag2.positions[:]
        axis_indices = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
        if self.axis not in axis_indices:
            raise ValueError("axis must be x, y or z")
        reference[axis_indices[self.axis]] = 0.
        temp_distances[:, axis_indices[self.axis][0]] = 0.
        temp_distances[:, axis_indices[self.axis][1]] = 0.
        self_distances = distance_array(
            temp_distances, reference, box=self.box)
        max_distances = np.max(self_distances, axis=1)
        self.results.distances.append(max_distances)


class EqualAxisIndex(AnalysisBase):
    """Selects atoms from a given AtomGroup based on their distance along a specified axis.

    Args:
        ag1 (mda.AtomGroup): The reference AtomGroup.
        ag2 (mda.AtomGroup): The AtomGroup to select atoms from.
        axis (str): The axis along which to measure distances. Must be "x", "y", or "z".
        box (mda.Box): The simulation box. Required if periodic boundary conditions are used.
        size (int): The number of atoms to select. If None, selects all atoms in ag2.
        **kwargs: Additional keyword arguments to pass to AnalysisBase.

    Raises:
        ValueError: If axis is not "x", "y", or "z".

    Attributes:
        ag1 (mda.AtomGroup): The reference AtomGroup.
        ag2 (mda.AtomGroup): The AtomGroup to select atoms from.
        universe (mda.Universe): The simulation universe.
        axis (str): The axis along which to measure distances.
        size (int): The number of atoms to select.
        box (mda.Box): The simulation box.
        results (AnalysisResults): The results of the analysis.

    """

    def __init__(self,
                 ag1: mda.AtomGroup,
                 ag2: mda.AtomGroup,
                 axis: str = "z",
                 box=None,
                 size=None,
                 **kwargs):
        self.ag1 = ag1
        self.ag2 = ag2
        self.universe = ag1.universe
        self.axis = axis
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
        """Prepares the analysis."""
        self.results.atom_groups = []
        self.results.indices = []
        if self.size is None:
            for _ in self.universe.trajectory:
                if self.size is None:
                    self.size = len(self.ag2)
                if len(self.ag2) < self.size:
                    self.size = len(self.ag2)

    def _single_frame(self):
        """Performs the analysis on a single frame."""
        reference = self.ag1.center_of_geometry()
        temp_distances = self.ag2.positions[:]
        axis_indices = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
        if self.axis not in axis_indices:
            raise ValueError("axis must be x, y or z")
        reference[axis_indices[self.axis]] = 0.
        temp_distances[:, axis_indices[self.axis][0]] = 0.
        temp_distances[:, axis_indices[self.axis][1]] = 0.
        self_distances = distance_array(
            temp_distances, reference, box=self.box)
        new_ag = self.ag1 | self.ag2[np.argsort(
            self_distances.flatten())[:self.size]]
        self.results.atom_groups.append(new_ag)
        self.results.indices.append(new_ag.indices)


class GroupsDistanceIndex(AnalysisBase):

    def __init__(self,
                 ag,
                 sliced_ags,
                 limitation: int = 1, 
                 box=None,
                 zone=None,
                 **kwargs):
        self.ag = ag
        self.sliced_ags = sliced_ags
        self.limitation = limitation
        self.universe = ag.universe
        self.box = box
        self.zone = zone
        if box and self.universe.dimensions is None:
            self.universe.dimensions = box
        super(GroupsDistanceIndex, self).__init__(
            ag.universe.trajectory, 
            **kwargs
        )

    def _prepare(self):
        self.results.atom_groups = []
        self.results.indices = []

    def _single_frame(self):
        import heapq as hq
        new_ag = self.ag
        
        reference = self.ag.center_of_geometry()
        reference[0] = 0.
        reference[1] = 0.
        
        min_queue = []
        
        for i, sl in enumerate(self.sliced_ags):
            temp_distances = sl.positions[:]
            temp_distances[:, 0] = 0.
            temp_distances[:, 1] = 0.
            self_distance = np.mean(distance_array(
                temp_distances, reference, box=self.box
            ))
            min_queue.append(self_distance)
        
        for i, j in hq.nsmallest(
            self.limitation, enumerate(min_queue), key=lambda x: x[1]
        ):
            new_ag = new_ag | self.sliced_ags[i]
        self.results.atom_groups.append(new_ag)
        self.results.indices.append(new_ag.indices)
