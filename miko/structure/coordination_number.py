import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable

from MDAnalysis import AtomGroup
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.distances import distance_array


def coordination_number_calculation(
    ag1: AtomGroup,
    ag2: AtomGroup,
    r0: float,
    d0: float = 0.0,
    n: int = 6,
    m: int = 12,
    box: Optional[NDArray] = None,
    switch_function: Optional[Callable] = None,
    mean: bool = True
):
    d = distance_array(ag1, ag2, box)
    if len(ag1) > 0:
        d = d[d != 0.].reshape((len(ag1), -1))
    else:
        return 0.

    if switch_function:
        cn = switch_function(d, r0, d0, n, m)
    else:
        cn = (1 - ((d - d0) / r0) ** n) / (1 - ((d - d0) / r0) ** m)
    if mean:
        return np.mean(np.sum(cn, axis=1))
    else:
        return np.sum(cn, axis=1)


class CoordinationNumber(AnalysisBase):
    def __init__(self,
                 g1: AtomGroup,
                 g2: AtomGroup,
                 r0: float,
                 d0: float = 0.0,
                 n: int = 6,
                 m: int = 12,
                 box: Optional[NDArray] = None,
                 switch_function: Optional[Callable] = None,
                 mean: bool = True,
                 **kwargs):
        super(CoordinationNumber, self).__init__(
            g1.universe.trajectory,
            **kwargs
        )
        self.g1 = g1
        self.g2 = g2
        self.r0 = r0
        self.d0 = d0
        self.n = n
        self.m = m
        self.box = box
        self.switch_function = switch_function
        self.mean = mean

    def _prepare(self):
        self.results.coordination_number = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store an example_result of `some_function` for a single frame
        self.results.coordination_number.append(
            coordination_number_calculation(
                ag1=self.g1,
                ag2=self.g2,
                r0=self.r0,
                d0=self.d0,
                n=self.n,
                m=self.m,
                box=self.box,
                switch_function=self.switch_function,
                mean=self.mean
            )
        )
