import numpy as np
from pathlib import Path

import pytest
from MDAnalysis import Universe
from miko.structure.coordination_number import CoordinationNumber

def test_coordination(shared_datadir):
    u = Universe(shared_datadir / 'test_case_O2Pt36.xyz')
    ag1 = u.select_atoms("name O")
    ag2 = u.select_atoms("name Pt")
    cn = CoordinationNumber(ag1, ag2, 3.2, box=[18.0, 18.0, 18.0, 90.0, 90.0, 90.0]) # type: ignore
    cn.run()
    assert np.array(cn.results.coordination_number).shape == (3,)
