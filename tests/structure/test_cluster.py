import numpy as np
from pathlib import Path

import pytest
from MDAnalysis import Universe
from miko.structure.cluster import Cluster

@pytest.fixture
def cluster(shared_datadir):
    cluster = Cluster(path=shared_datadir / 'test_case_O2Pt36.xyz')
    return cluster

def test_load_mda_trajectory(cluster):
    assert cluster.universe is not None

def test_convert_universe(cluster, shared_datadir):
    u = Universe(shared_datadir / 'test_case_O2Pt36.xyz')
    c = Cluster.convert_universe(u)
    assert len(cluster.universe.trajectory) == len(c.universe.trajectory)

def test_distance_to_com(cluster):
    distances = cluster.distance_to_com('name Pt')
    assert distances.shape == (3, 36)

def test_lindemann_per_frames(cluster):
    lindemann = cluster.lindemann_per_frames('name Pt')
    assert lindemann.shape == (3, 36)
