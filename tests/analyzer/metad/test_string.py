import numpy as np
import pytest

from catflow.analyzer.metad.fes import FreeEnergySurface
from catflow.analyzer.metad.string import StringMethod

@pytest.fixture
def fes():
    # Create a 2D energy landscape
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    V = np.sin(X) + np.cos(Y)
    make_fes = FreeEnergySurface.from_array(
        V, cv_min=[-5, -5], cv_max=[5, 5], periodic=[False, False], cv_name=["x", "y"], resolution=100
    )
    return make_fes

def test_compute_mep(fes):

    # Initialize StringMethod object
    sm = StringMethod(fes)

    # Compute MEP
    sm.compute_mep(begin=[-4, 0], end=[4, 0], n_points=10)

    # Check that the MEP has the correct shape
    assert sm.mep.shape == (10, 2)

def test_plot_mep(fes):
    # Initialize StringMethod object
    sm = StringMethod(fes)

    # Compute MEP
    sm.compute_mep(begin=[-4, 0], end=[4, 0], n_points=10)

    # Plot MEP
    fig, ax = sm.plot_mep()

    # Check that the plot has been created
    assert fig is not None
    assert ax is not None

def test_plot_mep_energy_profile(fes):
    # Initialize StringMethod object
    sm = StringMethod(fes)

    # Compute MEP
    sm.compute_mep(begin=[-4, 0], end=[4, 0], n_points=10)

    # Plot MEP energy profile
    fig, ax = sm.plot_mep_energy_profile()

    # Check that the plot has been created
    assert fig is not None
    assert ax is not None

def test_plot_string_evolution(fes):

    # Initialize StringMethod object
    sm = StringMethod(fes)

    # Compute MEP
    sm.compute_mep(begin=[-4, 0], end=[4, 0], n_points=10)

    # Plot string evolution
    fig, ax = sm.plot_string_evolution()

    # Check that the plot has been created
    assert fig is not None
    assert ax is not None
