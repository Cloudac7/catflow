import pytest
import numpy as np
import pandas as pd

from catflow.analyzer.metad.colvar import Colvar
from catflow.analyzer.metad.hills import Hills
from catflow.analyzer.metad.fes import FreeEnergySurface
from catflow.analyzer.metad.profile import FreeEnergyProfile

from matplotlib.testing.decorators import image_comparison

def read_hills(name, periodic, cv_per):
    return Hills(name, periodic=periodic, cv_per=cv_per)

def plumed_hills(filename, cvs, resolution=256):
    plumed_data = np.loadtxt(filename)
    plumed_data = np.reshape(plumed_data[:, cvs], [resolution] * cvs)
    plumed_data = plumed_data - np.min(plumed_data)
    return plumed_data

@pytest.mark.parametrize(
    "name, periodic, cv_per, resolution", [
        ("acealanme1d", [True], [[-np.pi, np.pi]], 256),
        ("acealanme", [True,True], [[-np.pi, np.pi], [-np.pi, np.pi]], 256),
        # ("acealanme3d", [True,True,True], [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], 64)
    ]
)
def test_sum_bias(shared_datadir, name, periodic, cv_per, resolution):
    hill_name = shared_datadir / "hills" / name
    hills = read_hills(
        name=hill_name, periodic=periodic, cv_per=cv_per
    )
    fes = FreeEnergySurface.from_hills(hills, resolution=resolution)
    _, fes_calc = fes.get_e_beta_c(resolution=resolution, return_fes=True)
    fes_calc = fes_calc.T

    plumed_name = shared_datadir / f"plumed/{name}.dat"
    plumed = plumed_hills(plumed_name, len(periodic), resolution=resolution)
    assert np.mean(fes_calc) == pytest.approx(np.mean(plumed), abs=1)
    assert np.allclose(fes_calc, plumed, atol=4)

@pytest.mark.parametrize(
    "name, periodic, cv_per, resolution", [
        ("acealanme1d", [True], [[-np.pi, np.pi]], 256),
        ("acealanme", [True,True], [[-np.pi, np.pi], [-np.pi, np.pi]], 256),
        # ("acealanme3d", [True,True,True], [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], 64)
    ]
)
def test_make_fes_original(shared_datadir, name, periodic, cv_per, resolution):
    hill_name = shared_datadir / "hills" / name
    hills = read_hills(
        name=hill_name, periodic=periodic, cv_per=cv_per
    )
    fes = FreeEnergySurface.from_hills(hills, resolution=resolution)
    fes_calc = fes.make_fes_original(resolution=resolution)
    fes_calc = fes_calc.T

    plumed_name = shared_datadir / f"plumed/{name}.dat"
    plumed = plumed_hills(plumed_name, len(periodic), resolution=resolution)
    assert np.mean(fes_calc) == pytest.approx(np.mean(plumed), abs=1e-3)
    assert np.allclose(fes_calc, plumed, atol=1e-2)

def test_minima(shared_datadir):
    hills = read_hills(
        name=shared_datadir / "hills/acealanme", 
        periodic=[True, True], 
        cv_per=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    fes = FreeEnergySurface.from_hills(hills, resolution=256)
    fes.make_fes_original(resolution=256)
    fes.find_minima()
    minima_df = fes.minima
    if type(minima_df) == pd.DataFrame:
        assert len(minima_df) == 5

def test_profile(shared_datadir):
    hills = read_hills(
        name=shared_datadir / "hills/acealanme", 
        periodic=[True, True], 
        cv_per=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    fes = FreeEnergySurface.from_hills(hills, resolution=256)
    fes.make_fes_original(resolution=256)
    fes.find_minima()
    profile = FreeEnergyProfile(fes, hills, profile_length=256)

    assert profile.free_profile.shape[0] == 257

def test_from_colvar(shared_datadir):
    import pandas as pd
    from catflow.analyzer.metad.colvar import Colvar

    # Create a mock Colvar object
    class MockColvar:
        def __init__(self, colvars):
            self.colvars = colvars

    # Define the input parameters for the test
    colvar_data = {
        "cv1": np.array([0.5, 1.0, 1.5, 2.0]),
        "cv2": np.array([1.0, 2.0, 3.0, 4.0])
    }
    colvar = MockColvar(pd.DataFrame(colvar_data))
    cv_name = ["cv1", "cv2"]
    periodic = [True, False]
    cv_per = [[-np.pi, np.pi], None]
    resolution = 128

    # Call the method under test
    fes = FreeEnergySurface.from_colvar(colvar, cv_name, periodic, cv_per, resolution) # type: ignore

    # Perform assertions
    assert fes.cvs == 2
    assert fes.cv_name == cv_name
    assert np.array_equal(fes.colvar.colvars["cv1"].to_numpy(), colvar_data["cv1"]) # type: ignore
    assert np.array_equal(fes.colvar.colvars["cv2"].to_numpy(), colvar_data["cv2"]) # type: ignore
    assert np.array_equal(fes.cv_min, [-np.pi, 0.55])
    assert np.array_equal(fes.cv_max, [np.pi, 4.45])
    assert np.array_equal(fes.periodic, [True, False])

@image_comparison(baseline_images=['fes_reweighting'], remove_text=True,
                  extensions=['png'], style='mpl20')
def test_make_fes_reweighting(shared_datadir):
    colvar = Colvar(shared_datadir / "colvar/COLVAR", "metad")
    resolution = 128
    cv_used_names = ["d1"]
    kb = 0.051704 / 600.0
    temp = 600.0

    # Create an instance of FreeEnergySurface
    fes = FreeEnergySurface.from_colvar(colvar, cv_used_names, resolution=resolution)

    # Call the method under test
    result = fes.make_fes_reweighting(
        resolution=resolution,
        kb=kb,
        temp=temp,
        cv_used_names=cv_used_names,
    )
    fig, ax = fes.plot(energy_unit="eV")
    fig.show()
