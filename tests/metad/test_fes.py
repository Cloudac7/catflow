import pytest
import numpy as np

from miko.metad.hills import Hills
from miko.metad.fes import FES

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
    fes = FES(hills, resolution=resolution)
    _, fes_calc = fes.get_e_beta_c(resolution=resolution)
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
    fes = FES(hills, resolution=resolution)
    fes_calc = fes.make_fes_original(resolution=resolution)
    fes_calc = fes_calc.T

    plumed_name = shared_datadir / f"plumed/{name}.dat"
    plumed = plumed_hills(plumed_name, len(periodic), resolution=resolution)
    assert np.mean(fes_calc) == pytest.approx(np.mean(plumed), abs=1e-3)
    assert np.allclose(fes_calc, plumed, atol=1e-2)
