import pytest
import numpy as np
import pandas as pd

from miko.metad.hills import Hills
from miko.metad.fes import FES
from miko.metad.minima import Minima

from .test_fes import read_hills

def test_minima(shared_datadir):
    hills = read_hills(
        name=shared_datadir / "hills/acealanme", 
        periodic=[True, True], 
        cv_per=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    fes = FES.from_hills(hills, resolution=256)
    fes.make_fes_original(resolution=256)
    minima = Minima(fes)
    minima_df = minima.minima
    if type(minima_df) == pd.DataFrame:
        assert len(minima_df) == 5
