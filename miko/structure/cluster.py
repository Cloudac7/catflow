import numpy as np
import pandas as pd

from scipy.spatial import distance
from scipy.optimize import curve_fit
from MDAnalysis import Universe


class Cluster(object):
    def __init__(self, path=None, **kwargs):
        self.path = path
        self.kwargs = kwargs
        if self.path is not None:
            self.universe = self.load_mda_trajectory(**self.kwargs)

    @classmethod
    def convert_universe(cls, u: Universe, **kwargs):
        """Convert mda.Universe instaince to a Cluster instance

        Args:
            u (Universe): Trajectory instance.
        
        Returns:
            Cluster: Cluster instance.
        """
        cluster = cls(**kwargs)
        cluster.universe = u
        return cluster

    def load_mda_trajectory(self, **kwargs) -> Universe:
        """Load trajectory as mda.Universe from file.

        Returns:
            Universe: Trajectory instance.
        """
        if kwargs.get("topology_format", None) is None:
            kwargs["topology_format"] = "XYZ"
        return Universe(self.path, **kwargs)
    
    def distance_to_com(self, selection_cluster: str):
        """Analyze cluster atoms by calculating distance to center of mass.

        Args:
            u (Universe): MDA trajectory instance.
            selection_cluster (str): Selection language to select atoms in cluster.

        Returns:
            _type_: _description_
        """
        u = self.universe
        cluster = u.select_atoms(selection_cluster, updating=True)
        distances = np.zeros((len(u.trajectory), len(cluster)))
        cg = cluster.center_of_geometry()
        for q, ts in enumerate(u.trajectory):
            for p, t in enumerate(cluster.positions):
                dis = distance.euclidean(t, cg)
                distances[q, p] = dis
        return distances
    
    def lindemann_per_frames(self, selection_cluster: str) -> np.ndarray:
        """Calculate the lindemann index for each atom AND FRAME

        Warning this can produce extremly large ndarrays in memory
        depending on the size of the cluster and the ammount of frames.

        Args:
            u (Universe): MDA trajectory instance.
            selection_cluster (str): Selection language to select atoms in cluster.

        Returns:
            np.ndarray: An ndarray of shape (len_frames, natoms)

        References:
            https://github.com/ybyygu/lindemann-index/blob/master/src/lindemann.pyx
            http://math.stackexchange.com/a/116344
        """
        u = self.universe
        sele_ori = u.select_atoms(selection_cluster)
        natoms = len(sele_ori)
        nframes = len(u.trajectory)
        len_frames = len(u.trajectory)
        array_mean = np.zeros((natoms, natoms))
        array_var = np.zeros((natoms, natoms))
        # array_distance = np.zeros((natoms, natoms))
        iframe = 1
        lindex_array = np.zeros((len_frames, natoms, natoms))
        cluster = u.select_atoms(selection_cluster, updating=True)
        for q, ts in enumerate(u.trajectory):
            # print(ts)
            coords = cluster.positions
            n, p = coords.shape
            array_distance = distance.cdist(coords, coords)
            
            # update mean and var arrays based on Welford algorithm (1962)
            for i in range(natoms):
                for j in range(i + 1, natoms):
                    xn = array_distance[i, j]
                    mean = array_mean[i, j]
                    var = array_var[i, j]
                    delta = xn - mean
                    # update mean
                    array_mean[i, j] = mean + delta / iframe
                    # update variance
                    array_var[i, j] = var + delta * (xn - array_mean[i, j])
            iframe += 1
            if iframe > nframes + 1:
                break

            for i in range(natoms):
                for j in range(i + 1, natoms):
                    array_mean[j, i] = array_mean[i, j]
                    array_var[j, i] = array_var[i, j]

            lindemann_indices = np.divide(
                np.sqrt(np.divide(array_var, iframe - 1)), array_mean
            )
            lindex_array[q] = lindemann_indices

        return np.array([np.nanmean(i, axis=1) for i in lindex_array])


def distance_to_cnt(u: Universe, selection_cluster: str, cnt_direction: str):
    """For carbon nanotube included trajectories, analyze cluster atoms.

    Args:
        u (Universe): MDA trajectory instance.
        selection_cluster (str): Selection language to select atoms in cluster.

    Returns:
        _type_: _description_
    """
    cnt = u.select_atoms('name C', updating=True) # C for carbon
    cluster = u.select_atoms(selection_cluster, updating=True)
    distances = np.zeros((len(u.trajectory), len(cluster)))
    for q, ts in enumerate(u.trajectory):
        cg = cnt.center_of_geometry()
        for p, t in enumerate(cluster.positions):
            if cnt_direction == "z":
                dis = distance.euclidean(t, [cg[0], cg[1], t[2]])
            elif cnt_direction == "y":
                dis = distance.euclidean(t, [cg[0], t[1], cg[2]])
            elif cnt_direction == "x":
                dis = distance.euclidean(t, [t[0], cg[1], cg[2]])
            else:
                raise ValueError("cnt_direction must be x, y or z")
            distances[q, p] = dis
    return distances


def fitting_lindemann_curve(
        temperature,
        lindemann,
        bounds,
        function='func2'
):
    """fit smooth curve from given lindemann index or free energy of each temperature.

    Args:
        temperature (_type_): list of temperatures. 
            e.g.: [200, 300, 400, 500, 600, 700, 800]
        lindemann (_type_): list of lindemann index calculated from 
            trajectories at each temperature
        bounds (_type_): upper and lower bounds of each param in functions.
            e.g. ([-np.inf, -np.inf, -np.inf, -np.inf, 400, 15.], 
                  [np.inf, np.inf, np.inf, np.inf, 700., 100.])
        function (str, optional): function used to fit the curve. Defaults to 'func2'.

    Returns:
        pd.DataFrame: _description_
    """

    def func(x, a, b, c, d, x0, dx):
        return b + (a - b) * x + d / (1 + np.exp((x - x0) / dx)) + c * x

    def func2(x, a, b, c, d, x0, dx):
        return (a * x + b) / (1 + np.exp((x - x0) / dx)) + \
               (c * x + d) / (1 + np.exp((x0 - x) / dx))

    if function == 'func2':
        fit_func = func2
    else:
        fit_func = func

    popt, pcov = curve_fit(
        fit_func, temperature, lindemann,
        bounds=bounds
    )
    x_pred = np.linspace(min(temperature), max(temperature), 100)

    df = pd.DataFrame({
        'temperature': x_pred,
        'fitting_curve': fit_func(x_pred, *popt)
    })
    return df
