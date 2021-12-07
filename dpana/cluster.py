from MDAnalysis import Universe
from ase.io import read, write
import numpy as np
from scipy.spatial import distance
from scipy.optimize import curve_fit
import pandas as pd


class Cluster(object):
    def __init__(self, path):
        self.path = path

    def trajectory(self, trj_type='ase', **kwargs):
        """
        load trajectory from file
        """
        if trj_type == 'mda':
            return self.load_mda_trajectory(**kwargs)
        else:
            return self.load_ase_trajectory(**kwargs)

    def load_mda_trajectory(self, **kwargs):
        topology_format = kwargs.get('topology_format', 'XYZ')
        return Universe(self.path, topology_format=topology_format)

    def load_ase_trajectory(self, **kwargs):
        topology_format = kwargs.get('topology_format', 'xyz')
        return read(self.path, format=topology_format)


def distance_to_cnt(u, selection_cluster, cluster_size):
    """
    For carbon nanotube included trajectories, analyze cluster atoms.

    Args:
         u: MDA trajectory instance.
         selection_cluster:
         cluster_size: size of clusters
    """
    distances = np.zeros((len(u.trajectory), cluster_size))
    cnt = u.select_atoms('name C', updating=True)
    pt = u.select_atoms(selection_cluster, updating=True)
    for q, ts in enumerate(u.trajectory):
        cg = cnt.center_of_geometry()
        for p, t in enumerate(pt.positions):
            dis = distance.euclidean(t, [cg[0], cg[1], t[2]])
            distances[q, p] = dis
    return distances


def lindemann_per_frames(u, select_lang):
    """
    Calculate the lindemann index for each atom AND FRAME
    Return a ndarray of shape (len_frames, natoms, natoms)
    Warning this can produce extremly large ndarrays in memory 
    depending on the size of the cluster and the ammount of frames.
    """
    # natoms = natoms
    sele_ori = u.select_atoms(select_lang)
    natoms = len(sele_ori)
    nframes = len(u.trajectory)
    len_frames = len(u.trajectory)
    array_mean = np.zeros((natoms, natoms))
    array_var = np.zeros((natoms, natoms))
    # array_distance = np.zeros((natoms, natoms))
    iframe = 1
    lindex_array = np.zeros((len_frames, natoms, natoms))
    cluster = u.select_atoms(select_lang, updating=True)
    for q, ts in enumerate(u.trajectory):
        # print(ts)
        coords = cluster.positions
        n, p = coords.shape
        array_distance = distance.cdist(coords, coords)

        #################################################################################
        # update mean and var arrays based on Welford algorithm suggested by Donald Knuth
        #################################################################################
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
            np.sqrt(np.divide(array_var, nframes)), array_mean
        )
        # lindemann_indices = np.nanmean(np.sqrt(array_var/nframes)/array_mean, axis=1)
        lindex_array[q] = lindemann_indices

    return np.array([np.nanmean(i, axis=1) for i in lindex_array])


def fitting_lindemann_curve(temperature, lindemann, bounds, function='func2'):
    """
    fit smooth curve from given lindemann index of each temperature.
    ----
    Args:
        temperature: list of temperatures. e.g.: [200, 300, 400, 500, 600, 700, 800]
        lindemann: list of lindemann index calculated from trajectories at each temperature
        bounds: upper and lower bounds of each param in functions.
            e.g. ([-np.inf, -np.inf, -np.inf, -np.inf, 400, 15.], [np.inf, np.inf, np.inf, np.inf, 700., 100.])
        function: function used to fit the curve

    Return:
        pd.DataFrame
    """
    def func(x, a, b, c, d, x0, dx):
        return b + (a - b) * x + d / (1 + np.exp((x - x0) / dx)) + c * x

    def func2(x, a, b, c, d, x0, dx):
        return (a * x + b) / (1 + np.exp((x - x0) / dx)) + (c * x + d) / (1 + np.exp((x0 - x) / dx))

    if function == 'func2':
        fit_func = func2
    else:
        fit_func = func

    popt, pcov = curve_fit(
        fit_func, temperature, lindemann,
        bounds=bounds
    )
    x_pred = np.linspace(min(temperature), max(temperature), )

    df = pd.DataFrame({
        'temperature': x_pred,
        'fitting_curve': fit_func(x_pred, *popt)
    })
    return df
