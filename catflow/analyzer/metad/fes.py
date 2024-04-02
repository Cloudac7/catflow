import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from joblib import Parallel, delayed
from typing import Optional, List, Tuple, Union, Dict, Any
from itertools import product
from numpy.typing import ArrayLike
from matplotlib.colors import Colormap

from catflow.analyzer.metad.hills import Hills
from catflow.analyzer.graph.plotting import canvas_style
from catflow.utils.config import parse_slice_string
from catflow.utils.log_factory import logger

dp2_cutoff_a = 1.0/(1.0 - np.exp(-6.25))
dp2_cutoff_b = - np.exp(-6.25)/(1.0 - np.exp(-6.25))

class FreeEnergySurface:
    """
    Computes the free energy surface corresponding to the provided Hills object.

    Usage:
    ```python
    from catflow.analyzer.metad.fes import FreeEnergySurface
    fes = FreeEnergySurface(hills)
    ```

    Args:
        hills (Hills): The Hills object used for computing the free energy surface.
        resolution (int, optional): \
            The resolution of the free energy surface. Defaults to 256.
        time_min (int): The starting time step of simulation. Defaults to 0.
        time_max (int, optional): The ending time step of simulation. Defaults to None.
    """

    fes: np.ndarray
    cvs: int
    cv_min: np.ndarray
    cv_max: np.ndarray
    periodic: np.ndarray
    resolution: int = 256
    hills: Optional[Hills] = None
    cv_fes_range: Optional[np.ndarray] = None
    cv_name: Optional[List[str]] = None
    minima: Optional[pd.DataFrame] = None

    def __init__(
        self,
        resolution: int = 256
    ):
        self.res = resolution

    @classmethod
    def from_hills(
        cls,
        hills: Hills,
        resolution: int = 256
    ):
        """Generate a FES object from a Hills object."""
        fes = cls(resolution=resolution)

        fes.cvs = hills.cvs

        fes.hills = hills
        fes.periodic = hills.periodic
        fes.cv_name = hills.cv_name

        fes.generate_cv_map()

        return fes

    def generate_cv_map(self):
        """generate CV map"""
        if self.hills is not None:
            cv_min = deepcopy(self.hills.cv_min)
            cv_max = deepcopy(self.hills.cv_max)
            cv_range = cv_max - cv_min
            self.cv_range = cv_range

            cv_min[~self.periodic] -= cv_range[~self.periodic] * 0.15
            cv_max[~self.periodic] += cv_range[~self.periodic] * 0.15
            cv_fes_range = np.abs(cv_max - cv_min)

            # generate remapped cv_min and cv_max
            self.cv_min = cv_min
            self.cv_max = cv_max
            self.cv_fes_range = cv_fes_range

    @classmethod
    def from_array(
        cls,
        fes_array: ArrayLike,
        cv_min: ArrayLike,
        cv_max: ArrayLike,
        periodic: ArrayLike,
        cv_name: List[str],
        resolution: int = 256
    ):
        fes = cls(resolution=resolution)
        fes.fes = np.array(fes_array)
        fes.cv_min = np.array(cv_min)
        fes.cv_max = np.array(cv_max)
        fes.cv_fes_range = np.array(cv_max) - np.array(cv_min)
        fes.periodic = np.array(periodic, dtype=bool)
        fes.cv_name = cv_name
        fes.res = resolution
        fes.cvs = len(cv_name)
        return fes

    def name_cv(self):
        if self.cv_name is None:
            self.cv_name = []
            for i in range(self.cvs):
                self.cv_name.append(f"CV{i+1}")

    def get_e_beta_c(
        self,
        resolution: Optional[int] = None,
        time_min: Optional[int] = None,
        time_max: Optional[int] = None,
        kb: float = 8.314e-3,
        temp: float = 300.0,
        bias_factor: float = 15.0,
        return_fes: bool = False
    ):
        """Function used internally for summing hills in Hills object with the fast Bias Sum Algorithm. 
        From which could also be quick to get e_beta_c for reweighting.

        Args:
            resolution (int, optional): \
                The resolution of the free energy surface. Defaults to 256.
            time_min (int): The starting time step of simulation. Defaults to 0.
            time_max (int, optional): The ending time step of simulation. Defaults to None.
            reweighting (bool, optional): \
                If True, the function of c(t) will be calculated and stored in `self.e_beta_c`.
                Defaults to False.
            kb (float, optional): The Boltzmann Constant in the energy unit. Defaults to 8.314e-3.
            temp (float, optional): The temperature of the simulation in Kelvins. Defaults to 300.0.
            bias_factor (float, optional): The bias factor used in the simulation. Defaults to 15.0.
        """

        if resolution is None:
            resolution = self.res

        if self.hills is None:
            raise ValueError("Hills not loaded yet.")

        if time_min is None:
            time_min = 0

        if time_max is None:
            time_max = len(self.hills.cv[:, 0])

        cvs = self.cvs

        cv_min = self.cv_min
        cv_max = self.cv_max
        if self.cv_fes_range is None:
            self.cv_fes_range = cv_max - cv_min
        cv_fes_range = self.cv_fes_range

        cv_bins = np.ceil(
            (self.hills.cv[time_min:time_max, :cvs] -
             cv_min) * resolution / cv_fes_range
        ).T.astype(int)

        sigma = self.hills.sigma[:cvs, 0]
        sigma_res = (sigma * resolution) / (cv_max - cv_min)

        gauss_res = np.ceil(8 * sigma_res).astype(int)
        gauss_res[gauss_res % 2 == 0] += 1

        gauss = self._gauss_kernel(gauss_res, sigma_res)
        gauss_center = np.array(gauss.shape) // 2

        fes = np.zeros([resolution] * cvs)
        e_beta_c = np.zeros(len(cv_bins[0]))

        for line in range(len(cv_bins[0])):
            fes_index_to_edit, delta_fes = \
                self._sum_bias(
                    gauss_center, gauss, cv_bins, line, cvs, resolution
                )
            fes[fes_index_to_edit] += delta_fes

            local_fes = fes - np.min(fes)
            exp_local_fes = np.exp(-local_fes / (kb * temp))
            exp_local_fes_bias = np.exp(-local_fes / (kb * temp * bias_factor))

            numerator = np.sum(exp_local_fes)
            denominator = np.sum(exp_local_fes_bias)
            e_beta_c[line] = numerator / denominator
        if return_fes:
            fes -= np.min(fes)
            return e_beta_c, fes
        else:
            return e_beta_c

    def save_fes_with_correction(
        self,
        resolution=None,
        time_min=None,
        time_max=None,
        kb: float = 8.314e-3,
        temp: float = 300.0,
        filename: Optional[str] = "fes_profile.hdf5"
    ):
        """
        Calculate the free energy surface (FES) with correction.
        Using eq.12 of doi:10.1021/jp504920s to get FES with correction,
        as error estimator of FES surface.

        Args:
            resolution (int, optional): The resolution of the FES grid. Defaults to None.
            time_min (int, optional): The minimum time index. Defaults to None.
            time_max (int, optional): The maximum time index. Defaults to None.
            kb (float, optional): The Boltzmann constant. Defaults to 8.314e-3.
            temp (float, optional): The temperature. Defaults to 300.0.
            return_fes (bool, optional): Whether to return the FES. Defaults to False.

        Returns:
            list: Returns a list containing the FES profile.
        """
        import h5py
        from tqdm import trange

        if resolution is None:
            resolution = self.res

        if self.hills is None:
            raise ValueError("Hills not loaded yet.")

        if time_min is None:
            time_min = 0

        if time_max is None:
            time_max = len(self.hills.cv[:, 0])

        cvs = self.cvs

        cv_min = self.cv_min
        cv_max = self.cv_max
        if self.cv_fes_range is None:
            self.cv_fes_range = cv_max - cv_min
        cv_fes_range = self.cv_fes_range

        cv_bins = np.ceil(
            (self.hills.cv[time_min:time_max, :cvs] -
             cv_min) * resolution / cv_fes_range
        ).T.astype(int)

        sigma = self.hills.sigma[:cvs, 0]
        sigma_res = (sigma * resolution) / (cv_max - cv_min)

        gauss_res = np.ceil(8 * sigma_res).astype(int)
        gauss_res[gauss_res % 2 == 0] += 1

        gauss = self._gauss_kernel(gauss_res, sigma_res)
        gauss_center = np.array(gauss.shape) // 2

        fes = np.zeros([resolution] * cvs)
        d_cv = np.prod(cv_fes_range / resolution)

        with h5py.File(filename, "w") as f:
            fes_data = f.create_dataset(
                "fes_data", 
                shape=(len(cv_bins[0]), *fes.shape), 
                dtype=np.float64, 
                chunks=(1, *fes.shape)
            )
            for line in trange(len(cv_bins[0])):
                fes_index_to_edit, delta_fes = \
                    self._sum_bias(
                        gauss_center, gauss, cv_bins, line, cvs, resolution
                    )
                fes[fes_index_to_edit] += delta_fes
                correction = np.sum(np.exp(- fes / kb / temp)) * d_cv
                fes_corrected = fes + kb * temp * np.log(correction)
                fes_data[line] = fes_corrected
        del fes, d_cv, fes_corrected

    def load_fes_with_correction(
        self, 
        filename: str,
        slice_string: Optional[str] = None
    ):
        import h5py

        with h5py.File(filename, 'r') as f:
            dataset = f["fes_data"]
            if slice_string:
                data_slice = parse_slice_string(slice_string)
                return dataset[data_slice] # type: ignore
            else:
                return dataset[:] # type: ignore

    def _gauss_kernel(self, gauss_res, sigma_res):

        gauss_center = gauss_res // 2
        grids = np.indices(gauss_res)

        grids_flatten = grids.reshape(gauss_res.shape[0], -1).T
        exponent = np.sum(
            -(grids_flatten - gauss_center)**2 / (2 * sigma_res**2),
            axis=1
        )
        gauss = -np.exp(exponent.T.reshape(gauss_res))
        return gauss

    def _sum_bias(
        self, gauss_center, gauss, cv_bins, line, cvs, resolution
    ):
        if self.hills is None:
            raise ValueError("Hills not loaded yet.")

        # create a meshgrid of the indexes of the fes that need to be edited
        fes_index_to_edit = np.indices(gauss.shape)

        # create a mask to avoid editing indexes outside the fes
        local_mask = np.ones_like(gauss, dtype=int)
        for d in range(cvs):
            fes_index_to_edit[d] += cv_bins[d][line] - gauss_center[d]
            if not self.periodic[d]:
                mask = np.where(
                    (fes_index_to_edit[d] < 0) + (
                        fes_index_to_edit[d] > resolution - 1)
                )[0]
                # if the cv is not periodic, remove the indexes outside the fes
                local_mask[mask] = 0
            # make sure the indexes are inside the fes
            fes_index_to_edit[d] = np.mod(fes_index_to_edit[d], resolution)
        delta_fes = gauss * local_mask * self.hills.heights[line]
        fes_index_to_edit = tuple(fes_index_to_edit)
        return fes_index_to_edit, delta_fes

    def reweighting(
        self,
        colvar_file: str,
        e_beta_c: ArrayLike,
        cv_indexes: Optional[List[int]] = None,
        resolution: int = 64,
        kb: float = 8.314e-3,
        temp: float = 300.0,
        bias_index: int = 2
    ):
        """
        Reweights the free energy surface based on the given collective variables.

        Args:
            colvar_file (str): The path to the file containing the collective variables.
            e_beta_c (ArrayLike): The array of e^(-beta*C(t)) values.
            cv_indexes (Optional[List[int]], optional): The indexes of the collective variables to use. Defaults to None.
            resolution (int, optional): The resolution of the free energy surface. Defaults to 64.
            kb (float, optional): The Boltzmann constant. Defaults to 8.314e-3.
            temp (float, optional): The temperature in Kelvin. Defaults to 300.0.

        Returns:
            np.ndarray: The reweighted free energy surface.
        """
        colvar = np.loadtxt(colvar_file)

        if cv_indexes is None:
            cvs = self.cvs
            colvar_value = colvar[:, 1:cvs+1]
        else:
            cvs = len(cv_indexes)
            colvar_value = np.vstack(
                [colvar[:, i + 1] for i in cv_indexes]
            ).T
        cv_array = colvar_value - np.min(colvar_value, axis=0)
        cv_range = np.max(cv_array)
        cv_array = np.floor((resolution - 1) * cv_array / cv_range).astype(int)

        bias = colvar[:, bias_index]
        probs = np.zeros([resolution] * cvs)

        e_beta_c = np.array(e_beta_c)
        reweighted_fes = np.zeros([len(e_beta_c)] + [resolution] * cvs)

        for i in np.arange(len(e_beta_c)):
            index = tuple(cv_array[i])
            probs[index] += np.exp(bias[i]/(kb * temp)) / e_beta_c[i]
            reweighted_fes[i] = -kb * temp * np.log(probs)

        return reweighted_fes

    def _calculate_dp2(self, index, cv, sigma, cv_min, cv_fes_range):

        dp2 = np.zeros(cv.shape[0])
        for cv_idx in range(cv.shape[1]):
            dist_cv = \
                cv[:, cv_idx] - \
                (cv_min[cv_idx] + index[cv_idx] * cv_fes_range[cv_idx] / self.res)
            if self.periodic[cv_idx]:
                dist_cv[dist_cv < -0.5*cv_fes_range[cv_idx]] += cv_fes_range[cv_idx]
                dist_cv[dist_cv > +0.5*cv_fes_range[cv_idx]] -= cv_fes_range[cv_idx]
            dp2_local = dist_cv ** 2 / \
                (2 * sigma[:, cv_idx] ** 2)
            dp2 += dp2_local

        return dp2

    def make_fes_original(
        self,
        resolution: Optional[int],
        time_min: Optional[int] = None,
        time_max: Optional[int] = None,
        n_workers: int = -1
    ):
        """
        Function internally used to sum Hills in the same way as Plumed `sum_hills`. 

        Args:
            resolution (int, optional): \
                The resolution of the free energy surface. Defaults to 256.
            time_min (int): The starting time step of simulation. Defaults to 0.
            time_max (int, optional): The ending time step of simulation. Defaults to None.
            n_workers (int, optional): Number of workers for parallelization. Defaults to 2.
        """
        if self.hills is None:
            raise ValueError("Hills not loaded yet.")

        if resolution is None:
            resolution = self.res

        cvs = self.cvs
        fes = np.zeros([resolution] * cvs)

        if time_min is None:
            time_min = 0
        if time_max is None:
            time_max = len(self.hills.cv[:, 0])
        time_limit = time_max - time_min

        if self.hills is None:
            raise ValueError("Hills not loaded yet.")

        cv_min = self.cv_min
        if self.cv_fes_range is None:
            self.cv_fes_range = self.cv_max - self.cv_min
        cv_fes_range = self.cv_fes_range
        cvs = self.cvs
        cv = self.hills.cv[time_min:time_max, :]
        sigma = self.hills.sigma[time_min:time_max, :]
        heights = self.hills.heights

        def calculate_fes(index, cv, sigma, cv_min, cv_fes_range, heights):
            dp2 = self._calculate_dp2(index, cv, sigma, cv_min, cv_fes_range)

            tmp = np.zeros(time_limit)
            tmp[dp2 < 6.25] = heights[dp2 < 6.25] * \
                (np.exp(-dp2[dp2 < 6.25]) * dp2_cutoff_a + dp2_cutoff_b)
            return index, -tmp.sum()

        results = Parallel(n_jobs=n_workers)(
            delayed(calculate_fes)(
                index, cv, sigma, cv_min, cv_fes_range, heights
            ) for index in np.ndindex(fes.shape)
        )
        if results is not None:
            for index, value in results:
                fes[index] = value
            fes -= np.min(fes)
        self.fes = fes
        return fes

    def remove_cv(
        self,
        CV: int,
        kb: Optional[float] = None,
        energy_unit: str = "kJ/mol",
        temp: float = 300.0
    ):
        """Remove a CV from an existing FES. 
        The function first recalculates the FES to an array of probabilities. 
        The probabilities are summed along the CV to be removed, 
        and resulting probability distribution with 1 less dimension 
        is converted back to FES. 

        Interactivity was working in jupyter notebook/lab with "%matplotlib widget".

        Args:
            CV (int): the index of CV to be removed. 
            energy_unit (str): has to be either "kJ/mol" or "kcal/mol". Defaults to be "kJ/mol".
            kb (float, optional): the Boltzmann Constant in the energy unit. \
                Defaults to be None, which will be set according to energy_unit.
            temp (float) = temperature of the simulation in Kelvins.

        Return:
            New `FES` instance without the CV to be removed.
        """

        logger.info(f"Removing CV {CV}.")

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

        if CV > self.hills.cvs:
            raise ValueError(
                "Error: The CV to remove is not available in this FES object.")

        if kb == None:
            if energy_unit == "kJ/mol":
                kb = 8.314e-3
            elif energy_unit == "kcal/mol":
                kb = 8.314e-3 / 4.184
            else:
                raise ValueError(
                    "Please give the Boltzmann Constant in the energy unit.")

        if self.cvs == 1:
            raise ValueError("Error: You can not remove the only CV. ")
        else:
            probabilities = np.exp(-self.fes / (kb * temp))
            new_prob = np.sum(probabilities, axis=CV)

            new_fes = FreeEnergySurface.from_hills(hills=self.hills)
            new_fes.fes = - kb * temp * np.log(new_prob)
            new_fes.fes = new_fes.fes - np.min(new_fes.fes)
            new_fes.res = self.res

            mask = np.ones(self.cvs, dtype=bool)
            mask[CV] = False
            new_fes.cv_min = self.cv_min[mask]
            new_fes.cv_max = self.cv_max[mask]
            new_fes.cv_fes_range = self.cv_fes_range[mask]
            new_fes.cv_name = [
                j for i, j in enumerate(self.cv_name) if mask[i]]
            new_fes.cvs = self.cvs - 1
            return new_fes

    def remove_cvs(
        self,
        CVs: List[int],
        kb: Optional[float] = None,
        energy_unit: str = "kJ/mol",
        temp: float = 300.0
    ):
        """
        Remove multiple collective variables (CVs) from the free energy surface (FES).

        Args:
            CVs (List[int]): The list of CVs to be removed.
            kb (Optional[float], optional): The Boltzmann constant. Defaults to None.
            energy_unit (str, optional): The unit of energy. Defaults to "kJ/mol".
            temp (float, optional): The temperature in Kelvin. Defaults to 300.0.

        Returns:
            Fes: The new FES object with the specified CVs removed.
        """
        fes = self.remove_cv(CVs[0], kb, energy_unit, temp)
        if len(CVs) > 1:
            for CV in CVs[1:]:
                if fes is not None:
                    fes = fes.remove_cv(CV, kb, energy_unit, temp)
        return fes

    def set_fes(self, fes: np.ndarray):
        self.fes = fes

    def find_minima(self, nbins=8):
        """Method for finding local minima on FES.

        Args:
            fes (Fes): The Fes object to find the minima on.
            nbins (int, default=8): The number of bins used to divide the FES.
        """
        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use make_fes_original() first."
            )

        if self.minima is not None:
            logger.warning("Minima already found.")
            return None

        import pandas as pd

        cv_min = self.cv_min
        cv_max = self.cv_max

        if int(nbins) != nbins:
            nbins = int(nbins)
            logger.info(
                f"Number of bins must be an integer, it will be set to {nbins}.")
        if self.res % nbins != 0:
            raise ValueError(
                "Resolution of FES must be divisible by number of bins.")
        if nbins > self.res/2:
            raise ValueError("Number of bins is too high.")

        bin_size = int(self.res/nbins)

        for index in np.ndindex(tuple([nbins] * self.cvs)):
            # index serve as bin number
            _fes_slice = tuple(
                slice(
                    index[i] * bin_size, (index[i] + 1) * bin_size
                ) for i in range(self.cvs)
            )
            fes_slice = self.fes[_fes_slice]
            bin_min = np.min(fes_slice)

            # indexes of global minimum of a bin
            bin_min_arg = np.unravel_index(
                np.argmin(fes_slice), fes_slice.shape
            )
            # indexes of that minima in the original fes (indexes +1)
            min_cv_b = np.array([
                bin_min_arg[i] + index[i] * bin_size for i in range(self.cvs)
            ], dtype=int)

            if (np.array(bin_min_arg, dtype=int) > 0).all() and \
                    (np.array(bin_min_arg, dtype=int) < bin_size - 1).all():
                # if the minima is not on the edge of the bin
                min_cv = (((min_cv_b+0.5)/self.res) * (cv_max-cv_min))+cv_min
                local_minima = np.concatenate([
                    [np.round(bin_min, 6)], min_cv_b, np.round(min_cv, 6)
                ])
                if self.minima is None:
                    self.minima = local_minima
                else:
                    self.minima = np.vstack((self.minima, local_minima))
            else:
                # if the minima is on the edge of the bin
                around = np.zeros(tuple([3] * self.cvs))

                for product_index in product(*[range(3)] * self.cvs):
                    converted_index = np.array(product_index, dtype=int) + \
                        np.array(min_cv_b, dtype=int) - 1
                    converted_index[self.periodic] = \
                        converted_index[self.periodic] % self.res

                    mask = np.where(
                        (converted_index < 0) + (converted_index > self.res - 1)
                    )[0]

                    if len(mask) > 0:
                        around[product_index] = np.inf

                    elif product_index == tuple([1] * self.cvs):
                        around[product_index] = np.inf

                    else:
                        around[product_index] = self.fes[tuple(
                            converted_index)]

                if (around > bin_min).all():
                    min_cv = (((min_cv_b+0.5)/self.res)
                              * (cv_max-cv_min))+cv_min
                    local_minima = np.concatenate([
                        [np.round(bin_min, 6)], min_cv_b, np.round(min_cv, 6)
                    ])
                    if self.minima is None:
                        self.minima = local_minima
                    else:
                        self.minima = np.vstack((self.minima, local_minima))

        if self.minima is None:
            logger.warning("No minima found.")
            return None

        if len(self.minima.shape) > 1:
            self.minima = self.minima[self.minima[:, 0].argsort()]

        if self.minima.shape[0] == 1:
            self.minima = np.concatenate((
                np.arange(0, self.minima.shape[0], dtype=int), self.minima
            ))
        else:
            self.minima = np.column_stack((
                np.arange(0, self.minima.shape[0], dtype=int), self.minima
            ))

        if self.cv_name is None:
            self.cv_name = [f"CV{i+1}" for i in range(self.cvs)]

        minima_df = pd.DataFrame(
            np.array(self.minima),
            columns=["Minimum", "free energy"] +
            [f"CV{i+1}bin" for i in range(self.cvs)] +
            [f"CV{i+1} - {self.cv_name[i]}" for i in range(self.cvs)]
        )
        minima_df["Minimum"] = minima_df["Minimum"].astype(int)
        self.minima = minima_df

    def plot(
        self,
        png_name: Optional[str] = None,
        cmap: Union[str, Colormap] = "RdYlBu_r",
        energy_unit: str = "kJ/mol",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        image_size: List[int] = [10, 7],
        levels: int = 20,
        dpi: int = 96,
        surface: bool = False,
        surface_params: dict = {},
        **kwargs
    ):
        """
        Visualizes the free energy surface (FES) using Matplotlib and PyVista.

        Usage:
        ```python
        fes.plot()
        ```

        Args:
            png_name (str, optional): If provided, the picture of FES will be saved under this name in the current working directory.
            cmap (str, default="RdYlBu_r"): The Matplotlib colormap used to color the 2D or 3D FES.
            energy_unit (str, default="kJ/mol"): The unit used in the description of the colorbar.
            xlabel, ylabel (str, optional): If provided, they will be used as labels for the graphs.
            image_size (List[int], default=[10,7]): The width and height of the picture.
            levels (int, optional): A list of free energy values for isosurfaces in FES. Defaults to be 20.
            dpi (int, default=96): The resolution of the picture.
            surface (bool, default=False): Whether to plot the 3D surface of the FES.
            surface_params (dict, optional): A dictionary of parameters to be passed to the PyVista plotter when plotting the 3D surface.
            **kwargs: Additional keyword arguments to be passed to the Matplotlib plotter.

        Returns:
            fig, ax: The Matplotlib figure and axis objects.
        """

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        if self.fes is None:
            raise ValueError(
                "FES not calculated yet. Use makefes() or makefes2() first.")

        if type(cmap) is str:
            cmap = cm.get_cmap(cmap)

        cvs = self.cvs

        if cvs == 1:
            fig, ax = PlottingFES._plot1d(
                self, image_size=image_size, dpi=dpi,
                energy_unit=energy_unit, xlabel=xlabel, **kwargs
            )

        elif cvs == 2:
            if surface:
                fig, ax = PlottingFES._surface_plot(
                    self,
                    cmap=cmap, image_size=image_size, dpi=dpi,
                    xlabel=xlabel, ylabel=ylabel,
                    energy_unit=energy_unit,
                    **surface_params, **kwargs
                )
            fig, ax = PlottingFES._plot2d(
                self,
                levels=levels, cmap=cmap, image_size=image_size, dpi=dpi,
                xlabel=xlabel, ylabel=ylabel, 
                energy_unit=energy_unit, **kwargs
            )

        else:
            raise ValueError("Only 1D and 2D FES are supported.")

        if png_name != None:
            fig.savefig(png_name)

        return fig, ax

    def plot_minima(self, mark_color="white", png_name=None, **kwargs):
        fig, ax = PlottingFES.plot_minima(self, mark_color, **kwargs)
        if png_name is not None:
            fig.savefig(png_name)
        return fig, ax


class PlottingFES:
    """
    A class that provides methods for plotting free energy surfaces (FES) from a FES object.
    """

    @staticmethod
    def _surface_plot(
        fes: FreeEnergySurface,
        cmap: Union[str, Colormap] = "RdYlBu",
        energy_unit: str = "kJ/mol",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        label_size: int = 12,
        image_size: List[int] = [10, 7],
        rstride: int = 1,
        cstride: int = 1,
        dpi: int = 96,
        **kwargs
    ):
        """
        Visualizes the 2D free energy surface (FES) as a 3D surface plot using Matplotlib.

        Note: Interactivity is currently limited to jupyter notebook or jupyter lab in `%matplotlib widget` mode. Otherwise, it is a static image of the 3D surface plot.

        Usage:
        ```python
        %matplotlib widget
        fes.surface_plot()
        ```

        Future plans include implementing this function using PyVista. However, in the current version of PyVista (0.38.5), there is an issue with labels on the 3rd axis for free energy showing wrong values.
        """

        if fes.cv_name is None:
            fes.name_cv()

        if fes.cvs == 2:
            cv_min = fes.cv_min
            cv_max = fes.cv_max

            x = np.linspace(cv_min[0], cv_max[0], fes.res)
            y = np.linspace(cv_min[1], cv_max[1], fes.res)

            X, Y = np.meshgrid(x, y)
            Z = fes.fes

            canvas_style(**kwargs)

            fig, ax = plt.subplots(
                figsize=image_size, dpi=dpi,
                subplot_kw={"projection": "3d"}
            )
            ax.plot_surface(X, Y, Z, cmap=cmap,  # type: ignore
                            rstride=rstride, cstride=cstride)

            if xlabel == None:
                ax.set_xlabel(
                    f'CV1 - {fes.cv_name[0]}', size=label_size)  # type: ignore
            else:
                ax.set_xlabel(xlabel, size=label_size)
            if ylabel == None:
                ax.set_ylabel(
                    f'CV2 - {fes.cv_name[1]}', size=label_size)  # type: ignore
            else:
                ax.set_ylabel(ylabel, size=label_size)
            if zlabel == None:
                # type: ignore
                ax.set_zlabel(f'Free energy ({energy_unit})', size=label_size)
            else:
                ax.set_zlabel(zlabel, size=label_size)  # type: ignore
        else:
            raise ValueError(
                f"Surface plot only works for FES with exactly two CVs, and this FES has {fes.cvs}."
            )
        return fig, ax

    @staticmethod
    def _plot1d(
        fes: FreeEnergySurface,
        image_size: List[int] = [10, 7],
        dpi: int = 96,
        energy_unit: str = 'kJ/mol',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        **kwargs
    ):
        canvas_style(**kwargs)
        fig, ax = plt.subplots(
            figsize=(image_size[0], image_size[1]),
            dpi=dpi
        )
        X = np.linspace(fes.cv_min[0], fes.cv_max[0], fes.res)
        ax.plot(X, fes.fes)
        if xlabel == None:
            ax.set_xlabel(
                f'CV1 - {fes.cv_name[0]}')  # type: ignore
        else:
            ax.set_xlabel(xlabel)
        if ylabel == None:
            ax.set_ylabel(f'Free Energy ({energy_unit})')
        else:
            ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def _plot2d(
        fes_obj: FreeEnergySurface,
        levels: int = 20,
        cmap: Union[str, Colormap] = "RdYlBu",
        image_size: List[int] = [10, 7],
        dpi: int = 96,
        energy_unit: str = 'kJ/mol',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        **kwargs
    ):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        canvas_style(**kwargs)
        fig, ax = plt.subplots(
            figsize=(image_size[0], image_size[1]),
            dpi=dpi
        )

        if fes_obj.fes is None:
            raise ValueError(
                "FES not calculated yet. Use fes.makefes() or fes.makefes2() first.")
        X = np.linspace(fes_obj.cv_min[0], fes_obj.cv_max[0], fes_obj.res)
        Y = np.linspace(fes_obj.cv_min[1], fes_obj.cv_max[1], fes_obj.res)
        fes = fes_obj.fes.T
        cs = ax.contourf(X, Y, fes, levels=levels, cmap=cmap)
        ax.contour(X, Y, fes, levels=levels, colors="black", alpha=0.2)

        if xlabel == None:
            ax.set_xlabel(
                f'CV1 - {fes_obj.cv_name[0]}')
        else:
            ax.set_xlabel(xlabel)
        if ylabel == None:
            ax.set_ylabel(
                f'CV2 - {fes_obj.cv_name[1]}')
        else:
            ax.set_ylabel(ylabel)
        cbar = fig.colorbar(cs)
        if zlabel == None:
            cbar.set_label(f'Free Energy ({energy_unit})')
        else:
            cbar.set_label(zlabel)
        return fig, ax

    @staticmethod
    def plot_minima(
        fes_obj: FreeEnergySurface,
        mark_color: str = "white",
        **kwargs
    ):
        """
        Function used to visualize the FES objects with the positions of local minima shown as letters on the graph.

        Usage:
        ```python
        minima.plot()
        ```
        """
        if fes_obj.minima is None:
            raise ValueError("No minima found.")

        fig, ax = fes_obj.plot(**kwargs)

        free_energy_range = fes_obj.fes.max() - fes_obj.fes.min()

        if fes_obj.cvs == 1:
            for m in range(len(fes_obj.minima.index)):
                ax.text(
                    float(fes_obj.minima.iloc[m, 3]),
                    float(fes_obj.minima.iloc[m, 1])+free_energy_range*0.05,
                    fes_obj.minima.iloc[m, 0],
                    horizontalalignment='center',
                    c=mark_color,
                    verticalalignment='bottom'
                )

        elif fes_obj.cvs == 2:
            for m in range(len(fes_obj.minima.index)):
                ax.text(
                    float(fes_obj.minima.iloc[m, 4]),
                    float(fes_obj.minima.iloc[m, 5]),
                    fes_obj.minima.iloc[m, 0],
                    horizontalalignment='center',
                    verticalalignment='center',
                    c=mark_color
                )
        return fig, ax
