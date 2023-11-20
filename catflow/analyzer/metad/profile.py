import numpy as np
import pandas as pd

from itertools import product
from typing import List, Optional, Union
from matplotlib.colors import Colormap

from catflow.analyzer.graph.plotting import canvas_style
from catflow.analyzer.metad.fes import FreeEnergySurface
from catflow.analyzer.metad.hills import Hills


class FreeEnergyProfile:
    """
    A class to calculate and visualize the free energy profile of a metadynamics simulation.
    """

    def __init__(
        self, 
        fes: FreeEnergySurface, 
        hills: Hills, 
        profile_length: Optional[int] = None
    ):
        """
        Initializes a FreeEnergyProfile object with the given FES and Hills objects.

        Args:
            fes (FES): A FES object containing the collective variables (CVs) and the free energy surface.
            hills (Hills): A Hills object containing the collective variables (CVs), the Gaussian hills, and the sigma values.
            profile_length (int, optional): The length of the free energy profile. Defaults to None, for which the length is the same as the number of hills.

        Raises:
            ValueError: If there is only one local minimum on the free energy surface.
        """

        self.cvs = fes.cvs
        self.res = fes.res

        if type(fes.minima) != pd.DataFrame:
            raise ValueError(
                "There is only one local minimum on the free energy surface."
            )
        self.minima = fes.minima

        self.periodic = fes.periodic
        self.heights = hills.heights

        self.cv_name = fes.cv_name
        self.cv_min = fes.cv_min
        self.cv_max = fes.cv_max
        self.sigma = hills.sigma
        self.cv = hills.cv

        self.make_free_energy_profile(profile_length)

    def make_free_energy_profile(
        self, 
        profile_length: Optional[int] = None
    ):
        """Internal method to calculate free energy profile.

        Raises:
            ValueError: If there is only one local minimum on the free energy surface.
        """
        hills_length = len(self.cv[:, 0])

        if profile_length == None:
            profile_length = hills_length
            scan_times = np.arange(hills_length, dtype=int)
        else:
            scan_times = np.linspace(0, hills_length-1, profile_length, dtype=int)

        number_of_minima = self.minima.shape[0]
        self.free_profile = np.zeros((self.minima["Minimum"].shape[0]+1))

        cvs = self.cvs
        cv_min, cv_max = self.cv_min, self.cv_max
        cv_fes_range = self.cv_max - self.cv_min

        fes = np.zeros((self.res, self.res))

        last_time = 0

        for time in scan_times:
            minima_cv_matrix = np.array(
                self.minima.iloc[:, cvs+2:2*cvs+2], dtype=float
            )
            for coords in product(*minima_cv_matrix.T):
                dist_cvs = []
                for i in range(cvs):
                    dist_cv = self.cv[:, i][last_time:time] - coords[i]
                    if self.periodic[i]:
                        dist_cv[dist_cv < -0.5*cv_fes_range[i]
                                ] += cv_fes_range[i]
                        dist_cv[dist_cv > +0.5*cv_fes_range[i]
                                ] -= cv_fes_range[i]
                    dist_cvs.append(dist_cv)
                dp2 = np.sum(np.array(dist_cvs)**2, axis=0) / \
                    (2*self.sigma[:, 0][last_time:time]**2)
                tmp = np.zeros(self.cv[:, 0][last_time:time].shape)
                tmp[dp2 < 2.5] = self.heights[last_time:time][dp2 < 2.5] * \
                    (np.exp(-dp2[dp2 < 2.5]) *
                     1.00193418799744762399 - 0.00193418799744762399)
                fes[tuple([
                    int((float(coords[i])-cv_min[i])*self.res/cv_fes_range[i]) for i in range(cvs)
                ])] -= tmp.sum()

            # save profile
            profile_line = [time]
            for m in range(number_of_minima):
                profile = fes[tuple([
                    int(float(self.minima.iloc[m, i+2])) for i in range(cvs)
                ])] - fes[tuple([
                    int(float(self.minima.iloc[0, i+2])) for i in range(cvs)
                ])]
                profile_line.append(profile)
            self.free_profile = np.vstack([self.free_profile, profile_line])

            last_time = time

    def plot(
        self,
        image_size: List[int] = [10, 7],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        time_unit: str = "ps",
        energy_unit: str = "kJ/mol",
        label_size: int = 12,
        cmap: Union[str, Colormap] = "RdYlBu",
        png_name: Optional[str] = None,
        **kwargs
    ):
        """
        Visualization function for Free Energy Profile (FEP).

        Usage:
        ```python
        fep.plot()
        ```

        Args:
            name (str): Name for the .png file to save the plot to. Defaults to be "FEProfile.png".
            image_size (List[int]): List of two dimensions of the picture. Defaults to be [10, 7].
            xlabel (str, optional): X-axis label. Defaults to be None.
            ylabel (str, optional): Y-axis label. Defaults to be None.
            label_size (int): Size of labels. Defaults to be 12.
            cmap (str): Matplotlib colormap used for coloring the line of the minima. Defaults to be "jet".
        """

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        canvas_style(**kwargs)

        fig, ax = plt.subplots(figsize=(image_size[0], image_size[1]))

        if type(cmap) == str:
            cmap = cm.get_cmap(cmap)

        # colors = cm.jet((self.minima.iloc[:,1].to_numpy()).astype(float)/\
        #                (np.max(self.minima.iloc[:,1].to_numpy().astype(float))))
        colors = cmap(np.linspace(0.15, 0.85, self.minima.shape[0]))
        for m in range(self.minima.shape[0]):
            ax.plot(
                self.free_profile[:, 0],
                self.free_profile[:, m+1],
                color=colors[m],
                label=f"Minima {self.minima.iloc[m, 0]}",
            )

        ax.legend()

        if xlabel == None:
            ax.set_xlabel(f'Time ({time_unit})')
        else:
            ax.set_xlabel(xlabel)
        if ylabel == None:
            ax.set_ylabel(f'Free energy difference ({energy_unit})')
        else:
            ax.set_ylabel(ylabel)

        if png_name != None:
            fig.savefig(png_name)
