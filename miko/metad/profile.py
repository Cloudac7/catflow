import numpy as np
import pandas as pd

from itertools import product
from typing import List, Optional, Union
from matplotlib.colors import Colormap

from miko.graph.plotting import canvas_style
from miko.metad.hills import Hills
from miko.metad.minima import Minima


class FEProfile:
    """
    Free energy profile is a visualization of differences between local 
    minima points during metadynamics simulation. If the values seem 
    to converge to a mean value of the difference, it suggests, 
    but not fully proof, that the FES did converge to the correct shape.

    Usage:
    ```python
    fep = miko.metad.FEProfile(minima, hillsfile)
    ```

    Args:
        minima (Minima): `Minima` to be calculated the free energy profile for.
        hillsfile (Hills): File of `Hills` to calculate the free energy profile from.

    """

    def __init__(self, minima: Minima, hills: Hills):
        self.cvs = minima.cvs
        self.res = minima.res

        if type(minima.minima) != pd.DataFrame:
            raise ValueError(
                "There is only one local minimum on the free energy surface."
            )
        self.minima = minima.minima

        self.periodic = minima.periodic
        self.heights = hills.get_heights()

        self.cv_name = minima.cv_name
        self.cv_min = minima.cv_min
        self.cv_max = minima.cv_max
        self.cv_per = minima.cv_per
        self.sigma = hills.sigma
        self.cv = hills.cv

        self.makefeprofile(hills)

    def makefeprofile(self, hills):
        """
        Internal method to calculate free energy profile.
        """
        hillslenght = len(hills.cv[:, 0])

        if hillslenght < 256:
            profilelenght = hillslenght
            scantimes = np.array(range(hillslenght))
        else:
            profilelenght = 256
            scantimes = np.array(
                ((hillslenght/(profilelenght))*np.array((range(1, profilelenght+1)))))
            scantimes -= 1
            scantimes = scantimes.astype(int)

        number_of_minima = self.minima.shape[0]

        self.feprofile = np.zeros((self.minima["Minimum"].shape[0]+1))

        cvs = self.cvs
        cv_min, cv_max = self.cv_min, self.cv_max
        cv_fes_range = self.cv_max - self.cv_min

        fes = np.zeros((self.res, self.res))

        lasttime = 0
        line = 0
        for time in scantimes:
            minima_cv_matrix = np.array(
                self.minima.iloc[:, cvs+2:2*cvs+2], dtype=float
            )
            for coords in product(*minima_cv_matrix.T):
                dist_cvs = []
                for i in range(cvs):
                    dist_cv = self.cv[:, i][lasttime:time] - coords[i]
                    if self.periodic[i]:
                        dist_cv[dist_cv < -0.5*cv_fes_range[i]
                                ] += cv_fes_range[i]
                        dist_cv[dist_cv > +0.5*cv_fes_range[i]
                                ] -= cv_fes_range[i]
                    dist_cvs.append(dist_cv)
                dp2 = np.sum(np.array(dist_cvs)**2, axis=0) / \
                    (2*self.sigma[:, 0][lasttime:time]**2)
                tmp = np.zeros(self.cv[:, 0][lasttime:time].shape)
                tmp[dp2 < 2.5] = self.heights[lasttime:time][dp2 < 2.5] * \
                    (np.exp(-dp2[dp2 < 2.5]) *
                     1.00193418799744762399 - 0.00193418799744762399)
                fes[tuple([
                    int((float(coords[i])-cv_min[i])*self.res/cv_fes_range[i]) for i in range(cvs)
                ])] -= tmp.sum()

            # save profile
            profileline = [time]
            for m in range(number_of_minima):
                profile = fes[tuple([
                    int(float(self.minima.iloc[m, i+2])) for i in range(cvs)
                ])] - fes[tuple([
                    int(float(self.minima.iloc[0, i+2])) for i in range(cvs)
                ])]
                profileline.append(profile)
            self.feprofile = np.vstack([self.feprofile, profileline])

            lasttime = time

    def plot(
        self,
        png_name: str = None,
        image_size: List[int] = [10, 7],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        time_unit: str = "ps",
        energy_unit: str = "kJ/mol",
        label_size: int = 12,
        cmap: Union[str, Colormap] = "RdYlBu",
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
                self.feprofile[:, 0],
                self.feprofile[:, m+1],
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
