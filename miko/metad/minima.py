import sys
import numpy as np
import pandas as pd

from itertools import product
from typing import Optional, List
from miko.metad.fes import FES, PlottingFES

from miko.utils.log_factory import logger

class Minima:
    """
    Represents an object of the Minima class used to find local free energy minima on a free energy surface (FES).

    The FES is divided into a specified number of bins (default is 8), and the absolute minima is found for each bin. 
    The algorithm then checks if each found point is a local minimum by comparing it to the surrounding points on the FES.

    The list of minima is stored as a pandas DataFrame.

    Usage:
    ```python
    minima = miko.metad.Minima(fes=f, nbins=8)
    ```

    The list of minima can be accessed using the `minima.minima` attribute:
    ```python
    print(minima.minima)
    ```

    Args:
        fes (Fes): The Fes object to find the minima on.
        nbins (int, default=8): The number of bins used to divide the FES.
    """

    def __init__(
        self, fes: FES, nbins=8
    ):
        if fes.fes is None:
            raise ValueError("FES is not defined.")
        self.fes_obj = fes
        self.fes = fes.fes
        self.periodic = fes.periodic
        self.cvs = fes.cvs
        self.res = fes.res

        self.cv_name = fes.cv_name

        # use remapped cv_min and cv_max
        self.cv_min = fes.cv_min
        self.cv_max = fes.cv_max

        self.cv_per = fes.hills.cv_per

        self.findminima(nbins=nbins)

    def findminima(self, nbins=8):
        """Method for finding local minima on FES.

        Args:
            fes (Fes): The Fes object to find the minima on.
            nbins (int, default=8): The number of bins used to divide the FES.
        """
        cv_min = self.cv_min
        cv_max = self.cv_max

        if int(nbins) != nbins:
            nbins = int(nbins)
            logger.info(
                f"Number of bins must be an integer, it will be set to {nbins}.")
        if self.res % nbins != 0:
            raise ValueError("Resolution of FES must be divisible by number of bins.")
        if nbins > self.res/2:
            raise ValueError("Number of bins is too high.")
        
        bin_size = int(self.res/nbins)

        self.minima = None

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
                    min_cv = (((min_cv_b+0.5)/self.res) * (cv_max-cv_min))+cv_min
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

        minima_df = pd.DataFrame(
            np.array(self.minima),
            columns=["Minimum", "free energy"] +
            [f"CV{i+1}bin" for i in range(self.cvs)] +
            [f"CV{i+1} - {self.cv_name[i]}" for i in range(self.cvs)]
        )
        minima_df["Minimum"] = minima_df["Minimum"].astype(int)
        self.minima = minima_df

    def plot(self, mark_color="white", png_name=None, **kwargs):
        fig, ax = PlottingMinima.plot(self, mark_color, **kwargs)
        if png_name is not None:
            fig.savefig(png_name)
        return fig, ax


class PlottingMinima:

    @staticmethod
    def plot(
        minima: Minima,
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
        fes_obj = minima.fes_obj
        fig, ax = fes_obj.plot(**kwargs)

        ferange = minima.fes.max() - minima.fes.min()

        if minima.minima is None:
            raise ValueError("No minima found.")

        if minima.cvs == 1:
            for m in range(len(minima.minima.index)):
                ax.text(
                    float(minima.minima.iloc[m, 3]), 
                    float(minima.minima.iloc[m, 1])+ferange*0.05, 
                    minima.minima.iloc[m, 0],
                    horizontalalignment='center', 
                    c=mark_color,
                    verticalalignment='bottom'
                )

        elif minima.cvs == 2:
            for m in range(len(minima.minima.index)):
                ax.text(
                    float(minima.minima.iloc[m, 4]), 
                    float(minima.minima.iloc[m, 5]), 
                    minima.minima.iloc[m, 0],
                    horizontalalignment='center',
                    verticalalignment='center', 
                    c=mark_color
                )
        return fig, ax
