from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def block_average(data: ArrayLike, block_size: int, axis: int = 0) -> Tuple[float, float]:
    data = np.array(data)
    if data.ndim == 1:
        data = np.reshape(data, (-1, 1))  # Reshape data into a 2D array
    N_b = data.shape[axis] // block_size
    # drop the last block if it is not full
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, block_size * N_b)
    reshaped_data = data[tuple(slices)].reshape(tuple(-1 if i == axis else data.shape[i]
                                                      for i in range(data.ndim - 1)) + (N_b,))
    blocked_data = np.mean(reshaped_data, axis=axis)
    mean = np.mean(blocked_data)
    var = np.std(blocked_data, ddof=1, axis=axis) / np.sqrt(N_b)
    return mean, var


def auto_correlation(data, tau_max):
    mean = np.mean(data)
    var = np.var(data)
    tau_array = np.arange(1, tau_max + 1)
    r_tau = np.array([[np.mean([(data[i] - mean) * (data[i + tau] - mean)
                     for i in np.arange(len(data) - tau)])] for tau in tau_array]) / var
    return tau_array, r_tau
