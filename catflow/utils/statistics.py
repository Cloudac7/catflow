from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def block_average(data: ArrayLike, block_size: int, axis: int = 0) -> Tuple[float, float]:
    data = np.array(data)
    if data.ndim == 1:
        data = np.reshape(data, (-1, 1))  # Reshape data into a 2D array
    N_b = data.shape[axis] // block_size
    if N_b == 0:
        raise ValueError(f"Cannot block average data: block size {block_size} is larger than "
                         f"the size of the array along axis {axis} ({data.shape[axis]})")
    # drop the last block if it is not full
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, block_size * N_b)

    # reshape data into blocks
    reshaped_data = reshape_array(data[tuple(slices)], N_b, axis)
    blocked_data = np.mean(reshaped_data, axis=axis+1)
    mean = np.mean(blocked_data, axis=axis)
    var = np.std(blocked_data, ddof=1, axis=axis) / np.sqrt(N_b)
    return mean, var


def auto_correlation(data, tau_max):
    mean = np.mean(data)
    var = np.var(data)
    tau_array = np.arange(1, tau_max + 1)
    r_tau = np.array([[np.mean([(data[i] - mean) * (data[i + tau] - mean)
                     for i in np.arange(len(data) - tau)])] for tau in tau_array]) / var
    return tau_array, r_tau


def reshape_array(data: np.ndarray, new_dim: int, axis: int) -> np.ndarray:
    shape = list(data.shape)
    old_dim = shape[axis]
    if old_dim % new_dim != 0:
        raise ValueError(f"Cannot reshape array: {old_dim} is not divisible by {new_dim}")
    shape[axis] = new_dim
    shape.insert(axis + 1, old_dim // new_dim)
    return data.reshape(shape)
