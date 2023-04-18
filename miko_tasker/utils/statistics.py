from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

def block_average(data: ArrayLike, block_size: int) -> Tuple[float, float]:
    data = np.array(data)
    N_b = len(data) // block_size
    # drop the last block if it is not full
    reshaped_data = data[ : block_size * N_b].reshape(-1, block_size)
    blocked_data = np.mean(reshaped_data, axis=1)
    mean = np.mean(blocked_data)
    var = np.std(blocked_data, ddof=1) / np.sqrt(N_b)
    return mean, var
