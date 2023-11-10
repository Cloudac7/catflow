import numpy as np

from joblib import Parallel, delayed
from .utils import load_reader

def atomic_ener_model_devi_atomic(*files, key_name, **kwargs):
    """Calculate the deviation of atomic energy from a model."""

    def _atomic_ener_model_devi(f, key_name, **kwargs):
        reader = load_reader(f, **kwargs)
        results = reader.read_atomic_property()
        return results[key_name]
    
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_atomic_ener_model_devi)(f, key_name, **kwargs) for f in files
    ) 
    results = np.array(results)

    return np.std(results, axis=0)

def atomic_ener_model_devi(*files, key_name, **kwargs):
    results = atomic_ener_model_devi_atomic(*files, key_name=key_name, **kwargs)
    return np.max(results, axis=1), np.min(results, axis=1), np.mean(results, axis=1)
