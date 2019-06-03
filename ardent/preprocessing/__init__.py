from .normalization import normalize_by_MAD
from .normalization import center_to_mean
from .normalization import pad

preprocessing_functions = [
    'normalize_by_MAD', 
    'center_to_mean', 
    'pad',
    ]

import numpy as np

def preprocess(data:np.ndarray, processes:list):
    """Given data and a list of processes from the preprocessing subpackage, 
    perform all such processes on data in the order provided and return the result."""

    # Verify data.
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a np.ndarray.\ntype(data): {type(data)}.")
    
    # Validate processes.
    processes = np.array(processes).astype(str)
    if not isinstance(processes, np.ndarray):
        raise TypeError(f"processes must be a type that can be cast to a np.ndarray of strings.\n"
            f"type(processes): {type(processes)}.")
    
    for process in processes:
        if process in preprocessing_functions:
            data = eval(f"{process}(data)")