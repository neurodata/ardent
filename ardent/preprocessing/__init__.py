from .normalization import normalize_by_MAD
from .normalization import center_to_mean
from .normalization import pad

from .resampling import downsample_image
from .resampling import change_resolution_to
from .resampling import change_resolution_by

# TODO: update preprocessing_functions.
preprocessing_functions = [
    'normalize_by_MAD', 
    'center_to_mean', 
    'pad',
    ]

"""
Preprocessing Pipeline
"""

import numpy as np

# TODO: incorporate arguments.
def preprocess(data:(np.ndarray, list), processes:list):
    """Given data and a list of processes from the preprocessing subpackage, 
    perform all such processes on data in the order provided and return the result.
    
    data can be either a np.ndarray or list of np.ndarrays."""

    # Verify data.
    if isinstance(data, list):
        if not all(isinstance(datam, np.ndarray) for datum in data):
            raise TypeError(f"If data is a list, all elements must be np.ndarrays.\n"
                f"type(data[0]): {type(data[0])}.")
    elif isinstance(data, np.ndarray):
        data = [data]
    else:
        # data is neither a list nor a np.ndarray.
        raise TypeError(f"data must be a np.ndarray or a list of np.ndarrays.")
    
    # Validate processes.
    try:
        processes = np.array(processes, str)
    except ValueError:
        raise ValueError(f"processes could not be cast to a string-type np.ndarray.\n"
            f"processes: {processes}.")
    if not isinstance(processes, np.ndarray):
        raise TypeError(f"processes must be a type that can be cast to a np.ndarray of strings.\n"
            f"type(processes): {type(processes)}.")
    
    # Process each np.ndarray.
    # If data was passed in as a single np.ndarray, 
    # then data is now a 1-element list containing that np.ndarray: [data].
    for data_index, datum in enumerate(data):
        for process in processes:
            if process in preprocessing_functions:
                data[data_index] = eval(f"{process}(datum)")
            else:
                raise ValueError(f"Process {process} not recognized.\n"
                    f"Recognized processes: {preprocessing_functions}.")

    # Return in a form appropriate to what was passed in.
    # i.e. list in, list out, np.ndarray in, np.ndarray out, empty list in, None out.
    return data if len(data) > 1 else data[0] if len(data) == 1 else None