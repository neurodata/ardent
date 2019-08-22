from .normalization import cast_to_typed_array
from .normalization import normalize_by_MAD
from .normalization import center_to_mean
from .normalization import pad

from .resampling import downsample_image
from .resampling import change_resolution_to
from .resampling import change_resolution_by

# TODO: update preprocessing_functions.
preprocessing_functions = [
    'cast_to_typed_array',
    'normalize_by_MAD',
    'center_to_mean',
    'pad',

    'downsample_image',
    'change_resolution_to',
    'change_resolution_by',
    ]

"""
Preprocessing Pipeline
"""

import numpy as np

# TODO: incorporate arguments.
def preprocess(data:(np.ndarray, list), processes:list):
    """
    Perform each preprocessing function in <processes>, in the order listed, 
    on the np.ndarray <data>, or on each np.ndarray in <data> if <data> is a list.
    
    Arguments:
        data {np.ndarray, list} -- The array or list of arrays to be preprocessed.
        processes {list} -- A list of strings, each of which must be the name of a function imported in this module.
    
    Raises:
        TypeError: If <data> is a list, each element must be of type np.ndarray.
        TypeError: <data> must be of type np.ndarray or of type list.
        ValueError: <processes> must be castable as a string-type np.ndarray.
        ValueError: Each element of <processes> must be the name of a function imported in this module.
    
    Returns:
        np.ndarray, list -- The result of applying each function listed in <processes>
        in the order provided to the array <data>, or to each element of <data> if <data> is a list of arrays.
    """

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

    # Return in a form appropriate to what was passed in, 
    # i.e. list in, list out, np.ndarray in, np.ndarray out.
    return data[0] if isinstance(data, list) and len(data) == 1 else data