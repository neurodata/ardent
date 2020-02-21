from .normalization import cast_to_typed_array
from .normalization import normalize_by_MAD
from .normalization import center_to_mean
from .normalization import pad

from .bias_and_artifact_correction import correct_bias_field


# TODO: update preprocessing_functions, include resample.
preprocessing_functions = [
    # from .normalization:
    'cast_to_typed_array',
    'normalize_by_MAD',
    'center_to_mean',
    'pad',
    # from .bias_and_artifact_correction:
    'correct_bias_field',
    ]

"""
Preprocessing Pipeline
"""

import numpy as np

def preprocess(data, processes, process_kwargs=None):
    """
    Perform each preprocessing function in processes, in the order listed, 
    on data if it is an array, or on each element in data if it is a list of arrays.
    
    Args:
        data (np.ndarray, list): The array or list of arrays to be preprocessed.
        processes (list): The list of strings, each corresponding to the name of a preprocessing function.
        process_kwargs (seq, optional): A sequence of dictionaries containing kwargs for each element of processes. Defaults to None.
    
    Raises:
        TypeError: Raised if data is a list whose elements are not all of type np.ndarray.
        TypeError: Raised if data is neither a np.ndarray or a list of np.ndarrays.
        ValueError: Raised if processes cannot be cast to a np.ndarray with dtype str.
        ValueError: Raised if any element of processes is not a recognized preprocessing function.
        ValueError: Raised if process_kwargs are provided with a length not matching that of processes.
        TypeError: Raised if process_kwargs contains an element other than a dictionary.
    
    Returns:
        np.ndarray, list: A copy of data after having each function in processes applied.
    """

    # Validate inputs.

    # Verify data.
    if isinstance(data, list):
        if not all(isinstance(datum, np.ndarray) for datum in data):
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

    # Verify process_kwargs.
    if process_kwargs is None:
        process_kwargs = np.full_like(processes, fill_value=dict(), dtype=dict)
    process_kwargs = np.array(process_kwargs)
    if len(process_kwargs) != len(processes):
        raise ValueError(f"If provided, process_kwargs must match the length of processes.\n"
                         f"len(process_kwargs): {len(process_kwargs)}.")
    for process_index, process_kwarg in enumerate(process_kwargs):
        if not isinstance(process_kwarg, dict):
            raise TypeError(f"If provided, process_kwargs must be a sequence containing only dictionaries.\n"
                            f"type(process_kwargs[{process_index}]): {type(process_kwargs[process_index])}.")
    
    # Process each np.ndarray.
    # If data was passed in as a single np.ndarray, 
    # then data is now a 1-element list containing that np.ndarray: [data].
    for data_index, datum in enumerate(data):
        for process_index, process in enumerate(processes):
            if process in preprocessing_functions:
                datum = eval(f"{process}(datum, **process_kwargs[process_index])")
            else:
                raise ValueError(f"Process {process} not recognized.\n"
                    f"Recognized processes: {preprocessing_functions}.")
        data[data_index] = datum

    # Return in a form appropriate to what was passed in, 
    # i.e. list in, list out, np.ndarray in, np.ndarray out.
    return data[0] if isinstance(data, list) and len(data) == 1 else data
