import numpy as np

from ardent.utilities import _validate_scalar_to_multi
from ardent.utilities import _validate_ndarray

"""
Test _validate_scalar_to_multi.
"""

def test__validate_scalar_to_multi():

    # Test proper use.

    kwargs = dict(value=1, size=1, dtype=float)
    correct_output = np.array([1], float)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=1, size=0, dtype=int)
    correct_output = np.array([], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=9.5, size=4, dtype=int)
    correct_output = np.full(4, 9, int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=[1, 2, 3.5], size=3, dtype=float)
    correct_output = np.array([1, 2, 3.5], float)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=[1, 2, 3.5], size=3, dtype=int)
    correct_output = np.array([1, 2, 3], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=(1, 2, 3), size=3, dtype=int)
    correct_output = np.array([1, 2, 3], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=np.array([1, 2, 3], float), size=3, dtype=int)
    correct_output = np.array([1, 2, 3], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    # Test improper use.

    kwargs = dict(value=[1, 2, 3, 4], size='size: not an int', dtype=float)
    expected_exception = TypeError
    match = "size must be interpretable as an integer."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=[], size=-1, dtype=float)
    expected_exception = ValueError
    match = "size must be non-negative."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=[1, 2, 3, 4], size=3, dtype=int)
    expected_exception = ValueError
    match = "The length of value must either be 1 or it must match size."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=np.arange(3*4, dtype=int).reshape(3,4), size=3, dtype=float)
    expected_exception = ValueError
    match = "value must not have more than 1 dimension."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=[1, 2, 'c'], size=3, dtype=int)
    expected_exception = ValueError
    match = "value and dtype are incompatible with one another."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value='c', size=3, dtype=int)
    expected_exception = ValueError
    match = "value and dtype are incompatible with one another."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

"""
Test _validate_ndarray.
"""

def test__validate_ndarray():

    # Test proper use.

    kwargs = dict(input=np.arange(3, dtype=int), dtype=float)
    correct_output = np.arange(3, dtype=float)
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(input=[[0,1,2], [3,4,5]], dtype=float)
    correct_output = np.arange(2*3, dtype=float).reshape(2,3)
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(input=np.array([0,1,2]), broadcast_to_shape=(2,3))
    correct_output = np.array([[0,1,2], [0,1,2]])
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(input=np.array(7), required_ndim=1)
    correct_output = np.array([7])
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    # Test improper use.

    # Validate arguments.

    kwargs = dict(input=np.arange(3), minimum_ndim=1.5)
    expected_exception = TypeError
    match = "minimum_ndim must be of type int."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.arange(3), minimum_ndim=-1)
    expected_exception = ValueError
    match = "minimum_ndim must be non-negative."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.arange(3), required_ndim=1.5)
    expected_exception = TypeError
    match = "required_ndim must be either None or of type int."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.arange(3), required_ndim=-1)
    expected_exception = ValueError
    match = "required_ndim must be non-negative."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.arange(3), dtype="not of type type")
    expected_exception = TypeError
    match = "dtype must be either None or a valid type."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)
    
    # Validate input.

    kwargs = dict(input=np.array(print), dtype=int)
    expected_exception = TypeError
    match = "input is of a type that is incompatible with dtype."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.array('string that is not an int'), dtype=int)
    expected_exception = ValueError
    match = "input has a value that is incompatible with dtype."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)
    
    kwargs = dict(input=np.array([[], 1]), dtype=None, forbid_object_dtype=True)
    expected_exception = TypeError
    match = "Casting input to a np.ndarray produces an array of dtype object while forbid_object_dtype == True and dtype != object."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.arange(3), required_ndim=2)
    expected_exception = ValueError
    match = "If required_ndim is not None, input.ndim must equal it unless input.ndim == 0 and required_ndin == 1."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(input=np.arange(3), minimum_ndim=2)
    expected_exception = ValueError
    match = "input.ndim must be at least equal to minimum_ndim."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

"""
Perform tests.
"""

if __name__ == "__main__":
    test__validate_scalar_to_multi()
    test__validate_ndarray()
