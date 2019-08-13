# jesus that's a lot lines of test code
# my god man

import pytest

import numpy as np
from scipy.interpolate import interpn

from ardent.preprocessing.resampling import _validate_scalar_to_multi
from ardent.preprocessing.resampling import _validate_xyz_resolution
from ardent.preprocessing.resampling import _validate_ndarray
from ardent.preprocessing.resampling import _compute_axes
from ardent.preprocessing.resampling import _compute_coords
from ardent.preprocessing.resampling import _resample
from ardent.preprocessing.resampling import _downsample_along_axis
from ardent.preprocessing.resampling import downsample_image
from ardent.preprocessing.resampling import change_resolution_to
from ardent.preprocessing.resampling import change_resolution_by

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
Test _validate_xyz_resolution.
"""

def test__validate_xyz_resolution():

    # Test proper use.
    
    kwargs = dict(ndim=1, xyz_resolution=2)
    correct_output = np.full(1, 2, float)
    assert np.array_equal(_validate_xyz_resolution(**kwargs), correct_output)

    kwargs = dict(ndim=4, xyz_resolution=1.5)
    correct_output = np.full(4, 1.5, float)
    assert np.array_equal(_validate_xyz_resolution(**kwargs), correct_output)

    kwargs = dict(ndim=3, xyz_resolution=np.ones(3, int))
    correct_output = np.ones(3, float)
    assert np.array_equal(_validate_xyz_resolution(**kwargs), correct_output)

    kwargs = dict(ndim=2, xyz_resolution=[3, 4])
    correct_output = np.array([3, 4], float)
    assert np.array_equal(_validate_xyz_resolution(**kwargs), correct_output)

    # Test improper use.

    kwargs = dict(ndim=2, xyz_resolution=[3, -4])
    expected_exception = ValueError
    match = "All elements of xyz_resolution must be positive."
    with pytest.raises(expected_exception, match=match):
        _validate_xyz_resolution(**kwargs)

    kwargs = dict(ndim=2, xyz_resolution=[3, 0])
    expected_exception = ValueError
    match = "All elements of xyz_resolution must be positive."
    with pytest.raises(expected_exception, match=match):
        _validate_xyz_resolution(**kwargs)

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
Test _compute_axes.
"""

def test__compute_axes():

    # Test proper use.

    # _compute_axes produces a list with a np.ndarray for each element in shape.

    kwargs = dict(shape=(0, 1, 2), xyz_resolution=1, origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((0, 1, 2), (1, 1, 1))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=(1, 2, 3, 4), xyz_resolution=1.5, origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((1, 2, 3, 4), (1.5, 1.5, 1.5, 1.5))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=(2, 3, 4), xyz_resolution=[1, 1.5, 2], origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((2, 3, 4), (1, 1.5, 2))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=5, xyz_resolution=1, origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((5,), (1,))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=5, xyz_resolution=1, origin='zero')
    correct_output = [np.arange(dim_size) * dim_res
        for dim_size, dim_res in zip((5,), (1,))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

"""
Test _compute_coords.
"""

def test__compute_coords():

    # Test proper use.

    kwargs = dict(shape=5, xyz_resolution=1, origin='center')
    correct_output = np.array([[-2], [-1], [0], [1], [2]])
    assert np.array_equal(_compute_coords(**kwargs), correct_output)

    kwargs = dict(shape=(3,4), xyz_resolution=1, origin='zero')
    correct_output = np.array([[[0,0], [0,1], [0,2], [0,3]], [[1,0], [1,1], [1,2], [1,3]], [[2,0], [2,1], [2,2], [2,3]]])
    assert np.array_equal(_compute_coords(**kwargs), correct_output)

"""
Test _resample.
"""

def test__resample():

    # Test proper use.

    shape = (3,4)
    resolution = 1.5
    origin = 'zero'
    attempted_scales = 1
    image = np.arange(np.prod(shape)).reshape(shape)
    real_axes = _compute_axes(image.shape, resolution, origin=origin)
    new_shape = np.floor(np.multiply(image.shape, attempted_scales))
    real_scales = np.divide(new_shape, image.shape)
    new_shape = np.floor(np.multiply(np.subtract(image.shape, 1), real_scales)) + 1
    new_real_coords = _compute_coords(new_shape, resolution / real_scales, origin=origin)
    correct_output = interpn(points=real_axes, values=image, xi=new_real_coords)
    assert np.array_equal(_resample(image, real_axes, new_real_coords), correct_output)

    shape = (3)
    resolution = 1
    origin = 'zero'
    attempted_scales = 1
    image = np.arange(np.prod(shape)).reshape(shape)
    real_axes = _compute_axes(image.shape, resolution, origin=origin)
    new_shape = np.floor(np.multiply(image.shape, attempted_scales))
    real_scales = np.divide(new_shape, image.shape)
    new_shape = np.floor(np.multiply(np.subtract(image.shape, 1), real_scales)) + 1
    new_real_coords = _compute_coords(new_shape, resolution / real_scales, origin=origin)
    correct_output = interpn(points=real_axes, values=image, xi=new_real_coords)
    assert np.array_equal(_resample(image, real_axes, new_real_coords), correct_output)

    shape = (3,4,5)
    resolution = (0.5,1,1.5)
    origin = 'center'
    attempted_scales = (1/2,1/3,1/4)

    image = np.arange(np.prod(shape)).reshape(shape)
    real_axes = _compute_axes(image.shape, resolution, origin=origin)
    new_shape = np.floor(np.multiply(image.shape, attempted_scales))
    real_scales = np.divide(new_shape, image.shape)
    new_shape = np.floor(np.multiply(np.subtract(image.shape, 1), real_scales)) + 1
    new_real_coords = _compute_coords(new_shape, resolution / real_scales, origin=origin)
    correct_output = interpn(points=real_axes, values=image, xi=new_real_coords)
    assert np.array_equal(_resample(image, real_axes, new_real_coords), correct_output)

    shape = (6,7,8,9)
    resolution = (0.5,1,1.5,2.5)
    origin = 'center'
    attempted_scales = (2,3,3.5,np.pi)

    image = np.arange(np.prod(shape)).reshape(shape)
    real_axes = _compute_axes(image.shape, resolution, origin=origin)
    new_shape = np.floor(np.multiply(image.shape, attempted_scales))
    real_scales = np.divide(new_shape, image.shape)
    new_shape = np.floor(np.multiply(np.subtract(image.shape, 1), real_scales)) + 1
    new_real_coords = _compute_coords(new_shape, resolution / real_scales, origin=origin)
    correct_output = interpn(points=real_axes, values=image, xi=new_real_coords)
    assert np.array_equal(_resample(image, real_axes, new_real_coords), correct_output)





    # # Test uniform resolutions.

    # real_axes = _compute_axes(image.shape, resolution, origin='center')
    
    # new_shape = np.floor(np.multiply(image.shape, xyz_scales))
    # real_scales = np.divide(new_shape, image.shape)
    # new_real_coords = _compute_coords(new_shape, resolution / real_scales, origin='center')
    
    # correct_output = interpn(points=real_axes, values=image, xi=new_real_coords)
    # assert np.array_equal(_resample(image, real_axes, new_real_coords), correct_output)

    # # Test non-uniform resolutions.

    # dynamic_resolution = np.arange(1, image.ndim + 1) * resolution
    # real_axes = _compute_axes(image.shape, dynamic_resolution, origin='center')
    
    # new_shape = np.floor(np.multiply(image.shape, xyz_scales))
    # real_scales = np.divide(new_shape, image.shape)
    # new_real_coords = _compute_coords(new_shape, dynamic_resolution / real_scales, origin='center')
    
    # correct_output = interpn(points=real_axes, values=image, xi=new_real_coords)
    # assert np.array_equal(_resample(image, real_axes, new_real_coords), correct_output)

# image = np.arange(3*4).reshape(3,4)
# resolution = 0.5
# xyz_scales = 1/3
# test__resample(image=image, resolution=resolution, xyz_scales=xyz_scales)

"""
Test _downsample_along_axis.
"""

def test__downsample_along_axis():

    # Test proper use.

    # Test identity.
    kwargs = dict(
        image=np.arange(3*4).reshape(3,4), 
        axis=0, 
        scale_factor=1, 
        truncate=False, 
    )
    correct_output = np.arange(3*4).reshape(3,4)
    assert np.array_equal(_downsample_along_axis(**kwargs), correct_output)

    # Test basic use with negative axis.
    kwargs = dict(
        image=np.arange(3*4).reshape(3,4), 
        axis=-1, 
        scale_factor=2, 
        truncate=False, 
    )
    correct_output = np.arange(0.5,12,2).reshape(3,2)
    assert np.array_equal(_downsample_along_axis(**kwargs), correct_output)

    # Test uneven case.
    kwargs = dict(
        image=np.arange(9*3,dtype=float).reshape(9,3), 
        axis=0, 
        scale_factor=4, 
        truncate=False, 
    )
    correct_output = np.stack(
        (
            np.average(np.arange(9*3).reshape(9,3).take(range(3),0),0), 
            np.average(np.arange(9*3).reshape(9,3).take(range(3,3+4),0),0),
            np.average(np.arange(9*3).reshape(9,3).take(range(3+4,9),0),0),
        ),0)
    assert np.array_equal(_downsample_along_axis(**kwargs), correct_output)
    
    # Test uneven case with dtype=int.
    kwargs = dict(
        image=np.arange(9*3,dtype=int).reshape(9,3), 
        axis=0, 
        scale_factor=4, 
        truncate=False, 
    )
    image = np.arange(9*3,dtype=int).reshape(9,3)
    intermediate_padded_image = np.concatenate((
        # np.pad(image.astype(int), mode='mean') rounds half to even before casting to int.
        np.round(np.mean(image.take(range(3),0),0,keepdims=True)), 
        np.arange(9*3,dtype=int).reshape(9,3), 
        np.round(np.mean(image.take(range(7,9),0),0,keepdims=True)), 
        np.round(np.mean(image.take(range(7,9),0),0,keepdims=True)), 
    ),0)
    correct_output = np.stack(
        (
            np.average(intermediate_padded_image.take(range(4),0),0), 
            np.average(intermediate_padded_image.take(range(4,4+4),0),0),
            np.average(intermediate_padded_image.take(range(4+4,12),0),0),
        ),0)
    assert np.array_equal(_downsample_along_axis(**kwargs), correct_output)
    
    # Test uneven case with truncation.
    kwargs = dict(
        image=np.arange(3*11,dtype=float).reshape(3,11), 
        axis=-1, 
        scale_factor=4, 
        truncate=True, 
    )
    correct_output = np.stack(
        (
            np.average(np.arange(3*11).reshape(3,11).take(range(1,1+4),1),1), 
            np.average(np.arange(3*11).reshape(3,11).take(range(1+4,1+4+4),1),1),
        ),1)
    assert np.array_equal(_downsample_along_axis(**kwargs), correct_output)

    # Test improper use.

    kwargs = dict(image=np.arange(3), axis=1.5, scale_factor = 1, truncate=False)
    expected_exception = TypeError
    match = "axis must be of type int."
    with pytest.raises(expected_exception, match=match):
        _downsample_along_axis(**kwargs)

    kwargs = dict(image=np.arange(3), axis=10, scale_factor = 1, truncate=False)
    expected_exception = ValueError
    match = "axis must be in range -image.ndim to image.ndim."
    with pytest.raises(expected_exception, match=match):
        _downsample_along_axis(**kwargs)

    kwargs = dict(image=np.arange(3), axis=0, scale_factor = 1.5, truncate=False)
    expected_exception = ValueError
    match = "scale_factor must be integer-valued."
    with pytest.raises(expected_exception, match=match):
        _downsample_along_axis(**kwargs)

    kwargs = dict(image=np.arange(3), axis=0, scale_factor = [], truncate=False)
    expected_exception = TypeError
    match = "scale_factor must be castable to type int."
    with pytest.raises(expected_exception, match=match):
        _downsample_along_axis(**kwargs)

    kwargs = dict(image=np.arange(3), axis=0, scale_factor = 0, truncate=False)
    expected_exception = ValueError
    match = "scale_factor must be at least 1."
    with pytest.raises(expected_exception, match=match):
        _downsample_along_axis(**kwargs)

"""
Test downsample_image.
"""

def test_downsample_image():

    # Test proper use.

    # Test identity.
    kwargs = dict(
        image=np.arange(3*4).reshape(3,4), 
        scale_factors=1, 
        truncate=False,
    )
    correct_output = np.arange(3*4).reshape(3,4)
    assert np.array_equal(downsample_image(**kwargs), correct_output)
    
    # Test 3-way downsample.
    kwargs = dict(
        image=np.arange(9*12*15).reshape(9,12,15), 
        scale_factors=[2,3,4], 
        truncate=False,
    )
    correct_output = _downsample_along_axis(
        image=_downsample_along_axis(
            image=_downsample_along_axis(
                image=np.arange(9*12*15).reshape(9,12,15), axis=0, scale_factor=2, truncate=False), 
            axis=1, scale_factor=3, truncate=False), 
        axis=2, scale_factor=4, truncate=False
    )
    assert np.array_equal(downsample_image(**kwargs), correct_output)

    # Test improper use.
    
    kwargs = dict(image=np.arange(3), scale_factors=0.5, truncate=False)
    expected_exception = ValueError
    match = "Every element of scale_factors must be at least 1."
    with pytest.raises(expected_exception, match=match):
        downsample_image(**kwargs)

"""
Test change_resolution_to.
"""

def test_change_resolution_to():

    # Test proper use.

    # Test identity with pad_to_match_res=True.
    kwargs = dict(
        image=np.arange(3*4).reshape(3,4), 
        xyz_resolution=[1,1.5], 
        desired_xyz_resolution=[1,1.5], 
        pad_to_match_res=True, 
        err_to_higher_res=True, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = np.arange(3*4).reshape(3,4)
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)

    # Test identity with pad_to_match_res=False.
    kwargs = dict(
        image=np.arange(3*4).reshape(3,4), 
        xyz_resolution=[1,1.5], 
        desired_xyz_resolution=[1,1.5], 
        pad_to_match_res=False, 
        err_to_higher_res=True, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = np.arange(3*4).reshape(3,4)
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)

    # Test basic downsampling resample.
    kwargs = dict(
        image=np.arange(3*4).reshape(3,4), 
        xyz_resolution=1, 
        desired_xyz_resolution=2, 
        pad_to_match_res=False, 
        err_to_higher_res=True, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = _resample(
        image=downsample_image(image=np.arange(3*4).reshape(3,4), scale_factors=(1,2), truncate=False),
        real_axes=_compute_axes(shape=(3,2), xyz_resolution=(1,2)), 
        new_real_coords=_compute_coords(shape=(2,2), xyz_resolution=(3/2,2))
    )
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)
        
    # Test larger downsampling resample.
    kwargs = dict(
        image=np.arange(10*13).reshape(10,13), 
        xyz_resolution=1, 
        desired_xyz_resolution=2.5, 
        pad_to_match_res=False, 
        err_to_higher_res=True, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = _resample(
        image=downsample_image(image=np.arange(10*13).reshape(10,13), scale_factors=(2,2), truncate=False),
        real_axes=_compute_axes(shape=(5,7), xyz_resolution=(2,2)), 
        new_real_coords=_compute_coords(shape=(4,6), xyz_resolution=(10/4,13/6))
    )
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)

    # Test downsampling resample with pad_to_match_res=True.
    kwargs = dict(
        image=np.arange(5*25).reshape(5,25), 
        xyz_resolution=[4,3], 
        desired_xyz_resolution=[3,17], 
        pad_to_match_res=True, 
        err_to_higher_res=True, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = _resample(
        image=downsample_image(
            image=np.pad(np.arange(5*25).reshape(5,25), pad_width=((1,1), (2,2)), mode='mean', stat_length=((1,1), (4,4))), 
            scale_factors=(1,5), 
            truncate=False
        ), 
        real_axes=_compute_axes(shape=(7,6), xyz_resolution=(4,15)), 
        new_real_coords=_compute_coords(shape=(7,5), xyz_resolution=(3,17))
    )
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)
    
    # Test return_true_resolution=True.
    kwargs = dict(
        image=np.arange(5*6*7).reshape(5,6,7), 
        xyz_resolution=1, 
        desired_xyz_resolution=2, 
        pad_to_match_res=False, 
        err_to_higher_res=True, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=True, 
    )
    correct_output = _resample(
        image=downsample_image(image=np.arange(5*6*7).reshape(5,6,7), scale_factors=(1,2,1), truncate=False),
        real_axes=_compute_axes(shape=(5,3,7), xyz_resolution=(1,2,1)), 
        new_real_coords=_compute_coords(shape=(3,3,4), xyz_resolution=np.array([5,6,7])/(3,3,4))
    ), np.divide((5,6,7), (3,3,4))
    actual_output = change_resolution_to(**kwargs)
    assert np.array_equal(correct_output[0], actual_output[0])
    assert np.array_equal(correct_output[1], actual_output[1])

    # Test err_to_higher_res=False.
    kwargs = dict(
        image=np.arange(5*6*7).reshape(5,6,7), 
        xyz_resolution=1, 
        desired_xyz_resolution=2, 
        pad_to_match_res=False, 
        err_to_higher_res=False, 
        average_on_downsample=True, 
        truncate=False, 
        return_true_resolution=True, 
    )
    correct_output = _resample(
        image=downsample_image(image=np.arange(5*6*7).reshape(5,6,7), scale_factors=2, truncate=False),
        real_axes=_compute_axes(shape=(3,3,4), xyz_resolution=(2,2,2)), 
        new_real_coords=_compute_coords(shape=(2,3,3), xyz_resolution=np.array([5,6,7])/(2,3,3))
    ), np.divide((5,6,7), (2,3,3))
    actual_output = change_resolution_to(**kwargs)
    assert np.array_equal(correct_output[0], actual_output[0])
    assert np.array_equal(correct_output[1], actual_output[1])

    # Test average_on_downsample=False.
    kwargs = dict(
        image=np.arange(10*13).reshape(10,13), 
        xyz_resolution=1, 
        desired_xyz_resolution=5, 
        pad_to_match_res=False, 
        err_to_higher_res=True, 
        average_on_downsample=False, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = _resample(
        image=np.arange(10*13).reshape(10,13), 
        real_axes=_compute_axes(shape=(10,13), xyz_resolution=1), 
        new_real_coords=_compute_coords(shape=(2,3), xyz_resolution=np.array([10,13])/(2,3))
    )
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)

    # Test nonuniform desired_xyz_resolution.
    kwargs = dict(
        image=np.arange(10*13).reshape(10,13), 
        xyz_resolution=1, 
        desired_xyz_resolution=(2,3.5), 
        pad_to_match_res=False, 
        err_to_higher_res=True, 
        average_on_downsample=False, 
        truncate=False, 
        return_true_resolution=False, 
    )
    correct_output = _resample(
        image=np.arange(10*13).reshape(10,13), 
        real_axes=_compute_axes(shape=(10,13), xyz_resolution=1), 
        new_real_coords=_compute_coords(shape=(5,4), xyz_resolution=np.array([10,13])/(5,4))
    )
    assert np.array_equal(change_resolution_to(**kwargs), correct_output)

    # Test warning.

    kwargs = dict(image=np.arange(3), xyz_resolution=7, desired_xyz_resolution=2, pad_to_match_res=False)
    expected_warning = RuntimeWarning
    match = "Could not exactly produce the desired_xyz_resolution."
    with pytest.warns(expected_warning, match=match):
        change_resolution_to(**kwargs)

"""
Test change_resolution_by.
"""

def test_change_resolution_by():

    # Test proper use.

    # Test identity.
    kwargs = dict(image=np.arange(3*4).reshape(3,4), xyz_scales=1, xyz_resolution=1)
    correct_output = np.arange(3*4).reshape(3,4)
    assert np.array_equal(change_resolution_by(**kwargs), correct_output)
    
    # Test uniform scaling.
    kwargs = dict(image=np.arange(3*4).reshape(3,4), xyz_scales=2, xyz_resolution=1)
    correct_output = change_resolution_to(
        image=np.arange(3*4).reshape(3,4), 
        xyz_resolution=1, 
        desired_xyz_resolution=1/2, 
    )
    assert np.array_equal(change_resolution_by(**kwargs), correct_output)
    
    # Test negative uniform scaling with nonunity xyz_resolution.
    kwargs = dict(image=np.arange(3*4).reshape(3,4), xyz_scales=-2, xyz_resolution=5)
    correct_output = change_resolution_to(
        image=np.arange(3*4).reshape(3,4), 
        xyz_resolution=5, 
        desired_xyz_resolution=10, 
    )
    assert np.array_equal(change_resolution_by(**kwargs), correct_output)

    # Test nonuniform scaling.
    kwargs = dict(image=np.arange(3*4*5*6).reshape(3,4,5,6), xyz_scales=[1,2,-2,0.5], xyz_resolution=1)
    correct_output = change_resolution_to(
        image=np.arange(3*4*5*6).reshape(3,4,5,6), 
        xyz_resolution=1, 
        desired_xyz_resolution=[1,1/2,2,2], 
    )
    out=change_resolution_by(**kwargs)
    assert np.array_equal(change_resolution_by(**kwargs), correct_output)

"""
Perform tests.
"""

if __name__ == "__main__":
    test__resample()
    test__validate_ndarray()
    test__validate_scalar_to_multi()
    test__validate_xyz_resolution()
    test__compute_axes()
    test__compute_coords
    test__downsample_along_axis()
    test_downsample_image()
    test_change_resolution_to()
    test_change_resolution_by()
