import pytest

import numpy as np
from scipy.interpolate import interpn

from ardent.preprocessing.resampling import _resample
from ardent.preprocessing.resampling import _validate_ndarray
from ardent.preprocessing.resampling import _validate_scalar_to_multi
from ardent.preprocessing.resampling import _validate_xyz_resolution
from ardent.preprocessing.resampling import _compute_coords
from ardent.preprocessing.resampling import _downsample_along_axis
from ardent.preprocessing.resampling import downsample_image
from ardent.preprocessing.resampling import change_resolution_to
from ardent.preprocessing.resampling import change_resolution_by

# Test _resample.

@pytest.mark.parametrize("image", list(map(lambda shape: np.arange(np.prod(shape)).reshape(shape), 
                                        [(3,4), (3,4,5), (3,4,6)])))
@pytest.mark.parametrize("resolution", [0.5, 1, 1.5, 2])
@pytest.mark.parametrize("xyz_scales", [1, 1/2, 1/3, 2, 3])
def test__resample(image, resolution, xyz_scales):

    # Test uniform resolutions.

    real_coords = _compute_coords(image.shape, resolution, origin='center')

    new_shape = np.mutiply(image.shape, xyz_scales)
    new_real_coords = _compute_coords(new_shape, resolution, origin='center')

    expected_output = interpn(points=real_coords, values=image, xi=new_real_coords)
    assert np.array_equal(_resample(image, real_coords, new_real_coords), expected_output)

    # Test non-uniform resolutions.

    dynamic_resolution = np.arange(image.ndim) * resolution
    new_real_coords = _compute_coords(new_shape, dynamic_resolution, origin='center')

    expected_output = interpn(points=real_coords, values=image, xi=new_real_coords)
    assert np.array_equal(_resample(image, real_coords, new_real_coords), expected_output)

# Test _validate_ndarray.

@pytest.mark.parametrize("kwargs, output", [
    (dict(input=np.arange(3, dtype=int), dtype=float), np.arange(3, dtype=float)), 
    (dict(input=[[0,1,2], [3,4,5]], dtype=float), np.arange(2*3).reshape(2,3).astype(float)), 
    (dict(input=np.array([0,1,2]), broadcast_to_shape=(2,3)), np.array([[0,1,2], [0,1,2]])), 
])
def test__validate_ndarray__correct(kwargs, output):
    assert np.array_equal(_validate_ndarray(**kwargs), output)


@pytest.mark.parametrize("kwargs, expected_exception, match", [
    # Verify arguments.
    (dict(input=np.arange(3), minimum_ndim=1.5), TypeError, "minimum_ndim must be of type int."), 
    (dict(input=np.arange(3), minimum_ndim=-1), ValueError, "minimum_ndim must be non-negative."), 
    (dict(input=np.arange(3), required_ndim=1.5), TypeError, "required_ndim must be either None or of type int."), 
    (dict(input=np.arange(3), required_ndim=-1), ValueError, "required_ndim must be non-negative."), 
    (dict(input=np.arange(3), dtype="not of type type"), TypeError, "dtype must be either None or a valid type."), 
    # Validate image.
    (dict(input=np.arange(print), dtype=int), TypeError, "input is not compatible with dtype."), 
    (dict(input=np.array([[], 1]), dtype=None, refuse_object_dtype=True), TypeError, "np.array(input).dtype == object while refuse_object_dtype == True and dtype != object."), 
    (dict(input=np.arange(3), minimum_ndim=2), ValueError, "input.ndim must be at least equal to minimum_ndim."), 
    (dict(input=np.arange(3), required_ndim=2), ValueError, "If required_ndim is not None, input.ndim must equal it."), 
    (dict(input=np.arange(3), required_ndim=2), ValueError, "If required_ndim is not None, input.ndim must equal it."), 
])
def test__validate_ndarray__raises(kwargs, expeced_exception, match):
    with pytest.raises(expeced_exception, match=match):
        _validate_ndarray(**kwargs)

# Test _validate_scalar_to_multi.

@pytest.mark.parametrize("kwargs, correct_output", [
    (dict(value=1, size=1, dtype=float), np.array([1], float)),
    (dict(value=9.5, size=4, dtype=int), np.full(4, 9, int)),
    (dict(value=[1, 2, 3.5], size=3, dtype=float), np.array([1, 2, 3.5], float)),
    (dict(value=[1, 2, 3.5], size=3, dtype=int), np.array([1, 2, 3], int)),
    (dict(value=(1, 2, 3), size=3, dtype=int), np.array([1, 2, 3], int)),
    (dict(value=np.array([1, 2, 3], float), size=3, dtype=int), np.array([1, 2, 3], int)),
])
def test__validate_scaler_to_multi__correct(kwargs, correct_output):
    assert np.array_equal(_validate_scaler_to_multi(**kwargs), correct_output)


@pytest.mark.parametrize("kwargs, expected_exception, match", [
    (dict(value=[1, 2, 3, 4], size='size: not an int', dtype=float), TypeError, "size must be interpretable as an integer."),
    (dict(value=[], size=0, dtype=float), ValueError, "size must be non-negative."),
    (dict(value=[1, 2, 3, 4], size=3, dtype=int), ValueError, "The length of value must be either 1 or it must match size."),
    (dict(value=np.arange(3*4, int).reshape(3,4), size=3, dtype=float), ValueError, "value must not have more than 1 dimension."),
    (dict(value=[1, 2, 'c'], size=3, dtype=int), ValueError, "value and dtype are incompatible with one another."),
    (dict(value='c', size=3, dtype=int), ValueError, "value and dtype are incompatible with one another."),
])
def test__validate_scaler_to_multi__raises(kwargs, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        _validate_scaler_to_multi(**kwargs)

# Test _validate_xyz_resolution.

@pytest.mark.parametrize("kwargs, output", [
    (dict(ndim=1, xyz_resolution=2), np.full(1, 2, float)),
    (dict(ndim=4, xyz_resolution=1.5), np.full(4, 1.5, float)),
    (dict(ndim=3, xyz_resolution=np.ones(3, int)), np.ones(3, float)),
    (dict(ndim=2, xyz_resolution=[3, 4]), np.array([3, 4], float)),
])
def test__validate_xyz_resolution__correct(kwargs, output):
    assert np.array_equal(_validate_xyz_resolution(**kwargs), output)


@pytest.mark.parametrize("kwargs, expeced_exception, match", [
    (dict(ndim=2, xyz_resolution=[3, -4]), ValueError, "All elements of xyz_resolution must be positive."),
    (dict(ndim=2, xyz_resolution=[3, 0]), ValueError, "All elements of xyz_resolution must be positive."),
])
def test__validate_xyz_resolution__raises(kwargs, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        _validate_xyz_resolution(**kwargs)

# Test _compute_coords.

@pytest.mark.parametrize("kwargs, output", [
    (dict(shape=(0, 1, 2), xyz_resolution=1, origin='center'), 
        [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((0, 1, 2), (1, 1, 1))]),
    (dict(shape=(1, 2, 3, 4), xyz_resolution=1.5, origin='center'), 
        [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((1, 2, 3, 4), (1.5, 1.5, 1.5, 1.5))]),
    (dict(shape=(2, 3, 4), xyz_resolution=[1, 1.5, 2], origin='center'), 
        [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((2, 3, 4), (1, 1.5, 2))]),
    (dict(shape=5, xyz_resolution=1, origin='center'), 
        [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((5), (1))]),
])
def test__compute_coords__correct(kwargs, output):
    # _compute_coords produces a list with a np.ndarray for each element in shape.
    for dim, coord in enumerate(_compute_coords(**kwargs)):
        assert np.array_equal(coord, output[dim])

# Test _downsample_along_axis.

@pytest.mark.parametrize("kwargs, output", [
    # Test identity.
    (
        # kwargs:
        dict(
            image=np.arange(3*4).reshape(3,4), 
            axis=0, 
            scale_factor=1, 
            truncate=False, 
            ), 
        # output:
        np.arange(3*4).reshape(3,4)
    ), 

    # Test basic use with negative axis.
    (
        # kwargs:
        dict(
            image=np.arange(3*4).reshape(3,4), 
            axis=-1, 
            scale_factor=2, 
            truncate=False, 
            ), 
        # output:
        np.arange(0.5,12,2).reshape(3,2)
    ), 

    # Test uneven case.
    (
        # kwargs:
        dict(
            image=np.arange(9,3).reshape(9,3), 
            axis=0, 
            scale_factor=4, 
            truncate=False, 
            ), 
        # output:
        np.stack(
            (
                np.average(np.arange(9*3).reshape(9,3).take(range(3),0),0), 
                np.average(np.arange(9*3).reshape(9,3).take(range(3,3+4),0),0),
                np.average(np.arange(9*3).reshape(9,3).take(range(3+4,9),0),0),
            ), 0)
    ), 
    
    # Test uneven case with truncation.
    (
        # kwargs:
        dict(
            image=np.arange(3*11).reshape(3,11), 
            axis=-1, 
            scale_factor=4, 
            truncate=True, 
            ), 
        # output:
        np.stack(
            (
                np.average(np.arange(3*11).reshape(3,11).take(range(1,1+4),1),1), 
                np.average(np.arange(3*11).reshape(3,11).take(range(1+4,1+4+4),1),1),
            ), 1)
    ), 
])
def test__downsample_along_axis__correct(kwargs, output):
    assert np.array_equal(_downsample_along_axis(**kwargs), output)


@pytest.mark.parametrize("kwargs, expected_exception, match", [
    (dict(image=np.arange(3), axis=1.5, scale_factor = 1, truncate=False), TypeError, "axis must be of type int."), 
    (dict(image=np.arange(3), axis=10, scale_factor = 1, truncate=False), ValueError, "axis must be in range(-image.ndim, image.ndim)."), 
    (dict(image=np.arange(3), axis=1, scale_factor = 1.5, truncate=False), TypeError, "scale_factor must be of type int."), 
    (dict(image=np.arange(3), axis=1, scale_factor = 0, truncate=False), ValueError, "scale_factor must be at least 1."), 
])
def test__downsample_along_axis__raises(kwargs, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        _downsample_along_axis(**kwargs)

# Test downsample_image.

@pytest.mark.parametrize("kwargs, output", [
    # Test identity.
    (
        # kwargs:
        dict(
            image=np.arange(3*4).reshape(3,4), 
            scale_factors=1, 
            truncate=False,
        ), 
        # output:
        np.arange(3*4).reshape(3,4)
    ),
    
    # Test 3-way downsample.
    (
        # kwargs:
        dict(
            image=np.arange(9*12*15).reshape(9,12,15), 
            scale_factors=[2, 3, 4], 
            truncate=False,
        ), 
        # output:
        _downsample_along_axis(
            image=_downsample_along_axis(
                image=_downsample_along_axis(
                    image=image, axis=0, scale_factor=2, truncate=False), 
            axis=1, scale_factor=3, truncate=False), 
        axis=2, scale_factor=4, truncate=False)
    ),
])
def test_downsample_image__correct(kwargs, output):
    assert np.array_equal(downsample_image(**kwargs), output)

@pytest.mark.parametrize("kwargs, expected_exception, match", [
    (dict(image=np.arange(3), scale_factors=0.5, truncate=False), ValueError, "Every element of scale_factors must be at least 1."), 
])
def test_downsample_image__raises(kwargs, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        downsample_image(**kwargs)

# Test change_resolution_to.

@pytest.mark.parametrize("kwargs, output", [
    # Test identity with pad_to_match_res=True.
    (
        # kwargs:
        dict(
            image=np.arange(3*4).reshape(3,4), 
            xyz_resolution=[1, 1.5, 2], 
            desired_xyz_resolution=[1, 1.5, 2], 
            pad_to_match_res=True, 
            err_to_higher_res=True, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=False, 
        ), 
        # output:
        np.arange(3*4).reshape(3,4)
    ), 

    # Test identity with pad_to_match_res=False.
    (
        # kwargs:
        dict(
            image=np.arange(3*4).reshape(3,4), 
            xyz_resolution=[1, 1.5, 2], 
            desired_xyz_resolution=[1, 1.5, 2], 
            pad_to_match_res=False, 
            err_to_higher_res=True, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=False, 
        ), 
        # output:
        np.arange(3*4).reshape(3,4)
    ), 
    
    # Test basic downsampling resample.
    (
        # kwargs:
        dict(
            image=np.arange(3*4).reshape(3,4), 
            xyz_resolution=1, 
            desired_xyz_resolution=2, 
            pad_to_match_res=False, 
            err_to_higher_res=True, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=False, 
        ), 
        # output:
        _resample(
            image=downsample_image(image=np.arange(3*4).reshape(3,4), scale_factors=2, truncate_downsample=False),
            real_coords=_compute_coords(shape=(3,4), xyz_resolution=1), 
            new_real_coords=_compute_coords(shape=(2,2), xyz_resolution=2))
    ), 
        
    # Test larger downsampling resample.
    (
        # kwargs:
        dict(
            image=np.arange(10,13).reshape(10,13), 
            xyz_resolution=1, 
            desired_xyz_resolution=5, 
            pad_to_match_res=False, 
            err_to_higher_res=True, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=False, 
        ), 
        # output:
        _resample(
            image=downsample_image(image=np.arange(10*13).reshape(10,13), scale_factors=5, truncate_downsample=False),
            real_coords=_compute_coords(shape=(10,13), xyz_resolution=1), 
            new_real_coords=_compute_coords(shape=(2,3), xyz_resolution=2))
    ), 

    # Test downsampling resample with pad_to_match_res=True.
    (
        # kwargs:
        dict(
            image=np.arange(5*25).reshape(5,25), 
            xyz_resolution=[4,3], 
            desired_xyz_resolution=[3,17], 
            pad_to_match_res=True, 
            err_to_higher_res=True, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=False, 
        ), 
        # output:
        _resample(
            image=downsample_image(
                image=np.pad(np.arange(5*25).reshape(5,25), pad_width=((0,1), (2,2)), mode='mean', stat_length=((1,1), (3,3))), 
                scale_factors=(1,5), 
                truncate_downsample=False),
            real_coords=_compute_coords(shape=(5,25), xyz_resolution=(4,3)), 
            new_real_coords=_compute_coords(shape=(7,5), xyz_resolution=(3,17)))
    ), 
    
    # Test return_true_resolution=True.
    (
        # kwargs:
        dict(
            image=np.arange(5*6*7).reshape(5,6,7), 
            xyz_resolution=1, 
            desired_xyz_resolution=2, 
            pad_to_match_res=False, 
            err_to_higher_res=True, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=True, 
        ), 
        # output:
        (_resample(
            image=downsample_image(image=np.arange(5*6*7).reshape(5,6,7), scale_factors=2, truncate_downsample=False),
            real_coords=_compute_coords(shape=(5,6,7), xyz_resolution=1), 
            new_real_coords=_compute_coords(shape=(3,3,4), xyz_resolution=2)), 
        np.divide((5,6,7), (3,3,4))), 
    ),
        
    # Test err_to_higher_res=False.
    (
        # kwargs:
        dict(
            image=np.arange(5*6*7).reshape(5,6,7), 
            xyz_resolution=1, 
            desired_xyz_resolution=2, 
            pad_to_match_res=False, 
            err_to_higher_res=False, 
            average_on_downsample=True, 
            truncate_downsample=False, 
            return_true_resolution=True, 
        ), 
        # output:
        (_resample(
            image=downsample_image(image=np.arange(5*6*7).reshape(5,6,7), scale_factors=2, truncate_downsample=False),
            real_coords=_compute_coords(shape=(5,6,7), xyz_resolution=1), 
            new_real_coords=_compute_coords(shape=(2,3,3), xyz_resolution=2)), 
        np.divide((5,6,7), (2,3,3))), 
    ),

    # Test average_on_downsample=False.
    (
        # kwargs:
        dict(
            image=np.arange(10,13).reshape(10,13), 
            xyz_resolution=1, 
            desired_xyz_resolution=5, 
            pad_to_match_res=False, 
            err_to_higher_res=True, 
            average_on_downsample=False, 
            truncate_downsample=False, 
            return_true_resolution=False, 
        ), 
        # output:
        _resample(
            image=np.arange(10*13).reshape(10,13), 
            real_coords=_compute_coords(shape=(10,13), xyz_resolution=1), 
            new_real_coords=_compute_coords(shape=(2,3), xyz_resolution=2))
    ), 
])
def test_change_resolution_to__correct(kwargs, output):
    assert np.array_equal(change_resolution_to(**kwargs), output)


@pytest.mark.parametrize("kwargs, expected_exeption, match", [
    (dict(image=np.arange(3), xyz_resolution=7, desired_xyz_resolution=2, pad_to_match_res=False), RuntimeWarning, "Could not exactly produce the desired_xyz_resolution."), 
])
def test_change_resolution_to__warns(kwargs, expeced_warning, match):
    with pytest.warns(expected_warning, match=match):
        change_resolution_to(**kwargs)

# Test change_resolution_by.

@pytest.mark.parametrize("kwargs, output", [
    # Test identity.
    (
        # kwargs:
        dict(image=np.arange(3*4).reshape(3,4), xyz_scales=1, xyz_resolution=1), 
        # output:
        np.arange(3*4).reshape(3,4)
    ), 
    
    # Test uniform scaling.
    (
        # kwargs:
        dict(image=np.arange(3*4).reshape(3,4), xyz_scales=2, xyz_resolution=1), 
        # output:
        change_resolution_to(
            image=np.arange(3*4).reshape(3,4), 
            xyz_resolution=1, 
            desired_xyz_resolution=1/2, 
            ), 
    ), 
    
    # Test negative uniform scaling with nonunity xyz_resolution.
    (
        # kwargs:
        dict(image=np.arange(3*4).reshape(3,4), xyz_scales=-2, xyz_resolution=5), 
        # output:
        change_resolution_to(
            image=np.arange(3*4).reshape(3,4), 
            xyz_resolution=1, 
            desired_xyz_resolution=10, 
            ), 
    ), 

    # Test nonuniform scaling.
    (
        # kwargs:
        dict(image=np.arange(3*4*5*6).reshape(3,4,5,6), xyz_scales=[1,2,-2,0.5], xyz_resolution=1), 
        # output:
        change_resolution_to(
            image=np.arange(3*4*5*6).reshape(3,4,5,6), 
            xyz_resolution=1, 
            desired_xyz_resolution=[1,1/2,2,2], 
            ), 
    ), 
])
def test_change_resolution_by(kwargs, output):
    assert np.array_equal(change_resolution_by(**kwargs), output)
