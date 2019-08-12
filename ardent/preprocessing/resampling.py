import numpy as np
import warnings
from scipy.interpolate import interpn

# TODO: will need one more function to do sinc upsampling for initializing a high res velocity field from a low res.

def _validate_scalar_to_multi(value, size=3, dtype=float):
    """
    If value's length is 1, upcast it to match size. 
    Otherwise, if it does not match size, raise error.

    Return a numpy array.
    """

    # Cast size to int.
    try:
        size = int(size)
    except (TypeError, ValueError):
        raise TypeError(f"size must be interpretable as an integer.\n"
            f"type(size): {type(size)}.")
    
    if size < 0:
        raise ValueError(f"size must be non-negative.\n"
            f"size: {size}.")
    
    # Cast value to np.ndarray.
    try:
        value = np.array(value, dtype)
    except ValueError:
        raise ValueError(f"value and dtype are incompatible with one another.")

    # Validate value's dimensionality and length.
    if value.ndim == 0:
        value = np.array([value])
    if value.ndim == 1:
        if len(value) == 1:
            # Upcast scalar to match size.
            value = np.full(size, value, dtype=dtype)
        elif len(value) != size:
            # value's length is incompatible with size.
            raise ValueError(f"The length of value must either be 1 or it must match size."
                f"len(value): {len(value)}.")    
    else:
        # value.ndim > 1.
        raise ValueError(f"value must not have more than 1 dimension."
            f"value.ndim: {value.ndim}.")
    
    # TODO: verify that this is necessary and rewrite/remove accordingly.
    # Check for np.nan values.
    if np.any(np.isnan(value)):
        raise NotImplementedError("np.nan values encountered. What input led to this result?"
            "Write in an exception as appropriate.")
        raise ValueError(f"value contains inappropriate values for the chosen dtype "
            f"and thus contains np.nan values.")
            
    return value


def _validate_xyz_resolution(ndim, xyz_resolution):
    """Validate xyz_resolution to assure its length matches the dimensionality of image."""

    xyz_resolution = _validate_scalar_to_multi(xyz_resolution, size=ndim)

    if np.any(xyz_resolution <= 0):
        raise ValueError(f"All elements of xyz_resolution must be positive.\n"
            f"np.min(xyz_resolution): {np.min(xyz_resolution)}.")

    return xyz_resolution


def _validate_ndarray(input, minimum_ndim=0, required_ndim=None, dtype=None, 
forbid_object_dtype=True, broadcast_to_shape=None):
    """Cast (a copy of) input to a np.ndarray if possible and return it 
    unless it is noncompliant with minimum_ndim, required_ndim, and dtype.
    
    Note:
    
    If required_ndim is None, _validate_ndarray will accept any object.
    If it is possible to cast to dtype, otherwise an exception is raised.

    If np.array(input).ndim == 0 and required_ndim == 1, input will be upcast to ndim 1.
    
    If forbid_object_dtype == True and the dtype is object, an exception is raised 
    unless object is the dtype.
    
    If a shape is provided to broadcast_to_shape, unless noncompliance is found with 
    required_ndim, input is broadcasted to that shape."""

    # Verify arguments.

    # Verify minimum_ndim.
    if not isinstance(minimum_ndim, int):
        raise TypeError(f"minimum_ndim must be of type int."
            f"type(minimum_ndim): {type(minimum_ndim)}.")
    if minimum_ndim < 0:
        raise ValueError(f"minimum_ndim must be non-negative."
            f"minimum_ndim: {minimum_ndim}.")

    # Verify required_ndim.
    if required_ndim is not None:
        if not isinstance(required_ndim, int):
            raise TypeError(f"required_ndim must be either None or of type int."
                f"type(required_ndim): {type(required_ndim)}.")
        if required_ndim < 0:
            raise ValueError(f"required_ndim must be non-negative."
                f"required_ndim: {required_ndim}.")

    # Verify dtype.
    if dtype is not None:
        if not isinstance(dtype, type):
            raise TypeError(f"dtype must be either None or a valid type."
                f"type(dtype): {type(dtype)}.")

    # Validate input.

    # Cast input to np.ndarray.
    # Validate compliance with dtype.
    try:
        input = np.array(input, dtype) # Side effect: breaks alias.
    except TypeError:
        raise TypeError(f"input is of a type that is incompatible with dtype.\n"
            f"type(input): {type(input)}, dtype: {dtype}.")
    except ValueError:
        raise ValueError(f"input has a value that is incompatible with dtype.\n"
            f"input: {input}, \ntype(input): {type(input)}, dtype: {dtype}.")

    # Verify compliance with forbid_object_dtype.
    if forbid_object_dtype:
        if input.dtype == object and dtype != object:
            raise TypeError(f"Casting input to a np.ndarray produces an array of dtype object while forbid_object_dtype == True and dtype != object.")

    # Validate compliance with required_ndim.
    if required_ndim is not None and input.ndim != required_ndim:
        # Upcast from ndim 0 to ndim 1 if appropriate.
        if input.ndim == 0 and required_ndim == 1:
            input = np.array([input])
        else:
            raise ValueError(f"If required_ndim is not None, input.ndim must equal it unless input.ndim == 0 and required_ndin == 1.\n"
                f"input.ndim: {input.ndim}, required_ndim: {required_ndim}.")

    # Verify compliance with minimum_ndim.
    if input.ndim < minimum_ndim:
        raise ValueError(f"input.ndim must be at least equal to minimum_ndim."
            f"input.ndim: {input.ndim}, minimum_ndim: {minimum_ndim}.")
    
    # Broadcast input if appropriate.
    if broadcast_to_shape is not None:
        input = np.copy(np.broadcast_to(array=input, shape=broadcast_to_shape))

    return input


def _compute_axes(shape, xyz_resolution=1, origin='center'):
    """Returns the real_axes defining an image with the given shape 
    at the given resolution as a list of numpy arrays.
    """

    # Validate shape.
    shape = _validate_ndarray(shape, dtype=int, required_ndim=1)

    # Validate xyz_resolution.
    xyz_resolution = _validate_xyz_resolution(len(shape), xyz_resolution)

    # Create axes.

    # axes is a list of arrays matching each shape element from shape, spaced by the corresponding xyz_resolution.
    axes = [np.arange(dim_size) * dim_res for dim_size, dim_res in zip(shape, xyz_resolution)]

    # List all presently recognized origin values.
    origins = ['center', 'zero']

    if origin == 'center':
        # Center each axes array to its mean.
        for xyz_index, axis in enumerate(axes):
            axes[xyz_index] -= np.mean(axis)
    elif origin == 'zero':
        # Allow each axis to increase from 0 along each dimension.
        pass
    else:
        raise NotImplementedError(f"origin must be one of these supported values: {origins}.\n"
            f"origin: {origin}.")
    
    return axes


def _compute_coords(shape, xyz_resolution=1, origin='center'):
    """Returns the real_coordinates of an image with the given shape 
    at the given resolution as a single numpy array of shape (*shape, len(shape))."""

    axes = _compute_axes(shape, xyz_resolution, origin)

    meshes = np.meshgrid(*axes, indexing='ij')

    return np.stack(meshes, axis=-1)


def _resample(image, real_axes, new_real_coords, **interpn_kwargs):
    """
    Resamples image by interpolation. 
    real_axes are the coordinates defining the image points, 
    new_real_coords are the points at which to resample the image.

    Both real_axes and new_real_coords are in real-valued coordinate positions as opposed to indices.
    """

    return interpn(points=real_axes, values=image, xi=new_real_coords, **interpn_kwargs)


def _downsample_along_axis(image, axis, scale_factor, truncate=False):
    """Average image along axis across intervals of length scale_factor."""

    # Validate arguments.

    # Validate image.
    image = _validate_ndarray(image)

    # Validate axis.
    if not isinstance(axis, int):
        raise TypeError(f"axis must be of type int."
            f"type(axis): {type(axis)}.")
    if axis not in range(image.ndim):
        if axis in range(-image.ndim, 0):
            # Roll axis around to accept negative values.
            axis = image.ndim + axis
        else:
            raise ValueError(f"axis must be in range -image.ndim to image.ndim.\n"
                f"axis: {axis}.")

    # Validate scale_factor.
    try:
        if scale_factor != int(scale_factor):
            raise ValueError(f"scale_factor must be integer-valued.\n"
                f"scale_factor: {scale_factor}.")
    except TypeError:
        raise TypeError(f"scale_factor must be castable to type int.\n"
            f"type(scale_factor): {type(scale_factor)}.")
    if scale_factor < 1:
        raise ValueError(f"scale_factor must be at least 1.\n"
            f"scale_factor: {scale_factor}.")
    
    # # In the event of the trivial case, perform this shortcut.
    # # A copy is returned to maintain non-aliasing behavior.
    # if scale_factor == 1:
    #     return np.copy(image)

    # scaled_shape[axis] will be rounded down if truncate, else it will be rounded up.
    scaled_shape = np.array(image.shape, float)
    scaled_shape[axis] /= scale_factor

    if truncate:
        # Trim (a copy of) image such that image.shape[axis] is reduced down to the nearest multiple of scale_factor.
        
        scaled_shape[axis] = np.floor(scaled_shape[axis])
        scaled_shape = scaled_shape.astype(int)
        scaled_image = np.zeros(scaled_shape)

        excess_shape_on_axis = image.shape[axis] % scale_factor
        # Slice image from np.floor(excess_shape_on_axis / 2) to np.ceil(excess_shape_on_axis / 2) along axis.
        image = image[tuple([*([slice(None)] * axis), slice(int(np.floor(excess_shape_on_axis / 2)), image.shape[axis] - int(np.ceil(excess_shape_on_axis / 2)))])]
    else:
        # Pad (a copy of) image on either side of the given axis with averages.
        # This brings image.shape[axis] up to the nearest multiple of scale_factor.
        
        scaled_shape[axis] = np.ceil(scaled_shape[axis])
        scaled_shape = scaled_shape.astype(int)
        scaled_image = np.zeros(scaled_shape)

        total_padding = -image.shape[axis] % scale_factor # The difference between image.shape[axis] and the nearest not-smaller multiple of scale_factor.
        on_axis_pad_width = np.array([np.floor(total_padding / 2), np.ceil(total_padding / 2)], int)
        pad_width = (*([(0,0)] * axis), on_axis_pad_width, *([(0,0)] * (image.ndim - axis - 1)))
        stat_length = scale_factor - on_axis_pad_width
        # Note: if image.dtype == int, the averaged pad values will be truncated accordingly.
        image = np.pad(image, pad_width=pad_width, mode='mean', stat_length=stat_length)

    # Take scaled_shape-shaped slices of image along axis, adding each slice to scaled_image.
    for scale_index in range(scale_factor):
        # TODO: pick one.
        # All of the following 3 lines do the same thing but with differing readability and speed.
        # scaled_image += image.take(range(scale_index, image.shape[axis], scale_factor), axis) # Simplest.
        # scaled_image += eval(f"image[{':,' * axis} slice(scale_index, image.shape[axis], scale_factor)]") # 2.5 times slower.
        scaled_image += image[tuple([*([slice(None)] * axis), slice(scale_index, scaled_shape[axis] * scale_factor, scale_factor)])] # 3 times faster.

    # Divide scaled_image by the number of points added at each coordinate.
    scaled_image /= scale_factor

    return scaled_image


def downsample_image(image, scale_factors, truncate=False):
    '''Downsample an image by averaging.
    down should either be a triple or a single number.'''
    
    # Validate arguments.

    # Validate image.
    image = _validate_ndarray(image)

    # Validate scale_factors.
    scale_factors = _validate_scalar_to_multi(scale_factors, image.ndim, int)
    # Verify that all scale_factors are at least 1.
    if np.any(scale_factors < 1):
        raise ValueError(f"Every element of scale_factors must be at least 1.\n"
            f"np.min(scale_factors): {np.min(scale_factors)}.")

    # Downsample a copy of image by averaging.

    scaled_image = np.copy(image) # Not necessary since _downsample_along_axis does not mutate.

    # Downsample image along each dimension.
    for dim, scale_factor in enumerate(scale_factors):
        scaled_image = _downsample_along_axis(scaled_image, dim, scale_factor, truncate) # Side effect: breaks alias.
    
    return scaled_image


def change_resolution_to(image, xyz_resolution, desired_xyz_resolution, 
pad_to_match_res=True, err_to_higher_res=True, average_on_downsample=True, 
truncate=False, return_true_resolution=False, **resample_kwargs):
    """
    Resamples image as close as possible to the desired_xyz_resolution. 
    If return_true_resolution, returns (resampled_image, true_resolution).

    If pad_to_match_res, pads image to guarantee that true_resolution == desired_xyz_resolution.
    Else, err_to_higher_res indicates whether the new_shape is rounded up or down to guarantee 
    the true shape of image is maintained (shape * resolution).

    If average_on_downsample, perform averaging before resampling 
    when downsampling by at least a factor of 2.

    truncate is used in downsample_image.
    """

    # Validate arguments.

    # Validate image.
    image = _validate_ndarray(image)

    # Validate resolutions.
    xyz_resolution = _validate_xyz_resolution(image.ndim, xyz_resolution)
    desired_xyz_resolution = _validate_xyz_resolution(image.ndim, desired_xyz_resolution)

    # Compute new_shape, and consequently true_resolution, after resampling.

    # image.shape * xyz_resolution == new_shape * new_xyz_resolution
    new_shape = image.shape * xyz_resolution / desired_xyz_resolution

    if pad_to_match_res:
        # Guarantee realization of desired_xyz_resolution at the possible expense of maintaining the true shape (shape * resolution).
        new_shape = np.ceil(new_shape)
        # Pad image evenly until image.shape * xyz_resolution >= new_shape * desired_xyz_resolution.
        minimum_image_padding = np.ceil((new_shape * desired_xyz_resolution - image.shape * xyz_resolution) / xyz_resolution)
        pad_width = np.array(list(zip(np.ceil(minimum_image_padding / 2), np.ceil(minimum_image_padding / 2))), int)
        old_true_shape = xyz_resolution * image.shape
        new_true_shape = desired_xyz_resolution * new_shape
        stat_length = np.maximum(1, np.ceil((desired_xyz_resolution - ((new_true_shape - old_true_shape) / 2)) / xyz_resolution)).astype(int)
        stat_length = np.broadcast_to(stat_length, pad_width.T.shape).T
        image = np.pad(image, pad_width=pad_width, mode='mean', stat_length=stat_length)
        # true_resolution has been guaranteed to equal desired_xyz_resolution.
        true_resolution = desired_xyz_resolution
    else:
        # Guarantee the true shape (shape * resolution) is maintained at the possible expense of achieving desired_xyz_resolution.
        if err_to_higher_res:
            # Round resolution up.
            new_shape = np.ceil(new_shape)
        else:
            # Round resolution down.
            new_shape = np.floor(new_shape)
        # Compute the achieved resultant resolution, or true_resolution.
        true_resolution = image.shape * xyz_resolution / new_shape
        # Warn the user if desired_xyz_resolution cannot be produced from image and xyz_resolution.
        if not np.array_equal(new_shape, image.shape * xyz_resolution / desired_xyz_resolution):
            warnings.warn(message=f"Could not exactly produce the desired_xyz_resolution.\n"
                f"xyz_resolution {xyz_resolution}.\n"
                f"desired_xyz_resolution: {desired_xyz_resolution}.\n"
                f"true_resolution: {true_resolution}.", category=RuntimeWarning)

    # Average if appropriate before resampling to lower resolution.

    if average_on_downsample:
        # Check resampling scales.
        downsampling_scale_factors = np.divide(image.shape, new_shape).astype(int)
        # downsampling_scale_factors[dim] is the maximum of 1 or factor by which resolution[dim] is multiplied.
        downsampling_scale_factors = np.maximum(
            np.ones_like(downsampling_scale_factors, dtype=int), downsampling_scale_factors)
        
        # Perform downsampling.
        image = downsample_image(image, downsampling_scale_factors, truncate=truncate)
        # Update xyz_resolution.
        xyz_resolution *= downsampling_scale_factors

    # Compute real_axes and new_real_coords.

    real_axes = _compute_axes(image.shape, xyz_resolution)
    # new_shape is recalculated assuming the image shape is 1 less than it really is along each dimension, 
    # with 1 added at the end. This is to account for interpn's coordinate interpretation of voxels, 
    # to ensure that new_real_coords all lie within the bounds of real_axes, but as close to filling them as possible.
    real_scales = np.divide(new_shape, image.shape)
    new_shape = np.floor(np.multiply(np.subtract(image.shape, 1), real_scales)) + 1
    new_real_coords = _compute_coords(new_shape, true_resolution)

    # Perform resampling.

    resampled_image = _resample(image, real_axes, new_real_coords, **resample_kwargs)

    if return_true_resolution:
        return resampled_image, true_resolution
    else:
        return resampled_image

    
def change_resolution_by(image, xyz_scales, xyz_resolution=1, 
pad_to_match_res=True, err_to_higher_res=True, average_on_downsample=True, 
truncate=False, return_true_resolution=False, **resample_kwargs):
    """
    Resample image such that its resolution is scaled by 1/xyz_scales[dim], 
    or by abs(xyz_scales[dim]) if it's negative, in dimension dim.

    If return_true_resolution, returns (resampled_image, true_resolution).
    return_true_resolution should be accompanied by xyz_resolution.
    If xyz_resolution is not provided, it is assumed to be all 1's.

    xyz_scales[dim] > 1 implies upsampling - increasing resolution and image size.
    xyz_scales[dim] = 1 implies unity - no change in resolution for this dimension.
    xyz_scales[dim] < 1 implies downsampling - decreasing resolution and image size.
    xyz_scales[dim] < 0 implies downsampling by this factor - cast to -1 / xyz_scales[dim].

    Examples:
    xyz_scales[dim] = 2 --> upsample by 2
    xyz_scales[dim] = 1 --> do nothing
    xyz_scales[dim] = 1/2 --> downsample by 2
    xyz_scales[dim] = -3 --> downsample by 3
    xyz_scales[dim] = -1/5 --> upsample by 5
    """

    # Validate arguments.

    # Validate image.
    image = _validate_ndarray(image)

    # Validate xyz_scales.
    xyz_scales = _validate_scalar_to_multi(xyz_scales, size=image.ndim)
    for dim, scale in enumerate(xyz_scales):
        if scale < 0:
            xyz_scales[dim] = -1 / xyz_scales[dim]

    # Validate xyz_resolution.
    xyz_resolution = _validate_xyz_resolution(image.ndim, xyz_resolution)

    # Compute desired_xyz_resolution.
    desired_xyz_resolution = xyz_resolution / xyz_scales

    change_resolution_to_kwargs = dict(
        image=image,
        xyz_resolution=xyz_resolution,
        desired_xyz_resolution=desired_xyz_resolution,
        pad_to_match_res=pad_to_match_res,
        err_to_higher_res=err_to_higher_res,
        average_on_downsample=average_on_downsample,
        truncate=truncate,
        return_true_resolution=return_true_resolution,
        **resample_kwargs
    )

    return change_resolution_to(**change_resolution_to_kwargs)

# TODO: reconcile use of scipy.interpolate.interpn vs scipy.misc.resize & skimage.transform.downscale_local_mean.

# TODO: isolate negative scale conversion into its own function.
# TODO: merge necessary new_shape transformations etc. into _resample.