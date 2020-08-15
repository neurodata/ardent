"""
Based on this:
https://github.com/mitragithub/Registration/blob/master/atlas_free_rigid_alignment.m
"""

'''

- take sequence of arrays, arbitrary shapes
- resample to max of dimensions per dimension into single array
- create a new 3D array where each slice is replaced with a weighted average of its n nearest neighbors
----- take a convolution kernel (gaussian of some size) and set the middle one to 0 (don't include self in weighted average)
------- blur image using decaying weights
- for each slice, do a 2D rigid registration (or constrained in some other way, but here we choose to use rigid)
- repeat num times

'''

import numpy as np
from skimage.transform import resize
from scipy.interpolate import interpn
from scipy.linalg import inv, solve, svd
from scipy.signal import fftconvolve

from ..lddmm._lddmm_utilities import _validate_scalar_to_multi
from ..lddmm._lddmm_utilities import _validate_ndarray
from ..lddmm._lddmm_utilities import _validate_resolution
from ..lddmm._lddmm_utilities import _multiply_coords_by_affine
from ..lddmm._lddmm_utilities import _compute_axes
from ..lddmm._lddmm_utilities import _compute_coords


def _slices_to_volume(slices):
    """
    Resample slices into a volume with the largest size per dimension of all slices.
    slices are stacked on dimension 0.
    """

    slice_shapes = np.array(list(map(lambda slice_: slice_.shape, slices)))
    maximum_slice_shape = np.max(slice_shapes, 0)
    # slices are stacked on the last dimension in volume.
    volume = np.empty((len(slices), *maximum_slice_shape), dtype=float)
    for slice_index, slice_ in enumerate(slices):
        volume[slice_index] = resize(slice_, maximum_slice_shape)

    return volume


def _volume_to_neighbor_averages(volume, sigma_gaussian, clip_gaussian_at_z):
    """
    Create a parallel volume whose slices are weighted averages of their neighbors in volume.
    """

    kernel = np.arange(1, sigma_gaussian * clip_gaussian_at_z + 1)
    kernel = np.concatenate((kernel[::-1], [0], kernel))
    kernel = np.exp(-kernel**2 / (2 * sigma_gaussian**2)) # / (sigma_gaussian * np.sqrt(2 * np.pi))
    kernel[len(kernel) // 2] = 0
    kernel /= kernel.sum()
    kernel = kernel.reshape(*[1] * slice_ndim, -1)
    volume_neighbors = fftconvolve(volume, kernel, axes=0)

    return volume_neighbors


def _compute_affine_inv_gradient(
    template,
    target,
    template_resolution,
    target_resolution,
    affine_inv,
    affine_inv_template,
    ):
    """
    Compute and return the appropriate gradient for affine_inv to follow using Gauss-Newton.
    """

    # Create coords and axes.
    template_axes = _compute_axes(template.shape, template_resolution, origin='center')
    target_coords = _compute_coords(target.shape, target_resolution, origin='center')

    # Compute the gradient of the original template.
    template_gradient = np.stack(np.gradient(template, *template_resolution), -1)

    # Apply the affine to each component of template_gradient.
    sample_coords = _multiply_coords_by_affine(affine_inv, target_coords)
    affine_inv_template_gradient = interpn(
        points=template_axes,
        values=template_gradient,
        xi=sample_coords,
        bounds_error=False,
        fill_value=None,
    )

    # Reshape and broadcast affine_inv_template_gradient, for a 3D example,
    # from shape (x,y,z,3) to (x,y,z,3,1) to (x,y,z,3,4).
    affine_inv_template_gradient_broadcast = np.repeat(np.expand_dims(affine_inv_template_gradient, -1), repeats=target.ndim + 1, axis=-1)

    # Construct homogenous_target_coords by appending 1's at the end of the last dimension throughout target_coords.
    ones = np.ones((*target.shape, 1))
    homogenous_target_coords = np.concatenate((target_coords, ones), -1)

    # For a 3D example:
    # affine_inv_template_gradient_broadcast has shape (x,y,z,3,4).
    # homogenous_target_coords has shape (x,y,z,4).
    # To repeat homogenous_target_coords along the 2nd-last dimension of affine_inv_template_gradient_broadcast, 
    # we reshape homogenous_target_coords from shape (x,y,z,4) to shape (x,y,z,1,4) and let that broadcast to shape (x,y,z,3,4).
    affine_inv_gradient = (
        affine_inv_template_gradient_broadcast
        * np.expand_dims(homogenous_target_coords, -2)
        * (affine_inv_template - target)
    )

    # Note: before implementing Gauss Newton below, 
    # affine_inv_gradient_reduction, as defined below, 
    # is the 1st order solution for affine_inv_gradient.
    # For 3D case, this has shape (3,4).
    affine_inv_gradient_reduction = np.sum(affine_inv_gradient, tuple(range(target.ndim)))

    # Apply Gauss-Newton for 2nd order solution.

    # Reshape to a single vector. For a 3D case this becomes shape (12,).
    affine_inv_gradient_reduction = affine_inv_gradient_reduction.ravel()

    # For a 3D case, affine_inv_gradient has shape (x,y,z,3,4).
    # For a 3D case, affine_inv_hessian_approx is affine_inv_gradient reshaped to shape (x,y,z,12,1), 
    # then matrix multiplied by itself transposed on the last two dimensions, then summed over the spatial dimensions
    # to resultant shape (12,12).
    affine_inv_hessian_approx = affine_inv_gradient.reshape(*affine_inv_gradient.shape[:-2], -1, 1)
    affine_inv_hessian_approx_tail_transpose = affine_inv_hessian_approx.reshape(*affine_inv_hessian_approx.shape[:-2], 1, -1)
    affine_inv_hessian_approx = affine_inv_hessian_approx @ affine_inv_hessian_approx_tail_transpose
    affine_inv_hessian_approx = np.sum(affine_inv_hessian_approx, tuple(range(target.ndim)))

    # Solve for affine_inv_gradient.
    try:
        affine_inv_gradient = solve(affine_inv_hessian_approx, affine_inv_gradient_reduction, assume_a='pos').reshape(affine_inv_gradient.shape[-2:])
    except np.linalg.LinAlgError as exception:
        raise RuntimeError(
            "The Hessian was not invertible in the Gauss-Newton computation of affine_inv_gradient. "
            "This may be because the image was constant along one or more dimensions. "
            "Consider removing any constant dimensions. "
            "Otherwise you may try using a smaller value for affine_stepsize."
        ) from exception
    # Append a row of zeros at the end of the 0th dimension.
    zeros = np.zeros((1, target.ndim + 1))
    affine_inv_gradient = np.concatenate((affine_inv_gradient, zeros), 0)

    return affine_inv_gradient


def _update_affine_inv(affine_inv, affine_inv_gradient, affine_stepsize, fixed_scale=None, rigid=False):
    """
    Apply affine_inv_gradient to affine_inv, possibly adjusting for fixed_scale or rigid if specified.
    Being redundant with one another, fixed_scale overrides rigid.
    """

    affine_inv -= affine_inv_gradient * affine_stepsize

    # Set scale of affine_inv if appropriate.
    if fixed_scale is not None:
        U, _, Vh = svd(affine_inv[:-1, :-1])
        affine_inv[:-1, :-1] = U @ np.diag([fixed_scale] * (len(affine_inv) - 1)) @ Vh
    # If fixed_scale was not provided (is None), project affine_inv to a rigid affine if rigid.
    elif rigid:
        U, _, Vh = svd(affine_inv[:-1, :-1])
        affine_inv[:-1, :-1] = U @ Vh

    return affine_inv


def apply_affine_to_image(image, resolution, affine):
    """
    Interpolates image at its centered identity coordinates multiplied by inv(affine).
    A copy of image so interpolated is returned.

    Args:
        image (ndarray): The image to be modified.
        resolution (float, seq): The per-dimension resolution of image. If provided as a scalar, 
            that scalar is interpreted as the resolution at every dimension.
        affine (ndarray): The affine array in homogenous coordinates to be applied to image.

    Returns:
        ndarray: A copy of image with affine applied to it.
    """

    # Validate inputs.
    image = _validate_ndarray(image, dtype=float)
    resolution = _validate_resolution(resolution, image.ndim)
    affine = _validate_ndarray(affine, required_shape=(image.ndim + 1, image.ndim + 1))

    affine_inv = inv(affine)

    image_axes = _compute_coords(image, resolution, origin='center')
    image_coords = _compute_coords(image, resolution, origin='center')
    
    # Apply affine_inv to image_coords by multiplication.
    affine_inv_coords = _multiply_coords_by_affine(affine_inv, image_coords)

    # Apply affine_inv_coords to image.
    affine_inv_image = interpn(
        points=image_axes,
        values=image_coords,
        xi=affine_inv_coords,
        bounds_error=False,
        fill_value=None,
    )

    return affine_inv_image


def affine_register(
    template,
    target,
    template_resolution,
    target_resolution,
    num_iterations,
    affine_stepsize=0.3,
    fixed_scale=None,
    rigid=False,
    initial_affine=None,
):
    """
    Iteratively compute an affine that aligns template and target.

    Args:
        template (ndarray): The image to be aligned to target.
        target (ndarray): The image template is aligned to.
        template_resolution (float, seq): The per-dimension resolution of template. 

        target_resolution (float, seq): The per-dimension resolution of target. 

        num_iterations (int): The number of iterations of registration to perform.
        affine_stepsize (float, optional): The Gauss-Newton stepsize in units of voxels. This should be between 0 and 1. Defaults to 0.3.
        fixed_scale (float, seq, optional): If provided, this fixes the scale of the affine and constrains it to be rigid. Defaults to None.
        rigid (bool, optional): If True, the affine is constrained to be rigid. Defaults to False.
        initial_affine (ndarray, optional): An optional initial guess of the affine that will transform template to align with target. Defaults to None.

    Returns:
        ndarray: The resultant affine that aligns template to target.
    """

    # Validate inputs.
    template = _validate_ndarray(template, dtype=float)
    target = _validate_ndarray(target, dtype=float, required_ndim=template.ndim)
    template_resolution = _validate_resolution(template_resolution, template.ndim, dtype=float)
    target_resolution = _validate_resolution(target_resolution, target.ndim, dtype=float)
    num_iterations = int(num_iterations)
    affine_stepsize = float(affine_stepsize)
    fixed_scale = _validate_scalar_to_multi(fixed_scale, template.ndim, dtype=float)
    if initial_affine is None:
        initial_affine = np.eye(template.ndim + 1)
    else:
        initial_affine = _validate_ndarray(initial_affine, required_shape=(template.ndim + 1, template.ndim + 1))

    # Initialize values.
    affine_inv = inv(initial_affine)
    affine_inv_template = apply_affine_to_image(template, template_resolution, initial_affine)

    # Iteratively optimize affine_inv.
    for iteration in range(num_iterations):

        # Compute affine_inv_gradient.
        affine_inv_gradient = _compute_affine_inv_gradient(
            template=template,
            target=target,
            template_resolution=template_resolution,
            target_resolution=target_resolution,
            affine_inv=affine_inv,
            affine_inv_template=affine_inv_template,
        )

        # Update affine_inv.
        affine_inv = _update_affine_inv(
            affine_inv=affine_inv,
            affine_inv_gradient=affine_inv_gradient,
            affine_stepsize=affine_stepsize,
            fixed_scale=fixed_scale,
            rigid=rigid,
        )

        # Apply affine_inv to template.
        affine_inv_template = apply_affine_to_image(template, template_resolution, inv(affine_inv))
    
    affine = inv(affine)

    return affine


def rigidly_align_slices(
    slices,
    slice_resolutions,
    num_iterations,
    sigma_gaussian,
    clip_gaussian_at_z=3,
):

    # Validate inputs.

    slices = list(slices)
    slice_ndim = np.array(slices[0]).ndim
    for slice_index, slice_ in enumerate(slices):
        slices[slice_index] = _validate_ndarray(slice_, required_ndim=slice_ndim)
    try:
        slice_resolutions = list(slice_resolutions)
    except TypeError:
        # slice_resolutions is a scalar.
        slice_resolutions = [slice_resolutions] * slice_ndim
    # slice_resolutions is a list.
    if np.array(slice_resolutions).ndim == 1:
        # slice_resolutions contains a single resolution to be applied to all slices.
        slice_resolutions = np.tile(slice_resolutions, [len(slices), 1])
    slice_resolutions = _validate_ndarray(slice_resolutions, minimum_ndim=2, required_shape=(len(slices), slice_ndim))
    num_iterations = int(num_iterations)
    sigma_gaussian = int(sigma_gaussian)
    clip_gaussian_at_z = int(clip_gaussian_at_z)

    # Resample slices into a volume with the largest size per dimension of all slices.
    volume = _slices_to_volume(slices)
    volume_resolutions

    # Create a parallel volume whose slices are weighted averages of their neighbors in volume.
    volume_neighbors = _volume_to_neighbor_averages(volume, sigma_gaussian, clip_gaussian_at_z)

    # Rigidly register each slice in volume to its corresponding slice in volume_neighbors.
    for slice_index in range(len(slices)):
        
        affine = affine_register(
            template=volume[slice_index],
            target=volume_neighbors[slice_index],
            num_iterations=num_iterations,
        )

        volume[slice_index] = apply_affine_to_image(volume[slice_index], RESOLUTION, affine)

    return volume