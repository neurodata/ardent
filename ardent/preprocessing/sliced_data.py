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
from scipy.interpolate import interpn
from scipy.linalg import inv, solve, svd
from scipy.signal import fftconvolve

from ..lddmm._lddmm_utilities import _validate_scalar_to_multi
from ..lddmm._lddmm_utilities import _validate_ndarray
from ..lddmm._lddmm_utilities import _validate_resolution
from ..lddmm._lddmm_utilities import _multiply_coords_by_affine
from ..lddmm._lddmm_utilities import _compute_axes
from ..lddmm._lddmm_utilities import _compute_coords


def apply_affine_to_image(image, image_resolution, affine, output_shape=None, output_resolution=None):
    """
    Interpolates image at its centered identity coordinates multiplied by inv(affine).
    A copy of image so interpolated is returned.

    Args:
        image (ndarray): The image to be modified.
        image_resolution (float, seq): The per-dimension resolution of image. If provided as a scalar, 
            that scalar is interpreted as the resolution at every dimension.
        affine (ndarray): The affine array in homogenous coordinates to be applied to image.
        output_shape (seq, optional): The shape to interpolate image at.
            If None, image.shape is used. Defaults to None.
        output_resolution (float, seq, optional): The resolution to interpolate the image at.
            If None, the image_resolution is used. Defaults to None.

    Returns:
        ndarray: A copy of image with affine applied to it.
    """

    # Validate inputs.
    image = _validate_ndarray(image, dtype=float)
    image_resolution = _validate_resolution(image_resolution, image.ndim)
    affine = _validate_ndarray(affine, required_shape=(image.ndim + 1, image.ndim + 1))
    if output_shape is None:
        output_shape = np.array(image.shape)
    output_shape = _validate_ndarray(output_shape, required_shape=image.ndim)
    if output_resolution is None:
        output_resolution = np.copy(image_resolution)
    output_resolution = _validate_resolution(output_resolution, image.ndim)

    affine_inv = inv(affine)

    image_axes = _compute_axes(image.shape, image_resolution, origin='center')
    output_coords = _compute_coords(output_shape, output_resolution, origin='center')
    
    # Apply affine_inv to image_coords by multiplication.
    affine_inv_output_coords = _multiply_coords_by_affine(affine_inv, output_coords)

    # Apply affine_inv_coords to image.
    affine_inv_image = interpn(
        points=image_axes,
        values=image,
        xi=affine_inv_output_coords,
        bounds_error=False,
        fill_value=None,
    )

    return affine_inv_image


def _slices_to_volume(slices, slice_resolutions, affines):
    """
    Resample slices into a volume with the largest real shape
    (physical length, i.e. number of voxels times voxel size) and smallest resolution 
    per dimension of all slices. Each affine in affines is applied to its corresponding slice.

    slices are stacked on dimension 0.
    """

    slice_shapes = np.array(list(map(lambda slice_: slice_.shape, slices)))
    slice_real_shapes = slice_shapes * slice_resolutions
    maximum_slice_real_shape = np.max(slice_real_shapes, 0)
    minimum_slice_resolution = np.min(slice_resolutions, 0)
    volume_slice_shape = np.ceil(maximum_slice_real_shape / minimum_slice_resolution).astype(int)
    volume_slice_resolution = maximum_slice_real_shape / volume_slice_shape

    # slices are stacked on the last dimension in volume.
    volume = np.empty((len(slices), *volume_slice_shape), dtype=float)
    for slice_index, slice_ in enumerate(slices):
        volume[slice_index] = apply_affine_to_image(
            image=slice_,
            image_resolution=slice_resolutions[slice_index],
            affine=affines[slice_index],
            output_shape=volume_slice_shape,
            output_resolution=volume_slice_resolution,
        )

    return volume, volume_slice_resolution


def _volume_to_neighbor_averages(volume, sigma_gaussian, clip_gaussian_at_z):
    """
    Create a parallel volume whose slices are weighted averages of their neighbors in volume.
    """

    slice_ndim = volume[0].ndim

    kernel = np.arange(1, int(sigma_gaussian * clip_gaussian_at_z + 1))
    kernel = np.concatenate((kernel[::-1], [0], kernel))
    kernel = np.exp(-kernel**2 / (2 * sigma_gaussian**2))
    kernel[len(kernel) // 2] = 0
    kernel /= kernel.sum()
    kernel = kernel.reshape(-1, *[1] * (volume.ndim - 1))
    pad_width = [[len(kernel) // 2] * 2, *[[0, 0] for _ in range(volume.ndim - 1)]]
    volume_neighbors = np.pad(volume, pad_width, mode='reflect')
    volume_neighbors = fftconvolve(volume_neighbors, kernel, mode='valid', axes=0)

    return volume_neighbors


def _compute_affine_inv_gradient(
    template,
    target,
    template_resolution,
    target_resolution,
    affine_inv,
    affine_inv_template,
    skip_gauss_newton=True,
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
    affine_inv_gradient_spatial = (
        affine_inv_template_gradient_broadcast
        * np.expand_dims(homogenous_target_coords, -2)
    )

    # Note: affine_inv_gradient_reduction contains the error term, affine_inv_gradient_spatial does not.

    # For a 3D case, affine_inv_gradient_reduction has shape (3,4).
    error = affine_inv_template - target
    affine_inv_gradient_reduction = affine_inv_gradient_spatial * error[..., None, None]
    affine_inv_gradient_reduction = np.sum(affine_inv_gradient_reduction, tuple(range(target.ndim)))
    if skip_gauss_newton:
        # Append a row of zeros at the end of the 0th dimension.
        zeros = np.zeros((1, target.ndim + 1))
        affine_inv_gradient = np.concatenate((affine_inv_gradient_reduction, zeros), 0)
        return affine_inv_gradient

    # Apply Gauss-Newton for 2nd order solution.

    # Reshape to a single vector. For a 3D case this becomes shape (12,).
    affine_inv_gradient_reduction = affine_inv_gradient_reduction.ravel()

    # For a 3D case, affine_inv_gradient_spatial has shape (x,y,z,3,4).
    # For a 3D case, affine_inv_hessian_approx is affine_inv_gradient_spatial reshaped to shape (x,y,z,12,1), 
    # then matrix multiplied by itself transposed on the last two dimensions, then summed over the spatial dimensions
    # to resultant shape (12,12).
    affine_inv_hessian_approx = affine_inv_gradient_spatial.reshape(*affine_inv_gradient_spatial.shape[:-2], -1, 1)
    affine_inv_hessian_approx_tail_transpose = affine_inv_hessian_approx.reshape(*affine_inv_hessian_approx.shape[:-2], 1, -1)
    affine_inv_hessian_approx = affine_inv_hessian_approx @ affine_inv_hessian_approx_tail_transpose
    affine_inv_hessian_approx = np.sum(affine_inv_hessian_approx, tuple(range(target.ndim)))

    # Solve for affine_inv_gradient.
    try:
        affine_inv_gradient = solve(affine_inv_hessian_approx, affine_inv_gradient_reduction, assume_a='pos').reshape(affine_inv_gradient_spatial.shape[-2:])
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


def _update_affine_inv(affine_inv, affine_inv_gradient, affine_stepsize, fixed_affine_scale=None, rigid=False):
    """
    Apply affine_inv_gradient to affine_inv, possibly adjusting for fixed_affine_scale or rigid if specified.
    Being redundant with one another, fixed_affine_scale overrides rigid.
    """

    affine_inv -= affine_inv_gradient * affine_stepsize

    # Set scale of affine_inv if appropriate.
    if fixed_affine_scale is not None:
        # Take reciprocal of fixed_affine_scale Since it is applied to affine_inv rather than affine.
        fixed_affine_scale = 1 / fixed_affine_scale
        # TODO: let fixed_affine_scale be per-dimension? 
        # --> svd replaced by polar decomposition
        # M = U * R, U = unitary, R = positive symmetric definite
        # adjust R
        U, _, Vh = svd(affine_inv[:-1, :-1])
        affine_inv[:-1, :-1] = U @ np.diag([fixed_affine_scale] * (len(affine_inv) - 1)) @ Vh
    # If fixed_affine_scale was not provided (is None), project affine_inv to a rigid affine if rigid.
    elif rigid:
        U, _, Vh = svd(affine_inv[:-1, :-1])
        affine_inv[:-1, :-1] = U @ Vh

    return affine_inv


def affine_register(
    template,
    target,
    template_resolution,
    target_resolution,
    num_iterations=100,
    affine_stepsize=0.3,
    fixed_affine_scale=None,
    rigid=False,
    initial_affine=None,
    skip_gauss_newton=False,
):
    """
    Iteratively compute an affine that aligns template and target.

    Args:
        template (ndarray): The image to be aligned to target.
        target (ndarray): The image template is aligned to.
        template_resolution (float, seq): The per-dimension resolution of template. 

        target_resolution (float, seq): The per-dimension resolution of target. 

        num_iterations (int, optional): The number of iterations of registration to perform. Defaults to 100.
        affine_stepsize (float, optional): The Gauss-Newton stepsize in units of voxels. This should be between 0 and 1. Defaults to 0.3.
        fixed_affine_scale (float, seq, optional): If provided, this fixes the scale of the affine and constrains it to be rigid. Defaults to None.
        rigid (bool, optional): If True, the affine is constrained to be rigid. Defaults to False.
        initial_affine (ndarray, optional): An optional initial guess of the affine that will transform template to align with target. Defaults to None.
        skip_gauss_newton (bool, optional): If True, the 2nd-order gauss-newton optimization is skipped.
            If True the appropriate value for affine_stepsize will change. Defaults to False.

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
    if fixed_affine_scale is not None:
        fixed_affine_scale = _validate_scalar_to_multi(fixed_affine_scale, template.ndim, dtype=float)
    if initial_affine is None:
        initial_affine = np.eye(template.ndim + 1)
    else:
        initial_affine = _validate_ndarray(initial_affine, required_shape=(template.ndim + 1, template.ndim + 1))
    skip_gauss_newton = bool(skip_gauss_newton)

    # Initialize values.
    affine_inv = inv(initial_affine)

    # Iteratively optimize affine_inv.
    for iteration in range(num_iterations):

        # Apply affine_inv to template.
        affine_inv_template = apply_affine_to_image(
            image=template,
            image_resolution=template_resolution,
            affine=inv(affine_inv),
            output_shape=target.shape,
            output_resolution=target_resolution,
        )

        # Compute affine_inv_gradient.
        affine_inv_gradient = _compute_affine_inv_gradient(
            template=template,
            target=target,
            template_resolution=template_resolution,
            target_resolution=target_resolution,
            affine_inv=affine_inv,
            affine_inv_template=affine_inv_template,
            skip_gauss_newton=skip_gauss_newton,
        )

        # Update affine_inv.
        affine_inv = _update_affine_inv(
            affine_inv=affine_inv,
            affine_inv_gradient=affine_inv_gradient,
            affine_stepsize=affine_stepsize,
            fixed_affine_scale=fixed_affine_scale,
            rigid=rigid,
        )
    
    affine = inv(affine_inv)

    return affine


def affine_align_slices_to_volume(
    slices,
    slice_resolutions,
    num_iterations,
    sigma_gaussian,
    clip_gaussian_at_z=3,
    # affine_register kwargs.
    num_iterations_per_registration=1,
    affine_stepsize=0.3,
    fixed_affine_scale=None,
    rigid=True,
    initial_affines=None,
    skip_gauss_newton=False,
    # Output specifiers.
    return_slice_resolution=False,
    return_affines=False,
):
    """
    Compute an affine registration between each slice and a weighted average of its neighboring slices 
    and apply it to each slice, resampled into a volume.

    Args:
        slices (seq): A sequence of arrays, all of the same dimensionality, each of which is considered a slice to align.
        slice_resolutions (float, seq): The resolution per dimension of each slice. 
            If provided as a scalar, it is interpreted as the isotropic resolution of all slices.
        num_iterations (int): The number of iterations of registering each slice to its neighbors.
        sigma_gaussian (float): The standard deviation of the gaussian weighting of neighboring slices in units of slices.
        clip_gaussian_at_z (float, optional): The number of standard deviations worth of neighboring slices to include around each slice. Defaults to 3.
        num_iterations_per_registration (int, optional): The number of registration iterations to use for each registration of each slice. Defaults to 1.
        affine_stepsize (float, optional): The unitless stepsize for affine adjustments. 
            Unless skip_gauss_newton == True, this should be between 0 and 1. Defaults to 0.3.
        fixed_affine_scale (float, optional): The scale to impose on the affine at all iterations. If None, no scale is imposed. 
            Otherwise, this has the effect of making the affine always rigid. Defaults to None.
        rigid (bool, optional): If True, all computed affines are projected onto a rigid affine. Redundant with fixed_affine_scale. Defaults to True.
        initial_affines (seq, optional): An initial guess to the affine applied to each slice. 
            If a single array is provided, it is used to initialize all affines. If None, identity is used. Defaults to None.
        skip_gauss_newton (bool, optional): If True, the 2nd order optimization of the affine registration is skipped over, 
            using only the 1st order solution. This changes the appropriate value, and range of plausible values, for affine_stepsize. Defaults to False.
        return_slice_resolution (bool, optional): If True, a tuple is returned whose 2nd element is the resolution of each slice in the volume. Defaults to False.
        return_affines (bool, optional): If True, a tuple is returned whose last element is the affines used to align slices. Defaults to False.

    Returns:
        ndarray, tuple: The volume produced by applying the computed affines to each slice and interpolating the result into a single ndarray. 
            Each slice in this volume has the maximum shape in physical units of all slices per dimension, 
            and at least the highest resolution of all slices per dimension. slices are stacked along dimension 0 of this volume. 
            If either return_slice_resolution or return_affines is True then volume will be returned as the first element of a tuple 
            whose other elements are the resolution of all of its slices and/or the affines used to align the slices, in that order, if requested.
    """

    # Validate inputs.
    slices = list(slices)
    slice_ndim = np.array(slices[0]).ndim
    for slice_index, slice_ in enumerate(slices):
        slices[slice_index] = _validate_ndarray(slice_, dtype=float, required_ndim=slice_ndim)
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
    if initial_affines is None:
        initial_affines = np.repeat(np.eye(slice_ndim + 1)[None], len(slices), 0)
    if isinstance(initial_affines, np.ndarray) and initial_affines.ndim == 2:
        # A single affine was provided for initial_affines.
        initial_affines = [initial_affines] * len(slices)
    else:
        initial_affines = list(initial_affines)
    initial_affines = _validate_ndarray(initial_affines, required_shape=(len(slices), slice_ndim + 1, slice_ndim + 1))

    # Initialize affines.
    affines = initial_affines

    # Iteratively register slices.
    for iteration in range(num_iterations):

        # Resample slices into a volume
        # with the largest real shape per dimension of all slices,
        # and at least the finest resolution per dimension of all slices.
        volume, volume_slice_resolution = _slices_to_volume(slices, slice_resolutions, affines)

        # Create a parallel volume whose slices are weighted averages of their neighbors in volume.
        volume_neighbors = _volume_to_neighbor_averages(volume, sigma_gaussian, clip_gaussian_at_z)
        # Note: slices in volume_neighbors have the same resolutions as volume, i.e. volume_slice_resolution.

        # Rigidly register each slice to its corresponding slice in volume_neighbors.
        for slice_index in range(len(slices)):
            
            # Compute affine between this slice and its convolved neighbors.
            affine = affine_register(
                template=slices[slice_index],
                target=volume_neighbors[slice_index],
                template_resolution=slice_resolutions[slice_index],
                target_resolution=volume_slice_resolution,
                num_iterations=num_iterations_per_registration,
                affine_stepsize=affine_stepsize,
                fixed_affine_scale=fixed_affine_scale,
                rigid=rigid,
                initial_affine=affines[slice_index],
                skip_gauss_newton=skip_gauss_newton,
            )

            # Update affines with the computed affine.
            affines[slice_index] = affine

    # Construct output.

    volume, volume_slice_resolution = _slices_to_volume(slices, slice_resolutions, affines)

    additional_outputs = []

    if return_slice_resolution:
        additional_outputs.append(volume_slice_resolution)
    if return_affines:
        additional_outputs.append(affines)
    
    if not return_slice_resolution and not return_affines:
        return volume
    else:
        return (volume, *additional_outputs)
