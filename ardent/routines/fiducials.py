import numpy as numpy
from scipy.interpolate import interpn

from ..utilities import _validate_ndarray
from ..utilities import _validate_xyz_resolution
from ..utilities import _compute_axes


def _find_transformed_points(position_field, position_field_resolution=1, points, origin='center'):
    """
    Return the positions of the transformed points.
    
    To transform points in the template space to the corresponding positions in the target space, 
    the position_field should be affine_phi (shape of the template).

    To transform points in the target space to the corresponding positions in the template space, 
    the position_field should be phi_inv_affine_inf (shape of the target).

    Transforming points - 
    template --> target: affine_phi
    target --> template: phi_inv_affine_inv
    """

    # Validate inputs.
    position_field = _validate_ndarray(position_field)
    position_field_resolution = _validate_xyz_resolution(position_field.ndim - 1, position_field_resolution)
    points = _validate_ndarray(points, required_ndim=2)
    if points.shape[1] != position_field.ndim - 1:
        raise ValueError(f"points.shape[1] must equal position_field.ndim - 1.\n"
                         f"points.shape[1]: {points.shape[1]}, position_field.ndim - 1: {position_field.ndim - 1}.")

    return interpn(
        points=_compute_axes(position_field.shape[:-1], position_field_resolution, origin), 
        values=position_field, 
        xi=points,
    )
