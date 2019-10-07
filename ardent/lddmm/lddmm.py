#%%
import numpy as np
from scipy.interpolate import interpn
from scipy.linalg import inv, solve
from skimage.transform import resize

from ardent.utilities import _validate_ndarray
from ardent.utilities import _validate_scalar_to_multi
from ardent.utilities import _compute_axes
from ardent.utilities import _compute_coords
from ardent.preprocessing.resampling import change_resolution_to

from pathlib import Path
import sys
utilities_path = Path('~/Documents/jhu/ardent/ardent/utilities.py').expanduser()

with open(utilities_path, 'r') as utilities_file:
    exec(utilities_file.read())

# from utilities import _validate_scalar_to_multi
# from utilities import _compute_axes
# from utilities import _compute_coords

class _Holder:
    """Container class for shared values and objects used in registration."""

    def __init__(self, template, target, template_resolution=1, target_resolution=1, 
    num_timesteps=5, affine=np.eye(4), contrast_order=1, sigmaM=None, smooth_length=None, sigmaR=None):
    
        # Images.
        self.template = template
        self.target = target

        # Resolution, axes, & coords.
        self.template_resolution = _validate_scalar_to_multi(template_resolution, self.template.ndim)
        self.template_axes = _compute_axes(self.template.shape, self.template_resolution)
        self.template_coords = _compute_coords(self.template.shape, self.template_resolution)
        self.target_resolution = _validate_scalar_to_multi(target_resolution, self.target.ndim)
        self.target_axes = _compute_axes(self.target.shape, self.target_resolution)
        self.target_coords = _compute_coords(self.target.shape, self.target_resolution)

        # Intermediates.
        self.num_timesteps = num_timesteps
        self.delta_t = 1 / self.num_timesteps
        self.contrast_order = contrast_order
        self.B = np.empty((self.target.size, self.contrast_order + 1))
        self.matching_weights = np.ones_like(self.target)
        self.deformed_template_to_t = []

        # Final outputs.
        if self.contrast_order < 1: raise ValueError(f"contrast_order must be at least 1.\ncontrast_order: {self.contrast_order}")
        self.contrast_coefficients = np.zeros(contrast_order + 1)
        self.contrast_coefficients[0] = np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template)
        self.contrast_coefficients[1] = np.std(self.target) / np.std(self.template)
        self.affine = _validate_ndarray(affine, required_ndim=2, broadcast_to_shape=(4, 4))
        self.affine_inv = inv(self.affine)
        self.velocity_fields = np.zeros((*self.template.shape, self.num_timesteps, self.template.ndim))
        self.phi = _compute_coords(self.template.shape, self.template_resolution)
        self.phi_inv = _compute_coords(self.template.shape, self.template_resolution)
        self.deformed_template = None
        self.contrast_deformed_template = None

        # Internals.
        self.sigmaM = sigmaM or np.std(self.target)
        self.sigmaR = sigmaR or 10 * np.max(self.template_resolution)
        self.smooth_length = smooth_length or 2 * np.max(self.template_resolution)
        self.fourier_velocity_fields = np.zeros_like(self.velocity_fields, np.complex128)

        self.fourier_high_pass_filter_power = 2
        fourier_velocity_fields_coords = _compute_coords(self.template.shape, 1 / (self.template_resolution * self.template.shape), origin='zero')
        self.fourier_high_pass_filter = (
            1 - self.smooth_length**2 
            * np.sum((-2  + 2 * np.cos(2 * np.pi * fourier_velocity_fields_coords * self.template_resolution)) / self.template_resolution**2, axis=-1)
        )**self.fourier_high_pass_filter_power


# f0I = torch.arange(self.nxI[0],dtype=self.dtype,device=self.device)/self.dxI[0]/self.nxI[0]
# f1I = torch.arange(self.nxI[1],dtype=self.dtype,device=self.device)/self.dxI[1]/self.nxI[1]
# f2I = torch.arange(self.nxI[2],dtype=self.dtype,device=self.device)/self.dxI[2]/self.nxI[2]
# F0I,F1I,F2I = torch.meshgrid(f0I, f1I, f2I)
# self.a = a
# self.p = p # fourier_high_pass_filter_power
# p=2
# Lhat = (1.0 - self.a**2*( 
#           (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[0]*F0I))/self.dxI[0]**2 
#         + (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[1]*F1I))/self.dxI[1]**2
#         + (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[2]*F2I))/self.dxI[2]**2 ) )**self.p

        self.phi_inv_affine_inv_target_coords = _compute_coords(self.target.shape, self.target_resolution)

        # Accumulators.
        self.matching_energies = []
        self.regularization_energies = []
        self.total_energies = []

        

    # @property
    


def register(template, target, template_resolution=1, target_resolution=1, 
    translation_stepsize=0, 
    linear_stepsize=0, 
    deformative_stepsize=0, 
    sigmaR=0, 
    num_iterations=200, 
    num_affine_only_iterations=50, 
    initial_affine=np.eye(4), initial_v=None, 
    num_timesteps=5, contrast_order=1, sigmaM=None, smooth_length=None):

    # Set up Holder instance.
    holder = _Holder(
        template=template,
        target=target,
        template_resolution=template_resolution,
        target_resolution=target_resolution,
        num_timesteps=num_timesteps,
        affine=initial_affine,
        contrast_order=contrast_order,
        sigmaM=sigmaM,
        smooth_length=smooth_length,
        sigmaR=sigmaR,
    )


    # Iteratively perform each step of the registration.
    for iteration in range(num_iterations):
        # Apply transforms to template.
        _apply_transforms(holder)
        # Contrast transform the deformed_template.
        _apply_contrast_map(holder)
        # Compute weights.
        _compute_weights(holder)
        # Compute cost.
        _compute_cost(holder)
        # Compute contrast map.
        _compute_contrast_map(holder)

        # Compute affine.

        # Compute velocity field gradients.
        # Update velocity fields.

        # Compute affine gradient.
        # Update affine array.

        # Do other things.
        pass

        return holder


def _apply_transforms(holder):
    """
    Calculate phi_inv from v
    Compose on the right with Ainv
    Apply phi_invAinv to template
    Transform the contrast of this deformed template

    requires:
        template
        target
        template_coords
        target_coords
        v
        A
        contrast_transform_coefficients --> produce outside

    results: 
        transformed_template_at_each_v_t
        contrast_transformed_deformed_template

        end result to user: phi_inv, phi_invAinv, contrast_transform_coefficients

    ==============================================================================

    used in holder:
        num_timesteps
        delta_t
        velocity_fields

        template_axes
        template_coords
        target_coords

        affine
        template

        contrast_coefficients
        

    set in holder:
        phi_inv
        affine_inv
        phi_inv_affine_inv_target_coords
        deformed_template
        contrast_deformed_template, via _apply_contrast_map


    """
    
    # Reset holder.deformed_template_to_t.
    holder.deformed_template_to_t = []

    for timestep in range(holder.num_timesteps):
        # Compute phi_inv.
        sample_coords = holder.template_coords - holder.delta_t * holder.velocity_fields[..., timestep, :]
        holder.phi_inv = interpn(
            points=holder.template_axes, 
            values=holder.phi_inv - holder.template_coords, 
            xi=sample_coords, 
            bounds_error=False, 
            fill_value=None, 
        ) + sample_coords

        # Compute deformed_template_to_t
        holder.deformed_template_to_t.append(interpn(
            points=holder.template_axes, 
            values=holder.template, 
            xi=holder.phi_inv, 
            bounds_error=False, 
            fill_value=None, 
        ))

        # End time loop.
    
    # Compute affine_inv.
    holder.affine_inv = inv(holder.affine)

    # Apply affine_inv to target_coords by multiplication.
    affine_inv_target_coords = _multiply_by_affine(holder.target_coords, holder.affine_inv)

    # Apply phi_inv to affine_inv_target_coords.
    holder.phi_inv_affine_inv_target_coords = interpn(
        points=holder.template_axes, 
        values=holder.phi_inv - holder.template_coords, 
        xi=affine_inv_target_coords, 
        bounds_error=False, 
        fill_value=None, 
    ) + affine_inv_target_coords

    # Apply phi_inv_affine_inv_target_coords to template.
    # deformed_template is sampled at the coordinates of the target.
    holder.deformed_template = interpn(
        points=holder.template_axes, 
        values=holder.template, 
        xi=holder.phi_inv_affine_inv_target_coords, 
        bounds_error=False, 
        fill_value=None, 
    )

        # DEBUG future tests pytest
        # print('max of phi_inv - template_coords:',np.max(holder.phi_inv - holder.template_coords))
        # print('max dif of phi_inv_affine_inv_target_coords:',np.max(holder.phi_inv_affine_inv_target_coords - holder.target_coords))


def _compute_contrast_map(holder):
    """
    Compute contrast_coefficients mapping deformed_template to target.

    used in holder:
        deformed_template
        matching_weights
        contrast_order

    set in holder:
        B
        contrast_coefficients

    """

    # Ravel necessary components.
    deformed_template_ravel = np.ravel(holder.deformed_template)
    target_ravel = np.ravel(holder.target)
    matching_weights_ravel = np.ravel(holder.matching_weights)

    # Set basis array B and create composites.
    for power in range(holder.contrast_order + 1):
        holder.B[:, power] = deformed_template_ravel**2
    B_transpose_B = np.matmul(holder.B.T * matching_weights_ravel, holder.B)
    B_transpose_target = np.matmul(holder.B.T * matching_weights_ravel, target_ravel)

    # Solve for contrast_coefficients.
    holder.contrast_coefficients = solve(B_transpose_B, holder.B.T * target_ravel, assume_a='pos')


def _apply_contrast_map(holder):
    """
    Apply contrast_coefficients to deformed_template to produce contrast_deformed_template.
    
    used in holder:
        contrast_coefficients
        B

    set in holder:
        contrast_deformed_template
    """

    holder.contrast_deformed_template = np.matmul(holder.B, holder.contrast_coefficients).reshape(holder.target.shape)


def _compute_weights(holder):
    """
    Compute the  matching_weights between the contrast_deformed_template and the target.

    used in holder:
        contrast_deformed_template
        target
        sigmaM
        sigmaA
        ca?

    set in holder:
        matching_weights
    """
    pass


def _compute_cost(holder):
    """
    Compute the matching cost using a weighted sum of square error.

    used in holder:
        contrast_deformed_template
        sigmaM
        sigmaR
        matching_weights
        template_resolution
        target_resolution
        fourier_velocity_fields
        fourier_high_pass_filter

    set in holder:
        matching_energies
        regularization_energies
        total_energies
    """

    matching_energy = (
        np.sum((holder.contrast_deformed_template - holder.target)**2 * holder.matching_weights) * 
        1/(2 * holder.sigmaR**2) * np.prod(holder.target_resolution)
    )

    # ER = torch.sum(torch.sum(torch.sum(torch.abs(self.vhat)**2,dim=(-1,1,0))*self.LLhat)) * (self.dt*torch.prod(self.dxI)/2.0/self.sigmaR**2/torch.numel(self.I))
    regularization_energy = (
        np.sum((np.abs(holder.fourier_velocity_fields) * holder.fourier_high_pass_filter[..., None, None])**2) * 
        1/(2 * holder.sigmaR**2) * np.prod(holder.template_resolution) / holder.delta_t / holder.template.size
    )

    total_energy = matching_energy + regularization_energy

    # Accumulate energies.
    holder.matching_energies.append(matching_energy)
    holder.regularization_energies.append(regularization_energy)
    holder.total_energies.append(total_energy)


def _gradient(params, affine_or_deformative):
    pass


def _affine_step(params):
    pass


def _deformative_step(holder):
    '''Backwards pass: one step of gradient descent for the velocity field.'''

    matching_error_prime = (holder.deformed_template - holder.target) * holder.matching_weights
    contrast_map_prime = np.zeros_like(holder.target)
    for power in range(1, holder.contrast_order + 1):
        contrast_map_prime += power * holder.deformed_template**(power - 1) * holder.contrast_coefficients[power]
    d_matching_d_deformed_template = matching_error_prime * contrast_map_prime


    # Compute phi_1t_inv
    # Apply affine
    

    # Set phi to identity. phi is secretly phi_1t_inv but at the end of the loop 
    # it will be phi_10_inv = phi_01 = phi.
    phi = np.copy(holder.template_coords)

    # Loop backwards across time.
    for timestep in range(holder.num_timesteps - 1, -1, -1):

        # Update phi.
        sample_coords = holder.template_coords + holder.velocity_fields[..., timestep, :] * holder.delta_t
        phi = interpn(
            points=holder.template_axes, 
            values=phi - holder.template_coords, 
            xi=sample_coords, 
            bounds_error=False, 
            fill_value=None, 
        ) + sample_coords

        # Apply affine by multiplication.
        affine_phi = _multiply_by_affine(phi, holder.affine)

        # Compute the determinant of the gradient of phi.


        # 



# TODO: use.
def _multiply_by_affine(array, affine):
    return np.stack(
        arrays=[
            np.sum(affine[0, :3] * array + affine[0, 3], axis=-1), 
            np.sum(affine[1, :3] * array + affine[1, 3], axis=-1), 
            np.sum(affine[2, :3] * array + affine[2, 3], axis=-1), 
        ],
        axis=-1,
    )


def _apply_transform(holder, subject:np.ndarray, subject_resolution, output_resolution=None, deform_to="template"):

    # Validate inputs.

    # Verify holder.
    if not isinstance(holder, _Holder):
        raise TypeError(f"holder must be an instance of class Holder.\n"
            f"type(holder): {type(holder)}.")
    # Validate subject.
    subject = _validate_ndarray(subject, required_ndim=holder.template.ndim)
    # Verify deform_to.AttributeError
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")
    # Validate output_resolution.
    if output_resolution is None:
        # Default to the resolution of the deform_to object.
        if deform_to == "template":
            output_resolution = holder.template_resolution
        elif deform_to == "target":
            output_resolution = holder.target_resolution
    elif output_resolution == "template":
        output_resolution = holder.template_resolution
    elif output_resolution == "target":
        output_resolution = holder.target_resolution
    else:
        output_resolution = _validate_scalar_to_multi(output_resolution, size=len(holder.template_resolution))
    
    # Create position field.
    if deform_to == "template":
        phi = _compute_coords(holder.template.shape, holder.template_resolution)
    elif deform_to == "target":
        phi_inv = _compute_coords(holder.template.shape, holder.template_resolution)

    # Integrate velocity field.
    for timestep in (range(holder.num_timesteps - 1, -1, -1) if deform_to == "template" 
        else range(0, holder.num_timesteps)):
        if deform_to == "template":
            sample_coords = holder.template_coords + holder.velocity_fields[..., timestep, :] * holder.delta_t
            phi = interpn(
                points=holder.template_coords,
                values=phi - holder.template_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            ) + sample_coords
        elif deform_to == "target":
            sample_coords = holder.template_coords - holder.velocity_fields[..., timestep, :] * holder.delta_t
            phi_inv = interpn(
                points=holder.template_coords,
                values=phi_inv - holder.template_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            ) + sample_coords

    # Apply the affine transform to the position field.
    if deform_to == "template":
        # Apply the affine by multiplication.
        affine_phi = _multiply_by_affine(phi, holder.affine)
        # affine_phi has the resolution of the template.
    elif deform_to == "target":
        # Apply the affine by interpolation.
        sample_coords = _multiply_by_affine(holder.target_coords, holder.affine_inv)
        phi_inv_affine_inv = interpn(
            points=holder.template_coords,
            values=phi_inv - holder.template_coords,
            xi=sample_coords,
            bounds_error=False,
            fill_value=None,
        ) + sample_coords
        # phi_inv_affine_inv has the resolution of the target.

    # Resample position field.
    if deform_to == "template":
        affine_phi = change_resolution_to(affine_phi, holder.template_resolution, output_resolution)
    elif deform_to == "target":
        phi_inv_affine_inv = change_resolution_to(phi_inv_affine_inv, holder.target_resolution, output_resolution)

    # Interpolate subject at position field.
    deformed_subject = interpn(
        points=_compute_axes(shape=subject.shape, xyz_resolution=subject_resolution),
        values=subject,
        xi=affine_phi if deform_to == "template" else phi_inv_affine_inv # if deform_to == "target"
    )

#%% Testing script.

template = np.zeros([12]*3, dtype=float)
r = 5
for i in range(template.shape[0]):
    for j in range(template.shape[1]):
        for k in range(template.shape[2]):
            if np.sqrt((i-6)**2 + (j-6)**2 + (k-6)**2) <= r:
                template[i,j,k] = 1

target = np.zeros([18, 18, 12], dtype=float)
a, b, c = 8, 8, 5
for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            if (i-9)**2 / a**2 + (j-9)**2 / b**2 + (k-6)**2 / c**2 <= 1:
                target[i,j,k] = 1

template = (template - np.mean(template)) / np.mean(template)
target = (target - np.mean(target)) / np.mean(target)

# holder = register(
#     template, target, 
#     num_iterations=2,
# )

holder = Holder(template, target, affine=[[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])

_apply_transform(subject=template, subject_resolution=1, output_resolution=None, deform_to="target", holder=holder)







'''
def register(template, target, template_resolution=1, target_resolution=1, 
    translation_stepsize=0, 
    linear_stepsize=0, 
    deformative_stepsize=0, 
    sigmaR=0, 
    num_iterations=200, 
    num_affine_only_iterations=50, 
    initial_affine=None, initial_v=None, 
    num_timesteps=5, contrast_order=1, sigmaM=None, smooth_length=None):
'''
