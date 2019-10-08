#%% Definition cell
import numpy as np
from scipy.interpolate import interpn
from scipy.linalg import inv, solve
from skimage.transform import resize

# from ardent.utilities import _validate_ndarray
# from ardent.utilities import _validate_scalar_to_multi
# from ardent.utilities import _validate_xyz_resolution
# from ardent.utilities import _compute_axes
# from ardent.utilities import _compute_coords
# from ardent.utilities import _multiply_by_affine
# from ardent.preprocessing.resampling import change_resolution_to

from matplotlib import pyplot as plt
%matplotlib inline
from pathlib import Path
import sys
utilities_path = Path('~/Documents/jhu/ardent/ardent/utilities.py').expanduser()
with open(utilities_path, 'r') as utilities_file:
    exec(utilities_file.read())
resampling_path = Path('~/Documents/jhu/ardent/ardent/preprocessing/resampling.py').expanduser()
with open(resampling_path, 'r') as resampling_file:
    exec(resampling_file.read())

# from utilities import _validate_scalar_to_multi
# from utilities import _compute_axes
# from utilities import _compute_coords

'''
  _            _       _                         
 | |          | |     | |                        
 | |        __| |   __| |  _ __ ___    _ __ ___  
 | |       / _` |  / _` | | '_ ` _ \  | '_ ` _ \ 
 | |____  | (_| | | (_| | | | | | | | | | | | | |
 |______|  \__,_|  \__,_| |_| |_| |_| |_| |_| |_|
                                                 
'''

class Lddmm:
    """Class for storing shared values and objects used in registration and performing the registration via methods. Not intended for direct user interaction."""

    def __init__(self, template, target, template_resolution=1, target_resolution=1, num_iterations=200, 
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
        self.num_itrations = num_iterations
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
        self.phi_inv_affine_inv_target_coords = _compute_coords(self.target.shape, self.target_resolution)

        # Accumulators.
        self.matching_energies = []
        self.regularization_energies = []
        self.total_energies = []


    def register(self):

        # Iteratively perform each step of the registration.
        for iteration in range(self.num_iterations):
            # Apply transforms to template.
            self._apply_transforms()
            # Contrast transform the deformed_template.
            self._apply_contrast_map()
            # Compute weights.
            self._compute_weights()
            # Compute cost.
            self._compute_cost()
            # Compute contrast map.
            self._compute_contrast_map()

            # Compute affine.

            # Compute velocity field gradients.
            # Update velocity fields.

            # Compute affine gradient.
            # Update affine array.

            # Do other things.
            pass

            affine_phi, phi_inv_affine_inv, template_resolution, target_resolution
            return dict(
                affine_phi=,
                phi_inv_affine_inv=,
                template_resolution=,
                target_resolution=,
                affine=,
                # Accumulators.
                matching_energies=,
                regularization_energies=,
                total_energies=,

            )


    def _apply_transforms(self):
        """
        Calculate phi_inv from v
        Compose on the right with Ainv
        Apply phi_invAinv to template

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


        """
        
        # Reset self.deformed_template_to_t.
        self.deformed_template_to_t = []

        for timestep in range(self.num_timesteps):
            # Compute phi_inv.
            sample_coords = self.template_coords - self.delta_t * self.velocity_fields[..., timestep, :]
            self.phi_inv = interpn(
                points=self.template_axes, 
                values=self.phi_inv - self.template_coords, 
                xi=sample_coords, 
                bounds_error=False, 
                fill_value=None, 
            ) + sample_coords

            # Compute deformed_template_to_t
            self.deformed_template_to_t.append(
                interpn(
                    points=self.template_axes, 
                    values=self.template, 
                    xi=self.phi_inv, 
                    bounds_error=False, 
                    fill_value=None, 
                )
            )
            
            # End time loop.
        
        # Compute affine_inv.
        self.affine_inv = inv(self.affine)

        # Apply affine_inv to target_coords by multiplication.
        affine_inv_target_coords = _multiply_by_affine(self.target_coords, self.affine_inv)

        # Apply phi_inv to affine_inv_target_coords.
        self.phi_inv_affine_inv_target_coords = interpn(
            points=self.template_axes, 
            values=self.phi_inv - self.template_coords, 
            xi=affine_inv_target_coords, 
            bounds_error=False, 
            fill_value=None, 
        ) + affine_inv_target_coords

        # Apply phi_inv_affine_inv_target_coords to template.
        # deformed_template is sampled at the coordinates of the target.
        self.deformed_template = interpn(
            points=self.template_axes, 
            values=self.template, 
            xi=self.phi_inv_affine_inv_target_coords, 
            bounds_error=False, 
            fill_value=None, 
        )

            # DEBUG future tests pytest
            # print('max of phi_inv - template_coords:',np.max(self.phi_inv - self.template_coords))
            # print('max dif of phi_inv_affine_inv_target_coords:',np.max(self.phi_inv_affine_inv_target_coords - self.target_coords))


    def _compute_contrast_map(self):
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
        deformed_template_ravel = np.ravel(self.deformed_template)
        target_ravel = np.ravel(self.target)
        matching_weights_ravel = np.ravel(self.matching_weights)

        # Set basis array B and create composites.
        for power in range(self.contrast_order + 1):
            self.B[:, power] = deformed_template_ravel**2
        B_transpose_B = np.matmul(self.B.T * matching_weights_ravel, self.B)
        B_transpose_target = np.matmul(self.B.T * matching_weights_ravel, target_ravel)

        # Solve for contrast_coefficients.
        self.contrast_coefficients = solve(B_transpose_B, self.B.T * target_ravel, assume_a='pos')


    def _apply_contrast_map(self):
        """
        Apply contrast_coefficients to deformed_template to produce contrast_deformed_template.
        
        used in holder:
            contrast_coefficients
            B

        set in holder:
            contrast_deformed_template
        """

        self.contrast_deformed_template = np.matmul(self.B, self.contrast_coefficients).reshape(self.target.shape)


    def _compute_weights(self):
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


    def _compute_cost(self):
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
            np.sum((self.contrast_deformed_template - self.target)**2 * self.matching_weights) * 
            1/(2 * self.sigmaR**2) * np.prod(self.target_resolution)
        )

        # ER = torch.sum(torch.sum(torch.sum(torch.abs(self.vhat)**2,dim=(-1,1,0))*self.LLhat)) * (self.dt*torch.prod(self.dxI)/2.0/self.sigmaR**2/torch.numel(self.I))
        regularization_energy = (
            np.sum((np.abs(self.fourier_velocity_fields) * self.fourier_high_pass_filter[..., None, None])**2) * 
            1/(2 * self.sigmaR**2) * np.prod(self.template_resolution) / self.delta_t / self.template.size
        )

        total_energy = matching_energy + regularization_energy

        # Accumulate energies.
        self.matching_energies.append(matching_energy)
        self.regularization_energies.append(regularization_energy)
        self.total_energies.append(total_energy)


    def _gradient(self, affine_or_deformative):
        pass


    def _affine_step(self):
        pass


    def _deformative_step(self):
        '''Backwards pass: one step of gradient descent for the velocity field.'''

        matching_error_prime = (self.deformed_template - self.target) * self.matching_weights
        contrast_map_prime = np.zeros_like(self.target)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += power * self.deformed_template**(power - 1) * self.contrast_coefficients[power]
        d_matching_d_deformed_template = matching_error_prime * contrast_map_prime


        # Compute phi_1t_inv
        # Apply affine
        

        # Set phi to identity. phi is secretly phi_1t_inv but at the end of the loop 
        # it will be phi_10_inv = phi_01 = phi.
        phi = np.copy(self.template_coords)

        # Loop backwards across time.
        for timestep in range(self.num_timesteps - 1, -1, -1):

            # Update phi.
            sample_coords = self.template_coords + self.velocity_fields[..., timestep, :] * self.delta_t
            phi = interpn(
                points=self.template_axes, 
                values=phi - self.template_coords, 
                xi=sample_coords, 
                bounds_error=False, 
                fill_value=None, 
            ) + sample_coords

            # Apply affine by multiplication.
            affine_phi = _multiply_by_affine(phi, self.affine)

            # Compute the determinant of the gradient of phi.

    # End Lddmm.
'''
  _    _                          __                          _     _                       
 | |  | |                        / _|                        | |   (_)                      
 | |  | |  ___    ___   _ __    | |_   _   _   _ __     ___  | |_   _    ___    _ __    ___ 
 | |  | | / __|  / _ \ | '__|   |  _| | | | | | '_ \   / __| | __| | |  / _ \  | '_ \  / __|
 | |__| | \__ \ |  __/ | |      | |   | |_| | | | | | | (__  | |_  | | | (_) | | | | | \__ \
  \____/  |___/  \___| |_|      |_|    \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
                                                                                            
'''

def _register(template, target, template_resolution=1, target_resolution=1, 
    translation_stepsize=0, 
    linear_stepsize=0, 
    deformative_stepsize=0, 
    sigmaR=0, 
    num_iterations=200, 
    num_affine_only_iterations=50, 
    initial_affine=np.eye(4), initial_v=None, 
    num_timesteps=5, contrast_order=1, sigmaM=None, smooth_length=None):

    # Set up Lddmm instance.
    lddmm = Lddmm(
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

    return lddmm.register()


def _generate_position_field(affine, velocity_fields, velocity_field_resolution, num_timesteps, 
template_shape, template_resolution, target_shape, target_resolution, deform_to="template"):

    # Validate inputs.

    # Validate velocity_fields.
    velocity_fields = _validate_ndarray(velocity_fields)
    # Validate velocity_field_resolution.
    velocity_field_resolution = _validate_xyz_resolution(velocity_fields.ndim - 2, velocity_field_resolution)
    # Validate affine.
    affine = _validate_ndarray(affine, required_ndim=2, broadcast_to_shape=(4, 4))
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")

    # Compute intermediates.
    delta_t = 1 / num_timesteps
    template_axes = _compute_axes(template_shape, template_resolution)
    template_coords = _compute_coords(template_shape, template_resolution)
    target_axes = _compute_axes(target_shape, target_resolution)
    target_coords = _compute_coords(target_shape, target_resolution)

    # Create position field.
    if deform_to == "template":
        phi = np.copy(template_coords)
    elif deform_to == "target":
        phi_inv = np.copy(template_coords)

    # Integrate velocity field.
    for timestep in (range(num_timesteps - 1, -1, -1) if deform_to == "template" 
        else range(0, num_timesteps)):
        if deform_to == "template":
            sample_coords = template_coords + velocity_fields[..., timestep, :] * delta_t
            phi = interpn(
                points=template_axes,
                values=phi - template_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            ) + sample_coords
        elif deform_to == "target":
            sample_coords = template_coords - velocity_fields[..., timestep, :] * delta_t
            phi_inv = interpn(
                points=template_axes,
                values=phi_inv - template_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            ) + sample_coords

    # Apply the affine transform to the position field.
    if deform_to == "template":
        # Apply the affine by multiplication.
        affine_phi = _multiply_by_affine(phi, affine)
        # affine_phi has the resolution of the template.
    elif deform_to == "target":
        # Apply the affine by interpolation.
        sample_coords = _multiply_by_affine(target_coords, affine_inv)
        phi_inv_affine_inv = interpn(
            points=template_axes,
            values=phi_inv - template_coords,
            xi=sample_coords,
            bounds_error=False,
            fill_value=None,
        ) + sample_coords
        # phi_inv_affine_inv has the resolution of the target.

    # return appropriate position field.
    if deform_to == "template":
        return affine_phi
    elif deform_to == "target":
        return phi_inv_affine_inv


def _apply_position_field(subject, subject_resolution, output_resolution, position_field, position_field_resolution):

    # Validate inputs.

    # Validate position_field.
    position_field = _validate_ndarray(position_field)
    # Validate position_field_resolution.
    position_field_resolution = _validate_xyz_resolution(position_field.ndim, position_field_resolution)
    # Validate subject.
    subject = _validate_ndarray(subject, required_ndim=position_field.ndim - 1)
    # Validate subject_resolution.
    subject_resolution = _validate_xyz_resolution(subject.ndim, subject_resolution)
    # Validate output_resolution.
    output_resolution = _validate_xyz_resolution(subject.ndim, output_resolution)

    # Resample position_field.
    position_field = change_resolution_to(position_field, [*position_field_resolution, 1], [*output_resolution, 1])

    # Interpolate subject at position field.
    deformed_subject = interpn(
        points=_compute_axes(shape=subject.shape, xyz_resolution=subject_resolution),
        values=subject,
        xi=position_field,
        bounds_error=False,
        fill_value=None,
    )

    return deformed_subject


def apply_transform(subject, subject_resolution, affine_phi, phi_inv_affine_inv, 
template_resolution, target_resolution, output_resolution=None, deform_to="template", **unused_kwargs):

    # Validate inputs: subject, subject_resolution, deform_to, & output_resolution.

    # Validate subject.
    subject = _validate_ndarray(subject)
    # Validate subject_resolution.
    subject_resolution = _validate_xyz_resolution(subject.ndim, subject_resolution)
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")
    # Validate output_resolution.
    if output_resolution is None and deform_to == "template" or output_resolution == "template":
        output_resolution = np.copy(template_resolution)
    elif output_resolution is None and deform_to == "target" or output_resolution == "target":
        output_resolution = np.copy(target_resolution)
    else:
        output_resolution = _validate_xyz_resolution(subject.ndim, output_resolution)

    # Define position_field and position_field_resolution.

    if deform_to == "template":
        position_field = affine_phi
        position_field_resolution = np.copy(template_resolution)
    elif deform_to == "target":
        position_field = phi_inv_affine_inv
        position_field_resolution = np.cpy(target_resolution)

    # Call _apply_position_field.

    deformed_subject = _apply_position_field(subject, subject_resolution, output_resolution, position_field, position_field_resolution)

    return deformed_subject

#%% Test cell

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

holder = _Holder(template, target)#, affine=[[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])

deformed_subject = _apply_transform(subject=template, subject_resolution=1, output_resolution=None, deform_to="target", holder=holder)







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
#%%
for i in range(0, len(template), len(template)//5):
    plt.imshow(template[i])
    plt.show()

#%%
ds = deformed_subject

for i in range(0, len(ds), len(ds)//5):
    plt.imshow(ds[i])
    plt.show()

#%%
