import numpy as np
from scipy.interpolate import interpn
from scipy.linalg import inv, solve, det
from skimage.transform import resize

from ardent.utilities import _validate_ndarray
from ardent.utilities import _validate_scalar_to_multi
from ardent.utilities import _validate_xyz_resolution
from ardent.utilities import _compute_axes
from ardent.utilities import _compute_coords
from ardent.utilities import _multiply_by_affine
from ardent.preprocessing.resampling import change_resolution_to

'''
  _            _       _                         
 | |          | |     | |                        
 | |        __| |   __| |  _ __ ___    _ __ ___  
 | |       / _` |  / _` | | '_ ` _ \  | '_ ` _ \ 
 | |____  | (_| | | (_| | | | | | | | | | | | | |
 |______|  \__,_|  \__,_| |_| |_| |_| |_| |_| |_|
                                                 
'''

class _Lddmm:
    """Class for storing shared values and objects used in registration and performing the registration via methods. Not intended for direct user interaction."""

    def __init__(self, template, target, template_resolution=1, target_resolution=1, check_artifacts=False, num_iterations=200, 
    num_timesteps=5, initial_affine=np.eye(4), initial_velocity_fields = None, contrast_order=1, sigmaM=None, sigmaA=None, smooth_length=None, sigmaR=None, 
    translational_stepsize=None, linear_stepsize=None, deformative_stepsize=None, contrast_stepsize=1):
    
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
        self.num_iterations = num_iterations
        self.num_timesteps = num_timesteps
        self.delta_t = 1 / self.num_timesteps
        self.contrast_order = contrast_order
        self.B = np.empty((self.target.size, self.contrast_order + 1))
        self.matching_weights = np.ones_like(self.target)
        self.deformed_template_to_t = []
        self.deformed_template = None
        self.contrast_deformed_template = None

        # Final outputs.
        self.contrast_coefficients = np.zeros(contrast_order + 1)
        self.contrast_coefficients[0] = np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template)
        self.contrast_coefficients[1] = np.std(self.target) / np.std(self.template)
        self.affine = _validate_ndarray(initial_affine, required_ndim=2, reshape_to_shape=(4, 4))
        self.affine_inv = inv(self.affine)
        self.velocity_fields = initial_velocity_fields or np.zeros((*self.template.shape, self.num_timesteps, self.template.ndim))
        self.phi = _compute_coords(self.template.shape, self.template_resolution)
        self.phi_inv = _compute_coords(self.template.shape, self.template_resolution)
        self.phi_inv_affine_inv = np.copy(self.target_coords)
        self.affine_phi = np.copy(self.template_coords)

        # Internals.
        self.check_artifacts = check_artifacts
        self.artifact_mean_value = None
        self.translational_stepsize = float(translational_stepsize)
        self.linear_stepsize = float(linear_stepsize)
        self.deformative_stepsize = float(deformative_stepsize)
        self.contrast_stepsize = contrast_stepsize
        self.contrast_order = int(contrast_order)
        if self.contrast_order < 1: raise ValueError(f"contrast_order must be at least 1.\ncontrast_order: {self.contrast_order}")
        self.sigmaM = sigmaM or np.std(self.target)
        self.sigmaA = sigmaA or 5 * self.sigmaM
        self.artifact_mean_value = np.max(self.target) if self.sigmaA is not None else 0 # TODO: verify this is right.
        self.sigmaR = sigmaR or 10 * np.max(self.template_resolution)
        self.smooth_length = smooth_length or 2 * np.max(self.template_resolution)
        self.fourier_velocity_fields = np.zeros_like(self.velocity_fields, np.complex128)

        self.fourier_high_pass_filter_power = 2
        fourier_velocity_fields_coords = _compute_coords(self.template.shape, 1 / (self.template_resolution * self.template.shape), origin='zero')
        self.fourier_high_pass_filter = (
            1 - self.smooth_length**2 
            * np.sum((-2  + 2 * np.cos(2 * np.pi * fourier_velocity_fields_coords * self.template_resolution)) / self.template_resolution**2, axis=-1)
        )**self.fourier_high_pass_filter_power
        self.phi_inv_affine_inv = _compute_coords(self.target.shape, self.target_resolution)

        # Accumulators.
        self.matching_energies = []
        self.regularization_energies = []
        self.total_energies = []


    def register(self):

        # Iteratively perform each step of the registration.
        for iteration in range(self.num_iterations):

            # Forward pass: apply transforms to the template and compute the costs.

            # Compute position_field from velocity_fields.
            self._update_and_apply_position_field()
            # Contrast transform the deformed_template.
            self._apply_contrast_map()
            # Compute weights. 
            # This is the expectation step of the expectation maximization algorithm.
            if self.check_artifacts and iteration % 1 == 0: self._compute_weights()
            # Compute cost.
            self._compute_cost()

            # Backward pass: update contrast map, affine, & velocity_fields.
            
            # Compute contrast map gradient.
            # Let the contrast map gradient be the optimal minus the current.
            contrast_map_gradient = self._compute_contrast_map_gradient()
            # Compute affine gradient.
            affine_gradient = self._compute_affine_gradient()
            # Compute velocity_fields gradient.
            velocity_fields_gradients = self._compute_velocity_fields_gradient()
            # Update contrast map.
            self._update_contrast_map(contrast_map_gradient)
            # Update affine.
            self._update_affine(affine_gradient)
            # Update velocity_fields.
            self._update_velocity_fields(velocity_fields_gradients)
            # Do other things.
            pass

            # TODO: save all below as attributes.
            return dict(
                affine=affine,
                phi=phi,
                phi_inv=phi_inv,
                affine_phi=affine_phi,
                phi_inv_affine_inv=phi_inv_affine_inv,
                contrast_coefficients=contrast_coefficients,

                # Helpers.
                template_resolution=template_resolution,
                target_resolution=target_resolution,

                # Accumulators.
                matching_energies=matching_energies,
                regularization_energies=matching_energies,
                total_energies=total_energies,
            )


    def _update_and_apply_position_field(self):
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
            phi_inv_affine_inv
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
        self.phi_inv_affine_inv = interpn(
            points=self.template_axes, 
            values=self.phi_inv - self.template_coords, 
            xi=affine_inv_target_coords, 
            bounds_error=False, 
            fill_value=None, 
        ) + affine_inv_target_coords

        # Apply phi_inv_affine_inv to template.
        # deformed_template is sampled at the coordinates of the target.
        self.deformed_template = interpn(
            points=self.template_axes, 
            values=self.template, 
            xi=self.phi_inv_affine_inv, 
            bounds_error=False, 
            fill_value=None, 
        )

            # DEBUG future tests pytest
            # print('max of phi_inv - template_coords:',np.max(self.phi_inv - self.template_coords))
            # print('max dif of phi_inv_affine_inv:',np.max(self.phi_inv_affine_inv - self.target_coords))


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
        # TODO: rename.
        
        self.artifact_mean_value = np.mean(self.target * (1 - self.matching_weights)) / np.mean(1 - self.matching_weights)
        
        likelihood_matching = np.exp((self.contrast_deformed_template - self.target)**2 * (-1/(2 * self.sigmaM**2))) / np.sqrt(2 * np.pi * self.sigmaM**2)
        likelihood_artifact = np.exp((self.artifact_mean_value        - self.target)**2 * (-1/(2 * self.sigmaA**2))) / np.sqrt(2 * np.pi * self.sigmaA**2)

        self.matching_weights = likelihood_matching / (likelihood_matching + likelihood_artifact)


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


    def _compute_contrast_map_gradient(self):
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
        optimal_contrast_coefficients = solve(B_transpose_B, self.B.T * target_ravel, assume_a='pos')

        # We do this so we can use the optimal gradient descent.
        # The format is used so that we could substitute a different approaach to computing the gradient.
        contrast_coefficients_gradient = optimal_contrast_coefficients - self.contrast_coefficients

        return contrast_coefficients_gradient


    def _update_contrast_map(contrast_map_gradient):

        self.contrast_coefficients += contrast_map_gradient * self.contrast_stepsize


    def _compute_affine_gradient(self):
        
        # matching_error_prime = (self.deformed_template - self.target) * self.matching_weights

        # # Energy gradient with respect to affine transform.
        # deformed_template_affine_gradient = np.gradient(self.deformed_template, self.target_resolution)

        # # TODO: wat?
        # deformed_template_affine_gradient_? = np.concatenate(deformed_template_affine_gradient, np.zeros_like(self.template))
        pass
        


    def _update_affine(affine_gradient):
        pass


    def _compute_velocity_fields_gradient(self):
        # TODO: verify contrast_deformed_template = fAphiI
        matching_error_prime = (self.contrast_deformed_template - self.target) * self.matching_weights
        contrast_map_prime = np.zeros_like(self.target)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += power * self.deformed_template**(power - 1) * self.contrast_coefficients[power]
        d_matching_d_deformed_template = matching_error_prime * contrast_map_prime

        # Set phi to identity. phi is secretly phi_1t_inv but at the end of the loop 
        # it will be phi_10_inv = phi_01 = phi.
        phi = np.copy(self.template_coords)

        # Loop backwards across time.
        d_matching_d_velocities = []
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
            # This transforms error in the target space back to time t.
            affine_phi = _multiply_by_affine(phi, self.affine)

            # Compute the determinant of the gradient of phi.
            grad_phi = np.gradient(phi, self.template_resolution)
            det_grad_phi = (
                grad_phi[0,0]*(grad_phi[1,1]*grad_phi[2,2] - grad_phi[1,2]*grad_phi[2,1]) -
                grad_phi[0,1]*(grad_phi[1,0]*grad_phi[2,2] - grad_phi[1,2]*grad_phi[2,0]) +
                grad_phi[0,2]*(grad_phi[1,0]*grad_phi[2,1] - grad_phi[1,1]*grad_phi[2,0])
            )

            # Transform error in target space back to time t.
            error_at_t = interpn(
                points=self.target_axes,
                values=d_matching_d_deformed_template,
                xi=affine_phi,
                bounds_error=False,
                fill_vallue=None,
            )

            # The gradient of the template image deformed to time t.
            deformed_template_to_t_gradient = np.gradient(self.deformed_template_to_t[timestep], np.template_resolution)

            # The derivative of the matching cost with respect to the velocity at time t
            # is the product of 
            # (the error deformed to time t), 
            # (the template gradient deformed to time t), 
            # & (the determinant of the jacobian of the transformation).
            d_matching_d_velocity_at_t = (error_at_t * det_grad_phi) * deformed_template_to_t_gradient * (-1.0/self.sigmaM**2) * det(self.affine)
            # To convert from derivative to gradient we smooth by applying a low-pass filter in the frequency domain.
            matching_cost_at_t_gradient = np.fft.fftn(d_matching_d_velocity_at_t, (0,1,2)) * self.low_pass_filter # TODO: define
            # Add the gradient of the regularization term.
            # TODO: grab from compute_cost.
            matching_cost_at_t_gradient += np.fft.fftn(self.velocity_fields[...,timestep,:], (0,1,2)) / self.sigmaR**2
            # TODO: save at each timestep.
            # Invert fourier transform back to the spatial domain.
            d_matching_d_velocity_at_t = np.fft.ifftn(matching_cost_at_t_gradient, (0,1,2)).real

            d_matching_d_velocities.insert(0, d_matching_d_velocity_at_t)

        return d_matching_d_velocities

    
    def _update_velocity_fields(velocity_fields_gradients):

            for timestep in range(self.num_timesteps):
                self.velocity_fields[...,timestep,:] -= velocity_fields_gradients[timestep] * self.deformative_stepsize

    # End _Lddmm.

'''
  _    _                          __                          _     _                       
 | |  | |                        / _|                        | |   (_)                      
 | |  | |  ___    ___   _ __    | |_   _   _   _ __     ___  | |_   _    ___    _ __    ___ 
 | |  | | / __|  / _ \ | '__|   |  _| | | | | | '_ \   / __| | __| | |  / _ \  | '_ \  / __|
 | |__| | \__ \ |  __/ | |      | |   | |_| | | | | | | (__  | |_  | | | (_) | | | | | \__ \
  \____/  |___/  \___| |_|      |_|    \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
                                                                                            
'''

def register(template, target, template_resolution=1, target_resolution=1, 
    translational_stepsize=0, 
    linear_stepsize=0, 
    deformative_stepsize=0, 
    sigmaR=0, 
    num_iterations=200, 
    num_affine_only_iterations=50, 
    initial_affine=np.eye(4), initial_velocity_fields=None, 
    num_timesteps=5, contrast_order=3, sigmaM=None, smooth_length=None):
    """
    Compute a registration between template and target, to be applied with apply_transform.
    
    Args:
        template (np.ndarray): The ideally clean template image being registered to the target.
        target (np.ndarray): The potentially messier target image being registered to.
        template_resolution (float, list, optional): A scalar or list of scalars indicating the resolution of the template. Defaults to 1.
        target_resolution (float, optional): A scalar or list of scalars indicating the resolution of the target. Defaults to 1.
        translational_stepsize (float, optional): The stepsize for translational adjustments. Defaults to 0.
        linear_stepsize (float, optional): The stepsize for linear adjustments. Defaults to 0.
        deformative_stepsize (float, optional): The stepsize for deformative adjustments. Defaults to 0.
        sigmaR (float, optional): A scalar indicating the freedom to deform. Defaults to 0.
        num_iterations (int, optional): The total number of iterations. Defaults to 200.
        num_affine_only_iterations (int, optional): The number of iterations at the start of the process without deformative adjustments. Defaults to 50.
        initial_affine (np.ndarray, optional): The affine array that the registration will begin with. Defaults to np.eye(4).
        initial_velocity_fields (np.ndarray, optional): The velocity fields that the registration will begin with. Defaults to None.
        num_timesteps (int, optional): The number of composed sub-transformations in the diffeomorphism. Defaults to 5.
        contrast_order (int, optional): The order of the polynomial fit between the contrasts of the template and target. Defaults to 3.
        sigmaM (float, optional): A measure of spread. Defaults to None.
        smooth_length (float, optional): The length scale of smoothing. Defaults to None.
    
    Returns:
        dict: A dictionary containing all important saved quantities computed during the registration.
    """

    # Set up Lddmm instance.
    lddmm = _Lddmm(
        template=template,
        target=target,
        template_resolution=template_resolution,
        target_resolution=target_resolution,
        num_timesteps=num_timesteps,
        initial_affine=initial_affine,
        initial_velocity_fields=initial_velocity_fields,
        contrast_order=contrast_order,
        sigmaM=sigmaM,
        smooth_length=smooth_length,
        sigmaR=sigmaR,
        translational_stepsize=translational_stepsize,
        linear_stepsize=linear_stepsize,
        deformative_stepsize=deformative_stepsize,
    )

    return lddmm.register()


def _generate_position_field(affine, velocity_fields, velocity_field_resolution, 
template_shape, template_resolution, target_shape, target_resolution, deform_to="template"):

    # Validate inputs.

    # Validate template_shape. Not rigorous.
    template_shape = _validate_ndarray(template_shape)
    # Validate target_shape. Not rigorous.
    target_shape = _validate_ndarray(target_shape)
    # Validate velocity_fields.
    velocity_fields = _validate_ndarray(velocity_fields, required_ndim=len(template_shape) + 2)
    if not np.all(velocity_fields.shape[:-2] == template_shape):
        raise ValueError(f"velocity_fields' initial dimensions must equal template_shape.\n"
            f"velocity_fields.shape: {velocity_fields.shape}, template_shape: {template_shape}.")
    # Validate velocity_field_resolution.
    velocity_field_resolution = _validate_xyz_resolution(velocity_fields.ndim - 2, velocity_field_resolution)
    # Validate affine.
    affine = _validate_ndarray(affine, required_ndim=2, reshape_to_shape=(4, 4))
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")

    # Compute intermediates.
    affine_inv = inv(affine)
    num_timesteps = velocity_fields.shape[-2]
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
    position_field_resolution = _validate_xyz_resolution(position_field.ndim - 1, position_field_resolution)
    # Validate subject.
    subject = _validate_ndarray(subject, required_ndim=position_field.ndim - 1)
    # Validate subject_resolution.
    subject_resolution = _validate_xyz_resolution(subject.ndim, subject_resolution)
    # Validate output_resolution.
    output_resolution = _validate_xyz_resolution(subject.ndim, output_resolution)

    # Resample position_field.
    position_field = change_resolution_to(position_field, [*position_field_resolution, 1], [*output_resolution, 1])

    # To make this fully general, accept an output_shape and adjust the position_field to match that shape by interpolating on a grid with arbitrary physical extent.
    # TODO: ^

    # Interpolate subject at position field.
    deformed_subject = interpn(
        points=_compute_axes(shape=subject.shape, xyz_resolution=subject_resolution),
        values=subject,
        xi=position_field,
        bounds_error=False,
        fill_value=None,
    )

    return deformed_subject


# TODO: confirm we don't want to make subject_resolution default to 1 as a kwarg.
def apply_transform(subject, subject_resolution, affine_phi, phi_inv_affine_inv, 
template_resolution, target_resolution, output_resolution=None, deform_to="template", **unused_kwargs):
    """
    Apply the transform, or position field affine_phi or phi_inv_affine_inv, to the subject 
    to deform it to either the template or the target.

    The user is expected to provide subject, and optionally subject_resolution, deform_to, and output_resolution.
    It is expected that the rest of the arguments will be provided by keyword argument from the output of the register function.

    Example use:
        register_output_dict = register(*args, **kwargs)
        deformed_subject = apply_transform(subject, subject_resolution, **register_output_dict)

    Args:
        subject (np.ndarray): The image to be deformed to the template or target from the results of the register function.
        subject_resolution (float, seq): The resolution of subject in each dimension, or just one scalar to indicate isotropy.
        affine_phi (np.ndarray): The position field in the shape of the template for deforming to the template.
        phi_inv_affine_inv ([np.ndarray): The position field in the shape of the target for deforming to the target.
        template_resolution (float, seq): The resolution of the template in each dimension, or just one scalar to indicate isotropy.
        target_resolution ([float, seq): The resolution of the target in each dimension, or just one scalar to indicate isotropy.
        output_resolution (NoneType, float, seq, optional): The resolution of the output deformed_subject in each dimension, 
            or just one scalar to indicate isotropy, or None to indicate the resolution of template or target based on deform_to. 
            Defaults to None.
        deform_to (str, optional): Either "template" or "target", indicating which position field to apply to subject. Defaults to "template".

    Raises:
        TypeError: Raised if deform_to is not of type str.
        ValueError: Raised if deform_to is a string other than "template" or "target".

    Returns:
        np.ndarray: The result of applying the appropriate position field to subject, deforming it based on deform_to.
    """

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
        position_field_resolution = np.copy(target_resolution)

    # Call _apply_position_field.

    deformed_subject = _apply_position_field(subject, subject_resolution, output_resolution, position_field, position_field_resolution)

    return deformed_subject
