# Transform is the primary class in ARDENT.

import numpy as np
from pathlib import Path
import pickle

from .presets import get_registration_preset
from .lddmm._lddmm import lddmm_register, apply_lddmm
# TODO: rename io as fileio to avoid conflict with standard library package io?
from . import io


class Transform:
    """
    Transform stores the deformation that is output by a registration 
    and provides methods for applying that transformation to various images.
    """
    
    def __init__(self):
        """
        Initialize Transform object. Sets attributes to None.
        """

        # Create attributes.

        # Saved for the continue_registration method.
        self._registration_parameters = None

        # lddmm_dict.

        # Core.
        self.affine=None,
        self.phi=None,
        self.phi_inv=None,
        self.affine_inv=None,
        self.phi_inv_affine_inv=None,
        self.contrast_coefficients=None,

        # Helpers.
        self.template_resolution=None,
        self.target_resolution=None,
        
        # Accumulators.
        self.matching_energies=None,
        self.regularization_energies=None,
        self.total_energies=None,

        # Debuggers.
        self.lddmm=None,
        
    def _update_lddmm_attributes(
        self,
        # Core.
        affine,
        phi,
        phi_inv,
        affine_inv,
        phi_inv_affine_inv,
        contrast_coefficients,
        # Helpers.
        template_resolution,
        target_resolution,
        # Accumulators.
        matching_energies,
        regularization_energies,
        total_energies,
        # Debuggers.
        lddmm,
    ):
        """Update attributes with the output dictionary from lddmm_register."""

        # Set attributes.

        # Core.
        self.affine=affine
        self.phi=phi
        self.phi_inv=phi_inv
        self.affine_phi=affine_phi
        self.phi_inv_affine_inv=phi_inv_affine_inv
        self.contrast_coefficients=contrast_coefficients

        # Helpers.
        self.template_resolution=template_resolution
        self.target_resolution=target_resolution

        # Accumulators.
        self.matching_energies=matching_energies
        self.regularization_energies=matching_energies
        self.total_energies=total_energies

        # Debuggers.
        self.lddmm=lddmm


    def get_lddmm_dict(self):
        """
        Constructs lddmm_dict, a dictionary of this object's attributes as populated by the return from lddmm_register.
        
        Returns:
            dict: The attributes of this Transform object, matching the dictionary returned from lddmm_register.
        """

        return dict(
            # Core.
            affine=self.affine,
            phi=self.phi,
            phi_inv=self.phi_inv,
            affine_inv=self.affine_inv,
            phi_inv_affine_inv=self.phi_inv_affine_inv,
            contrast_coefficients=self.contrast_coefficients,
            # Helpers.
            template_resolution=self.template_resolution,
            target_resolution=self.target_resolution,
            # Accumulators.
            matching_energies=self.matching_energies,
            regularization_energies=self.regularization_energies,
            total_energies=self.total_energies,
            # Debuggers.
            lddmm=self.lddmm,
        )


    def register(
        self,
        template,
        target,
        template_resolution=1,
        target_resolution=1,
        preset=None,
        # Registration parameters.
        translational_stepsize=0,
        linear_stepsize=0,
        deformative_stepsize=0,
        sigma_regularization=0,
        num_iterations=200,
        num_affine_only_iterations=50,
        initial_affine=None,
        initial_velocity_fields=None,
        initial_contrast_coefficients=None,
        num_timesteps=5,
        smooth_length=None,
        contrast_order=1,
        contrast_tolerance=1e-5,
        contrast_maxiter=100,
        sigma_contrast=1e-2,
        sigma_matching=None,
        spatially_varying_contrast_map=False,
        calibrate=False,
        track_progress_every_n=0,
    ):
        """
        Compute a registration between template and target, to be applied with apply_lddmm.
        
        Args:
            template (np.ndarray): The ideally clean template image being registered to the target.
            target (np.ndarray): The potentially messier target image being registered to.
            template_resolution (float, list, optional): A scalar or list of scalars indicating the resolution of the template. Defaults to 1.
            target_resolution (float, optional): A scalar or list of scalars indicating the resolution of the target. Defaults to 1.
            preset (string, Nontranslational_stepsizeype, optional): Preset of registration parameters. Overrides some subset of the registration parameters with preset values if provided.
                Supported options:
                    'identity'
                    'clarity, mouse'
                    'nissl, mouse'
                    'mri, human'
                Defaults to: None.
            translational_stepsize (float, optional): The stepsize for translational adjustments. Defaults to 0.
            linear_stepsize (float, optional): The stepsize for linear adjustments. Defaults to 0.
            deformative_stepsize (float, optional): The stepsize for deformative adjustments. Defaults to 0.
            sigma_regularization (float, optional): A scalar indicating the freedom to deform. Defaults to 0.
            num_iterations (int, optional): The total number of iterations. Defaults to 200.
            num_affine_only_iterations (int, optional): The number of iterations at the start of the process without deformative adjustments. Defaults to 50.
            initial_affine (np.ndarray, optional): The affine array that the registration will begin with. Defaults to np.eye(template.ndim + 1).
            initial_velocity_fields (np.ndarray, optional): The velocity fields that the registration will begin with. Defaults to None.
            initial_contrast_coefficients (np.ndarray, optional): The contrast coefficients that the registration will begin with. 
                If None, the 0th order coefficient(s) are set to np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template), 
                if self.contrast_order > 1, the 1st order coefficient(s) are set to np.std(self.target) / np.std(self.template), 
                and all others are set to zero. Defaults to None.
            num_timesteps (int, optional): The number of composed sub-transformations in the diffeomorphism. Defaults to 5.
            smooth_length (float, optional): The length scale of smoothing. Defaults to None.
            contrast_order (int, optional): The order of the polynomial fit between the contrasts of the template and target. Defaults to 3.
            contrast_tolerance (float, optional): The tolerance for convergence to the optimal contrast_coefficients if spatially_varying_contrast_map == True. Defaults to 1e-5.
            contrast_maxiter (int, optional): The maximum number of iterations to converge toward the optimal contrast_coefficients if spatially_varying_contrast_map == True. Defaults to 100.
            sigma_contrast (float, optional): The scale of variation in the contrast_coefficients if spatially_varying_contrast_map == True. Defaults to 1e-2.
            sigma_matching (float, optional): A measure of spread. Defaults to None.
            spatially_varying_contrast_map (bool, optional): If True, uses a polynomial per voxel to compute the contrast map rather than a single polynomial. Defaults to False.
            calibrate (bool, optional): A boolean flag indicating whether to accumulate additional intermediate values and display informative plots for calibration purposes. Defaults to False.
            track_progress_every_n (int, optional): If positive, a progress update will be printed every track_progress_every_n iterations of registration. Defaults to 0.
        """

        # Collect registration parameters.
        registration_parameters = dict(
            template=template,
            target=target,
            template_resolution=template_resolution,
            target_resolution=target_resolution,
            translational_stepsize=translational_stepsize,
            linear_stepsize=linear_stepsize,
            deformative_stepsize=deformative_stepsize,
            sigma_regularization=sigma_regularization,
            num_iterations=num_iterations,
            num_affine_only_iterations=num_affine_only_iterations,
            initial_affine=initial_affine,
            initial_velocity_fields=initial_velocity_fields,
            initial_contrast_coefficients=initial_contrast_coefficients,
            num_timesteps=num_timesteps,
            smooth_length=smooth_length,
            contrast_order=contrast_order,
            contrast_tolerance=contrast_tolerance,
            contrast_maxiter=contrast_maxiter,
            sigma_contrast=sigma_contrast,
            sigma_matching=sigma_matching,
            spatially_varying_contrast_map=spatially_varying_contrast_map,
            calibrate=calibrate,
            track_progress_every_n=track_progress_every_n,
        )

        # Fill unspecified parameters with presets if applicable.
        if preset is not None:
            registration_parameters.update(get_registration_preset(preset))

        # Perform registration.
        lddmm_dict = lddmm_register(**registration_parameters)

        # Save registration parameters for the continue_registration method, with the initial_affine, initial_velocity_fields, and initial_contrast_coefficients updated.
        registration_parameters.update(
            initial_affine=lddmm_dict['affine'],
            initial_velocity_fields=lddmm_dict['velocity_fields'],
            initial_contrast_coefficients=lddmm_dict['contrast_coefficients']
        )
        self._registration_parameters = registration_parameters

        # Update attributes.
        self._update_lddmm_attributes(**lddmm_dict)

    
    def continue_registration(self, **registration_parameter_updates):
        """
        Continue registering with all the same registration parameters from the previous call to the register method, 
        but with initial_affine, initial_velocity_fields, and initial_contrast_coefficients set to the affine, velocity_fields, and contrast_coefficients 
        most recently calculated in the register method, updated by registration_parameter_updates.

        Kwargs:
            registration_parameter_updates (key-value pairs, optional): registration parameters provided as kwargs 
                to overwrite the most recent registration_parameters used in the register method. Defaults to {}.
        
        Raises:
            RuntimeError: Raised if self._registration_parameters are not set (as they should be after a call to the register method).
        """

        if self._registration_parameters is None:
            raise RuntimeError(f"The continue_registration method cannot be called from an object that has not performed the register method first.")

        # Update most recently saved registration parameters with user-provided kwargs.
        self._registration_parameters.update(registration_parameter_updates)

        # Continue the registration.
        self.register(**self._registration_parameters)


    def apply_transform(self, subject, subject_resolution=1, output_resolution=None, deform_to="template", extrapolation_fill_value=None, save_path=None):
        """
        Apply the transformation--computed by the last call to self.register--to subject, 
        deforming it into the space of deform_to.
        
        Args:
            subject (np.ndarray): The image to deform.
            subject_resolution (float, seq, optional): The resolution of subject in each dimension, or just one scalar to indicate isotropy. Defaults to 1.
            deform_to (str, optional): Either 'template' or 'target' indicating which to deform subject to match. Defaults to: "template".
            output_resolution (NoneType, float, seq, optional): The resolution of the output deformed_subject in each dimension, 
                or just one scalar to indicate isotropy, or None to indicate the resolution of template or target based on deform_to. Defaults to None.
            extrapolation_fill_value (float, NoneType, optional): The fill_value kwarg passed to scipy.interpolate.interpn; it should be background intensity. 
            If None, this is set to a low quantile of the subject's 10**-subject.ndim quantile to estimate background. Defaults to None.
            save_path (str, Path, optional): The full path to save the output to. Defaults to: None.
        
        Returns:
            np.ndarray: The result of deforming subject to match deform_to.
        """

        deformed_subject = apply_lddmm(
            subject=subject,
            subject_resolution=subject_resolution,
            output_resolution=output_resolution,
            deform_to=deform_to,
            extrapolation_fill_value=extrapolation_fill_value,
            **self.get_lddmm_dict(),
        )
        
        if save_path is not None:
            io.save(deformed_subject, save_path)

        return deformed_subject

    
    def save(self, file_path):
        """
        Save the entire instance of this Transform object (self) to file.
        
        Args:
            file_path (str, Path): The full path to save self to.
        """

        io.save_pickled(self, file_path)


    def load(self, file_path):
        """
        Load an entire instance of a Transform object from memory, as from a file created with the save method, 
        and transplants all of its writeable attributes into self.
        
        Args:
            file_path (str, Path): The full path that a Transform object was saved to.
        """

        transform = io.load_pickled(file_path)

        self.__dict__.update(transform.__dict__)