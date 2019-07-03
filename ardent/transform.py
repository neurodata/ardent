# Transform is the primary class in ARDENT.

# import statements.
import numpy as np
import torch
from .presets import get_registration_presets
from .lddmm.transformer import Transformer
from .lddmm.transformer import torch_register
from .lddmm.transformer import torch_apply_transform
# TODO: rename io as fileio to avoid conflict with standard library package io?
# from .io import save as io_save
from . import io
from pathlib import Path
import pickle

class Transform():
    """transform stores the deformation that is output by a registration 
    and provides methods for applying that transformation to various images."""
    
    def __init__(self):
        """Initialize Transform object. If used without arguments, sets attributes 
        to None. 
        TODO: Add option to register on initialization?"""

        # Create attributes.
        self.phis = None
        self.phiinvs = None
        self.Aphis = None
        self.phiinvAinvs = None
        self.affine = None

        self.transformer = None # To be instantiated in the register method.
    
    @staticmethod
    def _handle_registration_parameters(preset:str, params:dict) -> dict:
        """Provides default parameters based on <preset>, superseded by the provided parameters params.
        Returns a dictionary with the resultant parameters."""

        # Get default registration parameters based on <preset>.
        preset_parameters = get_registration_presets(preset) # Type: dict.

        # Supplement and supplant with <params> from the caller.
        preset_parameters.update(params)

        return preset_parameters

    # TODO: argument validation and resolution scalar to triple correction.
    def register(self, template:np.ndarray, target:np.ndarray, template_resolution=[1,1,1], target_resolution=[1,1,1],
        preset=None, sigmaR=None, eV=None, eL=None, eT=None, 
        A=None, v=None, **kwargs) -> None:
        """Perform a registration using transformer between template and target. 
        Populates attributes for future calls to the apply_transform method.
        
        If used, <preset> will provide default values for sigmaR, eV, eL, and eT, 
        superseded by any such values that are provided.

        <preset> options:
        'clarity'
        """


        # Collect registration parameters from chosen caller.
        registration_parameters = dict(sigmaR=sigmaR, eV=eV, eL=eL, eT=eT, **kwargs)
        registration_parameters = {key : value for key, value in registration_parameters.items() if value is not None}
        # Fill unspecified parameters with presets if applicable.
        if preset is not None:
            registration_parameters = Transform._handle_registration_parameters(preset, registration_parameters)

        # Instantiate transformer as a new Transformer object.
        # self.affine and self.v will not be None if this Transform object was read with its load method or if its register method was already called.
        transformer = Transformer(I=template, J=target, Ires=template_resolution, Jres=target_resolution, 
                                    transformer=self.transformer, A=A, v=v)

        outdict = torch_register(template, target, transformer, **registration_parameters)
        '''outdict contains:
            - phis
            - phiinvs
            - Aphis
            - phiinvAinvs
            - A

            - transformer
        '''

        # Populate attributes.
        self.phis = outdict['phis']
        self.phiinvs = outdict['phiinvs']
        self.Aphis = outdict['Aphis']
        self.phiinvAinvs = outdict['phiinvAinvs']
        self.affine = outdict['A']

        self.transformer = outdict['transformer']


    def apply_transform(self, subject:np.ndarray, deform_to="template", save_path=None) -> np.ndarray:
        """Apply the transformation--computed by the last call to self.register--
        to <subject>, deforming it into the space of <deform_to>."""

        deformed_subject = torch_apply_transform(image=subject, deform_to=deform_to, transformer=self.transformer)
        
        if save_path is not None:
            io.save(deformed_subject, save_path)

        return deformed_subject

    
    def save(self, file_path):
        """Saves the entire self object instance to file."""

        io.save_pickled(self, file_path)


    def load(self, file_path):
        """Loads an entire object instance from memory and transplants all of its writeable attributes into self."""

        transform = io.load_pickled(file_path)

        self.__dict__.update(transform.__dict__)