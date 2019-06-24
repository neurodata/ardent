# Transform is the primary class in ARDENT.

# import statements.
import numpy as np
from .presets import get_registration_presets
from .lddmm.transformer import torch_register
from .lddmm.transformer import torch_apply_transform
from .io import save as io_save
# TODO: rename io as fileio.
from pathlib import Path

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

        self.transformer = None
    
    @staticmethod
    def _handle_registration_parameters(preset:str, params:dict) -> dict:
        """Provides default parameters based on <preset>, superseded by the provided parameters params.
        Returns a dictionary with the resultant parameters."""

        preset_parameters = get_registration_presets(preset) # Type: dict.

        final_parameters = preset_parameters.update(params)

        return final_parameters


    def register(self, template:np.ndarray, target:np.ndarray, 
        preset=None, sigmaR=None, eV=None, eL=None, eT=None, **kwargs) -> None:
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
            registration_parameters = _handle_registration_parameters(preset, registration_parameters)

        outdict = torch_register(template, target, **registration_parameters)
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

        deformed_subject = torch_apply_transform(image=subject, deform_to=deform_to, Aphis=self.Aphis, phiinvAinvs=self.phiinvAinvs, transformer=self.transformer)
        
        if save_path is not None:
            io_save(deformed_subject, save_path)

        return deformed_subject

    
    def save(self, file_path):
        """Saves the following attributes to file_path: phis, phiinvs, Aphis, phiinvAinvs, & A.
        The file is saved in .npz format."""

        attribute_dict = {
            'phis':self.phis,
            'phiinvs':self.phiinvs,
            'Aphis':self.Aphis,
            'phiinvAinvs':self.phiinvAinvs,
            'affine':self.affine
            }
        
        io_save(attribute_dict, file_path)

    def load(self, file_path):
        """Loads the following attributes from file_path: phis, phiinvs, Aphis, phiinvAinvs, & A.
        Presently they can only be accessed. This is not sufficient to run apply_transform 
        without first running register."""

        # Validate file_path.
        file_path = Path(file_path)
        if not file_path.suffix:
            file_path = file_path.with_suffix('.npz')
        elif file_path.suffix != '.npz':
            raise ValueError(f"file_path may not have an extension other than .npz.\n"
                f"file_path.suffix: {file_path.suffix}.")
        # file_path is appropriate.

        with np.load(file_path) as attribute_dict:
            self.phis = attribute_dict['phis']
            self.phiinvs = attribute_dict['phiinvs']
            self.Aphis = attribute_dict['Aphis']
            self.phiinvAinvs = attribute_dict['phiinvAinvs']
            self.affine = attribute_dict['affine']
        