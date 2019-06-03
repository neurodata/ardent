# Transform is the primary class in ARDENT.

# import statements.
import numpy as np
from .lddmm import torch_lddmm_wrapper

class Transform():
    """transform stores the deformation that is output by a registration 
    and provides methods for applying that transformation to various images."""
    
    def __init__(self):
        """Initialize Transform object. If used without arguments, sets attributes 
        to None. 
        TODO: Add option to register on initialization?"""

        # Create attributes.
        self.Aphis = None
        self.phiinvAinvs = None
        self.affine = None

        self.lddmm = None
    

    def register(self, template:np.ndarray, target:np.ndarray, sigmaR, eV, eL=0, eT=0, **kwargs) -> None:
        """Perform a registration using LDDMM between template and target. 
        Populates attributes for future calls to the apply_transform method."""

        outdict = torch_lddmm_wrapper.torch_register(template, target, sigmaR, eV, eL=eL, eT=eT, **kwargs)
        '''outdict contains:
            - Aphis
            - phis
            - phiinvs
            - phiinvAinvs
            - A

            - lddmm
        '''

        # Populate attributes.
        self.Aphis = outdict['Aphis']
        self.phiinvAinvs = outdict['phiinvAinvs']
        self.affine = outdict['A']

        self.lddmm = outdict['lddmm']


    def apply_transform(self, subject:np.ndarray, deform_to="template", save_path=None) -> np.ndarray:
        """Apply the transformation--computed by the last call to self.register--
        to <subject>, deforming it into the space of <deform_to>."""

        deformed_subject = torch_lddmm_wrapper.torch_apply_transform(image=subject, deform_to=deform_to, Aphis=self.Aphis, phiinvAinvs=self.phiinvAinvs, lddmm=self.lddmm)
        
        # TODO:
        # Perform I/O as needed.


        return deformed_subject
    
    