preset_parameters = {}

# Define registration parameter presets.
preset_parameters.update({'identity'          : dict(eT=0,    eL=0,     eV=0,    sigmaR=0, naffine=0, niter=1)})
preset_parameters.update({'clarity, mouse'    : dict(eT=1e-7, eL=1e-10, eV=2e0,  sigmaR=2e1)})
preset_parameters.update({'nissl, mouse'      : dict(eT=2e-9, eL=1e-13, eV=5e-4, sigmaR=1e0)})
preset_parameters.update({'mri, human'        : dict(eT=1e-9, eL=5e-13, eV=5e-4, sigmaR=1e0)})
# preset_parameters.update({'clarity' : dict(sigmaR=1e1, eV=5e-1, eL=2e-8, eT=2e-5)}) # TODO: remove deprecated 'clarity' preset.


def get_registration_presets():
    """
    Get the names of all registration presets.
    
    Returns:
        dict_keys: The keys of the dictionary mapping all registration preset names to the corresponding registration parameters.
    """

    return preset_parameters.keys()


def get_registration_preset(preset:str) -> dict:
    """
    If <preset> is recognized, returns a dictionary containing the registration parameters corresponding to <preset>.
    
    Args:
        preset (str): The name of a preset keyed to a particular dictionary of registration parameters.
    
    Raises:
        NotImplementedError: Raised if <preset> is not a recognized preset name.
    
    Returns:
        dict: The registration kwargs specified by <preset>.
    """

    preset = preset.strip().lower()

    if preset in preset_parameters:
        return dict(preset_parameters[preset]) # Recast as dict to provide a freely mutable copy.
    else:
        raise NotImplementedError(f"There is no preset for '{preset}'.\n"
            f"Recognized presets include:\n{list(preset_parameters.keys())}.")
