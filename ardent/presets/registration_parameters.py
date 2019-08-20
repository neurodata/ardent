preset_parameters = {}

# identity preset.
preset_parameters.update({'identity' : dict(sigmaR=0, eV=0, eL=0, eT=0, naffine=0, niter=1)})

# clarity preset.
preset_parameters.update({'clarity' : dict(sigmaR=1e1, eV=5e-1, eL=2e-8, eT=2e-5)})


def get_registration_presets():
    """
    Get the names of all registration presets.
    
    Returns:
        dict_keys -- The keys of the dictionary mapping all registration preset names to the corresponding registration parameters.
    """

    return preset_parameters.keys()


def get_registration_preset(preset:str) -> dict:
    """
    If <preset> is recognized, returns a dictionary containing the registration parameters corresponding to <preset>.
    
    Arguments:
        preset {str} -- The name of a preset keyed to a particular dictionary of registration parameters.
    
    Raises:
        NotImplementedError: Raised if <preset> is not a recognized preset name.
    
    Returns:
        dict -- The registration kwargs specified by <preset>.
    """

    preset = preset.strip().lower()

    if preset in preset_parameters:
        return preset_parameters[preset]
    else:
        raise NotImplementedError(f"There is no preset for '{preset}'.\n"
            f"Recognized presets include:\n{list(preset_parameters.keys())}.")
