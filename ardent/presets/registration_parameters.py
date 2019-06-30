preset_parameters = {}

# clarity preset.
preset_parameters.update({'clarity' : dict(sigmaR=1e4, eV=5e6, eL=5e-11, eT=5e-4)})

def get_registration_presets(preset:str) -> dict:
    """If <preset> is recognized, returns a dictionary containing Transform.register kwargs."""

    preset = preset.strip().lower()

    if preset in preset_parameters:
        return preset_parameters[preset]
    else:
        raise NotImplementedError(f"There is no preset for '{preset}'.\n"
            f"Recognized presets include:\n{list(preset_parameters.keys())}.")
