preset_parameters = {}

# identity preset.
preset_parameters.update({'identity' : dict(sigmaR=0, eV=0, eL=0, eT=0, naffine=0, niter=1)})

# clarity preset.
preset_parameters.update({'clarity' : dict(sigmaR=1e1, eV=5e-1, eL=2e-8, eT=2e-5)})

def get_registration_presets(preset:str) -> dict:
    """If <preset> is recognized, returns a dictionary containing Transform.register kwargs."""

    preset = preset.strip().lower()

    if preset in preset_parameters:
        return preset_parameters[preset]
    else:
        raise NotImplementedError(f"There is no preset for '{preset}'.\n"
            f"Recognized presets include:\n{list(preset_parameters.keys())}.")
