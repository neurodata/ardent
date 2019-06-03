import SimpleITK as sitk
from pathlib import Path

def _validate_inputs(**kwargs):
    """Accepts arbitrary kwargs. If recognized, they are validated.
    If they cannot be validated an exception is raised.
    A dictionary containing the validated kwargs is returned."""

    # Validate data.

    if 'data' in kwargs.keys():
        data = kwargs['data']
        if isinstance(data, list):
            # Verify that each element of the list is a np.ndarray object and recursion will stop at 1 level.
            if not all(map(lambda datum: isinstance(datum, np.ndarray), data)):
                raise ValueError(f"data must be either a np.ndarray or a list containing only np.ndarray objects.")
            # Recurse into each element of the list of np.ndarray objects.
            for index, datum in enumerate(data):
                data[index] = _validate_inputs(data=datum)['data']
        # If data is neither a list nor a np.ndarray, raise TypeError.
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a np.ndarray.\ntype(data): {type(data)}.")
        # data is a np.ndarray.
        kwargs.update(data=data)

    # Validate file_path.

    if 'file_path' in kwargs.keys():
        file_path = kwargs['file_path']
        file_path = Path(file_path).resolve() # Will raise exceptions if file_path is not an oppropriate type.
        if not file_path.parent.is_dir():
            raise FileNotFoundError(f"file_path corresponds to a location that does not presently exist.\n"
                f"file_path.parent: {file_path.parent}.")
        # Validate extension.
        if not file_path.suffix:
            default_extension = '.vtk'
            file_path = file_path.with_suffix(default_extension)
        # file_path is valid.
        kwargs.update(file_path=file_path)

    return kwargs


def save(data, file_path):
    """Save data to file_path."""
    # Validate inputs.
    inputs = {'data':data, 'file_path':file_path}
    validated_inputs = _validate_inputs(**inputs)
    data = validated_inputs['data']
    file_path = validated_inputs['file_path']

    # data is either a single np.ndarray or a list of np.ndarrays.

    if isinstance(data, np.ndarray):
        # Convert data to sitk.Image.
        data_Image = sitk.GetImageFromArray(data) # Side effect: breaks alias.
        # Save data to file_path.
        sitk.WriteImage(data_Image, str(file_path))
    elif isinstance(data, dict):
        np.savez(data, file_path.with_suffix('')) # '.npz' is appended.
    else:
        # _validate_inputs has failed.
        raise Exception(f"_validate_inputs has failed to prevent an improper type for data.\n"
            f"type(data): {type(data)}.")


def load(file_path):
    """Load data from file_path."""

    # Validate inputs.
    inputs = {'file_path':file_path}
    validated_inputs = _validate_inputs(**inputs)
    file_path = validated_inputs['file_path']

    if file_path.suffix == '.npz':
        data = np.load(file_path)
        # data is a dictionary.
        return data
    else:
        # Read in data as sitk.Image.
        data_Image = sitk.ReadImage(str(file_path))
        # Convert data_Image to np.ndarray.
        data = sitk.GetArrayFromImage(data_Image)
        # data is a np.ndarray.
        return data
    
    

    

# Raw unadulterated testing ground:

import numpy as np
import ardent
import matplotlib
# %matplotlib inline
# TODO: remove with ardent.io.
from pathlib import Path
import nibabel as nib 

directory_path = Path('/home/dcrowley/image_lddmm_tensorflow')
atlas_image_filename = 'average_template_50.img'
target_image_filename = '180517_Downsample.img'

atlasPath = directory_path / atlas_image_filename
targetPath = directory_path / target_image_filename

atlas = np.array(nib.load(str(atlasPath)).get_data()).astype(float).squeeze()
target = np.array(nib.load(str(targetPath)).get_data()).astype(float).squeeze()

atlas_saved_path = '/home/dcrowley/ARDENT_gpu_test/savetestdir/atlastarget'
# locals().update(_validate_inputs(file_path=atlas_saved_path))
# atlas_saved_path = _validate_inputs(file_path=atlas_saved_path)['file_path']


# save([atlas, target], atlas_saved_path)

# x = load(atlas_saved_path)

# print(type(x))
# print(x.shape)
