import SimpleITK as sitk
from pathlib import Path

def _validate_inputs(**kwargs):
    """Accepts arbitrary kwargs. If recognized, they are validated.
    If they cannot be validated an exception is raised.
    A dictionary containing the validated kwargs is returned."""

    # Validate data.

    if 'data' in kwargs.keys():
        data = kwargs['data']
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a np.ndarray.\ntype(data): {type(data)}.")
        # data is a np.ndarray.
        kwargs.update(data=data)

    # Validate file_path.

    if 'file_path' in kwargs.keys():
        file_path = kwargs['file_path']
        file_path = Path(file_path).resolve() # Will raise exceptions if file_path is not an oppropriate type.
        if not file_path.parent.is_dir():
            # Create directory if it does not exist, including all necessary parent directories.
            file_path.parent.mkdir(parents=True, exist_ok=True)
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
    locals().update(validated_inputs)
    # TODO: figure out why the above line didn't work.
    # Hotfix because the above line didn't work.
    file_path = validated_inputs['file_path']

    # Convert data to sitk.Image.
    data_Image = sitk.GetImageFromArray(data) # Side effect: breaks alias.

    # Save data to file_path.
    sitk.WriteImage(data_Image, str(file_path))


def load(file_path):
    """Load data from file_path."""

    # Validate inputs.
    inputs = {'file_path':file_path}
    validated_inputs = _validate_inputs(**inputs)
    locals().update(validated_inputs)
    # TODO: figure out why the above line didn't work.
    # Hotfix because the above line didn't work.
    file_path = validated_inputs['file_path']

    # Read in data as sitk.Image.
    data_Image = sitk.ReadImage(str(file_path))

    # Convert data_Image to np.ndarray.
    data = sitk.GetArrayFromImage(data_Image)

    return data
    
    

    
'''
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

atlas_saved_path = '/home/dcrowley/ARDENT_gpu_test/savetestdir/atlas'
# locals().update(_validate_inputs(file_path=atlas_saved_path))

save(atlas, atlas_saved_path)

x = load(atlas_saved_path)

print(type(x))
print(x.shape == atlas.shape)
print(np.all(x == atlas))
'''
