import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize, rescale


def correct_bias_field(image, correct_at_scale=1, as_float32=True, **kwargs):
    """
    Shifts image such that its minimum value is 1, computes the bias field after downsampling by correct_at_scale, 
    upsamples this bias field and applies it to the shifted image, then undoes the shift and returns the result.
    Computes bias field using sitk.N4BiasFieldCorrection (http://bit.ly/2oFwAun).
    
    Args:
        image (np.ndarray): The image to be bias corrected.
        correct_at_scale (float, optional): The scale by which the shape of image is reduced before computing the bias. Defaults to 4.
        as_float32 (bool, optional): If True, image is internally cast as a sitk.Image of type sitkFloat32. If False, it is of type sitkFloat64. Defaults to True.

    Kwargs:
        Any additional keyword arguments overwrite the default values passed to sitk.N4BiasFieldCorrection.
    
    Returns:
        np.ndarray: A copy of image after bias correction.
    """

    # Shift image such that its minimum value lies at 1.
    image_min = image.min()
    image = image - image_min + 1

    # Downsample image according to scale.
    downsampled_image = rescale(image, correct_at_scale)

    # Bias correct downsampled_image.
    N4BiasFieldCorrection_kwargs = dict(
        image=downsampled_image, 
        maskImage=np.ones_like(downsampled_image), 
        convergenceThreshold=0.001, 
        maximumNumberOfIterations=[50, 50, 50, 50], 
        biasFieldFullWidthAtHalfMaximum=0.15, 
        wienerFilterNoise=0.01, 
        numberOfHistogramBins=200,
        numberOfControlPoints=[4, 4, 4], 
        splineOrder=3, 
        useMaskLabel=True, 
        maskLabel=1, 
    )
    # Overwrite default arguments with user-supplied kwargs.
    N4BiasFieldCorrection_kwargs.update(kwargs)
    # Convert image and maskImage N4BiasFieldCorrection_kwargs from type np.ndarray to type sitk.Image.
    sitk_image = sitk.GetImageFromArray(N4BiasFieldCorrection_kwargs['image'])
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32 if as_float32 else sitk.sitkFloat64)
    sitk_maskImage = N4BiasFieldCorrection_kwargs['maskImage'].astype(np.uint8)
    sitk_maskImage = sitk.GetImageFromArray(sitk_maskImage)
    N4BiasFieldCorrection_kwargs.update(
        image=sitk_image,
        maskImage=sitk_maskImage,
    )
    bias_corrected_downsampled_image = sitk.N4BiasFieldCorrection(*N4BiasFieldCorrection_kwargs.values())
    bias_corrected_downsampled_image = sitk.GetArrayFromImage(bias_corrected_downsampled_image)

    # Compute bias from bias_corrected_downsampled_image.
    downsample_computed_bias = bias_corrected_downsampled_image / downsampled_image

    # Upsample bias.
    upsampled_bias = resize(downsample_computed_bias, image.shape)

    # Apply upsampled bias to original resolution shifted image.
    bias_corrected_image = image * upsampled_bias

    # Reverse the initial shift.
    bias_corrected_image += image_min - 1

    return bias_corrected_image


# TODO: complete function and import in preprocessing/__init__.py.
def remove_grid_artifact(image, z_axis=1, sigma=10, mask=None):
    """Remove the grid artifact from tiled data - tiles are stacked along z_axis."""

    if mask is None:
        mask = np.ones_like(image)
    elif mask == 'Otsu':
        sitk_Image_mask = sitk.OtsuThreshold(sitk.GetImageFromArray(image))
        mask = sitk.GetArrayFromImage(sitk_Image_mask)
        
    masked_image = image * mask

    # Shift masked_image above 1.
#     min_value = np.min(masked_image)
#     masked_image = masked_image - min_value + 1

    print(np.any(np.isnan(masked_image)))

    # Compute masked average.
    mean_across_z = np.average(masked_image, axis=z_axis)
    # Correct for the zero-valued elements included in the above average.
    z_mask_sum = np.sum(mask, axis=z_axis)
    mean_across_z *= masked_image.shape[z_axis] / np.where(z_mask_sum != 0, z_mask_sum, 1)
    np.nan_to_num(mean_across_z, copy=False, nan=0)
    
    print(np.any(np.isnan(mean_across_z)))

    bias_z_projection = gaussian_filter(mean_across_z, sigma) / mean_across_z
    bias_z_projection[ np.isinf(bias_z_projection) ] = 1.0
    
#     print(np.any(np.isnan(bias_z_projection)))
    print(np.any(np.isinf(bias_z_projection)))
    
    bias_z_image = np.expand_dims(bias_z_projection, z_axis)
    
#     print(np.any(np.isnan(bias_z_image)))
#     print(np.any(np.isnan(masked_image)))
#     print(masked_image.shape, bias_z_image.shape)
    
#     print(masked_image)
#     print(bias_z_image)

    corrected_masked_image = masked_image * bias_z_image
    
    print(corrected_masked_image.shape)
    print(np.any(np.isnan(corrected_masked_image)))

    # Shift corrected_masked_image to account for the initial shift.
#     corrected_masked_image = corrected_masked_image + min_value - 1

    return corrected_masked_image