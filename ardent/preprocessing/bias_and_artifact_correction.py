import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from ardent.preprocessing import change_resolution_by

def correct_bias_field(img, mask=None, scale=0.25, niters=[50, 50, 50, 50]):
    """Correct bias field in image using the N4ITK algorithm (http://bit.ly/2oFwAun)
    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input image with bias field.
    mask : {SimpleITK.SimpleITK.Image}, optional
        If used, the bias field will only be corrected within the mask. (the default is None, which results in the whole image being corrected.)
    scale : {float}, optional
        Scale at which to compute the bias correction. (the default is 0.25, which results in bias correction computed on an image downsampled to 1/4 of it's original size)
    niters : {list}, optional
        Number of iterations per resolution. Each additional entry in the list adds an additional resolution at which the bias is estimated. (the default is [50, 50, 50, 50] which results in 50 iterations per resolution at 4 resolutions)
    Returns
    -------
    SimpleITK.SimpleITK.Image
        Bias-corrected image that has the same size and spacing as the input image.
    """

    '''
    img_rescaled = img + 0.1 * stdev
    spacing = img_rescaled.spacing / scale
    img_resampled = resample(img_rescaled, spacing=spacing)

    img_bias_corrected = sitk.N4BiasFieldCorrection(img_resampled, mask, 0.001, niters)
    bias = img_bias_corrected / img_resampled
    bias = resample(bias, spacing=img.spacing, size=img.size)

    img_bias_corrected = img * bias
    '''



    '''
    img_rescaled = img + 0.1 * stdev
    spacing = img_rescaled.spacing / scale
    img_resampled = resample(img_rescaled, spacing=spacing)

    img_bias_corrected = sitk.N4BiasFieldCorrection(img_resampled, mask, 0.001, niters)
    bias = img_bias_corrected / img_resampled
    bias = resample(bias, spacing=img.spacing, size=img.size)

    img_bias_corrected = img * bias
    '''

# TODO: rewrite this function but better. Scale based on values within x stdevs and raise min to 1?
def scale_array(array):
    """Scale array to the range [1,2]."""

    array = np.copy(array)

    array -= np.min(array)
    array /= np.max(array)
    array += 1

    return array


def correct_bias_field(image, xyz_resolution, scale=0.25, **kwargs):
    '''Who cares if the image has 0 intensities?'''

    # Scale image to the interval [1,2].
    image = scale_array(image)

    # Downsample image according to <scale>.
    downsampled_image, downsampled_resolution = change_resolution_by(image, scale, return_true_resolution=True)

    # Bias-correct downsampled_image.
    N4BiasFieldCorrection_kwargs = dict(
        image=downsampled_image, 
        maskImage=None, 
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
    N4BiasFieldCorrection_kwargs.update(kwargs)
    bias_corrected_downsampled_image = sitk.N4BiasFieldCorrection(**N4BiasFieldCorrection_kwargs)
    bias_corrected_downsampled_image = sitk.GetArrayFromImage(bias_corrected_downsampled_image)

    # Compute bias from bias_corrected_downsampled_image.
    downsample_computed_bias = bias_corrected_downsampled_image / downsampled_image

    # Upsample bias.
    upsampled_bias = change_resolution_by(downsample_computed_bias, downsampled_resolution)

    # Apply upsampled bias to original image.
    bias_corrected_image = image * upsampled_bias

    return bias_corrected_image


def remove_grid_artifact(image, z_axis=1, sigma=10, mask=None):
    """Remove the grid artifact from tiled data."""

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