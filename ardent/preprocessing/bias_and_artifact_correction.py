import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize, rescale

from ..lddmm._lddmm_utilities import _validate_ndarray


def correct_bias_field(image, correct_at_scale=1, as_float32=True, **kwargs):
    """
    Shifts image such that its minimum value is 1, computes the bias field after downsampling by correct_at_scale, 
    upsamples this bias field and applies it to the shifted image, then undoes the shift and returns the result.
    Computes bias field using sitk.N4BiasFieldCorrection (http://bit.ly/2oFwAun).
    
    Args:
        image (np.ndarray): The image to be bias corrected.
        correct_at_scale (float, optional): The scale by which the shape of image is reduced before computing the bias. Defaults to 1.
        as_float32 (bool, optional): If True, image is internally cast as a sitk.Image of type sitkFloat32. If False, it is of type sitkFloat64. Defaults to True.

    Kwargs:
        Any additional keyword arguments overwrite the default values passed to sitk.N4BiasFieldCorrection.
    
    Returns:
        np.ndarray: A copy of image after bias correction.
    """

    # Verify correct_at_scale.
    correct_at_scale = float(correct_at_scale)
    if correct_at_scale < 1:
        raise ValueError(f"correct_at_scale must be equal to or greater than 1.\n"
                         f"correct_at_scale: {correct_at_scale}.")

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


def remove_grid_artifact(image, z_axis=0, sigma_blur=10, mask='Otsu'):
    """
    Remove the grid artifact from tiled data.
    
    Args:
        image (np.ndarray): The image with a grid artifact.
        z_axis (int, optional): The axis along which the tiles are stacked. Defaults to 0.
        sigma_blur (float, optional): The size of the blur used to compute the bias for grid edges in units of voxels. Should be approximately the size of the tiles. Defaults to 10.
        mask (np.ndarray, str, NoneType, optional): A mask of the valued voxels in the image. 
            Supported values are:
                a np.ndarray with a shape corresponding to image.shape, 
                None, indicating no mask (i.e. all voxels are considered in the artifact correction), 
                'Otsu', . Defaults to 'Otsu'.
    
    Returns:
        np.ndarray: A copy of image with its grid artifact removed.
    """
    test
    '''Remove the grid artifact from tiled data - tiles are stacked along z_axis.'''


    '''
    takae the mena across z
    blur the meaan with a gaussian
    sigma chosen such that when looking at the blurred mean you can no longer make out the gridlines
    image_corrective_factors = blurred_image / original_image 
    --> note: original image may have zeros. make it not so. when blurring, make sure all values are still positive. sigma is in units of voxels
    return image * image_corrective_factors
    '''

    # Construct masked_image as a ma.MaskedArray.

    # Interpret input mask.
    if mask is None:
        mask = np.ones_like(image)
    elif mask == 'Otsu':
        # Finds the optimal split threshold between the foreground anad background, by maximizing the interclass variance and minimizing the intraclass variance between voxel intensities.
        sitk_Image_mask = sitk.OtsuThreshold(sitk.GetImageFromArray(image))
        mask = sitk.GetArrayFromImage(sitk_Image_mask)
    else:
        mask = _validate_ndarray(mask, reshape_to_shape=image.shape)
    mask = mask.astype(bool)
    
    # Use the inverse of mask to create masked_image.
    masked_image = ma.masked_array(image, mask=~mask)

    # Shift masked_image so that its minimum lies at 1.
    masked_image_min = np.min(masked_image)
    masked_image -= masked_image_min - 1

    # Correct grid artifacts.

    # Take the average across z
    mean_across_z = np.mean(masked_image, axis=z_axis, keepdims=True)

    # Blur the mean with a gaussian.
    z_projection_bias = gaussian_filter(mean_across_z, sigma_blur) / mean_across_z

    # Apply the z_projection_bias to correct the masked_image.
    corrected_masked_image = masked_image * z_projection_bias

    # Reverse the shift to restore the original data window.
    corrected_masked_image += masked_image_min - 1

    # Return the unmasked array.
    return corrected_masked_image.data