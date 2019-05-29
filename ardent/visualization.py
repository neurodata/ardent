import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.plotting as niplot

def _scale_data(data, clip_mode=None, stdevs=4, quantile=0.01, limits=None):
    """Returns a copy of data scaled such that the bulk of the values are mapped to the range [0, 1].
    
    Upper and lower limits are chosen based on clip_mode and other arguments.
    These limits are the anchors by which <data> is scaled such that after scaling,
    they lie on either end of the range [0, 1].

    supported clip_mode values:
    - 'valid mode' | related_kwarg [default kwarg value] --> description
    - 'stdev' | stdevs [4] --> limits are the mean +/- <stdevs> standard deviations
    - 'quantile' | quantile [0.05] --> limits are the <quantile> and 1 - <quantile> quantiles

    If <limits> is provided as a 2-element iterable, it will override clip_mode 
    and be used directly as the anchoring limits:
    <limits> = (lower, upper)."""

    # Verify type of data.
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray.\ntype(data): {type(data)}.")

    # Copy data.
    data = np.copy(data)

    # Determine scaling limits.

    if limits is not None:
        try:
            if len(limits) != 2:
                raise ValueError(f"If provided, limits must have length 2.\n"
                    f"len(limits): {len(limits)}.")
        except TypeError:
            # If len(limits) raises a TypeError, raise informative TypeError.
            raise TypeError(f"If provided, limits must be a 2-element iterable.\n"
            f"type(limits): {type(limits)}.")
        # limits is a 2-element iterable.
        lower_lim, upper_lim = limits
    else:
        # limits is None. Use clip_mode to determine upper and lower limits.
        # List supported clip_mode values.
        supported_clip_modes = [None, 'stdev', 'quantile']
        # Handle default None value, with no clipping.
        if clip_mode is None:
            lower_lim = np.min(data)
            upper_lim = np.max(data)
        # Check clip_mode type.
        elif not isinstance(clip_mode, str):
            raise TypeError(f"If provided, clip_mode must be a string.\ntype(clip_mode): {type(clip_mode)}.")
        # clip_mode is a string.
        # Calculate limits appropriately.
        elif clip_mode == 'stdev':
            # Verify stdevs.
            if not isinstance(stdevs, (int, float)):
                raise TypeError(f"For clip_mode='stdev', <stdevs> must be of type int or float.\n"
                    f"type(stdevs): {type(stdevs)}.")
            if stdevs < 0:
                raise ValueError(f"For clip_mode='stdev', <stdevs> must be non-negative.\n"
                    f"stdevs: {stdevs}.")
            # Choose limits equal to the mean +/- <stdevs> standard deviations.
            stdev = np.std(data)
            mean = np.mean(data)
            lower_lim = mean - stdevs*stdev
            upper_lim = mean + stdevs*stdev
        elif clip_mode == 'quantile':
            # Verify quantile.
            if not isinstance(quantile, (int, float)):
                raise TypeError(f"For clip_mode='quantile', <quantile> must be of type int or float.\n"
                    f"type(quantile): {type(quantile)}.")
            if quantile < 0 or quantile > 1:
                raise ValueError(f"For clip_mode='quantile', <quantile> must be in the interval [0, 1].\n"
                    f"quantile: {quantile}.")
            # Choose limits based on quantiles
            lower_lim = np.quantile(data, quantile)
            upper_lim = np.quantile(data, 1 - quantile)
        else:
            raise ValueError(f"Unrecognized value for clip_mode. Supported values include {supported_clip_modes}.\n"
                f"clip_mode: {clip_mode}.")
    # lower_lim and upper_lim are set appropriately.

    # Create a clipped view of data and scale data based on it.
    clipped_data = data[(data - lower_lim >= 0) & (data - upper_lim <= 0)]
    data = data - np.min(clipped_data)
    data = data / np.max(clipped_data)

    # Return scaled copy of data.
    return data


def _validate_inputs(data, title, n_cuts, xcuts, ycuts, zcuts, figsize):
    """Returns a dictionary of the form {'inputName' : inputValue}.
    It has an entry for each argument that has been validated."""

    inputDict = {'data':data, 'figsize':figsize}

    # Validate data.

    supported_data_types = [np.ndarray]
    # Convert data to np.ndarray for supported types.
    # if isinstance(data, ):
    #     # Convert to np.ndarray.
    #     data = np.array(data)

    # If data is none of the supported types, attempt to cast it as a np.ndarray.
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except TypeError:
            # If a TypeError was raised on casting data to a np.ndarray, raise informative TypeError.
            raise TypeError(f"data is not one of the supported types {supported_data_types} and cannot be cast as a np.ndarray.\n"
                f"type(data): {type(data)}.")

    if data.ndim < 3: # <3
        raise ValueError(f"data must have at least 3 dimensions.\ndata.ndim: {data.ndim}.")
    # data is valid.
    inputDict.update(data=data)

    # Validate figsize.

    # TODO: dynamicize figsize.
    if figsize is None:
        # Compute figsize.
        raise NotImplementedError("This functionality has not yet been implemented. Please provide another figsize.")
    else:
        try:
            if len(figsize) != 2:
                raise ValueError(f"figsize must be an iterable with length 2.\n"
                    f"len(figsize): {len(figsize)}.")
        except TypeError:
            raise TypeError(f"figsize must be an iterable.\ntype(figsize): {type(figsize)}.")
    # figsize is valid.
    inputDict.update(figsize=figsize)

    return inputDict


# TODO: update work with interesting_cuts and new Image class
def _Image_to_Nifti2Image(image, affine=None):
    
    nifti2header = nib.Nifti2Header()
    nifti2header['dim'][1:1 + len(image.nxyz)] = image.nxyz
    nifti2header['pixdim'][1:1 + len(image.dxyz)] = image.dxyz
    
    nifti2image = nib.Nifti2Image(image.data, affine=affine, header=nifti2header)
    
    return nifti2image


def _get_cuts(data, xcuts, ycuts, zcuts, n_cuts=5, interesting_cuts=False):
    """Returns xcuts, ycuts, & zcuts. If any of these are provided, they are used. 
    If any are not provided, they are computed. 
    
    The default is to compute unprovided cuts as evenly spaced slices across that dimension.
    However, if interesting_cuts is True, then any dimension's cuts that are not specified 
    are computed using niplot.find_cut_slices."""

    if interesting_cuts is True:
        # TODO: update Image_to_Nifti2Image to allow for multiple input types, and find a way to give it image metadata.
        raise NotImplementedError("This functionality has not been fully implemented yet.")
        xcuts = xcuts or niplot.find_cut_slices(Image_to_Nifti2Image(atlas_Image, affine=np.eye(4)), direction='x', n_cuts=n_cuts).astype(int)
        ycuts = ycuts or niplot.find_cut_slices(Image_to_Nifti2Image(atlas_Image, affine=np.eye(4)), direction='y', n_cuts=n_cuts).astype(int)
        zcuts = zcuts or niplot.find_cut_slices(Image_to_Nifti2Image(atlas_Image, affine=np.eye(4)), direction='z', n_cuts=n_cuts).astype(int)
    else:
        xcuts = xcuts or np.linspace(0, data.shape[0], n_cuts + 2)[1:-1]
        ycuts = ycuts or np.linspace(0, data.shape[1], n_cuts + 2)[1:-1]
        zcuts = zcuts or np.linspace(0, data.shape[2], n_cuts + 2)[1:-1]
    
    return xcuts, ycuts, zcuts

# TODO: verify plotting works with xyzcuts provided with inconsistent lengths.
# TODO: allow n_cuts to be a triple.
def heatslices(data, title=None, n_cuts=5, xcuts=[], ycuts=[], zcuts=[], figsize=(10, 5)):

    # TODO: find a cleaner way to do this.
    # Validate inputs
    inputs = {'data':data, 'title':title, 'n_cuts':n_cuts, 'xcuts':xcuts, 'ycuts':ycuts, 'zcuts':zcuts, 'figsize':figsize}
    validated_inputs = _validate_inputs(**inputs)
    locals().update(validated_inputs)

    # Scale bulk of data to [0, 1].
    data = _scale_data(data, clip_mode='stdev', stdevs=4) # Side-effect: breaks alias.
    
    # Get cuts.
    xcuts, ycuts, zcuts = _get_cuts(data, xcuts, ycuts, zcuts, n_cuts)
    
    # maxcuts is the number of cuts in the dimension with the largest number of cuts.
    maxcuts = max(list(map(lambda cuts: len(cuts), [xcuts, ycuts, zcuts])))
    
    # TODO: properly scale subplots / axs such that scale is consistent across all images.
    fig, axs = plt.subplots(3, maxcuts, sharex='row', sharey='row', figsize=figsize)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)
    for ax in axs.ravel():
        pass
    plt.setp(axs, aspect='equal', xticks=[], yticks=[])
    for row, cuts in enumerate([xcuts, ycuts, zcuts]):
        for col, cut in enumerate(cuts):
            axs[row, col].grid(False)
            img = axs[row, col].imshow(data.take(cut, row), vmin=0, vmax=1, cmap='gray')
    cax = plt.axes([0.925, 0.1, 0.025, 0.77])
    plt.colorbar(img, cax=cax)
    fig.suptitle(title, fontsize=20)