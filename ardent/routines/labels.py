import numpy as np
from ..utilities import _validate_ndarray
from ..utilities import _validate_xyz_resolution

def compute_label_volumes(label_image, resolution=1):

    # Validate inputs.
    label_image = _validate_ndarray(label_image)
    resolution = _validate_xyz_resolution(label_image.ndim, resolution)

    # Count the instances of each unique label.
    unique_labels, label_counts = np.unique(label_image, return_counts=True)
    
    # Multiply by the volume of a single voxel to get label_volumes.
    voxel_volume = np.prod(resolution)
    label_volumes = label_counts * voxel_volume
    
    # Construct a dictionary of the form {unique_label : volume_of_label}.
    labels_to_volumes = dict(zip(unique_labels, label_volumes))

    return labels_to_volumes


def regional_stats(label_image, signal_image):
    """Return a dictionary mapping each unique label to its (mean, standard deviation) for the signal_image in that region."""

    # Validate inputs.
    label_image = _validate_ndarray(label_image)
    signal_image = _validate_ndarray(signal_image, reshape_to_shape=label_image.shape)

    # Count the instances of each unique label.
    unique_labels, label_counts = np.unique(label_image, return_counts=True)

    # Compute statistics for each label and include in final dictionary.
    labels_to_statistics = dict()
    for label in unique_labels:
        # Find the signal_image subset for this region.
        signal_image_subset = signal_image[label_image == label]

        # Compute the statistics for this label.
        label_mean = np.mean(signal_image_subset)
        label_standard_deviation = np.std(signal_image_subset)
        label_statistics = (label_mean, label_standard_deviation)

        labels_to_statistics[label] = label_statistics
    
    return labels_to_statistics