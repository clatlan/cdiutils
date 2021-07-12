import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import convolve, center_of_mass


def find_hull(volume, threshold=26):
    """
    Find the convex hull of a 3D volume object.

    :param volume: 3D np.array. The volume to get the hull from.
    :param threshold: threshold that selects what belongs to the
    hull or not (int). If threshold >= 27, the returned hull will be
    similar to volume.
    :returns: the convex hull of the shape accordingly to the given
    threshold (np.array).
    """

    kernel = np.ones(shape=(3, 3, 3))
    convolved_support = convolve(volume, kernel, mode='constant', cval=0.0)
    hull = np.where(
        ((0 < convolved_support) & (convolved_support <= threshold)),
        support,
        0)
    return hull


def unit_vector(vector):
    """Return a unit vector."""
    return vector / np.linalg.norm(vector)


def normalize_complex_array(array):
    """Normalize a array of complex numbers."""
    shifted_array = array - array.real.min() - 1j*array.imag.min()
    return shifted_array/np.abs(shifted_array).max()


def center(data, com=None):
    """
    Center 3D volume data such that the center of mass of data is at
    the very center of the 3D matrix.

    :param data: volume data (np.array). 3D numpy array which will be
    centered.
    :param com: center of mass coordinates(list, np.array). If no com is
    provided, com of the given data is computed (default: None).
    :returns: centered 3D numpy array.
    """
    shape = data.shape
    if com is None:
        com = [round(c) for c in center_of_mass(data)]
    centered_data = np.roll(data, eshape[0] // 2 - com[0], axis=0)
    centered_data = np.roll(centered_data, shape[1] // 2 - com[1], axis=1)
    centered_data = np.roll(centered_data, shape[2] // 2 - com[2], axis=2)

    return centered_data


def crop_at_center(data, final_shape=None):
    """int value.
    Crop 3D array data to match the final_shape. Center of the input
    data remains the center of cropped data.

    :param data: 3D array data to be cropped (np.array).
    :param final_shape: the targetted shape (list). If None, nothing
    happens.
    :returns: cropped 3D array (np.array).
    """
    if final_shape is None:
        print("No final shape specified, did not proceed to cropping")
        return data

    shape = data.shape
    final_shape = np.array(final_shape)

    if not (final_shape <= data.shape).all():
        print(
            "One of the axis of the final shape is larger than "
            "the initial axis (initial shape: {}, final shape: {}).\n"
            "Did not proceed to cropping.".format(shape, tuple(final_shape))
            )
        return data

    c = np.array(shape) // 2  # coordinates of the center
    to_crop = final_shape // 2  # indices to crop at both sides
    # if final_shape[i] is not even, center[i] must be at
    # final_shape[i] + 1
    plus_one = np.where((final_shape % 2 == 0), 0, 1)

    cropped = data[c[0] - to_crop[0]: c[0] + to_crop[0] + plus_one[0],
                   c[1] - to_crop[1]: c[1] + to_crop[1] + plus_one[1],
                   c[2] - to_crop[2]: c[2] + to_crop[2] + plus_one[2]]

    return cropped
