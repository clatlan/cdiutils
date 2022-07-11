import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import convolve, center_of_mass

# TODO:  Check out new parameters in function fund_hull


def size_up_support(support):
    kernel = np.ones(shape=(3, 3, 3))
    convolved_support = convolve(support, kernel, mode='constant', cval=0.0)
    return np.where(convolved_support > 3, 1, 0)


def find_hull(
        volume,
        threshold=18,
        kernel_size=3,
        boolean_value=False,
        nan_value=False):
    """
    Find the convex hull of a 3D volume object.
    :param volume: 3D np.array. The volume to get the hull from.
    :param threshold: threshold that selects what belongs to the
    hull or not (int). If threshold >= 27, the returned hull will be
    similar to volume.
    :returns: the convex hull of the shape accordingly to the given
    threshold (np.array).
    """

    kernel = np.ones(shape=(kernel_size, kernel_size, kernel_size))
    convolved_support = convolve(volume, kernel, mode='constant', cval=0.0)
    hull = np.where(
        ((0 < convolved_support) & (convolved_support <= threshold)),
        1 if boolean_value else convolved_support,
        np.nan if nan_value else 0)
    return hull


def make_support(data, isosurface=0.5, nan_value=False):
    return np.where(data >= isosurface, 1, np.nan if nan_value else 0)


def unit_vector(vector):
    """Return a unit vector."""
    return vector / np.linalg.norm(vector)


def normalize(data, zero_centered=True):
    if zero_centered:
        abs_max = np.max([np.abs(np.min(data)), np.abs(np.max(data))])
        vmin, vmax = -abs_max, abs_max
        ptp = vmax - vmin
    else:
        ptp = np.ptp(data)
        vmin = np.min(data)
    return (data - vmin) / ptp


def normalize_complex_array(array):
    """Normalize a array of complex numbers."""
    shifted_array = array - array.real.min() - 1j*array.imag.min()
    return shifted_array/np.abs(shifted_array).max()


def center(data, center=None, method="com"):
    """
    Center 3D volume data such that the center of mass of data is at
    the very center of the 3D matrix.
    :param data: volume data (np.array). 3D numpy array which will be
    centered.
    :param com: center of mass coordinates(list, np.array). If no com is
    provided, com of the given data is computed (default: None).
    :param method: what region to place at the center (str), either
    com or max.
    :returns: centered 3D numpy array.
    """
    shape = data.shape

    if method == "com":
        if center is None:
            xcenter, ycenter, zcenter = (
                int(round(c)) for c in center_of_mass(data)
                )
    elif method == "max":
        if center is None:
            xcenter, ycenter, zcenter = np.where(data == np.max(data))
    else:
        print("method unknown, please choose between ['com', 'max']")
        return data

    centered_data = np.roll(data, shape[0] // 2 - xcenter, axis=0)
    centered_data = np.roll(centered_data, shape[1] // 2 - ycenter, axis=1)
    centered_data = np.roll(centered_data, shape[2] // 2 - zcenter, axis=2)

    return centered_data


def crop_at_center(data, final_shape=None):
    """
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


def compute_distance_from_com(data, com=None):
    nonzero_coordinates = np.nonzero(data)
    distance_matrix = np.zeros(shape=data.shape)

    if com is None:
        com = center_of_mass(data)

    for x, y, z in zip(nonzero_coordinates[0],
                       nonzero_coordinates[1],
                       nonzero_coordinates[2]):
        distance = np.sqrt((x-com[0])**2 + (y-com[1])**2 + (z-com[2])**2)
        distance_matrix[x, y, z] = distance

    return distance_matrix


def zero_to_nan(data):
    return np.where(data == 0, np.nan, data)


def nan_to_zero(data):
    return np.where(np.isnan(data), 0, data)