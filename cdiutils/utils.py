from typing import Optional, Tuple, Union
import numpy as np
import matplotlib
import seaborn as sns
from scipy.ndimage import convolve, center_of_mass  
from scipy.stats import gaussian_kde
import textwrap
import xrayutilities as xu

from cdiutils.plot.formatting import plot_background



def pretty_print(text: str, max_char_per_line: int=80) -> None:
    """Print text with a frame of stars."""

    stars = "*" * max_char_per_line
    print("\n" + stars)
    print("*", end="")
    for i in range((max_char_per_line-len(text))//2 - 1):
        print(" ", end="")
    print(textwrap.fill(text, max_char_per_line), end="")
    for i in range((max_char_per_line-len(text))//2 - 1 + len(text)%2):
        print(" ", end="")
    print("*")
    print(stars + "\n")


def size_up_support(support):
    kernel = np.ones(shape=(3, 3, 3))
    convolved_support = convolve(support, kernel, mode='constant', cval=0.0)
    return np.where(convolved_support > 3, 1, 0)

# TODO:  Check out new parameters in function find_hull
def find_hull(
        volume,
        threshold=18,
        kernel_size=3,
        boolean_values=False,
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
        1 if boolean_values else convolved_support,
        np.nan if nan_value else 0)
    return hull


def make_support(data, isosurface=0.5, nan_values=False):
    return np.where(data >= isosurface, 1, np.nan if nan_values else 0)


def unit_vector(vector):
    """Return a unit vector."""
    return vector / np.linalg.norm(vector)


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def v1_to_v2_rotation_matrix(v1, v2):
    """ 
    Rotation matrix around axis v1xv2
    """
    vec_rot_axis = np.cross(v1, v2)
    normed_vrot = unit_vector(vec_rot_axis)

    theta = angle(v1, v2)

    n1, n2, n3 = normed_vrot    
    ct = np.cos(theta)
    st = np.sin(theta)
    
    r = np.array(((ct+n1**2*(1-ct), n1*n2*(1-ct)-n3*st, n1*n3*(1-ct)+n2*st),
                  (n1*n2*(1-ct)+n3*st, ct+n2**2*(1-ct), n2*n3*(1-ct)-n1*st),
                  (n1*n3*(1-ct)-n2*st, n2*n3*(1-ct)+n1*st, ct+n3**2*(1-ct))
                  ))
    return r


def normalize(data, zero_centered=False):
    if zero_centered:
        abs_max = np.max([np.abs(np.min(data)), np.abs(np.max(data))])
        vmin, vmax = -abs_max, abs_max
        ptp = vmax - vmin
    else:
        ptp = np.ptp(data)
        vmin = np.min(data)
    return (data - vmin) / ptp

def basic_filter(data, maplog_min_value=3.5):
    return np.power(xu.maplog(data, maplog_min_value, 0), 10)

def normalize_complex_array(array):
    """Normalize a array of complex numbers."""
    shifted_array = array - array.real.min() - 1j*array.imag.min()
    return shifted_array/np.abs(shifted_array).max()

def shape_for_safe_centered_cropping(
        data_shape: Union[tuple, np.ndarray, list],
        position: Union[tuple, np.ndarray, list],
        final_shape: Optional[tuple]=None
) -> tuple:
    """
    Utility function that find the smallest shape that allows a safe
    cropping, i.e. without moving data from one side to another when
    using the np.roll() function.
    """
    if not isinstance(data_shape, np.ndarray):
        data_shape = np.array(data_shape)
    if not isinstance(position, np.ndarray):
        position = np.array(position)

    secured_shape = 2 * np.min([position, data_shape - position], axis=0)

    if final_shape is None:
        return tuple(secured_shape)
    else:
        return tuple(np.min([secured_shape, final_shape], axis=0))
    

def center(
        data: np.ndarray,
        where="com",
        return_former_center=False
) -> Union[np.ndarray, tuple[np.ndarray, tuple]]:
    """
    Center 3D volume data such that the center of mass or max  of data
    is at the very center of the 3D matrix.
    :param data: volume data (np.array). 3D numpy array which will be
    centered.
    :param com: center of mass coordinates(list, np.array). If no com is
    provided, com of the given data is computed (default: None).
    :param method: what region to place at the center (str), either
    com or max.
    :returns: centered 3D numpy array.
    """
    shape = data.shape

    if where == "max":
        reference_position = np.unravel_index(data.argmax(), data.shape)
    elif where == "com":
        reference_position = tuple(int(e) for e in center_of_mass(data))
    elif isinstance(where, tuple) and len(where) == 3:
        reference_position = where
    else:
        raise ValueError(
            "method must be 'max', 'com' or tuple of 3 integer coordinates"
        )
    xcenter, ycenter, zcenter = reference_position

    centered_data = np.roll(data, shape[0] // 2 - xcenter, axis=0)
    centered_data = np.roll(centered_data, shape[1] // 2 - ycenter, axis=1)
    centered_data = np.roll(centered_data, shape[2] // 2 - zcenter, axis=2)

    if return_former_center:
        return centered_data, (xcenter, ycenter, zcenter)

    return centered_data


def symmetric_pad(data, final_shape=None, values=0):
    if final_shape is None:
        print("No final_shape given, data will not be padded")
        return data
    shape = data.shape

    axis0_pad_width = (final_shape[0] - shape[0]) // 2
    axis1_pad_width = (final_shape[1] - shape[1]) // 2
    axis2_pad_width = (final_shape[2] - shape[2]) // 2

    return np.pad(
        data,
        (
            (axis0_pad_width, axis0_pad_width + (final_shape[0] - shape[0]) % 2),
            (axis1_pad_width, axis1_pad_width + (final_shape[1] - shape[1]) % 2),
            (axis2_pad_width, axis2_pad_width + (final_shape[2] - shape[2]) % 2)
        ),
        mode="constant",
        constant_values=values
    )


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
    # if final_shape[i] is odd, center[i] must be at
    # final_shape[i] + 1 // 2
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


def zero_to_nan(data, boolean_value=False):
    return np.where(data == 0, np.nan, 1 if boolean_value else data)


def nan_to_zero(data, boolean_value=False):
    return np.where(np.isnan(data), 0, 1 if boolean_value else data)


def to_bool(data, nan_value=False):
    return np.where(np.isnan(data), np.nan if nan_value else 0, 1)


def nan_center_of_mass(data, indices=None):
    non_nan_coord = np.where(np.invert(np.isnan(data)))
    com = np.average(
        [non_nan_coord], axis=2,
        weights=data[non_nan_coord]
    )[0]
    return com


def compute_corrected_angles(
        inplane_angle: float,
        outofplane_angle: float,
        detector_coordinates: tuple,
        detector_distance: float,
        direct_beam_position: tuple,
        pixel_size: float=55e-6,
        verbose=False
):
    """
    Compute the detector corrected angles given the angles saved in the
    experiment data file and the position of interest in the detector frame

    :param inplane_angle: in-plane detector angle in degrees (float).
    :param outofplane_angle out-of-plane detector angle in degrees
    (float).
    :param detector_coordinates: the detector coordinates of the point
    of interest (tuple or list).
    :param detector_distance: the sample to detector distance
    :param direct_beam_position: the direct beam position in the
    detector frame (tuple or list).
    :param pixel_size: the pixel size (float).
    :param verbose: whether or not to print the corrections (bool).

    :return: the two corrected angles.
    """
    inplane_correction = np.rad2deg(
        np.arctan(
            (detector_coordinates[1] - direct_beam_position[0])
            * pixel_size
            / detector_distance
        )
    )

    outofplane_correction = np.rad2deg(
        np.arctan(
            (detector_coordinates[0] - direct_beam_position[1])
            * pixel_size
            / detector_distance
        )
    )

    corrected_inplane_angle = float(inplane_angle - inplane_correction)
    corrected_outofplane_angle = float(outofplane_angle - outofplane_correction)

    if verbose:
        print(
            f"current in-plane angle: {inplane_correction}\n"
            f"in-plane angle correction: {corrected_inplane_angle}\n"
            f"corrected in-plane angle: {corrected_inplane_angle}\n\n"
            f"current out-of-plane angle: {outofplane_angle}\n"
            f"out-of-plane angle correction: {outofplane_correction}\n"
            f"corrected out-of-plane angle: {corrected_outofplane_angle}"
        )
    return corrected_inplane_angle, corrected_outofplane_angle


def find_suitable_array_shape(
        support: np.array,
        padding: Optional[list]=[4, 4, 4],
        symmetrical_shape: Optional[bool]=True
) -> np.array:
    """Find a more suitable shape of an array"""

    hull = find_hull(support, boolean_values=True)
    coordinates = np.where(hull == 1)
    axis_0_range = np.ptp(coordinates[0]) + padding[0]
    axis_1_range = np.ptp(coordinates[1]) + padding[1]
    axis_2_range = np.ptp(coordinates[2]) + padding[2]

    if symmetrical_shape:
        return np.repeat(
            np.max(np.array([axis_0_range, axis_1_range, axis_2_range])),
            3
        )

    return np.array([axis_0_range, axis_1_range, axis_2_range])


def find_isosurface(
        amplitude: np.ndarray,
        nbins: Optional[int]=100,
        sigma_criterion: Optional[float]=2,
        show: Optional[bool]=False
) -> Tuple[float, matplotlib.axes.Axes]:
    """
    Estimate the isosurface from the amplitude distribution

    :param amplitude: the 3D amplitude volume (np.array)
    :param nbins: the number of bins to considerate when making the
    histogram (Optional, int)
    :param sigma_criterion: the factor to compute the isosurface wich is
    calculated as: mu - sigma_criterion * sigma. By default set to 2.
    (Optional, float)
    :param show: whether or not to show the the figure

    :return: the isosurface value and the figure in which the histogram
    was plotted
    """

    # normalize and flatten the amplitude
    flattened_amplitude = normalize(amplitude).ravel()

    counts, bins = np.histogram(flattened_amplitude, bins=nbins)

    # remove the background
    background_value = bins[np.where(counts == counts.max())[0]+1+ nbins//20]
    filtered_amplitude = flattened_amplitude[
        flattened_amplitude > background_value
    ]

    # redo the histogram with the filtered amplitude
    counts, bins = np.histogram(filtered_amplitude, bins=nbins, density=True)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    bin_size = bin_centres[1] - bin_centres[0]

    # fit the amplitude distribution
    kernel = gaussian_kde(filtered_amplitude)
    x = np.linspace(0, 1, 1000)
    fitted_counts = kernel(x)

    max_index = np.argmax(fitted_counts)
    right_gaussian_part = np.where(x >= x[max_index], fitted_counts, 0)
    
    # find the closest indexes
    right_HM_index = np.argmin(
        np.abs(right_gaussian_part - fitted_counts.max() / 2)
    )  
    left_HM_index = max_index - (right_HM_index - max_index)
    
    fwhm = x[right_HM_index] - x[left_HM_index]
    sigma_estimate = fwhm / 2*np.sqrt(2*np.log(2))
    isosurface = x[max_index] - sigma_criterion * sigma_estimate

    fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(8, 5))
    ax = plot_background(ax)
    ax.bar(
        bin_centres,
        counts,
        width=bin_size,
        color="dodgerblue",
        alpha=0.9,
        edgecolor=(0, 0, 0, 0.25),
        label="amplitude distribution"
    )
    sns.kdeplot(
        filtered_amplitude,
        ax=ax,
        alpha=0.3,
        fill=True,
        color="navy",
        label="density estimate"
    )
    ax.axvspan(
        x[left_HM_index],
        x[right_HM_index],
        edgecolor="k",
        facecolor="green",
        alpha=0.2,
        label="FWHM"
    )
    ax.plot(
        [isosurface, isosurface],
        [0, fitted_counts[(np.abs(x - isosurface)).argmin()]],
        solid_capstyle="round",
        color="lightcoral",
        lw=5,
        label=f"isosurface estimated at {isosurface:0.3f}"
    )

    ax.set_xlabel("normalized amplitude", size=14)
    ax.set_ylabel("counts",  size=14)
    ax.legend()
    fig.suptitle("Reconstructed amplitude distribution", size=16)
    fig.tight_layout()
    if show:
        matplotlib.pyplot.show()
    
    return isosurface, fig