from typing import Optional
import warnings

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


def size_up_support(support: np.ndarray) -> np.ndarray:
    kernel = np.ones(shape=(3, 3, 3))
    convolved_support = convolve(support, kernel, mode='constant', cval=0.0)
    return np.where(convolved_support > 3, 1, 0)

def find_hull(
        volume: np.ndarray,
        threshold: float=18,
        kernel_size: int=3,
        boolean_values: bool=False,
        nan_value: bool=False
) -> np.ndarray:
    """
    Find the convex hull of a 3D volume object.
    :param volume: 3D np.array. The volume to get the hull from.
    :param threshold: threshold that selects what belongs to the
    hull or not (int). If threshold >= 27, the returned hull will be
    similar to volume.
    :kernel_size: the size of the kernel used to convolute (int).
    :boolean_values: whether or not to return 1 and 0 np.ndarray
    or the computed coordination.

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


def make_support(
        data: np.ndarray,
        isosurface: float=0.5,
        nan_values: bool=False
) -> np.ndarray:
    """Create a support using the provided isosurface value."""
    data = normalize(data)
    return np.where(data >= isosurface, 1, np.nan if nan_values else 0)


def unit_vector(
        vector: tuple or list or np.ndarray
)->  np.ndarray:
    """Return a unit vector."""
    return np.array(vector) / np.linalg.norm(vector)


def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors."""
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def v1_to_v2_rotation_matrix(
        v1: np.ndarray,
        v2: np.ndarray
) -> np.ndarray:
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


def normalize(
        data: np.ndarray,
        zero_centered: bool=False
) -> np.ndarray:
    """Normalize a np.ndarray so the values are between 0 and 1."""
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

def normalize_complex_array(array: np.ndarray) -> np.ndarray:
    """Normalize a array of complex numbers."""
    shifted_array = array - array.real.min() - 1j*array.imag.min()
    return shifted_array/np.abs(shifted_array).max()


def find_max_pos(data: np.ndarray) -> tuple:
    """Find the index coordinates of the maximum value."""
    return np.unravel_index(data.argmax(), data.shape)


def shape_for_safe_centered_cropping(
        data_shape: tuple or np.ndarray or list,
        position: tuple or  np.ndarray or list,
        final_shape: Optional[tuple]=None
) -> tuple:
    """
    Utility function that finds the smallest shape that allows a safe
    cropping, i.e. without moving data from one side to another when
    using the np.roll() function.
    """
    if not isinstance(data_shape, np.ndarray):
        data_shape = np.array(data_shape)
    if not isinstance(position, np.ndarray):
        position = np.array(position)

    secured_shape = 2 * np.min([position, data_shape - position], axis=0)
    secured_shape = tuple(round(e) for e in secured_shape)

    if final_shape is None:
        return tuple(secured_shape)
    return tuple(np.min([secured_shape, final_shape], axis=0))


def _center_at_com(data: np.ndarray):
    shape = data.shape
    com = tuple(e for e in center_of_mass(data))
    print((np.array(shape)/2 == np.array(com)).all())
    com_to_center = np.array([
        int(np.rint(shape[i]/2 - com[i]))
        for i in range(3)
    ])
    if (com_to_center == np.array((0, 0, 0)).astype(int)).all():
        return data, com
    data = center(data, where=com)
    return _center_at_com(data)


def center(
        data: np.ndarray,
        where: str or tuple or  list or  np.ndarray="com",
        return_former_center: bool=False
) -> np.ndarray or tuple[np.ndarray, tuple]:
    """
    Center 3D volume data such that the center of mass or max  of data
    is at the very center of the 3D matrix.
    :param data: volume data (np.array). 3D numpy array which will be
    centered.
    :param com: center of mass coordinates(list, np.array). If no com is
    provided, com of the given data is computed (default: None).
    :param where: what region to place at the center (str), either
    com or max, or a tuple of the coordinates where to place the center
    at.
    :returns: centered 3D numpy array.
    """
    shape = data.shape

    if isinstance(where, (tuple, list, np.ndarray)) and len(where) == 3:
        reference_position = tuple(where)
    elif where == "max":
        reference_position = find_max_pos(data)
    elif where == "com":
        reference_position = tuple(e for e in center_of_mass(data))
    else:
        raise ValueError(
            "where must be 'max', 'com' or tuple or list of 3 floats "
            f"coordinates, can't be type: {type(where)} ({where}) "
        )
    xcenter, ycenter, zcenter = reference_position

    centered_data = np.roll(data, int(np.rint(shape[0] / 2 - xcenter)), axis=0)
    centered_data = np.roll(
        centered_data,
        int(np.rint(shape[1] / 2 - ycenter)),
        axis=1
    )
    centered_data = np.roll(
        centered_data,
        int(np.rint(shape[2] / 2 - zcenter)),
        axis=2
    )

    if return_former_center:
        return centered_data, (xcenter, ycenter, zcenter)

    return centered_data


def symmetric_pad(
        data: np.ndarray,
        final_shape: tuple or list or np.ndarray,
        values: float=0
) -> np.ndarray:
    """Return padded data so it matches the provided final_shape"""

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

def crop_at_center(
        data: np.ndarray,
        final_shape: list or tuple or  np.ndarray
) -> np.ndarray:
    """
    Crop 3D array data to match the final_shape. Center of the input
    data remains the center of cropped data.
    :param data: 3D array data to be cropped (np.array).
    :param final_shape: the targetted shape (list). If None, nothing
    happens.
    :returns: cropped 3D array (np.array).
    """
    shape = data.shape
    final_shape = np.array(final_shape)

    if not (final_shape <= data.shape).all():
        print(
            "One of the axis of the final shape is larger than "
            f"the initial axis (initial shape: {shape}, final shape: "
            f"{tuple(final_shape)}).\nDid not proceed to cropping."
        )
        return data

    c = np.array(shape) // 2  # coordinates of the center
    to_crop = final_shape // 2  # indices to crop at both sides
    # if final_shape[i] is odd, center[i] must be at
    # final_shape[i] + 1 // 2
    plus_one = np.where((final_shape % 2 == 0), 0, 1)

    cropped = data[
        c[0] - to_crop[0]: c[0] + to_crop[0] + plus_one[0],
        c[1] - to_crop[1]: c[1] + to_crop[1] + plus_one[1],
        c[2] - to_crop[2]: c[2] + to_crop[2] + plus_one[2]
    ]

    return cropped


def compute_distance_from_com(
        data: np.ndarray,
        com: tuple or list or np.ndarray=None
) -> np.ndarray:
    """
    Return a np.ndarray of the same shape of the provided data.
    (i, j, k) Value will correspond to the distance of the (i, j, k)
    voxel in data to the center of mass if that voxel is not null.
    """
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


def zero_to_nan(
        data: np.ndarray,
        boolean_values: bool=False
) -> np.ndarray:
    """Convert zero values to np.nan."""
    return np.where(data == 0, np.nan, 1 if boolean_values else data)


def nan_to_zero(
        data: np.ndarray,
        boolean_values: bool=False
) -> np.ndarray:
    """Convert np.nan values to 0."""
    return np.where(np.isnan(data), 0, 1 if boolean_values else data)


def to_bool(data: np.ndarray, nan_value: bool=False) -> np.ndarray:
    """Convert values to 1 (True) if not nan otherwise to 0 (False)"""
    return np.where(np.isnan(data), np.nan if nan_value else 0, 1)


def nan_center_of_mass(
        data: np.ndarray,
        return_int: bool=False
) -> np.ndarray:
    """
    Compute the center of mass of a np.ndarray that may contain
    nan values.
    """
    if not np.isnan(data).any():
        com = center_of_mass(data)

    non_nan_coord = np.where(np.invert(np.isnan(data)))
    com = np.average(
        [non_nan_coord], axis=2,
        weights=data[non_nan_coord]
    )[0]
    if return_int:
        return tuple([int(round(e)) for e in com])
    return tuple(com)


class CroppingHandler:
    """
    A class to handle data cropping. The class allows
    finding the requested position of the center and crop the data
    accordingly.
    """

    @staticmethod
    def get_position(
            data: np.ndarray, method: str or tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the position of the reference voxel based on the centering
        method.

        Args:
            data: Input data array.
            method: Centering method. Can be "max" for maximum
                intensity, "com" for center of mass, or a tuple of
                coordinates representing the voxel position.

        Returns:
            The position of the reference voxel as a tuple of
            coordinates.

        Raises:
        ValueError: If an invalid method is provided.
        """
        if method == "max":
            return np.unravel_index(np.argmax(data), data.shape)
        elif method == "com":
            com = center_of_mass(data)
            return tuple(int(round(e)) for e in com)
        elif (
            isinstance(method, (list, tuple))
            and all(isinstance(e, (int, np.int64)) for e in method)
        ):
            return tuple(method)
        else:
            raise ValueError(
                "Invalid method provided. Can be 'max', 'com' or a tuple of "
                "coordinates."
            )

    @classmethod
    def get_masked_data(
        cls, data: np.ndarray, roi: list[int]
    ) -> np.ma.MaskedArray:
        """
        Get the masked data array based on the region of interest (ROI).

        Args:
            data: Input data array.
            roi: Region of interest as a list of representing
                the cropped region ex: [start, end, start, end].

        Returns:
            The masked data array with the specified ROI.
        """

        mask = np.ones_like(data)
        mask[cls.roi_list_to_slices(roi)] = 0
        return np.ma.array(data, mask=mask)

    @staticmethod
    def roi_list_to_slices(roi: list[int]) -> tuple[slice, ...]:
        """
        Convert a ROI to a tuple of slices.

        Args:
            roi: Region of interest as a list of start and end values
                for each dimension.

        Returns:
            The ROI converted to a tuple of slices.
        """
        if len(roi) % 2 != 0:
            raise ValueError(
                "ROI should have start and end values for each dimension.")
        return tuple(
            slice(start, end) for start, end in zip(roi[::2], roi[1::2])
        )

    @classmethod
    def get_roi(
            cls,
            output_shape: tuple,
            where: tuple,
            input_shape: tuple=None
    ) -> list[int]:
        """
        Calculate the region of interest (ROI) for cropping the data
        based on the input shape, desired output shape, and reference
        voxel position.

        Args:
            output_shape: Desired output shape after cropping.
            where: Reference voxel position as a tuple of coordinates.
            input_shape: Shape of the input data array.

        Returns:
            The region of interest as a list of start and end values for
            each dimension.
        """

        # define how much to crop data
        # plus_one is whether or not to add one to the bounds.
        plus_one = np.where((np.array(output_shape) % 2 == 0), 0, 1)
        crop = [[e//2, e//2 + plus_one[i]] for i, e in enumerate(output_shape)]
        roi = []

        if input_shape is None:
            for i in range(len(output_shape)):
                roi.append(where[i]-crop[i][0])
                roi.append(where[i]+crop[i][1])
            return roi

        for i, s in enumerate(input_shape):
            # extend the roi to comply with the output_shape
            add_left = where[i]+crop[i][1]-s if where[i]+crop[i][1]>s else 0
            add_right = crop[i][0]-where[i] if where[i]-crop[i][0]<0 else 0

            roi.append(np.max([where[i]-crop[i][0], 0]) - add_left)
            roi.append(np.min([where[i]+crop[i][1], s]) + add_right)
        return roi

    @classmethod
    def chain_centering(
            cls,
            data: np.ndarray, output_shape: tuple[int, ...],
            methods: list[str or tuple[int, ...]]
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Apply sequential centering methods to the input data and return
        the cropped and centered data along with the position of the
        reference voxel in the newly cropped data frame.

        Args:
            data: Input data array.
            output_shape: Desired output shape after cropping.
            methods: list of sequential centering methods. Each method
                can be "max" for maximum intensity, "com" for center of
                mass, or a tuple of coordinates representing the voxel
                position.

        Returns:
            A tuple containing the cropped and centered data array, the
            position of the reference voxel in the original data frame,
            and the position of the reference voxel in the newly cropped
            data frame and the roi as 

        Raises:
            ValueError: If an invalid method is provided.
        """
        # For the first method the data are not masked
        masked_data = data
        position = None
        print("Chain centering:")
        for method in methods:
            # position is found in the masked data
            position = cls.get_position(masked_data, method)
            print(f"\t- {method}: {position}, value: {data[position]}")

            # get the roi
            roi = cls.get_roi(output_shape, position, data.shape)

            # mask the data in the out of roi 
            masked_data = cls.get_masked_data(data, roi=roi)
        if (
                methods[-1] == "com"
                and (position != cls.get_position(masked_data, "com"))
        ):
            warnings.warn(
                "The center of the final box does not correspond to the com.\n"
                "You might want to keep looking for it."
            )
        # actual position along which the data are centered using roi
        position = tuple(
            (start + stop) // 2
            for start, stop in zip(roi[::2], roi[1::2])
        )
        cropped_data = masked_data.data[cls.roi_list_to_slices(roi)]
        cropped_position = tuple(p - r for p, r in zip(position, roi[::2]))
        return cropped_data, position, cropped_position, roi


    @classmethod
    def force_centered_crop(
            cls,
            data: np.ndarray,
            output_shape: tuple,
            where: tuple
    ) -> np.ndarray:
        """
        Crop the data so the given reference position (where) is at
        the center of the final data frame no matter the output_shape.
        Therefore the real output shape might be different to the
        output_shape. 
        """
        position = cls.get_position(data, where)
        shape = data.shape
        safe_shape = shape_for_safe_centered_cropping(
            shape,
            position,
            output_shape
        )
        if safe_shape == output_shape:
            print("Do not require forced-centered cropping.")
        else:
            print(f"Required shape for cropping at the center is {safe_shape}")
        plus_one = np.where((safe_shape % 2 == 0), 0, 1)
        crop = [
            [safe_shape[i]//2, safe_shape[i]//2 + plus_one[i]]
            for i in range(len(safe_shape))
        ]
        roi = []
        for i, s in enumerate(shape):
            roi.append(np.max([where[i]-crop[i][0], 0]))
            roi.append(np.min([where[i]+crop[i][1], s]))

        return data[cls.roi_list_to_slices(roi)]



def compute_corrected_angles(
        inplane_angle: float,
        outofplane_angle: float,
        detector_coordinates: tuple,
        detector_distance: float,
        direct_beam_position: tuple,
        pixel_size: float=55e-6,
        verbose=False
) -> tuple[float, float]:
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
    corrected_outofplane_angle = float(
        outofplane_angle - outofplane_correction
    )

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
        padding: Optional[list]=None,
        symmetrical_shape: Optional[bool]=True
) -> np.array:
    """Find a more suitable shape of an array"""
    if padding is None:
        padding = [4, 4, 4]
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
        sigma_criterion: Optional[float]=3,
        plot: Optional[bool]=False,
        show: Optional[bool]=False
) -> tuple[float, matplotlib.axes.Axes] or float:
    """
    Estimate the isosurface from the amplitude distribution

    :param amplitude: the 3D amplitude volume (np.array)
    :param nbins: the number of bins to considerate when making the
    histogram (Optional, int)
    :param sigma_criterion: the factor to compute the isosurface wich is
    calculated as: mu - sigma_criterion * sigma. By default set to 3.
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

    if plot or show:
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
    return float(isosurface)