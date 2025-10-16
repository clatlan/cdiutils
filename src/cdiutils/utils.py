import inspect
import warnings

import numpy as np
from scipy.fft import fftshift, ifftshift, fftn
import matplotlib
import scipy.constants as cts
from scipy.ndimage import convolve, center_of_mass, median_filter


def bin_along_axis(
    data: np.ndarray | list,
    binning_factor: int,
    binning_method: str = "sum",
    axis: int = 0,
) -> np.ndarray:
    """
    Bin n-dimensional data along a specified axis using the specified
    method (mean, sum, median, max).

    Args:
        data (np.ndarray | list): n-dimensional data to be binned.
        binning_factor (int): the number of elements per bin.
        binning_method (str, optional): the binning method, either
            "mean", "sum", "median", or "max". Defaults to "sum".
        axis (int, optional): the axis along which to perform binning.
            Defaults to 0.

    Raises:
        ValueError: if the binning_method is unknown.

    Returns:
        np.ndarray: binned data.
    """
    if binning_factor == 1 or binning_factor is None:
        return data  # No binning required

    if isinstance(data, list):
        data = np.array(data)
        axis = 0

    original_dim = data.shape[axis]
    nb_of_bins = original_dim // binning_factor
    remaining = original_dim % binning_factor

    # Reshape data for easy binning, ignore leftover data for now.
    # Move the axis to the front for easier manipulation.
    reshaped_data = np.moveaxis(data, axis, 0)
    full_bins_data = reshaped_data[: nb_of_bins * binning_factor].reshape(
        nb_of_bins, binning_factor, *reshaped_data.shape[1:]
    )

    # Get the binning operation
    if binning_method == "mean":
        operation = np.mean
        binned_data = np.mean(full_bins_data, axis=1)
    elif binning_method == "sum":
        operation = np.sum
        binned_data = np.sum(full_bins_data, axis=1)
    elif binning_method == "max":
        operation = np.max
    elif binning_method == "median":
        operation = np.median
    else:
        raise ValueError(f"Unsupported binning method: {binning_method}")

    binned_data = operation(full_bins_data, axis=1)

    # Handle the remaining (leftover data that doesn't fit perfectly
    # into a bin)
    if remaining > 0:
        remaining_data = reshaped_data[-remaining:]
        remaining_bin = operation(remaining_data, axis=0)
        binned_data = np.concatenate(
            [binned_data, np.expand_dims(remaining_bin, axis=0)], axis=0
        )

    # Move the axis back to its original position
    return np.moveaxis(binned_data, 0, axis)


def get_prime_factors(n: int) -> list[int]:
    """Return the prime factors of the given integer."""
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def is_valid_shape(
    n: int, maxprime: int = 13, required_dividers: tuple[int] = (2,)
) -> bool:
    """Check if n meets Pynx shape constraints."""
    factors = get_prime_factors(n)
    return max(factors) <= maxprime and all(
        n % k == 0 for k in required_dividers
    )


def adjust_to_valid_shape(
    n: int,
    maxprime: int = 13,
    required_dividers: tuple[int] = (2,),
    decrease: bool = True,
) -> int:
    """Find the nearest valid shape value."""
    if maxprime < n:
        if n <= 1:
            raise ValueError("n<=1, cannot be adjusted.")
        while not is_valid_shape(n, maxprime, required_dividers):
            n = n - 1 if decrease else n + 1
            if n == 0:
                raise ValueError("No valid shape found")
    return n


def ensure_pynx_shape(
    shape: int | tuple | list | np.ndarray,
    maxprime: int = 13,
    required_dividers: tuple[int] = (2,),
    decrease: bool = True,
    verbose: bool = False,
) -> tuple | np.ndarray | list:
    """
    Ensure shape dimensions comply with Pynx constraints.
    This function has been adapted from the PyNX library. See:
    pynx.utils.math.smaller_primes.

    Args:
        shape (int | tuple | list | np.ndarray): the shape to adjust.
        maxprime (int, optional): the maximum prime factor allowed.
            Defaults to 13.
        required_dividers (tuple, optional): the required dividers.
            Defaults to (4,).
        decrease (bool, optional): whether to decrease the shape.
            Defaults to True.
        verbose (bool, optional): whether to print messages.

    Raises:
        TypeError: if the shape is not an int, list, tuple, or
            numpy array.

    Returns:
        tuple | np.ndarray | list: the adjusted shape.
    """
    if isinstance(shape, int):
        adjusted_shape = adjust_to_valid_shape(
            shape, maxprime, required_dividers, decrease
        )
    elif isinstance(shape, (list, tuple, np.ndarray)):
        adjusted_shape = [
            adjust_to_valid_shape(dim, maxprime, required_dividers, decrease)
            for dim in shape
        ]
        adjusted_shape = (
            np.array(adjusted_shape)
            if isinstance(shape, np.ndarray)
            else type(shape)(adjusted_shape)
        )
    else:
        raise TypeError("Shape must be an int, list, tuple, or numpy array")

    if verbose:
        if adjusted_shape != tuple(shape):
            print(
                f"PyNX needs a different shape to comply with gpu-based FFT, "
                f"requested shape {tuple(shape)} will be cropped to "
                f"{adjusted_shape}."
            )
        else:
            print("Shape already in agreement with pynx shape conventions.")
    return adjusted_shape


def energy_to_wavelength(energy: float) -> float:
    """
    Find the wavelength in metre (not angstrom!) that energy (in eV)
    corresponds to.

    Args:
        energy (float): the energy to convert.

    Returns:
        float: the wavelength in metre.
    """
    return (cts.c * cts.h) / (cts.e * energy)


def wavelength_to_energy(wavelength: float) -> float:
    """
    Find the energy in eV that wavelength in metre (not angstrom!)
    corresponds to.

    Args:
        wavelength (float): the wavelength in metre to convert.

    Returns:
        float: the energy in eV.
    """
    return (cts.c * cts.h) / (cts.e * wavelength)


def fill_up_support(support: np.ndarray) -> np.ndarray:
    """
    Fill up the support using the convex hull of the support.

    Args:
        support (np.ndarray): The binary support array to fill up.

    Returns:
        np.ndarray: The filled support array.
    """
    convex_support = np.zeros(support.shape)

    for axis in range(support.ndim):
        cumulative_sum = np.cumsum(support, axis=axis)
        reversed_cumulative_sum = np.flip(
            np.cumsum(np.flip(support, axis=axis), axis=axis), axis=axis
        )
        combined_support = cumulative_sum * reversed_cumulative_sum
        convex_support[combined_support != 0] = 1

    return convex_support


def size_up_support(support: np.ndarray) -> np.ndarray:
    kernel = np.ones(shape=(3, 3, 3))
    convolved_support = convolve(support, kernel, mode="constant", cval=0.0)
    return np.where(convolved_support > 3, 1, 0)


def find_hull(
    volume: np.ndarray,
    threshold: float = 18,
    kernel_size: int = 3,
    boolean_values: bool = False,
    nan_value: bool = False,
) -> np.ndarray:
    """
    Find the convex hull of a 2D or 3D object.

    :param volume: 2 or 3D np.ndarray. The volume to get the hull from.
    :param threshold: threshold that selects what belongs to the
        hull or not (int). If threshold >= 27, the returned hull will be
        similar to volume.
    :param kernel_size: the size of the kernel used to convolute (int).
    :param boolean_values: whether or not to return 1 and 0 np.ndarray
        or the computed coordination.

    :returns: the convex hull of the shape accordingly to the given
        threshold (np.array).
    """

    kernel = np.ones(shape=tuple(np.repeat(kernel_size, volume.ndim)))
    convolved_support = convolve(volume, kernel, mode="constant", cval=0.0)
    hull = np.where(
        ((0 < convolved_support) & (convolved_support <= threshold)),
        1 if boolean_values else convolved_support,
        np.nan if nan_value else 0,
    )
    return hull


def make_support(
    data: np.ndarray, isosurface: float = 0.5, nan_values: bool = False
) -> np.ndarray:
    """Create a support using the provided isosurface value."""
    data = normalise(data)
    return np.where(data >= isosurface, 1, np.nan if nan_values else 0)


def unit_vector(vector: tuple | list | np.ndarray) -> np.ndarray:
    """Return a unit vector."""
    return np.array(vector) / np.linalg.norm(vector)


def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors."""
    return np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def v1_to_v2_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Rotation matrix around axis v1xv2
    """
    vec_rot_axis = np.cross(v1, v2)
    normed_vrot = unit_vector(vec_rot_axis)

    theta = angle(v1, v2)

    n1, n2, n3 = normed_vrot
    ct = np.cos(theta)
    st = np.sin(theta)

    r = np.array(
        (
            (
                ct + n1**2 * (1 - ct),
                n1 * n2 * (1 - ct) - n3 * st,
                n1 * n3 * (1 - ct) + n2 * st,
            ),
            (
                n1 * n2 * (1 - ct) + n3 * st,
                ct + n2**2 * (1 - ct),
                n2 * n3 * (1 - ct) - n1 * st,
            ),
            (
                n1 * n3 * (1 - ct) - n2 * st,
                n2 * n3 * (1 - ct) + n1 * st,
                ct + n3**2 * (1 - ct),
            ),
        )
    )
    return r


def normalise(data: np.ndarray, zero_centered: bool = False) -> np.ndarray:
    """Normalise a np.ndarray so the values are between 0 and 1."""
    if zero_centered:
        abs_max = np.max([np.abs(np.min(data)), np.abs(np.max(data))])
        vmin, vmax = -abs_max, abs_max
        ptp = vmax - vmin
    else:
        ptp = np.ptp(data)
        vmin = np.min(data)
    return (data - vmin) / ptp


def normalise_complex_array(array: np.ndarray) -> np.ndarray:
    """Normalise a array of complex numbers."""
    shifted_array = array - array.real.min() - 1j * array.imag.min()
    return shifted_array / np.abs(shifted_array).max()


def find_max_pos(data: np.ndarray) -> tuple:
    """Find the index coordinates of the maximum value."""
    return np.unravel_index(data.argmax(), data.shape)


def shape_for_safe_centred_cropping(
    data_shape: tuple | np.ndarray | list,
    position: tuple | np.ndarray | list,
    final_shape: tuple = None,
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
    print((np.array(shape) / 2 == np.array(com)).all())
    com_to_center = np.array(
        [int(np.rint(shape[i] / 2 - com[i])) for i in range(3)]
    )
    if (com_to_center == np.array((0, 0, 0)).astype(int)).all():
        return data, com
    data = center(data, where=com)
    return _center_at_com(data)


def center(
    data: np.ndarray,
    where: str | tuple | list | np.ndarray = "com",
    return_former_center: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple]:
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
        centered_data, int(np.rint(shape[1] / 2 - ycenter)), axis=1
    )
    centered_data = np.roll(
        centered_data, int(np.rint(shape[2] / 2 - zcenter)), axis=2
    )

    if return_former_center:
        return centered_data, (xcenter, ycenter, zcenter)

    return centered_data


def symmetric_pad(
    data: np.ndarray,
    output_shape: tuple | list | np.ndarray,
    values: float = 0,
) -> np.ndarray:
    """Return padded data so it matches the provided final_shape"""

    if data.ndim != len(output_shape):
        raise ValueError(
            f"output_shape length ({len(output_shape)}) should match of input "
            f"data dimension ({data.ndim})."
        )
    pad_widths = []
    for current_s, output_s in zip(data.shape, output_shape):
        if output_s < current_s:
            pad_widths.append((0, 0))
        else:
            pad_left = (output_s - current_s) // 2
            pad_right = pad_left + (output_s - current_s) % 2
            pad_widths.append((pad_left, pad_right))
    return np.pad(
        data, pad_width=pad_widths, mode="constant", constant_values=values
    )


def crop_at_center(
    data: np.ndarray, final_shape: list | tuple | np.ndarray
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
        c[0] - to_crop[0] : c[0] + to_crop[0] + plus_one[0],
        c[1] - to_crop[1] : c[1] + to_crop[1] + plus_one[1],
        c[2] - to_crop[2] : c[2] + to_crop[2] + plus_one[2],
    ]

    return cropped


def compute_distance_from_com(
    data: np.ndarray, com: tuple | list | np.ndarray = None
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

    for x, y, z in zip(
        nonzero_coordinates[0], nonzero_coordinates[1], nonzero_coordinates[2]
    ):
        distance = np.sqrt(
            (x - com[0]) ** 2 + (y - com[1]) ** 2 + (z - com[2]) ** 2
        )
        distance_matrix[x, y, z] = distance

    return distance_matrix


def num_to_nan(data: np.ndarray, num: int | float = 0):
    """
    Replace all occurrences of 'num' in the array with np.nan.

    Args:
        data (np.ndarray): NumPy array.
        num (int | float, optional): The number to replace with np.nan
            Defaults to 0.

    Returns:
        A new NumPy array with 'num' replaced by np.nan.
    """
    return np.where(data == num, np.nan, data)


def zero_to_nan(data: np.ndarray, boolean_values: bool = False) -> np.ndarray:
    """Convert zero values to np.nan."""
    return np.where(data == 0, np.nan, 1 if boolean_values else data)


def nan_to_zero(data: np.ndarray, boolean_values: bool = False) -> np.ndarray:
    """Convert np.nan values to 0."""
    return np.where(np.isnan(data), 0, 1 if boolean_values else data)


def to_bool(data: np.ndarray, nan_value: bool = False) -> np.ndarray:
    """Convert values to 1 (True) if not nan otherwise to 0 (False)"""
    return np.where(np.isnan(data), np.nan if nan_value else 0, 1)


def nan_center_of_mass(
    data: np.ndarray, return_int: bool = False
) -> np.ndarray:
    """
    Compute the center of mass of a np.ndarray that may contain
    nan values.
    """
    if not np.isnan(data).any():
        com = center_of_mass(data)

    non_nan_coord = np.where(np.invert(np.isnan(data)))
    com = np.average([non_nan_coord], axis=2, weights=data[non_nan_coord])[0]
    if return_int:
        return tuple([int(round(e)) for e in com])
    return tuple(com)


def hybrid_gradient(
    data: np.ndarray,
    *d: list,
) -> list[np.ndarray] | np.ndarray:
    """
    Compute the gradient of a n-dim volume in each axis direction, 2nd order
    in the interior of the non-nan object, 1st order at the interface between
    the non-nan object and the surrounding nan values.

    Args:
        data (np.ndarray): the n-dim volume i to be derived
        d (list): the spacing in each
            direction

    Returns:
        list[np.ndarray] | np.ndarray: the gradients (in each direction) with
        the same shape as the input data
    """

    if isinstance(d, (int, float)):
        d = [d]
    if data.ndim != len(d):
        raise ValueError(f"Invalid shape for d ({d}), must match data.ndim")
    gradient = []
    for i in range(data.ndim):
        upper_slice = [slice(None)] * data.ndim
        upper_slice[i] = np.s_[1:]
        upper_slice = tuple(upper_slice)
        lower_slice = [slice(None)] * data.ndim
        lower_slice[i] = np.s_[:-1]
        lower_slice = tuple(lower_slice)

        # compute the first order gradient
        gradient.append((data[upper_slice] - data[lower_slice]) / d[i])

        # some warning is expecting here as mean of empty slices might occur
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # here is the trick, using the np.nanmean allows keeping the
            # first order derivative at the interface. The other values
            # correspond to second order gradient
            gradient[i] = np.nanmean(
                [
                    gradient[i][upper_slice],
                    gradient[i][lower_slice],
                ],
                axis=0,
            )
            pad = [(0, 0) for _ in range(data.ndim)]
            pad[i] = (1, 1)
            gradient[i] = np.pad(gradient[i], pad, constant_values=np.nan)
    if len(gradient) == 1:
        return gradient[0]
    return gradient


class CroppingHandler:
    """
    A class to handle data cropping. The class allows
    finding the requested position of the center and crop the data
    accordingly.
    """

    @staticmethod
    def get_position(data: np.ndarray, method: str | tuple[int]) -> tuple[int]:
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
        # if the data is complex, we take the absolute value
        if np.iscomplexobj(data):
            data = np.abs(data)
        if method == "max":
            return np.unravel_index(np.argmax(data), data.shape)
        elif method == "com":
            if isinstance(data, np.ma.MaskedArray):
                # data must be filled to account for the mask values
                data = data.filled(0)
            com = center_of_mass(data)
            return tuple(np.nan if np.isnan(e) else int(round(e)) for e in com)
        elif isinstance(method, (list, tuple)) and all(
            isinstance(e, (int, np.int64)) for e in method
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
                "ROI should have start and end values for each dimension."
            )
        return tuple(
            slice(start, end) for start, end in zip(roi[::2], roi[1::2])
        )

    @classmethod
    def get_roi(
        cls, output_shape: tuple, where: tuple, input_shape: tuple = None
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
        crop = [
            [e // 2, e // 2 + plus_one[i]] for i, e in enumerate(output_shape)
        ]
        roi = []

        if input_shape is None:
            for i in range(len(output_shape)):
                roi.append(where[i] - crop[i][0])
                roi.append(where[i] + crop[i][1])
            return roi

        for i, s in enumerate(input_shape):
            # extend the roi to comply with the output_shape
            add_left = (
                where[i] + crop[i][1] - s if where[i] + crop[i][1] > s else 0
            )
            add_right = (
                crop[i][0] - where[i] if where[i] - crop[i][0] < 0 else 0
            )

            roi.append(np.max([where[i] - crop[i][0], 0]) - add_left)
            roi.append(np.min([where[i] + crop[i][1], s]) + add_right)
        # for i in range(0, len(roi), 2):
        #     if roi[i] < 0:
        #         warnings.warn(
        #             f"The calculated roi contains a negative value "
        #             "({roi[i]}), will set it to 0, this might give "
        #             "inconsistent results."
        #         )
        #         roi[i+1] -= roi[i]
        #         roi[i] = 0
        return roi

    @classmethod
    def chain_centring(
        cls,
        data: np.ndarray,
        output_shape: tuple[int, ...],
        methods: list[str | tuple[int, ...]],
        verbose: bool = False,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Apply sequential centring methods to the input data and return
        the cropped and centred data along with the position of the
        reference voxel in the newly cropped data frame.

        Args:
            data: Input data array.
            output_shape: Desired output shape after cropping.
            methods: list of sequential centring methods. Each method
                can be "max" for maximum intensity, "com" for center of
                mass, or a tuple of coordinates representing the voxel
                position.
            verbose: (bool, optional) whether to print out messages.

        Returns:
            A tuple containing the cropped and centered data array, the
            position of the reference voxel in the original data frame,
            and the position of the reference voxel in the newly cropped
            data frame and the roi.

        Raises:
            ValueError: If an invalid method is provided.
        """
        # For the first method the data are not masked
        masked_data = data
        position = None
        msg = ""
        for method in methods:
            # position is found in the masked data
            position = cls.get_position(masked_data, method)
            msg += f"\t- {method}: {position}, value: {data[position]}\n"

            # get the roi
            roi = cls.get_roi(output_shape, position, data.shape)

            # mask the data values which are outside roi
            masked_data = cls.get_masked_data(data, roi=roi)
        if verbose:
            print("Chain centring:\n" + msg)

        # actual position along which the data are centered using roi
        position = tuple(
            (start + stop) // 2 for start, stop in zip(roi[::2], roi[1::2])
        )
        cropped_data = data[cls.roi_list_to_slices(roi)]
        cropped_position = tuple(p - r for p, r in zip(position, roi[::2]))
        return cropped_data.copy(), position, cropped_position, roi

    @classmethod
    def force_centred_cropping(
        cls,
        data: np.ndarray,
        where: str | tuple = "centre",
        output_shape: tuple = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Crop the data so the given reference position (where) is at
        the center of the final data frame no matter the output_shape.
        Therefore the real output shape might be different to
        output_shape.
        """
        if output_shape is None:
            output_shape = data.shape
        output_shape = np.array(output_shape)

        if where == "centre":
            where = tuple(e // 2 for e in data.shape)

        position = cls.get_position(data, where)
        shape = data.shape
        safe_shape = np.array(
            shape_for_safe_centred_cropping(shape, position, output_shape)
        )
        if verbose:
            print(f"Safe shape for cropping: {tuple(safe_shape)}")
            if np.all(safe_shape == output_shape):
                print("Does not require forced-centered cropping.")
            else:
                print(
                    "Required shape for cropping at the center is "
                    f"{tuple(safe_shape)}."
                )
        plus_one = np.where((safe_shape % 2 == 0), 0, 1)
        crop = [
            [safe_shape[i] // 2, safe_shape[i] // 2 + plus_one[i]]
            for i in range(len(safe_shape))
        ]
        roi = []
        for i, s in enumerate(shape):
            roi.append(np.max([position[i] - crop[i][0], 0]))
            roi.append(np.min([position[i] + crop[i][1], s]))

        return data[cls.roi_list_to_slices(roi)]


def compute_corrected_angles(
    inplane_angle: float,
    outofplane_angle: float,
    detector_coordinates: tuple,
    detector_distance: float,
    direct_beam_position: tuple,
    pixel_size: float = 55e-6,
    verbose=False,
) -> tuple[float, float]:
    """
    Compute the detector corrected angles given the angles saved in the
    experiment data file and the position of interest in the detector frame

    :param inplane_angle: in-plane detector angle in degrees (float).
    :param outofplane_angle: out-of-plane detector angle in degrees
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
    support: np.ndarray, pad: list = None, symmetrical: bool = True
) -> np.ndarray:
    """Find a smaller array shape of 2 or 3D array containing a support."""
    if pad is None:
        pad = np.repeat(4, support.ndim)
    pad = np.array(pad)

    if support.sum() <= 0:
        raise ValueError("support must contain 0 and 1.")

    def get_2d_shape(support_2d, pad):
        shape = []
        for k in range(2):
            lim = np.nonzero(support_2d.sum(axis=1 - k))[0][[0, -1]]
            s = np.ptp(lim) + pad[k] * 2
            shape.append(s)
        return shape

    shape = []
    if support.ndim == 3:
        shapes_2d = []
        for i in range(3):
            axes = tuple(k for k in range(3) if k != i)
            shapes_2d.append(
                get_2d_shape(support.sum(axis=i), np.squeeze(pad[[axes]]))
            )
        shape = (shapes_2d[1][0], shapes_2d[0][0], shapes_2d[0][1])
    elif support.ndim == 2:
        if len(pad) != 2:
            raise ValueError(
                f"Length of pad ({len(pad) = }) must be equal to support "  # noqa E251
                f"dimensions ({support.ndim = })"  # noqa E251
            )
        shape = get_2d_shape(support, pad)

    if symmetrical:
        return tuple(np.repeat(np.max(shape), support.ndim))
    return shape


def extract_reduced_shape(
    support: np.ndarray,
    pad: tuple | list | np.ndarray = None,
    symmetric: bool = False,
) -> tuple:
    if pad is None:
        pad = np.array([-10, 10])

    support_limits = []
    for i in range(support.ndim):
        limit = np.nonzero(support.sum(axis=i))[0][[0, -1]]
        limit += pad  # padding
        support_limits.append(limit)
    shape = [np.ptp(limit) for limit in support_limits]
    if symmetric:
        return tuple(np.repeat(np.max(shape), support.ndim))
    return shape


def get_oversampling_ratios(
    support: np.ndarray = None,
    direct_space_object: np.ndarray = None,
    isosurface: float = 0.3,
    plot: bool = False,
) -> np.ndarray:
    """
    Compute the oversampling ratio of a reconstruction.
    Function proposed by Ewen Bellec (ewen.bellec@esrf.fr)

    Args:
        support (np.ndarray, optional): the support of the
        reconstruction. Defaults to None.
        direct_space_object (np.ndarray, optional): the reconstructed
        object. Defaults to None.
        isosurface (float, optional): the isosurface to determine the
        support. Defaults to .3.
        plot (bool, optional):  whether to plot or not. Defaults to False.

    Raises:
        ValueError: If support is not provided, requires
        direct_space_object and isosurface (default to 0.3) value.

    Returns:
        np.ndarray: the oversampling ratio.
    """
    if support is None:
        if direct_space_object is None:
            raise ValueError(
                "If support is not provided, provide direct_space_object and "
                "isosurface (default to 0.3) value"
            )
        support = make_support(np.abs(direct_space_object), isosurface)
    if support.sum() < 1:
        return None
    support_indices = np.where(support == 1)
    size_per_dim = np.max(support_indices, axis=1) - np.min(
        support_indices, axis=1
    )
    oversampling_ratio = np.divide(np.array(support.shape), size_per_dim)

    if plot:
        _, ax = matplotlib.pyplot.subplots(
            1, support.ndim, figsize=(5 * support.ndim, 4)
        )
        for n in range(support.ndim):
            axes = tuple(np.delete(np.arange(3), n))
            proj = np.max(support, axis=axes)
            ax[n].plot(proj)
            title = (
                "oversampling along axis {n}\n{round(oversampling_ratio[n],2)}"
            )
            ax[n].set_title(title, fontsize=15)

    return oversampling_ratio


def oversampling_from_diffraction(
    data: np.ndarray,
    support_threshold: float = 0.1,
) -> np.ndarray:
    """
    Compute the oversampling ratios from diffraction data.
    Autocorrelation of the intensity is calculated to generate a support
    using support_threshold (isosurface).

    Args:
        data (np.ndarray): intensity (diffracted data)
        support_threshold (0.1): the threshold to build the support from
         a la pynx, default to 0.1.

    Returns:
        tuple[np.ndarray, np.ndarray]: the oversampling ratio and a
            suggestion for the rebin factors
    """
    autocorrelation = np.abs(ifftshift(fftn(fftshift(data))))
    support = autocorrelation > support_threshold * np.max(autocorrelation)

    oversampling_ratio = get_oversampling_ratios(support=support)

    return oversampling_ratio


def get_centred_slices(
    shape: tuple | list | np.ndarray, shift: tuple | list = None
) -> list:
    """
    Compute the slices that allows to select the centre of each axis. It
    returns a list of len(shape) tuples made of len(shape) slices. The
    shift allows to shift the centred the slices by the amount provided.

    Ex:

    * if shape = (25, 48), will return::

        [(12, slice(None, None, None)), (slice(None, None, None), 24)]

    * if shape = (25, 48, 50), will return::

        [(12, slice(None, None, None), slice(None, None, None)),
         (slice(None, None, None), 24, slice(None, None, None)),
         (slice(None, None, None), slice(None, None, None), 25)]

    Args:
        shape (tuple | list | np.ndarray): the shape of the np.ndarray
            of which you want to select the centre.
        shift (tuple | list): a shift in the slice selection with
            respect to the centre.

    Returns:
        list: the list of tuples made of slices.
    """
    if shift is None:
        shift = (0,) * len(shape)
    slices = []
    for i, element in enumerate(shape):
        s = [slice(None)] * len(shape)
        s[i] = element // 2 + shift[i]
        slices.append(tuple(s))
    return slices


def hot_pixel_filter(
    data: np.ndarray, threshold: float = 1e2, kernel_size: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove hot pixels using a median filter.

    Args:
        data (np.ndarray): the input data.
        threshold (float, optional): the threshold to determine the
            mask. Mask pixels that are threshold times higher than
            neighbouring pixels. Defaults to 1e2.
        kernel_size (int, optional): the size of the kernel to compute
            the median filter. It corresponds to the size parameter of
            scipy.ndimage.median_filter function. Defaults to 3.

    Returns:
        tuple[np.ndarray, np.ndarray]: the cleaned data, hot pixel are
            set to 0 and the mask (1 for hot pixel, 0 otherwise).
    """
    data_median = median_filter(data, size=kernel_size)
    mask = data < threshold * (data_median + 1)
    cleaned_data = data * mask
    return cleaned_data, np.logical_not(mask)


def valid_args_only(params: dict, function: callable) -> dict:
    return {
        k: v
        for k, v in params.items()
        if k in inspect.getfullargspec(function).args
    }
