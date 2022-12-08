"""Utility functions for phase processing in a BCDI framework.


Author:
    Clément Atlan - 12.08.2022
"""

import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from skimage.restoration import unwrap_phase
from sklearn.linear_model import LinearRegression
from typing import Optional, Union, Tuple
import warnings

from cdiutils.utils import (
    crop_at_center, symmetric_pad, normalize, make_support, nan_to_zero,
    zero_to_nan, find_suitable_array_shape, center
)


def remove_phase_ramp(phase: np.array) -> Tuple[np.array, np.array]:
    """
    Remove the phase ramp of a 3D volume phase.                                                                                                                                                              

    :param phase: the 3D volume phase (np.ndarray)
    :return_ramp: whether to return the 3D phase ramp (bool)

    :return: the phase without the computed ramp. Return the ramp if 
    return_ramp is true.
    """

    x, y, z = np.indices(phase.shape)
    non_nan_coordinates = np.where(np.logical_not(np.isnan(phase)))

    non_nan_phase = phase[non_nan_coordinates]
    x = x[non_nan_coordinates]
    y = y[non_nan_coordinates]
    z = z[non_nan_coordinates]

    X = np.swapaxes(np.array([x, y, z]), 0, 1)
    reg = LinearRegression().fit(X, non_nan_phase)

    x, y, z = np.indices(phase.shape)

    ramp = (
        reg.coef_[0] * x
        + reg.coef_[1] * y
        + reg.coef_[2] * z
        + reg.intercept_
    )

    return phase - ramp, ramp


def make_blackman_window(shape: tuple, normalization: float=1) -> np.array:
    """
    Make a 3D Blackman window of a given shape.

    :param shape: shape of the 3D window (tuple)
    :param normalization: value of the integral of the backman window
    
    :return: the 3D Blackman window
    """
    nx, ny, nz = shape 

    blackman_x = np.blackman(nx)
    blackman_y = np.blackman(ny)
    blackman_z = np.blackman(nz)

    blackman_slice = np.ones((nx, ny))
    blackman_cube = np.ones((nx, ny, nz))

    for x in range(nx):
        blackman_slice[x, :] = blackman_x[x] * blackman_y
        for y in range(ny):
            blackman_cube[x ,y] = blackman_slice[x, y] * blackman_z
    return blackman_cube / blackman_cube.sum() * normalization


def blackman_apodize(
        direct_space_data: np.array,
        initial_shape: Union[np.array, list, tuple]
) -> np.array:
    """
    Apodize in the Fourrier space a 3D volume in the direct space

    :param direct_space_data: the 3D volume of complex values to apodize
    (np.ndarray)
    :param initial_shape: the shape that had the object before cropping.
    The direct_space_data will be Fourrier transformed and the shape of
    the its initial corresponding Fourrier transform is better to get a
    relevant blackman winow.

    :return: the apodized data in the direct space with the same input
    shape
    """
    current_shape = direct_space_data.shape
    padded_data = symmetric_pad(direct_space_data, final_shape=initial_shape)

    q_space_data = ifftshift(fftn(fftshift(padded_data)))
    
    blackman_window = make_blackman_window(initial_shape)

    q_space_data = q_space_data * blackman_window

    return crop_at_center(
        ifftshift(ifftn(fftshift(q_space_data))), current_shape
    )


def flip_reconstruction(data: np.array) -> np.array:
    """
    Flip a direct space reconstruction.

    :param data: the 3D direct space reconstruction

    :return: the flipped reconstruction
    """

    return ifftshift(ifftn(np.conj(fftn(fftshift(data)))))



def hybrid_gradient(
        data: np.array,
        dx: float,
        dy: float,
        dz: float
) -> np.array:
    """
    Compute the gradient of a 3D volume in the 3 directions, 2 nd order 
    in the interior of the non-nan object, 1 st order at the interface between
    the non-nan object and the surrounding nan values.

    :param data: the 3D volume to be derived (3D np.ndarray)
    :param dx: the spacing in the x direction (axis 0)
    :param dy: the spacing in the y direction (axis 1)
    :param dz: the spacing in the z direction (axis 2)

    :return: a tuple, the three gradients (in each direction) with the
    same shape as the input data
    """

    # compute the first order gradient
    grad_x = (data[1:, ...] - data[:-1, ...]) / dx
    grad_y = (data[:, 1:, :] - data[:, :-1, :]) / dy
    grad_z = (data[..., 1:] - data[..., :-1]) / dz

    # some warning is expecting here as mean of empty slices might occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # here is the trick, using the np.nanmean allows keeping the 
        # first order derivative at the interface. The other values 
        # correspond to second order gradient
        grad_x = np.nanmean([grad_x[1:], grad_x[:-1]], axis=0)
        grad_y = np.nanmean([grad_y[: ,1:, :], grad_y[:, :-1, :]], axis=0)
        grad_z = np.nanmean([grad_z[..., 1:], grad_z[..., :-1]], axis=0)


    return (
        np.pad(grad_x, ((1, 1),  (0, 0), (0, 0)), constant_values=np.nan),
        np.pad(grad_y, ((0, 0),  (1, 1), (0, 0)), constant_values=np.nan),
        np.pad(grad_z, ((0, 0),  (0, 0), (1, 1)), constant_values=np.nan)
    )

def get_structural_properties(
        complex_object: np.array,
        isosurface: float,
        q_vector: float,
        hkl: Union[tuple, list],
        voxel_size: Union[np.array, tuple, list],
        final_shape: Optional[Union[np.array, list, tuple]]=None,
        normalize_amplitude: bool=True,
) -> dict:

    """
    Process the phase to get the structural properies of the 
    constructed nanocrystal

    :param complex_object: the reconstructed complex object
    (np.nrdarray)
    :param isosurface: the isosurface amplitude threshold to determine
    the surface
    :param qnorm: the norm of the q vector (float)
    :param hkl: the indexes of the probed Bragg peak (tuple, list)
    :voxe_size: the voxe size in nm ( np.array, tuple, list)
    :final_shape: the shape of the processed output arrays
    (np.array, list, tuple)
    :normalize_amplitude: wheter or not to normalize the amplitude in
    the final output (bool)

    :return: a dictionary of the following 3D arrays:
        - amplitude
        - support
        - phase
        - displacement
        - local_strain
        - numpy_local_strain
        - dspacing
        - lattice_constant
    """

    amplitude = np.abs(complex_object)
    phase = np.angle(complex_object) * (-1) # because it came out of pynx


    support = make_support(
        normalize(amplitude),
        isosurface=isosurface,
        nan_values=False
    )

    # center the arrays at the center of mass of the support
    support, c = center(support, method="com", return_former_center=True)
    amplitude  = center(amplitude, center_coordinates=c)
    phase = center(phase, center_coordinates=c)


    # if final_shape is not provided, find one
    if final_shape is None:
        print("[INFO] finding a new array shape")
        final_shape = find_suitable_array_shape(support)
        print(f"[INFO] new array shape is {final_shape}")

    print("[PROCESSING] Cropping the data: ", end="")
    # crop the data to the final shape
    amplitude = crop_at_center(amplitude, final_shape)
    support = crop_at_center(support, final_shape)
    phase = crop_at_center(phase, final_shape)
    print("done.")

    # process the phase
    print("[PROCESSING] Unwrapping the phase: ", end="")
    mask = np.where(support == 0, 1, 0)
    masked_phase = np.ma.masked_array(phase, mask=mask)
    unwrapped_phase = unwrap_phase(
        masked_phase,
        wrap_around=False,
        seed=1
    ).data
    print("done.")

    # create a nan support
    nan_support =  zero_to_nan(support)
    # remove the phase ramp
    print("[PROCESSING] Removing the phase ramp: ", end="")
    _, ramp = remove_phase_ramp(unwrapped_phase * nan_support)
    no_ramp_phase = unwrapped_phase - ramp
    print("done.")

    # set the origin of the phase to the phase mean
    print(
        "[PROCESSING] Setting the phase origin to the mean value: ",
        end=""
    )
    mean_centered_phase = no_ramp_phase - np.nanmean(
        no_ramp_phase * nan_support)
    print("done.")
    

    print(
        "[PROCESSING] Computing the strain, displacement, dspacing and "
        "lattice constant: ",
        end=""
    )
    q_norm = np.linalg.norm(q_vector)
    displacement = mean_centered_phase / q_norm

    dx, dy, dz = voxel_size # voxel size in nm

    # compute the strain along the axis 1
    _, numpy_local_strain, _, = np.gradient(displacement * 1e-1, dx, dy, dz)
    _, local_strain, _, = hybrid_gradient(
        nan_support * displacement * 1e-1, dx, dy, dz
    )

    # convert the strain in percent
    local_strain = 100 * nan_to_zero(local_strain)
    numpy_local_strain *= 100

    # compute the dspacing and lattice_constant
    dspacing =  2*np.pi / q_norm * (1 + local_strain/100)
    lattice_constant =  np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2) *dspacing

    print("done.")


    return {
        "amplitude": normalize(amplitude) 
            if normalize_amplitude else amplitude,
        "support": support,
        "phase": mean_centered_phase,
        "displacement": displacement,
        "local_strain": nan_to_zero(local_strain),
        "numpy_local_strain": numpy_local_strain,
        "dspacing": dspacing,
        "lattice_constant": lattice_constant,
        "hkl": hkl,
        "q_vector": q_vector,
        "q_norm": q_norm,
        "voxel_size": voxel_size,
    }