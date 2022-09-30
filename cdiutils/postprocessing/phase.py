import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from sklearn.linear_model import LinearRegression

from cdiutils.utils import crop_at_center, symmetric_pad


def remove_phase_ramp(phase, return_ramp=False):
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
    if return_ramp:
        return phase - ramp, ramp
    return phase - ramp


def make_blackman_window(shape: tuple, normalization: float=1):
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


def blackman_apodize(direct_space_data, initial_shape):
    """
    Apodize in the Fourrier space a 3D volume in the direct space

    :param direct_space_data: the 3D volume of complex values to apodize
    (np.ndarray)
    :param initial_shape: the shape that had the object before cropping.
    The direct_space_data will be Fourrier transformed and the shape of
    the its initial corresponding Fourrier transfrom is better to get a
    relevant blackman winow.

    :return: the apodized data in the direct space with the same input
    shape
    """
    current_shape = direct_space_data.shape
    padded_data = symmetric_pad(direct_space_data, final_shape=initial_shape)

    q_space_data = fftshift(fftn(padded_data))
    
    blackman_window = make_blackman_window(initial_shape)

    q_space_data = q_space_data * blackman_window

    return crop_at_center(ifftn(ifftshift(q_space_data)), current_shape)


def flip_reconstruction(data):
    """
    Flip a direct space reconstruction.

    :param data: the 3D direct space reconstruction

    :return: the flipped reconstruction
    """

    return ifftn(ifftshift(np.conj(fftshift(fftn(data)))))



def gradient(data, dx, dy, dz):
    """
    Compute the gradient of a 3D volume in the 3 directions.

    :param data: the 3D volume to be derived (3D np.ndarray)
    :param dx: the spacing in the x direction (axis 0)
    :param dy: the spacing in the y direction (axis 1)
    :param dz: the spacing in the z direction (axis 2)

    :return: a tupple of each gradient (for each direction) with the
    same shape as the input data
    """

    grad_x = (data[1:, ...] - data[:-1, ...]) / dx
    grad_y = (data[:, 1:, :] - data[:, :-1, :]) / dy
    grad_z = (data[..., 1:] - data[..., :-1]) / dz
    
    grad_x = np.nanmean([grad_x[1:], grad_x[:-1]], axis=0) 
    grad_y = np.nanmean([grad_y[: ,1:, :], grad_y[:, :-1, :]], axis=0) 
    grad_z = np.nanmean([grad_z[..., 1:], grad_z[..., :-1]], axis=0) 


    return (
        np.pad(grad_x, ((1, 1),  (0, 0), (0, 0)), constant_values=np.nan),
        np.pad(grad_y, ((0, 0),  (1, 1), (0, 0)), constant_values=np.nan),
        np.pad(grad_z, ((0, 0),  (0, 0), (1, 1)), constant_values=np.nan)
    )