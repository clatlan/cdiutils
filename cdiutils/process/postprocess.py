"""Utility functions for phase processing in a BCDI framework.


Author:
    Clément Atlan - 27.10.2023
"""

import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from skimage.restoration import unwrap_phase
from sklearn.linear_model import LinearRegression

from cdiutils.utils import (
    CroppingHandler,
    find_suitable_array_shape,
    zero_to_nan,
    nan_to_zero,
    normalize,
    make_support,
    hybrid_gradient
)


class PostProcessor:
    """
    A class to bundle all functions needed to post-process BCDI data.
    """

    @staticmethod
    def prepare_volume(
            complex_object: np.ndarray,
            isosurface,
            final_shape: np.ndarray or tuple or list = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the volume by finding a smaller array shape, centering
        at the center of mass of the support, and cropping

        Args:
            complex_object (np.ndarray): the complex object
            (rho e^{i phi})
            isosurface (bool): the isosurface that determines the
            support
            final_shape (np.ndarray or tuple or list, optional): the
            final shape of the array requested. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: the cropped complex_object
            and the associated support.
        """

        support = make_support(
            normalize(np.abs(complex_object)),
            isosurface=isosurface,
            nan_values=False
        )
        if final_shape is None:
            final_shape = find_suitable_array_shape(support, padding=[6, 6, 6])
            print(f"[INFO] new array shape is {final_shape}")
        # center the arrays at the center of mass of the support
        com = CroppingHandler.get_position(support, "com")
        complex_object = CroppingHandler.force_centered_cropping(
            complex_object,
            where=com,
            output_shape=final_shape
        )
        support = CroppingHandler.force_centered_cropping(
            support,
            where=com,
            output_shape=final_shape
        )
        return complex_object, support

    @staticmethod
    def flip_reconstruction(data: np.ndarray) -> np.ndarray:
        """
        Flip a direct space reconstruction.

        Args:
            data (np.ndarray): the 3D direct space reconstruction

        Returns:
            np.ndarray: the flipped reconstruction
        """
        return ifftshift(ifftn(np.conj(fftn(fftshift(data)))))

    @staticmethod
    def apodize(
            direct_space_data: np.ndarray,
            scale: float = 1
    ) -> np.ndarray:
        """
        Apodization on the direct space data using Blackman window.

        Args:
            direct_space_data (np.ndarray): the 3D volume data to
            apodize.
            scale (float, optional): value of the integral of the
            Blackman window. Defaults to None.

        Returns:
            np.ndarray: _description_
        """

        # first make the Blackman window
        ni, nj, nk = direct_space_data.shape
        blackman_i = np.blackman(ni)
        blackman_j = np.blackman(nj)
        blackman_k = np.blackman(nk)
        blackman_slice = np.ones((ni, nj))
        blackman_cube = np.ones((ni, nj, nk))

        for i in range(ni):
            blackman_slice[i, :] = blackman_i[i] * blackman_j
            for j in range(nj):
                blackman_cube[i, j] = blackman_slice[i, j] * blackman_k
        blackman_window = blackman_cube / blackman_cube.max() * scale

        q_space_data = ifftshift(fftn(fftshift(direct_space_data)))
        q_space_data = q_space_data * blackman_window

        return ifftshift(ifftn(fftshift(q_space_data)))

    @staticmethod
    def unwrap_phase(
            phase: np.ndarray,
            support: np.ndarray = None
    ) -> np.ndarray:
        """
        Unwrap phase for voxels that belong to the given support.

        Args:
            phase (np.ndarray): the phase to unwrap
            support (np.ndarray): the support where voxels of interest are

        Returns:
            np.ndarray: the unwrapped phase
        """
        if support is None:
            return unwrap_phase(
                phase,
                wrap_around=False,
                seed=1
            ).data
        support = nan_to_zero(support)
        mask = np.where(support == 0, 1, 0)
        phase = np.ma.masked_array(phase, mask=mask)
        return unwrap_phase(
            phase,
            wrap_around=False,
            seed=1
        ).data

    @staticmethod
    def remove_phase_ramp(phase: np.ndarray) -> np.ndarray:
        """
        Remove the phase ramp of a 3D volume phase.

        Args:
            phase (np.ndarray): the 3D volume phase

        Returns:
            np.ndarray: the phase without the computed ramp.
        """
        i, j, k = np.indices(phase.shape)
        non_nan_coordinates = np.where(np.logical_not(np.isnan(phase)))

        non_nan_phase = phase[non_nan_coordinates]
        i = i[non_nan_coordinates]
        j = j[non_nan_coordinates]
        k = k[non_nan_coordinates]

        indices = np.swapaxes(np.array([i, j, k]), 0, 1)
        reg = LinearRegression().fit(indices, non_nan_phase)

        i, j, k = np.indices(phase.shape)

        ramp = (
            reg.coef_[0] * i
            + reg.coef_[1] * j
            + reg.coef_[2] * k
            + reg.intercept_
        )

        return phase - ramp

    @staticmethod
    def phase_offset_to_zero(
            phase: np.ndarray,
            support: np.ndarray = None,
    ) -> np.ndarray:
        """
        Set the phase offset to the mean phase value.
        """
        return phase - np.nanmean(phase * support if support else 1)

    @staticmethod
    def get_displacement(
            phase: np.ndarray,
            g_vector: np.ndarray or tuple or list,
    ) -> np.ndarray:
        """
        Calculate the displacement from phase and g_vector.
        """
        return phase / np.linalg.norm(g_vector)

    @staticmethod
    def get_displacement_gradient(
            displacement: np.ndarray,
            voxel_size: np.ndarray or tuple or list,
            gradient_method: str = "hybrid"
    ) -> np.ndarray:
        """
        Calculate the gradient of the displacement.

        Args:
            displacement (np.ndarray): displacement array.
            voxel_size (np.ndarray or tuple or list): the voxel size of
            the array.
            gradient_method (str, optional): the method employed to
            compute the gradient. "numpy" is the traditional gradient.
            "hybrid" compute first order gradient at the surface and
            second order within the bulk of the reconstruction.
            Defaults to "hybrid".

        Raises:
            ValueError: If parsed method is unknown.

        Returns:
            np.ndarray: the gradient of the volume in the three
            directions.
        """
        if gradient_method == "numpy":
            grad_function = np.gradient
        elif gradient_method in ("hybrid", "h"):
            grad_function = hybrid_gradient
        else:
            raise ValueError("Unknown method for normal strain computation.")
        return grad_function(displacement, *voxel_size)

    @classmethod
    def get_het_normal_strain(
            cls,
            displacement: np.ndarray,
            g_vector: np.ndarray or tuple or list,
            voxel_size: np.ndarray or tuple or list,
            gradient_method: str = "hybrid",
    ) -> np.ndarray:
        """
        Compute the heterogeneous normal strain, i.e. the gradient of
        the displacement projected along the measured Bragg peak
        direction.

        Args:
            displacement (np.ndarray): the displacement array
            g_vector (np.ndarray or tuple or list): the position of the
            measured Bragg peak (com or max of the intensity).
            voxel_size (np.ndarray or tuple or list): voxel size of the
            array
            gradient_method (str, optional): the method employed to
            compute the gradient. "numpy" is the traditional gradient.
            "hybrid" compute first order gradient at the surface and
            second order within the bulk of the reconstruction.
            Defaults to "hybrid".

        Returns:
            np.ndarray: the heterogeneous normal strain
        """
        displacement_gradient = cls.get_displacement_gradient(
            displacement,
            voxel_size,
            gradient_method
        )
        displacement_gradient = np.moveaxis(
            np.asarray(displacement_gradient),
            source=0,
            destination=3
        )
        return np.dot(
            displacement_gradient,
            g_vector / np.linalg.norm(g_vector)
        )

    @classmethod
    def get_structural_properties(
            cls,
            complex_object: np.ndarray,
            isosurface: np.ndarray,
            g_vector: np.ndarray or tuple or list,
            hkl: tuple or list,
            voxel_size: np.ndarray or tuple or list,
            phase_factor: int = -1,
            handle_defects: bool = False,
    ) -> np.ndarray:
        """
        Main method used in the post-processing workflow. The method
        computes all the structural properties of interest in BCDI
        (amplitude, phase, displacement, displacement gradient,
        heterogeneous strain d-spacing and lattice parameter maps.)

        Args:
            complex_object (np.ndarray): the reconstructed object
            (rho e^(i phi))
            g_vector (np.ndarray or tuple or list): the reciprocal space
            node on which the displacement gradient must be projected.
            hkl (tuple or list): _description_
            voxel_size (np.ndarray or tuple or list): the voxel size of
            the 3D array.
            phase_factor (int, optional): the factor the phase should
            should be multiplied by, depending on the FFT convention
            used. Defaults to -1 (PyNX convention in Phase Retrieval,
            in PyNX scattering, use 1).
            handle_defects (bool, optional): whether a defect is present
            in the reconstruction, in this case phasing processing and
            strain computation is different. Defaults to False.

        Returns:
            np.ndarray: _description_
        """
        complex_object, support = cls.prepare_volume(
            complex_object, isosurface=isosurface)
        if handle_defects:
            pass
        else:
            # extract phase and amplitude
            amplitude = np.abs(complex_object)
            phase = np.angle(complex_object) * phase_factor

            phase = cls.unwrap_phase(phase, support)
            support = zero_to_nan(support)  # 0 values must be nan now
            phase = phase * support
            phase_with_ramp = phase.copy()  # save the 'ramped' phase for later
            phase = cls.remove_phase_ramp(phase)
            phase = cls.phase_offset_to_zero(phase)

            # compute the displacement
            displacement = cls.get_displacement(phase, g_vector)
            displacement_with_ramp = cls.get_displacement(
                phase_with_ramp, g_vector
            )

            # compute the displacement gradient
            displacement_gradient = cls.get_displacement_gradient(
                displacement,
                voxel_size,
                gradient_method="hybrid"
            )

            # compute the various strain quantities
            numpy_het_strain = cls.get_het_normal_strain(
                displacement * 1e-1,  # displacement values converted in nm.
                g_vector,
                voxel_size,
                gradient_method="numpy"
            )
            het_strain = cls.get_het_normal_strain(
                displacement * 1e-1,
                g_vector,
                voxel_size,
                gradient_method="hybrid"
            )
            het_strain_with_ramp = cls.get_het_normal_strain(
                displacement_with_ramp * 1e-1,
                g_vector,
                voxel_size,
                gradient_method="hybrid"
            )

            # compute the dspacing and lattice_parameter
            dspacing = (
                2 * np.pi
                / np.linalg.norm(g_vector)
                * (1 + het_strain_with_ramp)
            )
            lattice_parameter = (
                np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
                * dspacing
            )
            dspacing_mean = np.nanmean(dspacing)
            het_strain_from_dspacing = (dspacing - dspacing_mean)/dspacing_mean

            # all strains are saved in percent
            return {
                "amplitude": normalize(amplitude),
                "support": nan_to_zero(support),
                "phase": nan_to_zero(phase),
                "displacement": displacement,
                "displacement_gradient": displacement_gradient,
                "het_strain": nan_to_zero(het_strain)*100,
                "het_strain_with_ramp": nan_to_zero(het_strain_with_ramp)*100,
                "het_strain_from_dspacing": nan_to_zero(
                    het_strain_from_dspacing)*100,
                "numpy_het_strain": numpy_het_strain*100,
                "dspacing": dspacing,
                "lattice_parameter": lattice_parameter,
                "hkl": hkl,
                "g_vector": g_vector,
                "voxel_size": voxel_size
            }
