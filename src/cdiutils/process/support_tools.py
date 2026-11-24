"""
Interactive support manipulation tools for BCDI workflows.

This module provides the SupportTools class for creating, extracting,
and optimising supports (binary masks) in Bragg Coherent Diffractive
Imaging (BCDI) reconstructions. It is designed for interactive use
in Jupyter notebooks.
"""

from ast import literal_eval

import numpy as np
from scipy.ndimage import gaussian_filter

from cdiutils.utils import resize_at_center

# Optional dependencies for interactive features
try:
    import tables as tb
    from IPython.display import clear_output

    HAS_SUPPORT_TOOLS_DEPS = True
except ImportError:
    HAS_SUPPORT_TOOLS_DEPS = False
    clear_output = None  # Will not be used if dependencies missing


if HAS_SUPPORT_TOOLS_DEPS:

    class SupportTools:
        """
        A utility class for creating, extracting, and
        optimising supports in BCDI workflows.

        This class provides methods to:
        - Extract a support (binary mask) from reconstructed
        BCDI data files (e.g., .cxi, .npz).
        - Create a support from reconstructed electronic
        density data using a threshold.
        - Smooth a support using Gaussian convolution to
        remove holes and refine boundaries.
        - Resize a support (crop and/or pad around its centre) to the
        shape of new data.

        The support is typically a 3D binary array (0s and 1s)
        representing the region of interest in the reconstructed object.
        The class is designed to work with common BCDI file formats
        and supports interactive and automated workflows.

        Attributes:
            path_to_data: Path to the reconstructed object data file.
            path_to_support: Path to the support file.
            saving_directory: Directory where results (e.g., supports, figures)
                are saved.

        Example:
            >>> # Extract support from a .cxi file
            >>> sup = SupportTools(path_to_data="reconstruction.cxi")
            >>> sup.extract_support()

            >>> # Create a support from data using a threshold
            >>> sup = SupportTools(path_to_data="reconstruction.cxi")
            >>> sup.compute_support(threshold=0.1)

            >>> # Smooth an existing support
            >>> sup = SupportTools(path_to_support="support.npz")
            >>> sup.gaussian_convolution(sigma=0.5, threshold=0.2)

            >>> # Resize a support to the shape of new data
            >>> sup = SupportTools(path_to_support="support.npz")
            >>> sup.resize_support(new_shape=(160, 180, 180))
        """

        def __init__(
            self,
            path_to_data=None,
            path_to_support=None,
            saving_directory=None,
        ):
            """
            Initialise the SupportTools class.

            Args:
                path_to_data: Path to the reconstructed object data file.
                    Supported formats: .cxi, .h5, .npz.
                path_to_support: Path to the support file.
                    Supported formats: .npz, .npy.
                saving_directory: Directory to save results.
                    Defaults to the directory of `path_to_data` or
                    `path_to_support`.
            """
            self.path_to_data = path_to_data
            self.path_to_support = path_to_support
            if saving_directory is None:
                try:
                    self.saving_directory = self.path_to_data.replace(
                        self.path_to_data.split("/")[-1], ""
                    )
                except AttributeError:
                    try:
                        self.saving_directory = self.path_to_support.replace(
                            self.path_to_support.split("/")[-1], ""
                        )
                    except AttributeError:
                        raise AttributeError(
                            "Please provide a saving_directory."
                        )
            else:
                self.saving_directory = saving_directory

        @staticmethod
        def load_support(path: str) -> np.ndarray:
            """
            Load a support from a npz, npy, cxi or h5 file.

            Args:
                path: path to the file. For a npz file, the arrays
                    "support", "data" and "mask" are tried in this order,
                    then the only array of the file. For a cxi or h5 file
                    (PyNX reconstruction), the support is read at
                    /entry_1/image_1/mask.

            Raises:
                ValueError: if the format is not supported or the support
                    is not found in the file.

            Returns:
                The support array.
            """
            if path.endswith(".npy"):
                return np.load(path)
            if path.endswith(".npz"):
                with np.load(path) as file:
                    for key in ("support", "data", "mask"):
                        if key in file:
                            return file[key]
                    if len(file.files) == 1:
                        return file[file.files[0]]
                    raise ValueError(
                        f"No 'support', 'data' or 'mask' array in {path} "
                        f"(arrays: {file.files})."
                    )
            if path.endswith((".cxi", ".h5")):
                try:
                    with tb.open_file(path, "r") as file:
                        return file.root.entry_1.image_1.mask[:]
                except tb.NoSuchNodeError as error:
                    raise ValueError(
                        f"No support at /entry_1/image_1/mask in {path}."
                    ) from error
            raise ValueError(
                f"Unsupported support file {path}, expected npz, npy, cxi "
                "or h5."
            )

        def extract_support(self, compute=True):
            """
            Extract a support (binary mask) from a reconstructed object file.

            For .cxi files, the support is extracted from the `/entry_1/image_1/mask` dataset.
            The extracted support is saved as a compressed .npz file.

            Args:
                compute (bool, optional): If True, perform the extraction. Defaults to True.

            Raises:
                tb.NoSuchNodeError: If the file structure is not as expected.
                ValueError: If the file format is not supported.
            """
            # Import plot_3d_slices here to avoid circular imports
            from cdiutils.interactive import plot_3d_slices

            if compute:
                if self.path_to_data.endswith(".cxi"):
                    try:
                        with tb.open_file(self.path_to_data, "r") as f:
                            support = f.root.entry_1.image_1.mask[:]
                            print(
                                "\n###################"
                                "#####################"
                                "#####################"
                                "#####################"
                            )
                            np.savez_compressed(
                                self.saving_directory
                                + "extracted_support.npz",
                                support=support,
                            )
                            print(
                                f"Saved support in {self.saving_directory} as:"
                            )
                            print("\textracted_support.npz")
                            print(
                                "#####################"
                                "#####################"
                                "#####################"
                                "###################\n"
                            )
                            plot_3d_slices(support, log="interact")
                    except tb.NoSuchNodeError:
                        print("Data type not supported")
                else:
                    print("Data type not supported")
            else:
                clear_output(True)
                print("Set compute to true to continue")

        def gaussian_convolution(self, sigma, threshold, compute=True):
            """
            Apply Gaussian convolution to a support to smooth its boundaries.

            This method helps remove holes and refine the support by applying a Gaussian filter
            and thresholding the result. The smoothed support is saved as a compressed .npz file.

            Args:
                sigma (float): Standard deviation for the Gaussian kernel.
                threshold (float): Threshold for binarizing the convolved
                    support, which is scaled to 100 before the filter:
                    the threshold is on a 0 to 100 scale.
                compute (bool, optional): If True, perform the convolution. Defaults to True.

            Raises:
                KeyError: If the support file does not contain a 'support' or 'data' array.
                ValueError: If the file format is not supported.
            """
            # Import plot_3d_slices here to avoid circular imports
            from cdiutils.interactive import plot_3d_slices

            if compute:
                try:
                    old_support = self.load_support(self.path_to_support)
                except ValueError as error:
                    print(error)
                    return
                try:
                    # The support (0 or 1) is scaled to 100 before the
                    # filter, so the threshold is on a 0-100 scale
                    bigdata = 100 * old_support
                    conv_support = np.where(
                        gaussian_filter(bigdata, sigma) > threshold, 1, 0
                    )
                    print(
                        "\n###################"
                        "#####################"
                        "#####################"
                        "#####################"
                    )
                    np.savez_compressed(
                        f"{self.saving_directory}filter_sig{sigma}_t{threshold}",
                        oldsupport=old_support,
                        support=conv_support,
                    )
                    print(f"Saved support in {self.saving_directory} as:")
                    print(f"\tfilter_sig{sigma}_t{threshold}")
                    print(
                        "#####################"
                        "#####################"
                        "#####################"
                        "###################\n"
                    )
                    plot_3d_slices(conv_support, log="interact")
                except UnboundLocalError:
                    pass
            else:
                clear_output(True)
                print("Set compute to true to continue")

        def compute_support(self, threshold, compute=True):
            """
            Create a support from reconstructed electronic density data using a threshold.

            The support is created by thresholding the amplitude of the electronic density.
            The resulting support is saved as a compressed .npz file.

            Args:
                threshold (float): Threshold for binarizing the amplitude (0 to 1).
                compute (bool, optional): If True, perform the computation. Defaults to True.

            Raises:
                tb.HDF5ExtError: If the file format is not supported.
            """
            # Import plot_3d_slices here to avoid circular imports
            from cdiutils.interactive import plot_3d_slices

            if compute:
                try:
                    with tb.open_file(self.path_to_data, "r") as f:
                        if self.path_to_data.endswith(".cxi"):
                            electronic_density = f.root.entry_1.data_1.data[:]
                        elif self.path_to_data.endswith(".h5"):
                            electronic_density = f.root.entry_1.data_1.data[:][
                                0
                            ]
                        print(
                            "\n###################"
                            "#####################"
                            "#####################"
                            "#####################"
                        )
                        print(
                            "Shape of real space complex electronic density array:"
                        )
                        print(f"\t{np.shape(electronic_density)}")
                        amp = np.abs(electronic_density)
                        print(
                            f"\tMaximum value in amplitude array: {amp.max()}"
                        )
                        support = np.where(amp < threshold * amp.max(), 0, 1)
                        # Fraction of the voxels in the support
                        print("Percentage of 3D array occupied by support:")
                        print(f"\t{100 * support.mean():.3f} %")
                        np.savez_compressed(
                            self.saving_directory + "computed_support.npz",
                            support=support,
                        )
                        print(f"Saved support in {self.saving_directory} as:")
                        print("\tcomputed_support.npz")
                        print(
                            "#####################"
                            "#####################"
                            "#####################"
                            "###################\n"
                        )
                        plot_3d_slices(support, log="interact")
                except tb.HDF5ExtError:
                    print("Data type not supported")
            else:
                clear_output(True)
                print("Set compute to true to continue")

        def resize_support(self, new_shape, compute=True):
            """
            Crop and/or pad a support around its centre to new_shape.

            Use it when the data were preprocessed with a different
            shape, e.g. a support from a (200, 200, 200) reconstruction
            for data of shape (160, 180, 180). The voxel size is kept, so
            the data must have the same voxel size (same binning and
            preprocessing). The centre voxel (index n // 2 along each
            axis) stays at the centre. The resized support is saved as
            support_<n0>_<n1>_<n2>.npz, under the "support" key, which
            the phase retrieval GUI reads.

            Args:
                new_shape: the output shape, a tuple of ints, or a string
                    such as "(160, 180, 180)" (from a text widget).
                compute (bool, optional): If True, perform the resizing.
                    Defaults to True.
            """
            # Import plot_3d_slices here to avoid circular imports
            from cdiutils.interactive import plot_3d_slices

            if not compute:
                clear_output(True)
                print("Set compute to true to continue")
                return

            try:
                if isinstance(new_shape, str):
                    new_shape = literal_eval(new_shape)
                new_shape = tuple(int(s) for s in new_shape)
                support = self.load_support(self.path_to_support)
                resized = resize_at_center(support, new_shape)
            except (ValueError, TypeError, SyntaxError) as error:
                print(f"Could not resize the support: {error}")
                return

            # Support voxels outside the new box are lost
            nb_lost = int(np.count_nonzero(support)) - int(
                np.count_nonzero(resized)
            )
            print(f"Support resized from {support.shape} to {resized.shape}.")
            if nb_lost > 0:
                print(
                    f"Warning: {nb_lost} support voxels "
                    f"({100 * nb_lost / np.count_nonzero(support):.2f} %) "
                    "were outside the new shape and are cropped."
                )
            file_name = "support_" + "_".join(str(s) for s in new_shape)
            np.savez_compressed(
                f"{self.saving_directory}{file_name}.npz", support=resized
            )
            print(f"Saved support in {self.saving_directory} as:")
            print(f"\t{file_name}.npz")
            plot_3d_slices(resized, log="interact")

else:
    # Placeholder class when dependencies are not available
    class SupportTools:
        """
        Placeholder for SupportTools when optional dependencies are
        not installed.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SupportTools requires tables package. "
                "Can be installed with pip install tables, or with interactive"
                " option: pip install cdiutils[interactive]"
            )
