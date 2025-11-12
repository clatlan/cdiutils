"""
Interactive support manipulation tools for BCDI workflows.

This module provides the SupportTools class for creating, extracting,
and optimising supports (binary masks) in Bragg Coherent Diffractive
Imaging (BCDI) reconstructions. It is designed for interactive use
in Jupyter notebooks.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

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
                threshold (float): Threshold for binarizing the convolved support (0 to 1).
                compute (bool, optional): If True, perform the convolution. Defaults to True.

            Raises:
                KeyError: If the support file does not contain a 'support' or 'data' array.
                ValueError: If the file format is not supported.
            """
            # Import plot_3d_slices here to avoid circular imports
            from cdiutils.interactive import plot_3d_slices

            if compute:
                try:
                    old_support = np.load(self.path_to_support)["support"]
                except KeyError:
                    try:
                        old_support = np.load(self.path_to_support)["data"]
                    except KeyError:
                        try:
                            old_support = np.load(self.path_to_support)
                        except Exception as E:
                            print(
                                "Could not load 'data' or 'support' array from file."
                            )
                            raise E
                except ValueError:
                    print("Data type not supported")
                try:
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
                        rocc = np.where(support == 1)
                        rnocc = np.where(support == 0)
                        print("Percentage of 3D array occupied by support:")
                        print(f"\t{np.shape(rocc)[1] / np.shape(rnocc)[1]}")
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
