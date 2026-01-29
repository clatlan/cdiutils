"""
Plotter class for interactive data visualization from files or arrays.

This module provides the Plotter class for loading and visualizing various
data formats commonly used in BCDI experiments.

Note: This module requires ipywidgets. Dependency checking is handled at
the package level (cdiutils.interactive.__init__).
"""

# Standard library
import os
from typing import Literal

# Third-party
import h5py as h5
import ipywidgets
import numpy as np
from ipywidgets import interact

# Import plotting functions from the same package
from .plotting import plot_3d_slices, plot_data


class Plotter:
    """
    Class to plot data from files, NumPy arrays, or layered 3D datasets.

    This class provides a unified interface for visualizing data commonly
    produced in BCDI experiments, supporting both static matplotlib-based
    plots and interactive multi-layer 3D visualization.

    Args:
        data:
            Data to plot. One of:
            - str: path to a file (.npy, .npz, .cxi, .h5)
            - np.ndarray: array to plot
            - dict[str, np.ndarray]: required when plot='layers'

        plot:
            Plot type. One of:
            - '2D'
            - '1D'
            - 'slices'
            - 'phase_slices'
            - 'contour_slices'
            - 'sum_slices'
            - 'sum_contour_slices'
            - '3D'
            - 'layers' (interactive multi-layer 3D viewer)

            Default is 'slices'.

        log:
            Display data in logarithmic scale (static plots only).

        cmap:
            Colormap name (used for static plots only).

        figsize:
            Figure size in inches.

        fontsize:
            Base font size for labels, ticks, and titles.

        title:
            Optional plot title.

        layers_kwargs:
            Keyword arguments forwarded to `MultiVolumeViewer`
            (only used when plot='layers').

            Supported keys:
                voxel_size: tuple[float, float, float]
                    Physical voxel size along each axis.

                PLOT_ORDER: Literal['xyz', 'zyx']
                    Axis ordering convention.

                CBAR_LEN: float
                    Relative colorbar length.

                render_workers: int | None
                    Number of parallel render workers (for animation).

                render_in_flight: int | None
                    Maximum number of in-flight render tasks.

                rendering_mode: Literal['safe', 'fast', 'process']
                    Rendering backend strategy (for animation).

            Ignored for all other plot modes.

    Attributes:
        data_array:
            NumPy array containing the loaded data (static plot modes).

        data_dict:
            Dictionary of named 3D arrays (used when plot='layers').

        plot:
            Selected plot type.

        log:
            Whether logarithmic scaling is enabled.

        cmap:
            Selected colormap.

        figsize:
            Figure size in inches.

        fontsize:
            Base font size.

        title:
            Plot title.

        layers_kwargs:
            Keyword arguments forwarded to the multi-layer viewer.

        filename:
            Name of the loaded file when data is provided as a path.
    """

    def __init__(
        self,
        data: str | np.ndarray | dict[str, np.ndarray],
        plot: Literal[
            "2D",
            "slices",
            "phase_slices",
            "contour_slices",
            "sum_slices",
            "sum_contour_slices",
            "3D",
            "1D",
            "layers",
        ] = "slices",
        log: bool = False,
        cmap: str = "turbo",
        figsize: tuple[int, int] = (10, 10),
        fontsize: int = 15,
        title: str | None = None,
        layers_kwargs: dict | None = None,
    ):
        """Initialise the Plotter class.

        Args:
            data:
                Data to plot. One of:
                - str: path to a file (.npy, .npz, .cxi, .h5)
                - np.ndarray: array to plot
                - dict[str, np.ndarray]: required when plot='layers'

            plot:
                Plot type. One of:
                - '2D'
                - '1D'
                - 'slices'
                - 'phase_slices'
                - 'contour_slices'
                - 'sum_slices'
                - 'sum_contour_slices'
                - '3D'
                - 'layers' (multi-layer 3D viewer)

            log:
                Display data in logarithmic scale.

            cmap:
                Colormap name. (for static plot)

            figsize:
                Figure size in inches.

            fontsize:
                Base font size for labels, ticks, and titles.

            title:
                Optional plot title.

            layers_kwargs:
                Keyword arguments forwarded to `MultiVolumeViewer`
                (only used when plot='layers').

                Supported keys:
                    voxel_size: tuple[float, float, float]
                        Physical voxel size along each axis.

                    PLOT_ORDER: Literal['xyz', 'zyx']
                        Axis ordering convention.

                    CBAR_LEN: float
                        Relative colorbar length.

                    render_workers: int | None
                        Number of parallel render workers (for animation).

                    render_in_flight: int | None
                        Maximum number of in-flight render tasks.

                    rendering_mode: Literal['safe', 'fast', 'process']
                        Rendering backend strategy (for animation).
        """
        # ---- legacy behaviour ----
        self.data_array = None
        self.data_dict = None
        self.plot = plot
        self.log = log
        self.cmap = cmap
        self.figsize = figsize
        self.fontsize = fontsize
        self.title = title
        self.layers_kwargs = layers_kwargs or {}

        # 🔒 STRICT: MultiVolumeViewer only accepts dict
        if self.plot == "layers":
            if not isinstance(data, dict):
                print(
                    "[Plotter] MultiVolumeViewer requires a dict[str, np.ndarray].\n"
                    "Example:\n"
                    "    {'density': density_3d, 'phase': phase_3d}"
                )
                return

            self.data_dict = data
            self.init_plot()
            return

        # Get data array from any of the supported files
        if isinstance(data, str) and os.path.isfile(data):
            self.filename = data
            self.get_data_array()

        elif isinstance(data, np.ndarray):
            self.data_array = data
            self.init_plot()

        else:
            print(
                "Please provide either a valid filename (arg filename)"
                " or directly an np.ndarray (arg data_array)."
            )

    def init_plot(self):
        """
        Initialise a plot of the data stored in the `data_array` attribute.

        The type of plot and the parameters are specified in the class
        constructor. The plot can be a 2D plot, 3D slices, contour
        plots of slices, sum of slices, sum of contour plots of slices,
        or a 3D plot. The specific plot type is determined by the value
        of the `plot` attribute. If the number of dimensions of the
        `data_array` is not compatible with the specified plot type, the
        function simply prints the number of dimensions and shape of the
        `data_array`.

        Attributes:
            data_array: An array containing the data to be plotted.
            plot: The type of plot to be generated, which can be one of the following: "2D", "slices", "phase_slices",
                "contour_slices", "sum_slices", "sum_contour_slices", or "3D".
            figsize: The size of the plot in inches.
            fontsize: The font size of the plot.
            log: If True, plot the data in logarithmic scale.
            cmap: The colourmap to be used for the plot.
            title: The title of the plot.
        """
        # Import ThreeDViewer here to avoid circular imports
        from .viewer_3d import ThreeDViewer

        if self.plot == "2D":
            plot_data(
                data_array=self.data_array,
                figsize=self.figsize,
                fontsize=self.fontsize,
                log=self.log,
                cmap=self.cmap,
                title=self.title,
            )

        elif self.plot == "slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize,
                title=self.title,
                figsize=None,
                log=self.log,
                cmap=self.cmap,
                contour=False,
                sum_over_axis=False,
            )

        elif self.plot == "phase_slices" and self.data_array.ndim == 3:
            amp = np.abs(self.data_array)
            phase = np.angle(self.data_array)
            max_amp = np.max(amp)
            phase_in_support = np.where(amp > 0.05 * max_amp, phase, np.nan)

            plot_3d_slices(
                data_array=phase_in_support,
                fontsize=self.fontsize,
                title=self.title,
                figsize=None,
                log=self.log,
                cmap=self.cmap,
                contour=False,
                sum_over_axis=False,
            )

        elif self.plot == "contour_slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize,
                title=self.title,
                figsize=None,
                log=self.log,
                cmap=self.cmap,
                contour=True,
                sum_over_axis=False,
            )

        elif self.plot == "sum_slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize,
                title=self.title,
                figsize=None,
                log=self.log,
                cmap=self.cmap,
                contour=False,
                sum_over_axis=True,
            )

        elif self.plot == "sum_contour_slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize,
                title=self.title,
                figsize=None,
                log=self.log,
                cmap=self.cmap,
                contour=True,
                sum_over_axis=True,
            )

        elif self.plot == "3D" and self.data_array.ndim == 3:
            viewer = ThreeDViewer(self.data_array)
            viewer.show()
        elif self.plot == "layers":
            from .multiviewer_3d import MultiVolumeViewer

            self.layers_kwargs.pop("figsize", None)
            self.layers_kwargs.pop("fontsize", None)

            viewer = MultiVolumeViewer(
                self.data_dict,
                fontsize=self.fontsize,
                figsize=self.figsize,
                **self.layers_kwargs,
            )
            viewer.show()
            return
        elif self.plot == "1D" and self.data_array.ndim == 1:
            print(self.data_array)
            plot_data(
                data_array=self.data_array,
                figsize=self.figsize,
                fontsize=self.fontsize,
                log=self.log,
                cmap=self.cmap,
                title=self.title,
            )
        else:
            print(
                "#########################################################"
                "########################################################\n"
                f"Loaded data array\n"
                f"\tNb of dimensions: {self.data_array.ndim}\n"
                f"\tShape: {self.data_array.shape}\n"
                "\n#########################################################"
                "########################################################"
            )

    def get_data_array(self):
        """
        Return the data array stored in the class instance by reading
        the specified file.

        The file must have a .npy, .cxi, .h5, or .npz extension. If the
        file is a .npy or .h5 file, the data array is directly loaded.
        If the file is a .cxi file, the data array is loaded from
        `f.root.entry_1.data_1.data[:]` or
        `f.root.entry_1.image_1.data[:]`, following cxi conventions.
        If the file is a .npz file, the user is prompted to select the
        data array from a dropdown list of arrays stored in the .npz
        file.

        If the file extension is supported and the data array is
        successfully loaded, the `init_plot` function is called.

        Return:
            A Numpy array representing the data stored
            in the class, or None if the file could not be loaded.

        Raises:
            KeyError: If the file is a .cxi or .h5 file, and the data
            could not be found in either `f.root.entry_1.data_1.data[:]`
            or `f.root.entry_1.image_1.data[:]`.
        """
        # No need to select data array interactively
        if self.filename.endswith((".npy", ".h5", ".cxi")):
            if self.filename.endswith(".npy"):
                try:
                    self.data_array = np.load(self.filename)

                except ValueError:
                    print("Could not load data ... ")

            elif self.filename.endswith(".cxi"):
                try:
                    self.data_array = h5.File(self.filename, mode="r")[
                        "entry_1/data_1/data"
                    ][()]

                except (KeyError, OSError):
                    try:
                        self.data_array = h5.File(self.filename, mode="r")[
                            "entry_1/image_1/data"
                        ][()]
                    except (KeyError, OSError):
                        print(
                            "The file could not be loaded, verify that you are"
                            "loading a file with an hdf5 architecture (.nxs, "
                            ".cxi, .h5, ...) and that the file exists."
                            "Otherwise, verify that the data is saved in "
                            "f.root.entry_1.data_1.data[:],"
                            "or f.root.entry_1.image_1.data[:], as it should be"
                            "following cxi conventions."
                        )

            elif self.filename.endswith(".h5"):
                try:
                    self.data_array = h5.File(self.filename, mode="r")[
                        "entry_1/data_1/data"
                    ][()]
                    if self.data_array.ndim == 4:
                        self.data_array = self.data_array[0]
                    # Due to labelling of axes x,y,z and not z,y,x
                    self.data_array = np.swapaxes(self.data_array, 0, 2)

                except (KeyError, OSError):
                    try:
                        self.data_array = h5.File(self.filename, mode="r")[
                            "entry_1/image_1/data"
                        ][()]
                        if self.data_array.ndim == 4:
                            self.data_array = self.data_array[0]
                        # Due to labelling of axes x,y,z and not z,y,x
                        self.data_array = np.swapaxes(self.data_array, 0, 2)
                    except (KeyError, OSError):
                        raise KeyError(
                            "The file could not be loaded, verify that you are"
                            "loading a file with an hdf5 architecture (.nxs, "
                            ".cxi, .h5, ...) and that the file exists."
                            "Otherwise, verify that the data is saved in "
                            "f.root.entry_1.data_1.data[:],"
                            "or f.root.entry_1.image_1.data[:], as it should be"
                            "following cxi conventions."
                        )

            # Plot data
            self.init_plot()

        # Need to select data array interactively
        elif self.filename.endswith(".npz"):
            # Open npz file and allow the user to pick an array
            try:
                rawdata = np.load(self.filename)

                @interact(
                    file=ipywidgets.Dropdown(
                        options=rawdata.files,
                        value=rawdata.files[0],
                        description="Pick an array to load:",
                        style={"description_width": "initial"},
                    )
                )
                def open_npz(file):
                    # Pick an array
                    try:
                        self.data_array = rawdata[file]
                    except ValueError:
                        print("Key not valid, is this an array ?")

                    # Plot data
                    self.init_plot()

            except ValueError:
                print("Could not load data.")

        else:
            print("Data type not supported.")
