"""
Interactive data browser widget for file loading and visualization.

This module provides the TabPlotData class for browsing and visualizing
various data file formats in BCDI workflows.

Note: This module requires ipywidgets and h5glance. Dependency checking
is handled at the package level (cdiutils.interactive.__init__).
"""

import glob
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
from h5glance import H5Glance
from IPython.display import Image, clear_output, display
from ipywidgets import interactive

from cdiutils.process.support_tools import SupportTools

from .plotter import Plotter


class TabPlotData(widgets.VBox):
    """
    A widget for interactive data visualisation and
    support manipulation in BCDI workflows.

    This class provides a GUI tab for loading, plotting,
    and manipulating data files (e.g., .npy, .npz, .cxi, .h5,
    .nxs, .png) in the context of Bragg Coherent Diffractive Imaging (BCDI).
    It allows users to:
    - Browse and select data files from a directory.
    - Visualise 1D, 2D, and 3D data interactively.
    - Create, extract, smooth and resize supports for BCDI
      reconstructions.
    - Display HDF5 file trees and delete selected files.

    The tab is designed as a vertical box of interactive
    widgets, enabling users to select files, choose colourmaps,
    and specify plotting or support operations.

    Attributes:
        header: A brief description of the tab's purpose.
        box_style: Optional styling for the widget box.
        parent_folder: Dropdown to select the parent data directory.
        button_refresh_files: Button to look again for the folders and
            files.
        filename: Multi-select widget for compatible data files.
        cmap: Dropdown to select a colourmap for plots.
        support_shape: Text field with the shape of the resized
            support, e.g. "(160, 180, 180)".
        data_use: Toggle buttons for data operations (plot, support, etc.).
        children: Ordered collection of child widgets.

    Example:
        >>> tab = TabPlotData(work_dir="/path/to/data")
        >>> tab.stand_alone()  # Display the interactive GUI
    """

    def __init__(
        self, work_dir: str | None = None, box_style: str | None = None
    ):
        """
        Initialise the TabPlotData widget.

        Args:
            box_style: CSS style for the widget box. Defaults to "".
            work_dir: Working directory path. Defaults to current directory.
        """
        super(TabPlotData, self).__init__()
        # Brief header describing the tab
        self.header = "Plot data"
        self.box_style = box_style if box_style is not None else ""
        # Define widgets
        self.unused_label_plot = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
                Loads data files and displays it in the GUI",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )
        if work_dir is None:
            work_dir = os.getcwd()
        self.work_dir = work_dir

        folders = self._list_sub_folders(work_dir)
        self.parent_folder = widgets.Dropdown(
            options=folders,
            value=folders[0],
            description="Data folder:",
            continuous_update=False,
            layout=widgets.Layout(width="75%"),
            style={"description_width": "initial"},
        )
        self.button_refresh_files = widgets.Button(
            description="Refresh file list",
            icon="refresh",
            tooltip="Look again for the folders and files",
            layout=widgets.Layout(width="15%", height="35px"),
        )
        self.button_refresh_files.on_click(self.refresh_file_list)
        self.filename = widgets.SelectMultiple(
            options=self._list_files(folders[0]),
            rows=20,
            description="Compatible file list",
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )
        # Shape used by the "Resize support" tool
        self.support_shape = widgets.Text(
            value="(160, 180, 180)",
            placeholder="(160, 180, 180)",
            description="Resized support shape:",
            continuous_update=False,
            layout=widgets.Layout(width="40%"),
            style={"description_width": "initial"},
        )
        self.cmap = widgets.Dropdown(
            options=plt.colormaps(),
            value="turbo",
            description="Colour map:",
            continuous_update=False,
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )
        self.data_use = widgets.ToggleButtons(
            options=[
                ("Clear/ Reload folder", False),
                ("1D plot", "1D"),
                ("2D plot", "2D"),
                ("Plot slices", "slices"),
                ("Plot phase slices", "phase_slices"),
                ("Plot contour slices", "contour_slices"),
                ("Plot sum over axes", "sum_slices"),
                ("Plot contour of sum over axes", "sum_contour_slices"),
                ("3D plot", "3D"),
                ("Create support", "create_support"),
                ("Extract support", "extract_support"),
                ("Smooth support", "smooth_support"),
                ("Resize support", "resize_support"),
                ("Display .png image", "show_image"),
                ("Display hdf5 tree", "hf_glance"),
                ("Delete selected files", "delete"),
            ],
            value=False,
            description="Load data",
            tooltips=[
                "Clear the output and unload data from GUI, saves RAM",
                "Load data and plot vector",
                "Load data and plot data slice interactively",
                "Load data and plot phase slice interactively",
                "Load data and plot data slices for each dimension in its middle",
                "Load data and plot data contours of slices for each dimension in its middle",
                "Load data and plot data summed for each dimension",
                "Load data and plot contours of data summed for each dimension in its middle",
                "Load data and plot 3D data interactively",
                "Load data and allow for the creation of a support interactively",
                "Load data and allow for the creation of a support automatically",
                "Load support and smooth its boundaries",
                "Load support and crop/pad it around its centre to a new "
                "shape",
                "Display .png image",
                "Display hdf5 tree",
                "Delete selected files, careful !!",
            ],
            button_style="",
            icon="fast-forward",
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )
        # Define children
        self.children = (
            self.unused_label_plot,
            widgets.HBox([self.parent_folder, self.button_refresh_files]),
            self.filename,
            self.cmap,
            self.support_shape,
            self.data_use,
        )
        # Assign handlers
        self._is_refreshing = False
        self.parent_folder.observe(self.plot_folder_handler, names="value")

    # File extensions listed in the file list
    FILE_PATTERNS = ("*.npy", "*.npz", "*.cxi", "*.h5", "*.nxs", "*.png")

    @staticmethod
    def _list_sub_folders(work_dir: str) -> list[str]:
        """
        List work_dir and its sub-folders, without Jupyter checkpoints.

        Args:
            work_dir: the top folder.

        Returns:
            The folder paths, each ending with a "/", work_dir first.
        """
        top = os.path.join(work_dir, "")
        sub_folders = sorted(
            os.path.join(root, "")
            for root, _, _ in os.walk(work_dir)
            if ".ipynb" not in root
        )
        return [top] + [f for f in sub_folders if f != top]

    @classmethod
    def _list_files(cls, folder: str) -> list[str]:
        """
        List the compatible files of a folder, newest first.

        Args:
            folder: the folder to look into.

        Returns:
            The file names (without folder).
        """
        paths = set()
        for pattern in cls.FILE_PATTERNS:
            paths.update(glob.glob(os.path.join(folder, pattern)))
        return [
            os.path.basename(path)
            for path in sorted(paths, key=os.path.getmtime, reverse=True)
        ]

    def _set_controls_disabled(self, disabled: bool) -> None:
        """
        Disable or enable the controls above the operation buttons.

        Args:
            disabled: True to disable them (while a tool is open).
        """
        for widget in self.children[:-1]:
            # Widgets in a box are reached through the box
            if isinstance(widget, widgets.Box):
                inner_widgets = widget.children
            else:
                inner_widgets = (widget,)
            for inner_widget in inner_widgets:
                if not isinstance(inner_widget, widgets.HTML):
                    inner_widget.disabled = disabled

    def plot_folder_handler(self, change):
        """
        Update the file list when the parent folder changes.

        Args:
            change: The change event triggered by the observer.
        """
        # The folder list itself is being refreshed
        if self._is_refreshing:
            return
        self.refresh_file_list(keep_selection=False)

    def refresh_file_list(self, button=None, keep_selection=True):
        """
        Look again for the sub-folders and the compatible files.

        The lists are only changed if folders or files appeared or
        disappeared, so that the selected plot is not drawn again.

        Args:
            button: the button that was clicked, if any (unused).
            keep_selection: keep the selected files that still exist.
        """
        # Folders: look for new sub-folders, keep the current one
        folders = self._list_sub_folders(self.work_dir)
        if list(self.parent_folder.options) != folders:
            self._is_refreshing = True
            try:
                current_folder = self.parent_folder.value
                self.parent_folder.options = folders
                self.parent_folder.value = (
                    current_folder if current_folder in folders else folders[0]
                )
            finally:
                self._is_refreshing = False

        files = self._list_files(self.parent_folder.value)
        if list(self.filename.options) == files:
            return
        selected = self.filename.value if keep_selection else ()
        # Changing the options empties the selection, restore it after
        # (hold_trait_notifications does not keep it reliably)
        self.filename.options = files
        self.filename.value = tuple(f for f in selected if f in files)

    def show(self):
        """
        Display the interactive GUI as a standalone widget.
        """
        init_plot_tab_gui = interactive(
            self.init_plot_data_tab,
            parent_folder=self.parent_folder,
            filename=self.filename,
            cmap=self.cmap,
            support_shape=self.support_shape,
            data_use=self.data_use,
        )
        display(init_plot_tab_gui)

    def init_plot_data_tab(
        self,
        parent_folder,
        filename,
        cmap,
        support_shape,
        data_use,
    ):
        """
        Execute the selected data operation (plot, support, etc.).

        Args:
            parent_folder: Parent folder path.
            filename: Selected filename(s).
            cmap: Colourmap for plots.
            support_shape: Shape of the resized support, e.g.
                "(160, 180, 180)" ("resize_support" only).
            data_use: Operation to perform (e.g., "2D", "3D",
                "create_support").
        """
        if data_use in ("2D", "1D"):
            # Plot 2D data
            for p in filename:
                print(f"Showing {p}")
                Plotter(
                    parent_folder + "/" + p,
                    plot=data_use,
                    log="interact",
                    cmap=cmap,
                )
        elif data_use == "3D" and len(filename) == 1:
            # Plot 3D data
            Plotter(
                parent_folder + "/" + filename[0],
                plot=data_use,
                log="interact",
                cmap=cmap,
            )
        elif data_use in [
            "slices",
            "contour_slices",
            "sum_slices",
            "sum_contour_slices",
            "phase_slices",
        ]:
            # Plot slices or sums
            for p in filename:
                print(f"Showing {p}")
                Plotter(
                    parent_folder + "/" + p,
                    plot=data_use,
                    log="interact",
                    cmap=cmap,
                )
        elif data_use == "create_support" and len(filename) == 1:
            # Create support interactively
            self._set_controls_disabled(True)
            sup = SupportTools(path_to_data=parent_folder + "/" + filename[0])
            window_support = interactive(
                sup.compute_support,
                threshold=widgets.FloatText(
                    value=0.05,
                    step=0.001,
                    max=1,
                    min=0.001,
                    continuous_update=False,
                    description="Threshold:",
                    readout=True,
                    layout=widgets.Layout(width="20%"),
                    style={"description_width": "initial"},
                ),
                compute=widgets.ToggleButton(
                    value=False,
                    description="Compute support ...",
                    button_style="",
                    icon="step-forward",
                    layout=widgets.Layout(width="45%"),
                    style={"description_width": "initial"},
                ),
            )

            def support_handler(change):
                if not change.new:
                    window_support.children[0].disabled = False
                if change.new:
                    window_support.children[0].disabled = True

            window_support.children[1].observe(support_handler, names="value")
            display(window_support)
        elif data_use == "extract_support" and len(filename) == 1:
            # Extract support from data
            self._set_controls_disabled(True)
            sup = SupportTools(path_to_data=parent_folder + "/" + filename[0])
            sup.extract_support()
        elif data_use == "smooth_support" and len(filename) == 1:
            # Smooth support
            self._set_controls_disabled(True)
            sup = SupportTools(
                path_to_support=parent_folder + "/" + filename[0]
            )
            window_support = interactive(
                sup.gaussian_convolution,
                sigma=widgets.FloatText(
                    value=0.05,
                    step=0.001,
                    max=1,
                    min=0.001,
                    continuous_update=False,
                    description="Sigma:",
                    readout=True,
                    layout=widgets.Layout(width="20%"),
                    style={"description_width": "initial"},
                ),
                threshold=widgets.FloatText(
                    value=0.05,
                    step=0.001,
                    max=1,
                    min=0.001,
                    continuous_update=False,
                    description="Threshold:",
                    readout=True,
                    layout=widgets.Layout(width="20%"),
                    style={"description_width": "initial"},
                ),
                compute=widgets.ToggleButton(
                    value=False,
                    description="Compute support ...",
                    button_style="",
                    icon="step-forward",
                    layout=widgets.Layout(width="45%"),
                    style={"description_width": "initial"},
                ),
            )

            def support_handler(change):
                if not change.new:
                    window_support.children[0].disabled = False
                if change.new:
                    window_support.children[0].disabled = True

            window_support.children[1].observe(support_handler, names="value")
            display(window_support)
        elif data_use == "resize_support" and len(filename) == 1:
            # Crop/pad the support around its centre to support_shape
            self._set_controls_disabled(True)
            sup = SupportTools(
                path_to_support=parent_folder + "/" + filename[0]
            )
            sup.resize_support(new_shape=support_shape)
        elif data_use == "show_image":
            # Display PNG image
            try:
                for p in filename:
                    print(f"Showing {p}")
                    display(Image(filename=parent_folder + "/" + p))
            except (FileNotFoundError, ValueError):
                print("Could not load image from file.")
        elif data_use == "hf_glance":
            # Display HDF5 tree
            for p in filename:
                try:
                    print(f"Showing {p}")
                    display(H5Glance(parent_folder + "/" + p))
                except TypeError:
                    print("This tool supports .nxs, .cxi or .hdf5 files only.")
        elif (
            data_use
            in [
                "3D",
                "create_support",
                "extract_support",
                "smooth_support",
                "resize_support",
            ]
            and len(filename) != 1
        ):
            print("Please select only one file.")
        elif data_use == "delete":
            # Delete selected files
            self._set_controls_disabled(True)
            button_delete_data = widgets.Button(
                description="Delete files ?",
                button_style="",
                layout=widgets.Layout(width="70%"),
                style={"description_width": "initial"},
                icon="step-forward",
            )

            @button_delete_data.on_click
            def action_button_delete_data(selfbutton):
                for p in filename:
                    try:
                        os.remove(parent_folder + "/" + p)
                        print(f"Removed {p}")
                    except FileNotFoundError:
                        print(f"Could not remove {p}")
                # The deleted files leave the list
                self.refresh_file_list()

            display(button_delete_data)
        elif data_use is False:
            # Clear output
            plt.close()
            self._set_controls_disabled(False)
            self.refresh_file_list()
            print("Cleared window.")
            clear_output(True)
