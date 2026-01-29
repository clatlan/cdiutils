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
    - Create, extract, and smooth supports for BCDI reconstructions.
    - Display HDF5 file trees and delete selected files.

    The tab is designed as a vertical box of interactive
    widgets, enabling users to select files, choose colourmaps,
    and specify plotting or support operations.

    Attributes:
        header: A brief description of the tab's purpose.
        box_style: Optional styling for the widget box.
        parent_folder: Dropdown to select the parent data directory.
        filename: Multi-select widget for compatible data files.
        cmap: Dropdown to select a colourmap for plots.
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

        options = sorted([x[0] + "/" for x in os.walk(work_dir)])
        for root in options:
            if ".ipynb" in root:
                options.remove(root)

        self.parent_folder = widgets.Dropdown(
            options=options,
            value=work_dir + "/",
            placeholder=work_dir + "/",
            description="Data folder:",
            continuous_update=False,
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )
        self.filename = widgets.SelectMultiple(
            options=[""]
            + [
                os.path.basename(f)
                for f in sorted(
                    glob.glob(os.getcwd() + "/*.npy")
                    + glob.glob(os.getcwd() + "/*.npz")
                    + glob.glob(os.getcwd() + "/*.cxi")
                    + glob.glob(os.getcwd() + "/*.h5")
                    + glob.glob(os.getcwd() + "/*.nxs")
                    + glob.glob(os.getcwd() + "/*.png"),
                    key=os.path.getmtime,
                )
            ],
            rows=20,
            description="Compatible file list",
            layout=widgets.Layout(width="90%"),
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
            self.parent_folder,
            self.filename,
            self.cmap,
            self.data_use,
        )
        # Assign handlers
        self.parent_folder.observe(self.plot_folder_handler, names="value")

    def plot_folder_handler(self, change):
        """
        Update the filename dropdown when the parent folder changes.

        Args:
            change: Dictionary containing the new folder path.
        """
        if hasattr(change, "new"):
            change = change.new
        options = [""] + [
            os.path.basename(f)
            for f in sorted(
                glob.glob(change + "/*.npy")
                + glob.glob(change + "/*.npz")
                + glob.glob(change + "/*.cxi")
                + glob.glob(change + "/*.h5")
                + glob.glob(change + "/*.nxs")
                + glob.glob(change + "/*.png"),
                key=os.path.getmtime,
            )
        ]
        self.filename.options = [os.path.basename(f) for f in options]

    def show(self):
        """
        Display the interactive GUI as a standalone widget.
        """
        init_plot_tab_gui = interactive(
            self.init_plot_data_tab,
            parent_folder=self.parent_folder,
            filename=self.filename,
            cmap=self.cmap,
            data_use=self.data_use,
        )
        display(init_plot_tab_gui)

    def init_plot_data_tab(
        self,
        parent_folder,
        filename,
        cmap,
        data_use,
    ):
        """
        Execute the selected data operation (plot, support, etc.).

        Args:
            parent_folder: Parent folder path.
            filename: Selected filename(s).
            cmap: Colourmap for plots.
            data_use: Operation to perform (e.g., "2D", "3D", "create_support").
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
            for w in self.children[:-1]:
                if not isinstance(w, widgets.HTML):
                    w.disabled = True
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
            for w in self.children[:-1]:
                if not isinstance(w, widgets.HTML):
                    w.disabled = True
            sup = SupportTools(path_to_data=parent_folder + "/" + filename[0])
            sup.extract_support()
        elif data_use == "smooth_support" and len(filename) == 1:
            # Smooth support
            for w in self.children[:-1]:
                if not isinstance(w, widgets.HTML):
                    w.disabled = True
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
                    display(H5Glance(parent_folder + "/" + filename[0]))
                except TypeError:
                    print("This tool supports .nxs, .cxi or .hdf5 files only.")
        elif (
            data_use
            in [
                "3D",
                "create_support",
                "extract_support",
                "smooth_support",
            ]
            and len(filename) != 1
        ):
            print("Please select only one file.")
        elif data_use == "delete":
            # Delete selected files
            for w in self.children[:-2]:
                if not isinstance(w, widgets.HTML):
                    w.disabled = True
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

            display(button_delete_data)
        elif data_use is False:
            # Clear output
            plt.close()
            for w in self.children[:-2]:
                if not isinstance(w, widgets.HTML):
                    w.disabled = False
            self.plot_folder_handler(change=parent_folder)
            print("Cleared window.")
            clear_output(True)
