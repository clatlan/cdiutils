import warnings
import numpy as np
import h5py as h5
import os

from tornado.ioloop import PeriodicCallback

from skimage.measure import marching_cubes
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import ipywidgets as widgets
from ipywidgets import interact, fixed
from IPython.display import display, HTML
import ipyvolume as ipv

from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper
from bokeh.models import Slider, RadioButtonGroup
import bokeh.palettes as bp
import panel as pn


pn.extension()
warnings.filterwarnings("ignore")


class Plotter:
    """
    Class to plot data from files or Numpy arrays.

    Parameters
    ----------
    data : str or np.ndarray
        The data to plot. This can either be the path to a file, or a
        Numpy array directly.
    plot : str, optional
        Specifies the type of plot to create. Available options are:
        '2D', 'slices', 'contour_slices', 'sum_slices',
        'sum_contour_slices', '3D', by default 'slices'.
    log : bool, optional
        Whether to display the plot in log scale, by default False.
    cmap : str, optional
        The colormap to use for the plot, by default 'turbo'.
    figsize : tuple, optional
        The size of the figure in inches, by default (10, 10).
    fontsize : int, optional
        The font size to use in the plot, by default 15.
    title : str, optional
        The title of the plot, by default None.

    Attributes
    ----------
    data_array : np.ndarray
        The Numpy array with the data.
    plot : str
        The type of plot specified.
    log : bool
        Whether to display the plot in log scale.
    cmap : str
        The colormap specified.
    figsize : tuple
        The size of the figure specified.
    fontsize : int
        The font size specified.
    title : str
        The title of the plot.
    filename : str
        The name of the file if data is given as a path.
    """

    def __init__(
            self,
            data: str | np.ndarray,
            plot: str = "slices",
            log: bool = False,
            cmap: str = "turbo",
            figsize: tuple[int, int] = (10, 10),
            fontsize: int = 15,
            title: str = None,
    ):
        """Initialize the Plotter class.

        Parameters
        ----------
            data : str | np.ndarray
                The data to plot. This can either be the path to a file,
                or a Numpy array directly.
            plot : str, optional
                Specifies the type of plot to create. Available options
                are: '2D', 'slices', 'contour_slices', 'sum_slices',
                'sum_contour_slices', '3D', by default 'slices'.
            log : bool, optional
                Whether to display the plot in log scale, by default
                False.
            cmap : str, optional
                The colormap to use for the plot, by default 'turbo'.
            figsize : tuple, optional
                The size of the figure in inches, by default (10, 10).
            fontsize : int, optional
                The font size to use in the plot, by default 15.
            title : str, optional
                The title of the plot, by default None.
        """
        self.data_array = None
        self.plot = plot
        self.log = log
        self.cmap = cmap
        self.figsize = figsize
        self.fontsize = fontsize
        self.title = title

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
        Initialize a plot of the data stored in the `data_array`
        attribute.

        The type of plot and the parameters are specified in the class
        constructor. The plot can be a 2D plot, 3D slices, contour
        plots of slices, sum of slices, sum of contour plots of slices,
        or a 3D plot. The specific plot type is determined by the value
        of the `plot` attribute. If the number of dimensions of the
        `data_array` is not compatible with the specified plot type, the
        function simply prints the number of dimensions and shape of the
        `data_array`.

        Attributes:
        -----------
        data_array : numpy.ndarray
            An array containing the data to be plotted.
        plot : str
            The type of plot to be generated, which can be one of the following: "2D", "slices", "contour_slices",
            "sum_slices", "sum_contour_slices", or "3D".
        figsize : tuple
            The size of the plot in inches.
        fontsize : int
            The font size of the plot.
        log : bool
            If True, plot the data in logarithmic scale.
        cmap : str
            The colormap to be used for the plot.
        title : str
            The title of the plot.

        Raises:
        -------
        None

        Returns:
        -------
        None
        """
        if self.plot == "2D":
            plot_data(
                data_array=self.data_array,
                figsize=self.figsize, fontsize=self.fontsize,
                log=self.log, cmap=self.cmap, title=self.title,
            )

        elif self.plot == "slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize, title=self.title,
                figsize=None, log=self.log, cmap=self.cmap,
                contour=False, sum_over_axis=False,
            )

        elif self.plot == "contour_slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize, title=self.title,
                figsize=None, log=self.log, cmap=self.cmap,
                contour=True, sum_over_axis=False,
            )

        elif self.plot == "sum_slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize, title=self.title,
                figsize=None, log=self.log, cmap=self.cmap,
                contour=False, sum_over_axis=True,
            )

        elif self.plot == "sum_contour_slices" and self.data_array.ndim == 3:
            plot_3d_slices(
                data_array=self.data_array,
                fontsize=self.fontsize, title=self.title,
                figsize=None, log=self.log, cmap=self.cmap,
                contour=True, sum_over_axis=True,
            )

        elif self.plot == "3D" and self.data_array.ndim == 3:
            ThreeDViewer(self.data_array)

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
        Returns the data array stored in the class instance by reading
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

        Returns:
            numpy.ndarray: A Numpy array representing the data stored
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
                    self.data_array = h5.File(self.filename, mode='r')[
                        'entry_1/data_1/data'][()]

                except (KeyError, OSError):
                    try:
                        self.data_array = h5.File(self.filename, mode='r')[
                            'entry_1/image_1/data'][()]
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
                    self.data_array = h5.File(self.filename, mode='r')[
                        'entry_1/data_1/data'][()]
                    if self.data_array.ndim == 4:
                        self.data_array = self.data_array[0]
                    # Due to labelling of axes x,y,z and not z,y,x
                    self.data_array = np.swapaxes(self.data_array, 0, 2)

                except (KeyError, OSError):
                    try:
                        self.data_array = h5.File(self.filename, mode='r')[
                            'entry_1/image_1/data'][()]
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
                    file=widgets.Dropdown(
                        options=rawdata.files,
                        value=rawdata.files[0],
                        description='Pick an array to load:',
                        style={'description_width': 'initial'}))
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


class ThreeDViewer(widgets.Box):
    """
    Widget to display 3D objects from CDI optimisation, loaded from a result
    CXI file or a mode file.

    Simplified from the widgets class in PyNX @Vincent Favre Nicolin (ESRF)
    """

    def __init__(self, input_file=None, html_width=None):
        """
        Initialize the output and widgets

        :param input_file: the data filename or directly the 3D data array.
        :param html_width: html width in %. If given, the width of the
         notebook will be changed to that value (e.g. full width with 100)
        """
        super(ThreeDViewer, self).__init__()

        if html_width is not None:
            # flake8: noqa
            # type: ignore
            display(
                HTML(# type: ignore
                    fr""" 
                    <style>.container \{ width:{int(html_width)}% \
                    !important; \}</style>
                    """
                )
            )
            # type: ignore

        # focus_label = widgets.Label(value='Focal distance (cm):')
        self.threshold = widgets.FloatSlider(
            value=5,
            min=0,
            max=20,
            step=0.02,
            description='Contour.',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.01f',
        )
        self.toggle_phase = widgets.ToggleButtons(
            options=['Abs', 'Phase'],
            description='',
            disabled=False,
            value='Phase',
            button_style='',
        )
        self.toggle_rotate = widgets.ToggleButton(
            value=False,
            description='Rotate',
            tooltips='Rotate',
        )
        self.pcb_rotate = None
        hbox1 = widgets.HBox([self.toggle_phase, self.toggle_rotate])

        self.toggle_dark = widgets.ToggleButton(
            value=False,
            description='Dark',
            tooltips='Dark/Light theme',
        )
        self.toggle_box = widgets.ToggleButton(
            value=True,
            description='Box',
            tooltips='Box ?',
        )
        self.toggle_axes = widgets.ToggleButton(
            value=True,
            description='Axes',
            tooltips='Axes ?',
        )
        hbox_toggle = widgets.HBox(
            [self.toggle_dark, self.toggle_box, self.toggle_axes])

        # Colormap widgets
        self.colormap = widgets.Dropdown(
            options=['Cool', 'Gray', 'Gray_r', 'Hot', 'Hsv',
                     'Inferno', 'Jet', 'Plasma', 'Rainbow', 'Viridis'],
            value='Jet',
            description='Colors:',
            disabled=True,
        )
        self.colormap_range = widgets.FloatRangeSlider(
            value=[20, 80],
            min=0,
            max=100,
            step=1,
            description='Range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )

        # Progress bar
        self.progress = widgets.IntProgress(
            value=10,
            min=0,
            max=10,
            description='Processing:',
            bar_style='',
            style={'bar_color': 'green'},
            orientation='horizontal'
        )

        # Set observers
        self.threshold.observe(self.on_update_plot)
        self.toggle_phase.observe(self.on_change_scale)
        self.colormap.observe(self.on_update_plot)
        self.colormap_range.observe(self.on_update_plot)

        self.toggle_dark.observe(self.on_update_style)
        self.toggle_box.observe(self.on_update_style)
        self.toggle_axes.observe(self.on_update_style)

        self.toggle_rotate.observe(self.on_animate)

        # Future attributes
        self.mesh = None
        self.color = None
        self.d0 = None

        # Create final vertical box with all the widgets
        self.vbox = widgets.VBox([
            self.threshold, hbox1, hbox_toggle, self.colormap,
            # self.colormap_range, # useless for one contour
            self.progress,
        ])

        # Load data
        if isinstance(input_file, np.ndarray):
            # We create an output for ipyvolume
            self.output_view = widgets.Output()
            with self.output_view:
                self.fig = ipv.figure()
                self.set_data(input_file)
                display(self.fig)

            self.window = widgets.HBox([self.output_view, self.vbox])
            display(self.window)

        else:
            print("Could not load data")

    def on_update_plot(self, change=None):
        """
        Update the plot according to parameters. The points are
        re-computed.

        :param change: used to update the values
        :return:
        """
        if change is not None and change['name'] != 'value':
            return
        self.progress.value = 7

        try:
            verts, faces, _, _ = marching_cubes(
                abs(self.data),
                level=self.threshold.value,
                step_size=1,
            )
            vals = self.rgi(verts)

            # Phase colouring
            if self.toggle_phase.value == "Phase":
                self.colormap.disabled = True
                rgba = complex2rgbalin(vals)
                color = rgba[..., :3] / 256

            # Linear or log colouring
            elif self.toggle_phase.value in ['Abs', 'log10(Abs)']:
                self.colormap.disabled = False
                cs = cm.ScalarMappable(
                    norm=Normalize(
                        vmin=self.colormap_range.value[0],
                        vmax=self.colormap_range.value[1]),
                    cmap=self.colormap.value.lower())
                color = cs.to_rgba(abs(vals))[..., :3]
            else:
                # TODO: Gradient
                gx, gy, gz = self.rgi_gx(verts), self.rgi_gy(
                    verts), self.rgi_gz(verts)
                color = np.empty((len(vals), 3), dtype=np.float32)
                color[:, 0] = abs(gx)
                color[:, 1] = abs(gy)
                color[:, 2] = abs(gz)
                color *= 100
                self.color = color
            x, y, z = verts.T
            self.mesh = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)
            self.fig.meshes = [self.mesh]

        # Keep general exception for debugging purposes
        except Exception as E:
            print(E)

        # Update progress bar
        self.progress.value = 10

    def on_update_style(self, change):
        """
        Update the plot style - for all parameters which
        do not involved recomputing
        the displayed object.
        :param change: dict from widget
        :return:
        """
        if change['name'] == 'value':
            if self.toggle_dark.value:
                ipv.pylab.style.set_style_dark()
            else:
                ipv.pylab.style.set_style_light()
                # Fix label colours (see self.fig.style)
                ipv.pylab.style.use(
                    {'axes': {'label': {'color': 'black'},
                              'ticklabel': {'color': 'black'}}})
            if self.toggle_box.value:
                ipv.pylab.style.box_on()
            else:
                ipv.pylab.style.box_off()
            if self.toggle_axes.value:
                ipv.pylab.style.axes_on()
            else:
                ipv.pylab.style.axes_off()

    def on_change_scale(self, change):
        """Change scale between logarithmic and linear"""
        if change['name'] == 'value':
            if isinstance(change['old'], str):
                newv = change['new']
                oldv = change['old']

                # linear scale
                if 'log' in oldv and 'log' not in newv:
                    data = self.d0
                    self.set_data(data, threshold=10 ** self.threshold.value)

                # log scale
                elif 'log' in newv and 'log' not in oldv:
                    self.d0 = self.data
                    data = np.log10(np.maximum(0.1, abs(self.d0)))
                    self.set_data(data, threshold=np.log10(
                        self.threshold.value))
                    return
            self.on_update_plot()

    def set_data(self, data, threshold=None):
        """
        Check if data is complex or not

        :param data: data 3d array, complex ot not, to be plotted
        :param threshold: threshold for contour, if None set to max/2
        """
        # Update progress bar
        self.progress.value = 5

        # Save data
        self.data = data

        # Change scale options depending on data
        self.toggle_phase.unobserve(self.on_change_scale)

        if np.iscomplexobj(data):
            if self.toggle_phase.value == 'log10(Abs)':
                self.toggle_phase.value = 'Abs'
            self.toggle_phase.options = ('Abs', 'Phase')
        else:
            if self.toggle_phase.value == 'Phase':
                self.toggle_phase.value = 'Abs'
            self.toggle_phase.options = ('Abs', 'log10(Abs)')
        self.toggle_phase.observe(self.on_change_scale)

        # Set threshold
        self.threshold.unobserve(self.on_update_plot)
        self.colormap_range.unobserve(self.on_update_plot)
        self.threshold.max = abs(self.data).max()
        if threshold is None:
            self.threshold.value = self.threshold.max / 2
        else:
            self.threshold.value = threshold

        # Set colormap
        self.colormap_range.max = abs(self.data).max()
        self.colormap_range.value = [0, abs(self.data).max()]
        self.threshold.observe(self.on_update_plot)
        self.colormap_range.observe(self.on_update_plot)

        nz, ny, nx = self.data.shape
        z, y, x = np.arange(nz), np.arange(ny), np.arange(nx)

        # Interpolate probe to object grid
        self.rgi = RegularGridInterpolator(
            (z, y, x),
            self.data,
            method='linear',
            bounds_error=False,
            fill_value=0,
        )

        # Also prepare the phase gradient
        gz, gy, gx = np.gradient(self.data)
        a = np.maximum(abs(self.data), 1e-6)
        ph = self.data / a
        gaz, gay, gax = np.gradient(a)
        self.rgi_gx = RegularGridInterpolator(
            (z, y, x), ((gx - gax * ph) / (ph * a)).real,
            method='linear', bounds_error=False, fill_value=0)
        self.rgi_gy = RegularGridInterpolator(
            (z, y, x), ((gy - gay * ph) / (ph * a)).real,
            method='linear', bounds_error=False, fill_value=0)
        self.rgi_gz = RegularGridInterpolator(
            (z, y, x), ((gz - gaz * ph) / (ph * a)).real,
            method='linear', bounds_error=False, fill_value=0)

        # Fix extent otherwise weird things happen
        ipv.pylab.xlim(0, self.data.shape[0])
        ipv.pylab.ylim(0, self.data.shape[1])
        ipv.pylab.zlim(0, self.data.shape[2])
        # ipv.squarelim()
        self.on_update_plot()

    def on_animate(self, v):
        """Trigger the animation (rotation around vertical axis)"""
        if self.pcb_rotate is None:
            self.pcb_rotate = PeriodicCallback(self.callback_rotate, 50.)
        if self.toggle_rotate.value:
            self.pcb_rotate.start()
        else:
            self.pcb_rotate.stop()

    def callback_rotate(self):
        """Used for periodic rotation."""
        # ipv.view() only supports a rotation against
        # the starting azimuth and elevation
        # ipv.view(azimuth=ipv.view()[0]+1)

        # Use a quaternion and the camera's 'up' as rotation axis
        x, y, z = self.fig.camera.up
        n = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        a = np.deg2rad(2.5) / 2  # angular step
        sa, ca = np.sin(a / 2) / n, np.cos(a / 2)
        r = Rotation.from_quat((sa * x, sa * y, sa * z, ca))
        self.fig.camera.position = tuple(r.apply(self.fig.camera.position))


# Methods

def plot_data(
    data_array: np.ndarray,
    figsize: tuple[int, int] = (10, 10),
    fontsize: int = 15,
    log: bool | str = "interact",
    cmap: str = "turbo",
    title: str | list[str] = None,
) -> None:
    """
    Plot the data contained in a numpy array.

    Parameters
    ----------
    data_array : np.ndarray
        The data to be plotted, contained in a numpy array.
    figsize : tuple[int, int], optional
        The size of the figure, by default (10, 10).
    fontsize : int, optional
        The font size to use for plot labels and titles, by default 15.
    log : bool | str, optional
        If True, plot the data using a logarithmic scale. If False, plot
        using a linear scale.
        If "interact", create an interactive toggle button to switch
        between linear and logarithmic scales.
    cmap : str, optional
        The color map to use for 2D data, by default "turbo".
    title : str | list[str], optional
        The title(s) to be used for the plot. If data_array is 2D, title
        can be either a string or a list of strings with one title per
        subplot. If data_array is 1D, title should be a string.

    Returns
    -------
    None
    """
    # Get dimensions
    data_dimensions = data_array.ndim

    if data_dimensions == 1:
        plt.close()
        fig, ax = plt.subplots(figsize=figsize)

        # Depends on log scale
        if log:
            ax.plot(np.log(data_array))
            plt.title(title, fontsize=fontsize+2)
            plt.tight_layout()
            plt.show()

        elif log is False:
            ax.plot(data_array)
            plt.title(title, fontsize=fontsize+2)
            plt.tight_layout()
            plt.show()

        elif log == "interact":
            @interact(
                scale=widgets.ToggleButtons(
                    options=["linear", "logarithmic"],
                    value="linear",
                    description='Scale',
                    disabled=False,
                    style={'description_width': 'initial'}),
            )
            def plot_with_interactive_scale(scale, figsize):
                # Create figure
                if not figsize:
                    figsize = (7, 7)

                _, ax = plt.subplots(figsize=figsize)

                # Get scale
                log = scale == "logarithmic"

                if log:
                    ax.plot(np.log(data_array))
                else:
                    ax.plot(data_array)

                plt.title(title, fontsize=fontsize+2)
                plt.tight_layout()
                plt.show()

    elif data_dimensions == 2:
        # Depends on log scale
        if isinstance(log, bool):
            # Create figure
            if not figsize:
                figsize = (10, 10)

            fig, ax = plt.subplots(figsize=figsize)

            img = plot_2d_image(
                data_array, log=log, fig=fig, ax=ax, cmap=cmap,
                title=title, fontsize=fontsize
            )

            # Create axis for colorbar
            cbar_ax = make_axes_locatable(ax).append_axes(
                position='right', size='5%', pad=0.1)

            # Create colorbar
            fig.colorbar(mappable=img, cax=cbar_ax)

            # Show figure
            plt.tight_layout()
            plt.show()

        elif log == "interact":
            @interact(
                scale=widgets.ToggleButtons(
                    options=["linear", "logarithmic"],
                    value="linear",
                    description='Scale',
                    disabled=False,
                    style={'description_width': 'initial'}),
                figsize=fixed(figsize)
            )
            def plot_with_interactive_scale(scale, figsize):
                # Create figure
                if not figsize:
                    figsize = (10, 10)

                fig, ax = plt.subplots(figsize=figsize)

                # Get scale
                log = scale == "logarithmic"

                # Plot
                img = plot_2d_image(
                    data_array, log=log, fig=fig, ax=ax, cmap=cmap
                )

                # Create axis for colorbar
                cbar_ax = make_axes_locatable(ax).append_axes(
                    position='right', size='5%', pad=0.1)

                # Create colorbar
                fig.colorbar(mappable=img, cax=cbar_ax)

                # Show figure
                plt.tight_layout()
                plt.show()

    elif data_dimensions == 3:

        # Define function used to get data slice
        def get_data_slice(
            data: np.ndarray,
            axis: str = "x",
            index: int = 0,
            data_type: str = "Module",
            scale: str = "linear",
        ) -> np.ndarray:
            """
            Get a slice of 3D data along a specific axis and index.

            Parameters
            ----------
            data: np.ndarray
                The input 3D data.
            axis: str
                The axis along which to slice the data. Must be one of
                "x", "y", or "z".
            index: int
                The index along the specified axis at which to slice the
                data.
            data_type: str
                The data type to be returned. Must be one of "Real",
                "Imaginary", "Module", or "Phase".
            scale: str
                The scale to apply to the data. Must be one of "linear"
                or "logarithmic".

            Returns
            -------
                np.ndarray: The sliced data.
            """
            # Project on specific index
            if axis == "x":
                dt = data[index, :, :]
            elif axis == "y":
                dt = data[:, index, :]
            elif axis == "z":
                dt = data[:, :, index]

            # Data type
            if data_type == "Real":
                dt = np.real(dt)
            elif data_type == "Imaginary":
                dt = np.imag(dt)
            elif data_type == "Module":
                dt = np.abs(dt)
            elif data_type == "Phase":
                dt = np.angle(dt)

            # Scale
            if scale == "logarithmic":
                dt = np.where(dt > 0, np.log(dt), 0)

            lin_color_mapper.high = np.max(dt)

            return dt

        output_notebook()

        # Get bokeh palette from colormap
        TOOLTIPS = [
            ("x", "$x"),
            ("y", "$y"),
            ("value", "@data"),
        ]

        # List of compatible cmaps in bokeh
        bokey_cmaps = [
            p for p in bp.__palettes__ if p.endswith("256")
        ]

        palette = "Magma256"
        for p in bokey_cmaps:
            if cmap[1:] in p:  # skip capital letter
                palette = p
                print("Changing cmap to", palette)

        # Figure
        fig = figure(
            x_axis_label="x",
            y_axis_label="y",
            toolbar_location="above",
            toolbar_sticky=False,
            tools=(
                "pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair,"
                "hover, save"
            ),
            active_scroll="wheel_zoom",
            active_tap="auto",
            active_drag="box_zoom",
            active_inspect="auto",
            tooltips=TOOLTIPS
        )

        # Define source
        source = ColumnDataSource(
            data=dict(
                data=[np.abs(data_array[0, :, :])],
                dw=[data_array.shape[2]],
                dh=[data_array.shape[1]],
                index=[0],
                axis=["x"],
                data_type=["Module"],
                scale=["linear"],
            )
        )

        # Index
        def callback_change_index(attr, old, new):
            # Compute data
            dt = get_data_slice(
                data=data_array,
                axis=source.data["axis"][0],
                index=new,
                data_type=source.data["data_type"][0],
                scale=source.data["scale"][0],
            )

            # Save new values
            source.data["data"] = [dt]
            source.data["index"] = [new]

        slider_index = Slider(
            start=0,
            end=data_array.shape[0]-1,
            value=0,
            step=1,
            title="Position",
            orientation="horizontal",
        )
        slider_index.on_change('value', callback_change_index)

        # Axis
        def callback_change_axis(attr, old, new):
            # Get axis for projection
            new_axis = ["x", "y", "z"][new]

            # Compute data
            dt = get_data_slice(
                data=data_array,
                axis=new_axis,
                index=source.data["index"][0],
                data_type=source.data["data_type"][0],
                scale=source.data["scale"][0],
            )

            # Save new values
            source.data["data"] = [dt]
            source.data["axis"] = [new_axis]
            source.data["dw"][0] = dt.shape[1]
            source.data["dh"][0] = dt.shape[0]

            # Get new axis names and slider range
            if new_axis == "x":
                slider_range = data_array.shape[0]
                fig.axis[0].axis_label = "y"
                fig.axis[1].axis_label = "z"

            elif new_axis == "y":
                slider_range = data_array.shape[1]
                fig.axis[0].axis_label = "x"
                fig.axis[1].axis_label = "z"

            elif new_axis == "z":
                slider_range = data_array.shape[2]
                fig.axis[0].axis_label = "x"
                fig.axis[1].axis_label = "y"

            # Change slider range
            slider_index.end = slider_range-1

        select_axis = RadioButtonGroup(
            labels=["x", "y", "z"],
            active=0,
        )
        select_axis.on_change('active', callback_change_axis)

        # Data type
        def callback_change_data_type(attr, old, new):
            # Get data type
            new_data_type = ["Real", "Imaginary", "Module", "Phase"][new]

            # Compute data
            dt = get_data_slice(
                data=data_array,
                axis=source.data["axis"][0],
                index=source.data["index"][0],
                data_type=new_data_type,
                scale=source.data["scale"][0],
            )

            # Save new values
            source.data["data"] = [dt]
            source.data["data_type"] = [new_data_type]

        select_data_type = RadioButtonGroup(
            labels=["Real", "Imaginary", "Module", "Phase"],
            active=2,
        )
        select_data_type.on_change('active', callback_change_data_type)

        # Color bar
        def callback_change_cbar(attr, old, new):
            # Get new cbar
            new_cbar = ["linear", "logarithmic"][new]

            # Compute data
            dt = get_data_slice(
                data=data_array,
                axis=source.data["axis"][0],
                index=source.data["index"][0],
                data_type=source.data["data_type"],
                scale=new_cbar,
            )

            # Save new values
            source.data["data"] = [dt]
            source.data["scale"] = [new_cbar]

        select_cbar = RadioButtonGroup(
            labels=["linear", "logarithmic"],
            active=0,
        )
        select_cbar.on_change('active', callback_change_cbar)

        # Background
        fig.background_fill_color = "white"
        fig.background_fill_alpha = 0.5

        # Title
        fig.title.text_font = "futura"
        fig.title.text_font_style = "bold"
        fig.title.text_font_size = "15px"

        # Color bars
        lin_color_mapper = LinearColorMapper(
            palette=palette,
            low=0,
            high=np.max(source.data["data"][0]),
        )
        lin_color_bar = ColorBar(color_mapper=lin_color_mapper)

        fig.add_layout(lin_color_bar, 'right')

        # Image
        fig.image(
            image="data",
            source=source,
            x=0,
            y=0,
            dw="dw",
            dh="dh",
            color_mapper=lin_color_mapper,
        )

        # Create app layout
        app = pn.Column(
            pn.pane.Bokeh(
                row(
                    select_axis,
                    select_cbar,
                )
            ),
            pn.pane.Bokeh(
                row(
                    select_data_type,
                    slider_index,
                )
            ),
            pn.pane.Bokeh(
                row(
                    fig,
                )
            )
        )

        display(app)


def plot_2d_image(
    two_d_array: np.ndarray,
    fontsize: int = 15,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axes.Subplot = None,
    log: bool = False,
    cmap: str = "turbo",
    title: str = None,
    x_label: str = "x",
    y_label: str = "y",
    contour: bool = False,
) -> matplotlib.image.AxesImage:
    """
    Plots a 2D image of the input data.

    Parameters
    ----------
    two_d_array: np.ndarray
        A 2D array of data to be plotted.
    fontsize: (int, optional)
        The font size for the x and y labels, title, and colorbar
        (default 15).
    fig: matplotlib.figure.Figure, optional
        A matplotlib figure object to be used for plotting.
    ax: matplotlib.axes.Subplot, optional
        A matplotlib axes object to be used for plotting.
    log: bool, optional
        If True, the plot will be on a logarithmic scale
        (default False).
    cmap: str, optional
        The color map to be used for the image (default "turbo").
    title: str, optional
        The title for the plot (default None).
    x_label: str, optional
        The x label for the plot (default "x").
    y_label: str, optional
        The y label for the plot (default "y").
    contour: bool, optional
        If True, the plot will be a contour plot (default False).

    Returns:
    --------
    matplotlib.image.AxesImage
        An image object if the plot was successful, None otherwise.
    """

    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    scale = "logarithmic" if log else "linear"

    try:
        if np.iscomplex(two_d_array).any():
            print(
                "Using complex data, switching to array module for plot."
            )
            two_d_array = np.abs(two_d_array)
        if contour:
            img = ax.contourf(
                two_d_array,
                norm={"linear": None, "logarithmic": LogNorm()}[
                    scale],
                cmap=cmap,
                origin="lower",
            )
        else:
            img = ax.imshow(
                two_d_array,
                norm={"linear": None, "logarithmic": LogNorm()}[
                    scale],
                cmap=cmap,
                origin="lower",
            )
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        if isinstance(title, str):
            ax.set_title(title, fontsize=fontsize + 2)

        return img

    except ValueError:
        plt.close()
        if scale == "logarithmic":
            print("Log scale can not handle this kind of data ...")
        else:
            pass
        return None

    except TypeError:
        print("You probably took a slice on a detector gap ...")
        return None


def plot_3d_slices(
    data_array: np.ndarray,
    fontsize: int = 15,
    figsize: tuple[int, int] = None,
    log: bool = False,
    cmap: str = "turbo",
    title: str = None,
    contour: bool = False,
    sum_over_axis: bool = False,
) -> None:
    """
    Plot slices of a 3D array as images.

    The function takes a 3D `data_array` and plots three images, one for
    each axis, with the slices being taken at the middle of the
    corresponding dimension. The slices can be plotted with a
    logarithmic scale or linear scale and with a specified colormap. The
    images can also be plotted with or without contour lines. If
    `sum_over_axis` is set to True, the images will be obtained by
    summing the values over one of the dimensions.

    Parameters
    ----------
    data_array: numpy.ndarray
        The input 3D array.
    fontsize: int, optional
        The font size for the plot titles and axis labels. Default is 15.
    figsize: tuple, optional
        The size of the figure in inches. Default is None, meaning a
        default figure size is used.
    log: bool or None, optional
        If True, the images will be plotted on a logarithmic scale. If
        False, the images will be plotted on a linear scale. If None, a
        toggle widget will appear to switch between the two scales.
        Default is False.
    cmap: str, optional
        The colormap to use when plotting the images. Default is "turbo".
    title: str or tuple or list, optional
        The title for the figure or the titles for each of the three
        images. If a string is passed, it will be used as the title for
        the figure. If a list or tuple of three strings is passed, each
        string will be used as the title for one of the images. Default
        is None.
    contour: bool, optional
        If True, contour lines will be plotted over the images. Default
        is False.
    sum_over_axis: bool, optional
        If True, the images will be obtained by summing the values over
        one of the dimensions.
        Default is False.

    Returns
    -------
    None
        The function displays the plots, but returns nothing.
    """
    if isinstance(log, bool):
        # Create figure
        if not figsize:
            figsize = (15, 7)

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # Add titles
        if isinstance(title, str):
            fig.suptitle(title, fontsize=fontsize + 2, y=0.95)
            titles = [None, None, None]
        elif isinstance(title, tuple) and len(title) == 3\
                or isinstance(title, list) and len(title) == 3:
            titles = title
        else:
            titles = [None, None, None]

        # 3D array shape
        shape = data_array.shape

        # Plot first image
        if sum_over_axis:  # Compute sum
            two_d_array = np.sum(np.nan_to_num(data_array), axis=(0))

        else:  # Get middle slice
            two_d_array = data_array[shape[0]//2, :, :]
        img_x = plot_2d_image(
            two_d_array, fig=fig, title=titles[0], ax=axs[0], log=log,
            cmap=cmap, fontsize=fontsize, x_label="z", y_label="y",
            contour=contour
        )

        # Create axis for colorbar
        cbar_ax = make_axes_locatable(axs[0]).append_axes(
            position='right', size='5%', pad=0.1)

        # Create colorbar
        fig.colorbar(mappable=img_x, cax=cbar_ax)

        # Plot second image
        if sum_over_axis:  # Compute sum
            two_d_array = np.sum(np.nan_to_num(data_array), axis=(1))

        else:  # Get middle slice
            two_d_array = data_array[:, shape[1]//2, :]
        img_y = plot_2d_image(
            two_d_array, fig=fig, title=titles[1], ax=axs[1], log=log,
            cmap=cmap, fontsize=fontsize, x_label="z", y_label="x",
            contour=contour
        )

        # Create axis for colorbar
        cbar_ax = make_axes_locatable(axs[1]).append_axes(
            position='right', size='5%', pad=0.1)

        # Create colorbar
        fig.colorbar(mappable=img_y, cax=cbar_ax)

        # Plot third image
        if sum_over_axis:  # Compute sum
            two_d_array = np.sum(np.nan_to_num(data_array), axis=(2))

        else:  # Get middle slice
            two_d_array = data_array[:, :, shape[2]//2]
        img_z = plot_2d_image(
            two_d_array, fig=fig, title=titles[2], ax=axs[2], log=log,
            cmap=cmap, fontsize=fontsize, x_label="y", y_label="x",
            contour=contour
        )

        # Create axis for colorbar
        cbar_ax = make_axes_locatable(axs[2]).append_axes(
            position='right', size='5%', pad=0.1)

        # Create colorbar
        fig.colorbar(mappable=img_z, cax=cbar_ax)

        # Show figure
        fig.tight_layout()
        fig.show()

    else:
        @interact(
            scale=widgets.ToggleButtons(
                options=[
                    ("linear", False),
                    ("logarithmic", True),
                ],
                value=False,
                description='Scale',
                disabled=False,
                style={'description_width': 'initial'}),
            figsize=fixed(figsize)
        )
        def plot_with_interactive_scale(scale, figsize):
            try:
                plot_3d_slices(
                    data_array=data_array,
                    fontsize=fontsize,
                    figsize=figsize,
                    log=scale,
                    cmap=cmap,
                    title=title,
                    contour=contour,
                    sum_over_axis=sum_over_axis,
                )
            except IndexError:
                plt.close()
                print("Is this a 3D array?")


def complex2rgbalin(
    s: np.ndarray,
    gamma: float = 1.0,
    smax: float = None,
    smin: float = None,
    percentile: tuple[float, float] = (None, None),
    alpha: tuple[float, float] = (0, 1),
    final_type: str = 'uint8'
) -> np.ndarray:
    """
    Convert a complex valued array into an RGBA image with the magnitude
    encoded in the alpha channel.

    Parameters
    ----------
    s : np.ndarray
        The complex valued array to be converted.
    gamma : float, optional
        The gamma correction to apply to the magnitude, by default 1.0.
    smax : float, optional
        The maximum value to clip the magnitude to, by default None.
    smin : float, optional
        The minimum value to clip the magnitude to, by default None.
    percentile : tuple[float, float], optional
        The percentiles used to compute smin and smax, by default
        (None, None).
    alpha : tuple[float, float], optional
        The minimum and maximum values to use for the alpha channel,
        by default (0, 1).
    final_type : str, optional
        The type of the output image, either 'float' or 'uint8', by
        default 'uint8'.

    Returns
    -------
    np.ndarray
        The RGBA image with the magnitude encoded in the alpha channel.
    """
    rgba = phase2rgb(s)
    a = np.abs(s)
    if percentile is not None:
        if percentile[0] is not None:
            smin = np.percentile(a, percentile[0])
        if percentile[1] is not None:
            smax = np.percentile(a, percentile[1])
        if smax is not None and smin is not None and smin > smax:
            smin, smax = smax, smin
    if smax is not None:
        a = (a - smax) * (a <= smax) + smax
    if smin is not None:
        a = (a - smin) * (a >= smin)
    a /= a.max()
    a = a ** gamma
    rgba[..., 3] = alpha[0] + alpha[1] * a
    if final_type == 'float':
        return rgba
    return (rgba * 255).astype(np.uint8)


def phase2rgb(s: np.ndarray) -> np.ndarray:
    """
    Convert a complex numpy array into an RGBA image, color-coding the
    phase.

    Parameters
    ----------
    s : np.ndarray
        A complex numpy array.

    Returns
    -------
    np.ndarray
        An RGBA numpy array with an added dimension.

    Notes
    -----
    The conversion is based on code from PyNX.
    """
    ph = np.angle(s)
    t = np.pi / 3
    rgba = np.zeros(list(s.shape) + [4])
    rgba[..., 0] = (ph < t) * (ph > -t) + (ph > t) * (ph < 2 * t) * \
        (2 * t - ph) / t + (ph > -2 * t) * (ph < -t) * (
        ph + 2 * t) / t
    rgba[..., 1] = (ph > t) + (ph < -2 * t) * (-2 * t - ph) / \
        t + (ph > 0) * (ph < t) * ph / t
    rgba[..., 2] = (ph < -t) + (ph > -t) * (ph < 0) * (-ph) / \
        t + (ph > 2 * t) * (ph - 2 * t) / t
    return rgba
