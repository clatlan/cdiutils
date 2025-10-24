"""
3D viewer widget for interactive visualization of CDI reconstruction data.

This module provides the ThreeDViewer class for displaying 3D objects
from CDI optimization results using ipyvolume.
"""

import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
import ipyvolume as ipv
from skimage.measure import marching_cubes
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
from matplotlib import cm
from matplotlib.colors import Normalize
from tornado.ioloop import PeriodicCallback

from cdiutils.plot.colormap import complex_to_rgb


class ThreeDViewer(widgets.Box):
    """
    Widget to display 3D objects from CDI optimisation, loaded from a result
    CXI file or a mode file.

    Simplified from the widgets class in PyNX @Vincent Favre Nicolin (ESRF)
    """

    cmap_options = (
        "turbo",
        "viridis",
        "spectral",
        "inferno",
        "magma",
        "plasma",
        "cividis",
        "RdBu",
        "coolwarm",
        "Blues",
        "Greens",
        "Greys",
        "Purples",
        "Oranges",
        "Reds",
        "cet_CET_D13",
        "cet_CET_C9s_r",
        "cet_CET_D1A",
        "jch_const",
        "jch_max",
    )

    def __init__(self, input_file=None, html_width=None):
        """
        Initialise the output and widgets.

        Args:
            input_file: The data filename or directly the 3D data array.
            html_width: html width in %. If given, the width of the
                notebook will be changed to that value (e.g. full width with 100).
        """
        super(ThreeDViewer, self).__init__()

        if html_width is not None:
            # flake8: noqa
            # type: ignore
            html_code = f"""
                    <style>.container {{
                        width:{int(html_width)}%
                    !important; }}</style>
                    """
            display(HTML(html_code))  # type: ignore
            # type: ignore

        # focus_label = widgets.Label(value='Focal distance (cm):')
        self.threshold = widgets.FloatSlider(
            value=5,
            min=0,
            max=20,
            step=0.02,
            description="Contour.",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".01f",
        )
        self.toggle_phase = widgets.ToggleButtons(
            options=["Abs", "Phase"],
            description="",
            disabled=False,
            value="Phase",
            button_style="",
        )
        self.toggle_rotate = widgets.ToggleButton(
            value=False,
            description="Rotate",
            tooltips="Rotate",
        )
        self.pcb_rotate = None
        hbox1 = widgets.HBox([self.toggle_phase, self.toggle_rotate])

        self.toggle_dark = widgets.ToggleButton(
            value=False,
            description="Dark",
            tooltips="Dark/Light theme",
        )
        self.toggle_box = widgets.ToggleButton(
            value=True,
            description="Box",
            tooltips="Box ?",
        )
        self.toggle_axes = widgets.ToggleButton(
            value=True,
            description="Axes",
            tooltips="Axes ?",
        )
        hbox_toggle = widgets.HBox(
            [self.toggle_dark, self.toggle_box, self.toggle_axes]
        )

        # Colormap widgets
        self.colormap = widgets.Dropdown(
            options=self.cmap_options,
            value="turbo",
            description="Colors:",
            disabled=True,
        )
        self.colormap_range = widgets.FloatRangeSlider(
            value=[20, 80],
            min=0,
            max=100,
            step=1,
            description="Range:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
        )

        # Progress bar
        self.progress = widgets.IntProgress(
            value=10,
            min=0,
            max=10,
            description="Processing:",
            bar_style="",
            style={"bar_color": "green"},
            orientation="horizontal",
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
        self.vbox = widgets.VBox(
            [
                self.threshold,
                hbox1,
                hbox_toggle,
                self.colormap,
                # self.colormap_range, # useless for one contour
                self.progress,
            ]
        )

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
        recomputed.

        Args:
            change: Used to update the values.
        """
        if change is not None and change["name"] != "value":
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
                rgb = complex_to_rgb(
                    vals, cmap="jch_const", output_type="float"
                )
                color = rgb  # Already in [0, 1] range

            # Linear or log colouring
            elif self.toggle_phase.value in ["Abs", "log10(Abs)"]:
                self.colormap.disabled = False
                cs = cm.ScalarMappable(
                    norm=Normalize(
                        vmin=self.colormap_range.value[0],
                        vmax=self.colormap_range.value[1],
                    ),
                    cmap=self.colormap.value.lower(),
                )
                color = cs.to_rgba(abs(vals))[..., :3]
            else:
                # TODO: Gradient
                gx, gy, gz = (
                    self.rgi_gx(verts),
                    self.rgi_gy(verts),
                    self.rgi_gz(verts),
                )
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
        do not involve recomputing the displayed object.

        Args:
            change: Dict from widget.
        """
        if change["name"] == "value":
            if self.toggle_dark.value:
                ipv.pylab.style.set_style_dark()
            else:
                ipv.pylab.style.set_style_light()
                # Fix label colours (see self.fig.style)
                ipv.pylab.style.use(
                    {
                        "axes": {
                            "label": {"color": "black"},
                            "ticklabel": {"color": "black"},
                        }
                    }
                )
            if self.toggle_box.value:
                ipv.pylab.style.box_on()
            else:
                ipv.pylab.style.box_off()
            if self.toggle_axes.value:
                ipv.pylab.style.axes_on()
            else:
                ipv.pylab.style.axes_off()

    def on_change_scale(self, change):
        """Change scale between logarithmic and linear."""
        if change["name"] == "value":
            if isinstance(change["old"], str):
                newv = change["new"]
                oldv = change["old"]

                # linear scale
                if "log" in oldv and "log" not in newv:
                    data = self.d0
                    self.set_data(data, threshold=10**self.threshold.value)

                # log scale
                elif "log" in newv and "log" not in oldv:
                    self.d0 = self.data
                    data = np.log10(np.maximum(0.1, abs(self.d0)))
                    self.set_data(
                        data, threshold=np.log10(self.threshold.value)
                    )
                    return
            self.on_update_plot()

    def set_data(self, data, threshold=None):
        """
        Check if data is complex or not.

        Args:
            data: Data 3d array, complex or not, to be plotted.
            threshold: Threshold for contour, if None set to max/2.
        """
        # Update progress bar
        self.progress.value = 5

        # Save data
        self.data = data

        # Change scale options depending on data
        self.toggle_phase.unobserve(self.on_change_scale)

        if np.iscomplexobj(data):
            if self.toggle_phase.value == "log10(Abs)":
                self.toggle_phase.value = "Abs"
            self.toggle_phase.options = ("Abs", "Phase")
        else:
            if self.toggle_phase.value == "Phase":
                self.toggle_phase.value = "Abs"
            self.toggle_phase.options = ("Abs", "log10(Abs)")
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
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

        # Also prepare the phase gradient
        gz, gy, gx = np.gradient(self.data)
        a = np.maximum(abs(self.data), 1e-6)
        ph = self.data / a
        gaz, gay, gax = np.gradient(a)
        self.rgi_gx = RegularGridInterpolator(
            (z, y, x),
            ((gx - gax * ph) / (ph * a)).real,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        self.rgi_gy = RegularGridInterpolator(
            (z, y, x),
            ((gy - gay * ph) / (ph * a)).real,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        self.rgi_gz = RegularGridInterpolator(
            (z, y, x),
            ((gz - gaz * ph) / (ph * a)).real,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

        # Fix extent otherwise weird things happen
        ipv.pylab.xlim(0, self.data.shape[0])
        ipv.pylab.ylim(0, self.data.shape[1])
        ipv.pylab.zlim(0, self.data.shape[2])
        # ipv.squarelim()
        self.on_update_plot()

    def on_animate(self, v):
        """Trigger the animation (rotation around vertical axis)."""
        if self.pcb_rotate is None:
            self.pcb_rotate = PeriodicCallback(self.callback_rotate, 50.0)
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
        n = np.sqrt(x**2 + y**2 + z**2)
        a = np.deg2rad(2.5) / 2  # angular step
        sa, ca = np.sin(a / 2) / n, np.cos(a / 2)
        r = Rotation.from_quat((sa * x, sa * y, sa * z, ca))
        self.fig.camera.position = tuple(r.apply(self.fig.camera.position))
