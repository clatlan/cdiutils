"""
3D viewer widget for interactive visualisation of CDI reconstruction data.

This module provides the ThreeDViewer class for displaying 3D objects
from CDI optimisation results using Plotly.
"""

import ipywidgets as widgets
import numpy as np
from IPython.display import HTML, display

try:
    import plotly.graph_objects as go
    from scipy.interpolate import RegularGridInterpolator
    from skimage.measure import marching_cubes

    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False

# check if volume module is available for shared utilities
try:
    from .volume import (
        _extract_isosurface_with_values,
        colorcet_to_plotly,
    )

    HAS_VOLUME_UTILS = True
except ImportError:
    HAS_VOLUME_UTILS = False
    import matplotlib.pyplot as plt

    def colorcet_to_plotly(cmap_name: str, n_colors: int = 256):
        """Fallback colormap converter."""
        if cmap_name not in plt.colormaps():
            raise ValueError(f"Colormap '{cmap_name}' not found.")
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
        return [
            [
                i / (n_colors - 1),
                f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})",
            ]
            for i, c in enumerate(colors)
        ]


class ThreeDViewer(widgets.Box):
    """
    Widget to display 3D objects from CDI optimisation using Plotly.

    This class provides interactive 3D visualisation of volumetric data
    with controls for threshold, phase/amplitude display, and colormap
    selection.

    Interactive controls:
        - Threshold slider: controls the isosurface level
        - Phase/Amplitude toggle: switches between phase and amplitude
          display
        - Colormap dropdown: selects the colormap for the surface colour
        - Auto-scale checkbox: automatically scales the colorbar to data
          range
        - Symmetric checkbox: forces the colorbar to be symmetric around
          zero
        - Set limits checkbox: enables manual vmin/vmax input fields
        - Replace NaN with mean checkbox: replaces NaN values in the
          displayed quantity with the mean value to avoid weird
          colouring artefacts
    """

    # colormaps (1D - standard matplotlib/colorcet)
    cmap_options = (
        "turbo",
        "viridis",
        "inferno",
        "magma",
        "plasma",
        "cividis",
        "RdBu",
        "coolwarm",
        "twilight",
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

    def __init__(
        self,
        input_file: np.ndarray | None = None,
        html_width: int | None = None,
        voxel_size: tuple = (1, 1, 1),
        figsize: tuple = (9, 6),
    ):
        """
        Initialise the 3D viewer with Plotly backend.

        Args:
            input_file (np.ndarray | None, optional): 3D complex array
                to visualise. Defaults to None.
            html_width (int | None, optional): HTML width in %. If
                given, the width of the notebook will be changed to that
                value (e.g. full width with 100). Defaults to None.
            voxel_size (tuple, optional): voxel size (dx, dy, dz) for
                proper scaling. Defaults to (1, 1, 1).
            figsize (tuple, optional): figure size in inches
                (width, height). Defaults to (9, 6).

        Raises:
            ImportError: if plotly or required packages are not
                installed.
        """
        if not IS_PLOTLY_AVAILABLE:
            raise ImportError(
                "ThreeDViewer requires plotly, scikit-image, and scipy. "
                "Install with: pip install cdiutils[interactive]"
            )

        super().__init__()

        if html_width is not None:
            html_code = f"""
                <style>.container {{
                    width:{int(html_width)}%
                !important; }}</style>
                """
            display(HTML(html_code))

        # store parameters
        self.voxel_size = np.array(voxel_size)
        self.figsize = figsize

        # create plotly figure
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            template="plotly_white",
            scene=dict(
                xaxis=dict(showbackground=True, title="x"),
                yaxis=dict(showbackground=True, title="y"),
                zaxis=dict(showbackground=True, title="z"),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    # improve zoom sensitivity
                    projection=dict(type="perspective"),
                ),
            ),
            width=figsize[0] * 96,
            height=figsize[1] * 96,
            dragmode="orbit",
        )

        # create control widgets
        self.threshold = widgets.FloatSlider(
            value=5,
            min=0,
            max=20,
            step=0.02,
            description="Threshold:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        )

        self.toggle_phase = widgets.ToggleButtons(
            options=["Amplitude", "Phase"],
            description="Display:",
            disabled=False,
            value="Phase",
            button_style="",
        )

        self.toggle_rotate = widgets.ToggleButton(
            value=False,
            description="Rotate",
            tooltips="Rotate view",
        )

        # colormap dropdown - default depends on mode
        self.colormap = widgets.Dropdown(
            options=self.cmap_options,
            value="cet_CET_C9s_r",
            description="Colormap:",
            disabled=False,
        )

        self.theme_toggle = widgets.ToggleButton(
            value=False,
            description="Dark Theme",
            tooltips="Toggle dark/light theme",
        )

        # colorbar control checkboxes
        self.auto_scale = widgets.Checkbox(
            value=True,
            description="Auto-scale colorbar",
            tooltips="Scale colorbar to min/max of current plot",
            indent=False,
        )

        self.symmetric_scale = widgets.Checkbox(
            value=False,
            description="Symmetric colorbar",
            tooltips="Center colorbar at 0 (for strain, phase, etc.)",
            indent=False,
        )

        self.replace_nan = widgets.Checkbox(
            value=False,
            description="Replace NaN with mean",
            tooltips="Replace NaN values with mean (fixes weird colouring)",
            indent=False,
        )

        # set observers
        self.threshold.observe(self._on_update_plot, names="value")
        self.toggle_phase.observe(self._on_change_display, names="value")
        self.colormap.observe(self._on_update_plot, names="value")
        self.theme_toggle.observe(self._on_update_style, names="value")
        self.toggle_rotate.observe(self._on_animate, names="value")
        self.auto_scale.observe(self._on_update_plot, names="value")
        self.symmetric_scale.observe(self._on_update_plot, names="value")
        self.replace_nan.observe(self._on_update_plot, names="value")

        # internal state
        self.data = None
        self.rgi = None  # interpolator
        self._rotation_angle = 0
        self._rotation_callback = None

        # create layout
        controls_row1 = widgets.HBox([self.threshold])
        controls_row2 = widgets.HBox([self.toggle_phase, self.toggle_rotate])
        controls_row3 = widgets.HBox([self.colormap, self.theme_toggle])
        controls_row4 = widgets.HBox(
            [self.auto_scale, self.symmetric_scale, self.replace_nan]
        )

        self.vbox = widgets.VBox(
            [controls_row1, controls_row2, controls_row3, controls_row4]
        )

        # load data if provided
        if isinstance(input_file, np.ndarray):
            self.set_data(input_file)

        # set children for the Box widget
        self.children = [self.fig, self.vbox]

    def show(self) -> None:
        """Display the 3D viewer widget."""
        display(self)

    def set_data(
        self, data: np.ndarray, threshold: float | None = None
    ) -> None:
        """
        Set the 3D data to visualise.

        Args:
            data (np.ndarray): 3D complex array to visualise.
            threshold (float | None, optional): initial threshold value.
                If None, uses current slider value. Defaults to None.
        """
        self.data = data

        # create interpolator for getting values at mesh vertices
        nz, ny, nx = data.shape
        grid_z = np.arange(nz)
        grid_y = np.arange(ny)
        grid_x = np.arange(nx)
        self.rgi = RegularGridInterpolator(
            (grid_z, grid_y, grid_x),
            data,
            bounds_error=False,
            fill_value=0,
        )

        # update threshold range if needed, handle NaN values
        amp = np.abs(data)
        self.threshold.max = float(np.nanmax(amp))
        if threshold is not None:
            self.threshold.value = threshold

        # initial plot
        self._on_update_plot()

    def _on_update_plot(self, change=None) -> None:
        """
        Update the plot according to parameters.

        Args:
            change: widget change event (not used but required by
                observer).
        """
        if self.data is None:
            return

        try:
            # use shared helper function if available
            if HAS_VOLUME_UTILS:
                verts_scaled, faces, vals = _extract_isosurface_with_values(
                    self.data,
                    self.data,
                    self.threshold.value,
                    self.voxel_size,
                    use_interpolator=True,  # needed for complex
                )
            else:
                # fallback: inline implementation
                verts, faces, _, _ = marching_cubes(
                    np.abs(self.data),
                    level=self.threshold.value,
                    step_size=1,
                )
                verts_scaled = verts * self.voxel_size
                vals = self.rgi(verts)

            # optionally replace NaN values with mean to fix weird
            # colouring
            if self.replace_nan.value and np.any(np.isnan(vals)):
                mean_val = np.nanmean(vals)
                vals = np.where(np.isnan(vals), mean_val, vals)

            # determine colours based on display mode
            if self.toggle_phase.value == "Phase":
                # get phase values
                phase_vals = np.angle(vals)

                # determine colour range based on settings
                if self.symmetric_scale.value:
                    # symmetric around 0, use actual phase values
                    intensity = phase_vals
                    cmin, cmax = -np.pi, np.pi
                    colorbar = dict(
                        title="Phase (rad)",
                        tickmode="array",
                        tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                        ticktext=["-π", "-π/2", "0", "π/2", "π"],
                        len=0.7,
                        x=0.85,
                        showticklabels=True,
                        thickness=20,
                        lenmode="fraction",
                        xanchor="left",
                    )
                elif self.auto_scale.value:
                    # auto-scale to actual data range, handle NaN
                    intensity = phase_vals
                    cmin, cmax = (
                        float(np.nanmin(phase_vals)),
                        float(np.nanmax(phase_vals)),
                    )
                    colorbar = dict(
                        title="Phase (rad)",
                        len=0.7,
                        x=0.85,
                        showticklabels=True,
                        thickness=20,
                        lenmode="fraction",
                        xanchor="left",
                    )
                else:
                    # normalise to [0, 1] for full range
                    intensity = (phase_vals + np.pi) / (2 * np.pi)
                    cmin, cmax = 0, 1
                    colorbar = dict(
                        title="Phase (rad)",
                        tickmode="array",
                        tickvals=[0, 0.25, 0.5, 0.75, 1],
                        ticktext=["-π", "-π/2", "0", "π/2", "π"],
                        len=0.7,
                        x=0.85,
                        showticklabels=True,
                        thickness=20,
                        lenmode="fraction",
                        xanchor="left",
                    )

                vertex_colors = None
                colorscale = colorcet_to_plotly(self.colormap.value)

            else:  # amplitude
                # use actual amplitude values (not normalised)
                intensity = np.abs(vals)

                # determine colour range based on settings
                if self.symmetric_scale.value:
                    # symmetric around 0 - doesn't make much sense
                    # for amplitude but keep for consistency; centre
                    # at mean
                    mean_val = float(np.nanmean(intensity))
                    max_dev = float(
                        max(
                            np.nanmax(intensity) - mean_val,
                            mean_val - np.nanmin(intensity),
                        )
                    )
                    cmin = mean_val - max_dev
                    cmax = mean_val + max_dev
                elif self.auto_scale.value:
                    # auto-scale to actual data range, handle NaN
                    cmin, cmax = (
                        float(np.nanmin(intensity)),
                        float(np.nanmax(intensity)),
                    )
                else:
                    # use full range from data, handle NaN
                    cmin, cmax = (
                        float(np.nanmin(intensity)),
                        float(np.nanmax(intensity)),
                    )

                vertex_colors = None
                colorscale = colorcet_to_plotly(self.colormap.value)
                colorbar = dict(
                    title="Amplitude",
                    len=0.7,
                    x=0.85,
                    showticklabels=True,
                    thickness=20,
                    lenmode="fraction",
                    xanchor="left",
                )

            # update or create mesh
            with self.fig.batch_update():
                if len(self.fig.data) == 0:
                    # first time - add mesh
                    mesh_args = dict(
                        x=verts_scaled[:, 0],
                        y=verts_scaled[:, 1],
                        z=verts_scaled[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        intensity=intensity,
                        vertexcolor=vertex_colors,
                        colorscale=colorscale,
                        colorbar=colorbar,
                        cmin=cmin,
                        cmax=cmax,
                        opacity=1.0,
                        flatshading=False,
                        lighting=dict(
                            ambient=0.85,
                            diffuse=0.1,
                            specular=0.5,
                            roughness=0.2,
                            fresnel=0.5,
                        ),
                    )
                    self.fig.add_trace(go.Mesh3d(**mesh_args))

                else:
                    # update existing mesh
                    self.fig.data[0].x = verts_scaled[:, 0]
                    self.fig.data[0].y = verts_scaled[:, 1]
                    self.fig.data[0].z = verts_scaled[:, 2]
                    self.fig.data[0].i = faces[:, 0]
                    self.fig.data[0].j = faces[:, 1]
                    self.fig.data[0].k = faces[:, 2]
                    self.fig.data[0].intensity = intensity
                    self.fig.data[0].vertexcolor = vertex_colors
                    self.fig.data[0].colorscale = colorscale
                    self.fig.data[0].colorbar = colorbar
                    self.fig.data[0].cmin = cmin
                    self.fig.data[0].cmax = cmax

        except Exception as e:
            print(f"Error updating plot: {e}")

    def _on_change_display(self, change) -> None:
        """
        Handle display mode change (amplitude/phase).

        Args:
            change: widget change event.
        """
        if change["name"] == "value":
            # switch default colormap based on mode
            if change["new"] == "Phase":
                # use cyclic colormap for phase (good defaults)
                if self.colormap.value not in [
                    "twilight",
                    "cet_CET_C9s_r",
                    "jch_const",
                    "jch_max",
                ]:
                    self.colormap.value = "cet_CET_C9s_r"
                # enable symmetric scale for phase (centered at 0)
                if not self.symmetric_scale.value:
                    self.symmetric_scale.value = True
            else:
                # use sequential colormap for amplitude
                if self.colormap.value in [
                    "twilight",
                    "cet_CET_C9s_r",
                    "jch_const",
                    "jch_max",
                ]:
                    self.colormap.value = "turbo"
                # disable symmetric scale for amplitude (usually not needed)
                if self.symmetric_scale.value:
                    self.symmetric_scale.value = False

            # update plot
            self._on_update_plot()

    def _on_update_style(self, change) -> None:
        """
        Update the plot style (theme).

        Args:
            change: widget change event.
        """
        if change["name"] == "value":
            if self.theme_toggle.value:
                self.fig.update_layout(template="plotly_dark")
            else:
                self.fig.update_layout(template="plotly_white")

    def _on_animate(self, change) -> None:
        """
        Handle rotation animation toggle.

        Args:
            change: widget change event.
        """
        if change["name"] == "value":
            if change["new"]:  # start rotation
                self._start_rotation()
            else:  # stop rotation
                self._stop_rotation()

    def _start_rotation(self) -> None:
        """Start continuous rotation animation."""
        import asyncio

        async def rotate():
            """Async rotation loop."""
            while self.toggle_rotate.value:
                self._rotation_angle += 2
                # update camera azimuth
                eye_x = 1.5 * np.cos(np.radians(self._rotation_angle))
                eye_y = 1.5 * np.sin(np.radians(self._rotation_angle))
                eye_z = 1.5

                with self.fig.batch_update():
                    self.fig.layout.scene.camera.eye = dict(
                        x=eye_x, y=eye_y, z=eye_z
                    )

                await asyncio.sleep(0.05)  # ~20 FPS

        # create and run task
        loop = asyncio.get_event_loop()
        self._rotation_callback = loop.create_task(rotate())

    def _stop_rotation(self) -> None:
        """Stop rotation animation."""
        if self._rotation_callback is not None:
            self._rotation_callback.cancel()
            self._rotation_callback = None
