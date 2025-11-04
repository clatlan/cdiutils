"""
Interactive 3D volume visualisation tools for BCDI data.

This module provides interactive widgets for visualising 3D volumetric data
with different backends:
- ThreeDViewer: Plotly-based interactive 3D viewer class
  (requires plotly, ipywidgets - included in: pip install cdiutils[interactive])
- plot_3d_isosurface: Plotly-based isosurface function
  (requires plotly, ipywidgets - included in: pip install cdiutils[interactive])
- VolumeViewer: PyVista/Trame-based visualisation
  (requires pyvista, trame - install with: pip install cdiutils[pyvista])

The Plotly-based functions are recommended for most use cases as they provide
excellent interactive performance and are included in the standard interactive
dependencies. PyVista/Trame is available for specialised workflows.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# PyVista/Trame availability checked in __init__.py, but we need the imports
try:
    import pyvista as pv
    from pyvista.trame.ui.vuetify3 import divider, slider, select

    IS_PYVISTA_AVAILABLE = True
except ImportError:
    IS_PYVISTA_AVAILABLE = False
    pv = None

# Plotly and related imports
try:
    import plotly.graph_objects as go
    from skimage import measure
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import RegularGridInterpolator

    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False
    go = None


def plot_3d_isosurface(
    amplitude: np.ndarray,
    quantities: dict[str, np.ndarray],
    voxel_size: tuple[float, float, float] | None = None,
    initial_quantity: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    convention: str | None = None,
    figsize: tuple[int, int] = (9, 6),
    title: str | None = None,
    lighting_params: dict[str, float] | None = None,
    camera_position: dict | None = None,
    theme: str = "plotly_white",
):
    """
    Plot an interactive 3D isosurface using Plotly and ipywidgets.

    This function creates an interactive 3D visualisation where users can
    adjust the isosurface threshold, switch between different scalar
    quantities, change colormaps, and control colourbar scaling using
    interactive widgets. The camera view is preserved when updating.

    Interactive Controls:
        - Isosurface slider: Adjust threshold level (0-1 normalised)
        - Quantity dropdown: Switch between different quantities
        - Colourmap dropdown: Change the colourmap on-the-fly
        - Set limits checkbox: Enable manual colourbar limits
          (default: OFF)
          * When OFF: Auto-scales to min/max of current plot
          * When ON: Enables vmin/vmax input fields for manual control
        - vmin/vmax inputs: Set custom colourbar limits (disabled
          unless "Set limits" is checked)
        - Symmetric colourbar: Centre at 0 for strain/phase data
          (default: OFF)
        - Replace NaN with mean: Replace NaN values with mean to avoid
          weird colouring artefacts (default: OFF)

    Args:
        amplitude (np.ndarray): 3D array for determining isosurface
            levels.
        quantities (dict[str, np.ndarray]): Dictionary of 3D arrays to
            visualise, with keys as quantity names and values as numpy
            arrays.
        voxel_size (tuple[float, float, float] | None, optional): Size
            of voxels (dx, dy, dz) for proper scaling. Defaults to
            (1.0, 1.0, 1.0).
        initial_quantity (str | None, optional): Name of the quantity to
            display initially. Must be a key in quantities dict. If None,
            uses the first key in quantities. Defaults to None.
        cmap (str | None, optional): Initial colourmap name. Defaults to
            "viridis".
        vmin (float | None, optional): Initial minimum value for colour
            scale. If None, auto-scales to data minimum. Defaults to
            None.
        vmax (float | None, optional): Initial maximum value for colour
            scale. If None, auto-scales to data maximum. Defaults to
            None.
        convention (str | None, optional): Coordinate convention
            ("cxi" or "xu"). Defaults to "cxi".
        figsize (tuple[int, int], optional): Figure size in inches
            (width, height). Defaults to (9, 6).
        title (str | None, optional): Plot title. Defaults to
            "Interactive 3D Isosurface".
        lighting_params (dict[str, float] | None, optional): Plotly
            lighting parameters. Defaults to preset values.
        camera_position (dict | None, optional): Initial camera position.
            Defaults to eye=dict(x=1.5, y=1.5, z=1.5).
        theme (str, optional): Plotly theme. Defaults to "plotly_white".

    Returns:
        VBox: ipywidgets VBox containing controls and the figure widget.

    Raises:
        PlotlyImportError: if plotly or required packages are not
            installed.
        ValueError: if amplitude and quantities have different shapes, or
            if initial_quantity is not in quantities dict.

    Example:
        >>> import numpy as np
        >>> amp = np.random.rand(50, 50, 50)
        >>> strain = np.random.randn(50, 50, 50) * 0.01
        >>> phase = np.random.randn(50, 50, 50) * np.pi
        >>> widget = plot_3d_isosurface(
        ...     amp,
        ...     {"het_strain": strain, "phase": phase},
        ...     voxel_size=(1.0, 1.0, 1.0),
        ...     initial_quantity="het_strain",
        ...     cmap="cet_CET_D13"
        ... )
        >>> display(widget)
    """
    if not IS_PLOTLY_AVAILABLE:
        raise PlotlyImportError()

    try:
        from ipywidgets import (
            FloatSlider,
            Dropdown,
            VBox,
            HBox,
            Checkbox,
            FloatText,
        )
    except ImportError as e:
        raise PlotlyImportError(
            f"Required packages not available: {e}. "
            "Install with: pip install cdiutils[interactive]"
        )

    # validate inputs
    for name, quantity in quantities.items():
        if amplitude.shape != quantity.shape:
            raise ValueError(
                f"amplitude and quantity '{name}' must have the same "
                f"shape. Got {amplitude.shape} and {quantity.shape}"
            )

    # set defaults
    if voxel_size is None:
        voxel_size = (1.0, 1.0, 1.0)
    if lighting_params is None:
        lighting_params = dict(
            ambient=0.90,
            diffuse=0.05,
            specular=0.5,
            roughness=0.2,
            fresnel=0.5,
        )
    if camera_position is None:
        camera_position = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    if title is None:
        title = "Interactive 3D Isosurface"
    if convention is None:
        convention = "cxi"

    # determine initial quantity to display
    quantity_names = list(quantities.keys())
    if initial_quantity is None:
        initial_qty_name = quantity_names[0]
    else:
        if initial_quantity not in quantities:
            raise ValueError(
                f"initial_quantity '{initial_quantity}' not found in "
                f"quantities dict. Available: {quantity_names}"
            )
        initial_qty_name = initial_quantity

    # set initial colourmap
    if cmap is None:
        initial_cmap = "viridis"
    else:
        initial_cmap = cmap

    # create initial isosurface using shared helper
    isosurface_default = 0.5

    # handle NaN values in amplitude when calculating threshold
    amplitude_max = np.nanmax(amplitude)
    verts_scaled, faces, quantity_at_verts = _extract_isosurface_with_values(
        amplitude,
        quantities[initial_qty_name],
        isosurface_default * amplitude_max,
        voxel_size,
    )

    # set initial colourbar limits, handle NaN values
    if vmin is None:
        initial_vmin = np.nanmin(quantity_at_verts)
    else:
        initial_vmin = vmin

    if vmax is None:
        initial_vmax = np.nanmax(quantity_at_verts)
    else:
        initial_vmax = vmax

    # convert colourmap to plotly format
    initial_plotly_cmap = colorcet_to_plotly(initial_cmap)

    # Create FigureWidget (allows in-place updates)
    fig = go.FigureWidget(
        data=[
            go.Mesh3d(
                x=verts_scaled[:, 0],
                y=verts_scaled[:, 1],
                z=verts_scaled[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=quantity_at_verts,
                colorscale=initial_plotly_cmap,
                cmin=initial_vmin,
                cmax=initial_vmax,
                colorbar=dict(title=initial_qty_name),
                opacity=1.0,
                flatshading=False,
                lighting=lighting_params,
                hovertemplate=(
                    "<b>Position:</b><br>"
                    "x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<br>"
                    f"<b>{initial_qty_name}:</b> %{{intensity:.3f}}<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"{title} - {initial_qty_name} (iso={isosurface_default:.2f})",
        template=theme,
        scene=dict(
            xaxis=dict(showbackground=True, title="x"),
            yaxis=dict(showbackground=True, title="y"),
            zaxis=dict(showbackground=True, title="z"),
            aspectmode="data",
            camera=camera_position,
        ),
        width=figsize[0] * 96,
        height=figsize[1] * 96,
        dragmode="orbit",
    )

    # Create widgets
    isosurface_slider = FloatSlider(
        value=isosurface_default,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Isosurface:",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    quantity_dropdown = Dropdown(
        options=quantity_names,
        value=initial_qty_name,
        description="Quantity:",
        style={"description_width": "initial"},
    )

    # Colormap dropdown
    cmap_options = [
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
    ]
    colormap_dropdown = Dropdown(
        options=cmap_options,
        value=initial_cmap,
        description="Colormap:",
        style={"description_width": "initial"},
    )

    # Colorbar control checkboxes and inputs
    set_limits_checkbox = Checkbox(
        value=False,
        description="Set limits:",
        tooltip="Enable manual colorbar limits",
        indent=False,
        style={"description_width": "80px"},
    )

    vmin_input = FloatText(
        value=initial_vmin,
        description="vmin:",
        disabled=True,  # disabled by default
        style={"description_width": "50px"},
        layout={"width": "150px"},
    )

    vmax_input = FloatText(
        value=initial_vmax,
        description="vmax:",
        disabled=True,  # disabled by default
        style={"description_width": "50px"},
        layout={"width": "150px"},
    )

    symmetric_checkbox = Checkbox(
        value=False,
        description="Symmetric colorbar",
        tooltip="Center colorbar at 0 (for strain, phase, etc.)",
        indent=False,
        style={"description_width": "initial"},
    )

    replace_nan_checkbox = Checkbox(
        value=False,
        description="Replace NaN with mean",
        tooltip="Replace NaN values with mean (fixes weird colouring)",
        indent=False,
        style={"description_width": "initial"},
    )

    def toggle_limit_inputs(change=None) -> None:
        """
        Enable/disable limit input fields based on checkbox.

        Args:
            change: widget change event (not used but required by
                observer).
        """
        if set_limits_checkbox.value:
            vmin_input.disabled = False
            vmax_input.disabled = False
        else:
            vmin_input.disabled = True
            vmax_input.disabled = True

    def update_mesh(change=None) -> None:
        """
        Update mesh when slider or dropdown changes.

        Handles NaN values in quantity data using np.nanmin/np.nanmax
        for robust limit calculation. Can optionally replace NaN with
        mean value to avoid weird colouring.

        Args:
            change: widget change event (not used but required by
                observer).
        """
        iso_level = isosurface_slider.value
        qty_name = quantity_dropdown.value
        quantity = quantities[qty_name]

        # use shared helper function to extract isosurface
        verts_scaled, faces, quantity_at_verts = (
            _extract_isosurface_with_values(
                amplitude,
                quantity,
                iso_level * np.nanmax(amplitude),  # handle NaN in amplitude
                voxel_size,
            )
        )

        # optionally replace NaN values with mean to fix weird colouring
        if replace_nan_checkbox.value:
            if np.any(np.isnan(quantity_at_verts)):
                mean_val = np.nanmean(quantity_at_verts)
                quantity_at_verts = np.where(
                    np.isnan(quantity_at_verts),
                    mean_val,
                    quantity_at_verts,
                )

        # get colourmap from dropdown
        cmap = colormap_dropdown.value
        plotly_cmap = colorcet_to_plotly(cmap)

        # determine colour scale limits based on checkbox settings
        if symmetric_checkbox.value:
            # symmetric around 0, handle NaN values
            min_val = np.nanmin(quantity_at_verts)
            max_val = np.nanmax(quantity_at_verts)
            max_abs = max(abs(min_val), abs(max_val))
            vmin = -max_abs
            vmax = max_abs
            # update input fields to show current values
            # (but keep disabled if unchecked)
            vmin_input.value = vmin
            vmax_input.value = vmax
        elif set_limits_checkbox.value:
            # use manual limits from input fields
            vmin = vmin_input.value
            vmax = vmax_input.value
        else:
            # default: auto-scale to actual data range, handle NaN
            vmin = float(np.nanmin(quantity_at_verts))
            vmax = float(np.nanmax(quantity_at_verts))
            # update input fields to show current values
            # (but keep disabled)
            vmin_input.value = vmin
            vmax_input.value = vmax

        # update mesh data in-place (preserves camera view!)
        with fig.batch_update():
            fig.data[0].x = verts_scaled[:, 0]
            fig.data[0].y = verts_scaled[:, 1]
            fig.data[0].z = verts_scaled[:, 2]
            fig.data[0].i = faces[:, 0]
            fig.data[0].j = faces[:, 1]
            fig.data[0].k = faces[:, 2]
            fig.data[0].intensity = quantity_at_verts
            fig.data[0].colorscale = plotly_cmap
            fig.data[0].cmin = vmin
            fig.data[0].cmax = vmax
            fig.data[0].colorbar.title = qty_name
            fig.data[0].hovertemplate = (
                "<b>Position:</b><br>"
                "x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<br>"
                f"<b>{qty_name}:</b> %{{intensity:.3f}}<br>"
                "<extra></extra>"
            )
            fig.layout.title = f"{title} - {qty_name} (iso={iso_level:.2f})"

    # Attach observers to widgets
    isosurface_slider.observe(update_mesh, names="value")
    quantity_dropdown.observe(update_mesh, names="value")
    colormap_dropdown.observe(update_mesh, names="value")
    set_limits_checkbox.observe(toggle_limit_inputs, names="value")
    set_limits_checkbox.observe(update_mesh, names="value")
    vmin_input.observe(update_mesh, names="value")
    vmax_input.observe(update_mesh, names="value")
    symmetric_checkbox.observe(update_mesh, names="value")
    replace_nan_checkbox.observe(update_mesh, names="value")

    # create layout
    controls_row1 = HBox(
        [isosurface_slider, quantity_dropdown, colormap_dropdown]
    )
    controls_row2 = HBox(
        [
            set_limits_checkbox,
            vmin_input,
            vmax_input,
            symmetric_checkbox,
            replace_nan_checkbox,
        ]
    )
    widget = VBox([controls_row1, controls_row2, fig])

    return widget


class PyVistaImportError(ImportError):
    """Custom exception to handle PyVista/Trame import error."""

    def __init__(self, msg: str = None) -> None:
        """
        Initialise PyVistaImportError with informative message.

        Args:
            msg (str, optional): additional error message. Defaults to None.
        """
        _msg = (
            "PyVista and Trame packages are not installed. "
            "Install with: pip install cdiutils[pyvista]"
        )
        if msg is not None:
            _msg += "\n" + msg
        super().__init__(_msg)


class PlotlyImportError(ImportError):
    """Custom exception to handle Plotly import error."""

    def __init__(self, msg: str = None) -> None:
        """
        Initialise PlotlyImportError with informative message.

        Args:
            msg (str, optional): additional error message. Defaults to None.
        """
        _msg = (
            "Plotly and required packages are not installed. "
            "Install with: pip install cdiutils[interactive]"
        )
        if msg is not None:
            _msg += "\n" + msg
        super().__init__(_msg)


def _extract_isosurface_with_values(
    amplitude: np.ndarray,
    quantity: np.ndarray,
    isosurface_level: float,
    voxel_size: tuple = (1.0, 1.0, 1.0),
    use_interpolator: bool = False,
):
    """
    Extract isosurface and interpolate quantity values at vertices.

    This is a shared utility function used by both ThreeDViewer and
    plot_3d_isosurface to avoid code duplication.

    Args:
        amplitude (np.ndarray): 3D array for determining isosurface.
        quantity (np.ndarray): 3D array of values to interpolate at
            surface vertices (can be complex).
        isosurface_level (float): threshold value for marching cubes.
        voxel_size (tuple, optional): voxel size (dx, dy, dz) for
            scaling. Defaults to (1.0, 1.0, 1.0).
        use_interpolator (bool, optional): if True, use
            RegularGridInterpolator (needed for complex arrays).
            If False, use map_coordinates (faster for real arrays).
            Defaults to False.

    Returns:
        tuple: (verts_scaled, faces, quantity_at_verts) where:
            - verts_scaled: nx3 array of scaled vertex positions
            - faces: mx3 array of triangle face indices
            - quantity_at_verts: length-n array of interpolated values

    Raises:
        PlotlyImportError: if required packages not available.
    """
    if not IS_PLOTLY_AVAILABLE:
        raise PlotlyImportError()

    # extract isosurface using marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        np.abs(amplitude),
        level=isosurface_level,
        step_size=1,
    )

    # scale vertices
    verts_scaled = verts * voxel_size

    # interpolate quantity values at vertices
    if use_interpolator or np.iscomplexobj(quantity):
        # use RegularGridInterpolator for complex arrays
        nz, ny, nx = quantity.shape
        grid_z = np.arange(nz)
        grid_y = np.arange(ny)
        grid_x = np.arange(nx)
        rgi = RegularGridInterpolator(
            (grid_z, grid_y, grid_x),
            quantity,
            bounds_error=False,
            fill_value=0,
        )
        quantity_at_verts = rgi(verts)
    else:
        # use map_coordinates for real arrays (faster)
        quantity_at_verts = map_coordinates(
            quantity, verts.T, order=1, mode="nearest"
        )

    return verts_scaled, faces, quantity_at_verts


def colorcet_to_plotly(cmap_name: str, n_colors: int = 256) -> list[list]:
    """
    Convert a colorcet or matplotlib colormap to a Plotly colorscale.

    Args:
        cmap_name (str): name of the colorcet or matplotlib colormap
            (e.g., 'rainbow', 'fire', 'cet_CET_D13').
        n_colors (int, optional): number of colour samples to extract
            from the colormap. Defaults to 256.

    Returns:
        list[list]: Plotly colorscale as a list of
            [position, 'rgb(r,g,b)'] entries with positions in [0.0, 1.0].

    Raises:
        ValueError: if the specified colormap name is not found in the
            matplotlib/colorcet colormaps.
    """
    # get the colorcet colormap
    if cmap_name not in plt.colormaps():
        raise ValueError(
            f"Colormap '{cmap_name}' not found in matplotlib/colorcet "
            f"colormaps."
        )
    cmap = plt.get_cmap(cmap_name)

    # sample colours from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]

    # convert to Plotly format
    plotly_colorscale = [
        [
            i / (n_colors - 1),
            f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})",
        ]
        for i, c in enumerate(colors)
    ]

    return plotly_colorscale


class VolumeViewer:
    """
    A class to plot volume in 3D with Trame and PyVista.

    This class provides interactive 3D visualization of volumetric data
    using PyVista's Trame backend for Jupyter notebooks.

    Raises:
        PyVistaImportError: if Trame or PyVista are not installed.
    """

    generic_params = {
        "amplitude": {"cmap": "turbo", "centred_clim": False, "clim": [0, 1]},
        "support": {"cmap": "viridis", "centred_clim": False},
        "phase": {"cmap": "cet_CET_C9s_r", "centred_clim": True},
        "displacement": {"cmap": "cet_CET_D1A", "centred_clim": True},
        "het_strain": {"cmap": "cet_CET_D13", "centred_clim": True},
        "het_strain_from_dspacing": {
            "cmap": "cet_CET_D13",
            "centred_clim": True,
        },
        "lattice_parameter": {"cmap": "turbo", "centred_clim": False},
        "dspacing": {"cmap": "turbo", "centred_clim": False},
        "isosurface": 0.50,
        "cmap": "turbo",
    }

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

    @classmethod
    def _generate_toolbar_tools(
        cls, initial_scalar: str, available_scalars: list[str], **kwargs
    ) -> callable:
        """
        Generate toolbar widgets for the Trame interface.

        Args:
            initial_scalar (str): Initial scalar field to display.
            available_scalars (list[str]): List of available scalar fields.

        Returns:
            callable: Toolbar function for Trame UI.
        """

        def toolbar_tools() -> None:
            divider(vertical=True, classes="mx-1")

            # isosurface slider
            slider(
                model=("isosurface_value", cls.generic_params["isosurface"]),
                tooltip="Adjust isosurface threshold",
                min=0.0,
                max=1.0,
                step=0.01,
                dense=True,
                hide_details=False,
                style="width: 250px",
                classes="my-0 py-0 ml-1 mr-1",
            )

            divider(vertical=True, classes="mx-1")

            # scalar field dropdown
            select(
                model=("scalar_field", initial_scalar),
                tooltip="Choose scalar field for coloring",
                items=("available_scalars", available_scalars),
                hide_details=True,
                dense=True,
                outlined=True,
            )
            divider(vertical=True, classes="mx-1")

            # colourmap dropdown
            select(
                model=("cmap", cls.generic_params[initial_scalar]["cmap"]),
                tooltip="Choose a colourmap",
                items=("cmap_options", cls.cmap_options),
                hide_details=True,
                dense=True,
                outlined=True,
            )

        return toolbar_tools

    @classmethod
    def contour_plot(
        cls,
        data_path: str | None = None,
        initial_active_scalar: str = "het_strain",
        **data: np.ndarray,
    ):
        """
        Generate a contour plot application using PyVista.

        Args:
            data_path (str | None, optional): Path to a .vti file
                containing the data. Defaults to None.
            initial_active_scalar (str, optional): Initial scalar field
                to display. Defaults to "het_strain".
            **data (np.ndarray): Dictionary of numpy arrays to visualize.

        Raises:
            PyVistaImportError: if Trame or PyVista are not installed.
            ValueError: If the path is not a .vti file.
            ValueError: If initial_active_scalar is not available.
            NotImplementedError: When parsing np.ndarray directly
                (reserved for future use).

        Returns:
            The widget viewer for display in Jupyter notebooks.
        """
        if not IS_PYVISTA_AVAILABLE:
            raise PyVistaImportError()

        if data_path is not None:
            # ignoring the **data
            if not data_path.endswith(".vti"):
                raise ValueError(
                    "The provided data_path should point to a .vti file"
                )
            structure_grid = pv.read(data_path)
            available_scalars = structure_grid.array_names

        elif len(data) < 1:
            raise NotImplementedError(
                "Either np.ndarray or data_path must be provided."
            )
        else:
            initial_active_scalar = list(data.keys())[0]
            mesh = np.meshgrid(
                *[np.arange(s) for s in data[initial_active_scalar].shape],
                indexing="ij",
            )
            structure_grid = pv.StructuredGrid(*mesh)
            available_scalars = list(data.keys())
            for key, d in data.items():
                structure_grid.point_data[key] = d.flatten()

        plotter = pv.Plotter(notebook=True)

        # generate the initial isosurface
        contours = structure_grid.contour(
            [cls.generic_params["isosurface"]], scalars="amplitude"
        )
        if initial_active_scalar not in available_scalars:
            raise ValueError(
                f"initial_active_scalar (={initial_active_scalar}) "
                "cannot be found in the provided data."
            )
        contours.set_active_scalars(initial_active_scalar)
        initial_clim = cls.generic_params[initial_active_scalar].get("clim")
        if cls.generic_params[initial_active_scalar]["centred_clim"]:
            initial_clim = (
                -np.max(data[initial_active_scalar]),
                np.max(data[initial_active_scalar]),
            )

        mesh_actor = plotter.add_mesh(
            contours,
            scalars=initial_active_scalar,
            cmap=cls.generic_params[initial_active_scalar]["cmap"],
            clim=initial_clim,
            scalar_bar_args={
                "title": initial_active_scalar.replace("_", " ").capitalize()
            },
        )

        plotter.add_axes()

        # get the IPython widget
        widget = plotter.show(
            jupyter_kwargs={
                "add_menu_items": cls._generate_toolbar_tools(
                    initial_active_scalar, available_scalars
                )
            },
            return_viewer=True,
        )

        # connect Trame state with PyVista
        state = widget.viewer.server.state
        ctrl = widget.viewer.server.controller
        state.isosurface_value = cls.generic_params["isosurface"]
        state.scalar_field = initial_active_scalar
        state.cmap = cls.generic_params[initial_active_scalar]["cmap"]

        ctrl.view_update = widget.viewer.update

        # Trame Callbacks
        @state.change("isosurface_value")
        def update_isosurface(isosurface_value, **kwargs):
            """Update isosurface when slider changes."""
            new_contours = structure_grid.contour(
                [isosurface_value], scalars="amplitude"
            )
            new_contours.set_active_scalars(state.scalar_field)
            mesh_actor.mapper.dataset = new_contours
            ctrl.view_update()

        @state.change("scalar_field")
        def update_scalar_field(scalar_field, **kwargs):
            """Change the active scalar field dynamically."""
            contours.set_active_scalars(scalar_field)
            mesh_actor.mapper.array_name = scalar_field

            cmap = cls.generic_params[scalar_field]["cmap"]
            centred_clim = cls.generic_params[scalar_field]["centred_clim"]

            clim_range = list(contours.get_data_range(scalar_field))
            if centred_clim:
                max_abs = max(abs(clim_range[0]), abs(clim_range[1]))
                clim_range = [-max_abs, max_abs]
            else:
                clim_range = cls.generic_params[scalar_field].get("clim")

            mesh_actor.mapper.scalar_range = clim_range
            state.cmap = cmap
            mesh_actor.mapper.lookup_table = pv.LookupTable(cmap)

            plotter.remove_scalar_bar()
            plotter.add_scalar_bar(
                title=scalar_field.replace("_", " ").capitalize(), n_labels=5
            )

            ctrl.view_update()

        @state.change("cmap")
        def update_colourmap(cmap, **kwargs):
            """Update the colourmap dynamically."""
            state.cmap = cmap
            mesh_actor.mapper.lookup_table = pv.LookupTable(cmap)
            plotter.remove_scalar_bar()
            plotter.add_scalar_bar(
                title=state.scalar_field.replace("_", " ").capitalize(),
                n_labels=5,
            )
            ctrl.view_update()

        return widget

    @staticmethod
    def multi_mesh(
        scalar_field: np.ndarray,
        isosurfaces: list[float] | np.ndarray,
        initial_view: dict[str, float] = None,
        kwargs_mesh: dict[str, float | str | bool] = None,
        scalar_field_name: str = "Values",
        window_size: list[int] = None,
        plot_title: str = "3D view",
        interactive: bool = True,
        jupyter_backend: str = "client",
    ) -> None:
        """
        Visualise a 3D scalar field using PyVista with isosurfaces.

        This function creates a structured 3D grid from a scalar field
        and generates isosurfaces (contours) for the specified values.

        Args:
            scalar_field (np.ndarray): 3D array representing the scalar
                field to visualize.
            isosurfaces (list[float] | np.ndarray): List of scalar values
                for which isosurfaces will be generated.
            initial_view (dict[str, float], optional): Dictionary
                specifying the initial camera position. Defaults to None.
            kwargs_mesh (dict, optional): Keyword arguments for PyVista's
                add_mesh function. Defaults to None.
            scalar_field_name (str, optional): Name for the scalar field.
                Defaults to "Values".
            window_size (list[int], optional): Window size in pixels.
                Defaults to [1100, 700].
            plot_title (str, optional): Title for the plot window.
                Defaults to "3D view".
            interactive (bool, optional): Enable interactive mode.
                Defaults to True.
            jupyter_backend (str, optional): Backend for Jupyter display.
                Defaults to "client".

        Returns:
            None: Displays the 3D plot.

        Raises:
            PyVistaImportError: if PyVista is not installed.
        """
        if not IS_PYVISTA_AVAILABLE:
            raise PyVistaImportError()

        if window_size is None:
            window_size = [1100, 700]

        if kwargs_mesh is None:
            kwargs_mesh = {
                "cmap": "viridis",
                "opacity": 0.2,
                "show_edges": False,
                "style": "wireframe",
                "log_scale": False,
            }

        # Create grid for PyVista
        nx, ny, nz = scalar_field.shape
        x = np.arange(nx, dtype=np.float32)
        y = np.arange(ny, dtype=np.float32)
        z = np.arange(nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        grid = pv.StructuredGrid(X, Y, Z)

        grid.point_data[scalar_field_name] = scalar_field.flatten(order="F")

        # Generate contours for different isosurfaces
        contours = grid.contour(isosurfaces=isosurfaces, method="contour")

        plotter = pv.Plotter()
        plotter.add_mesh(contours, **kwargs_mesh)

        # Set the initial view if provided
        if initial_view:
            if "azimuth" in initial_view:
                plotter.camera.Azimuth(initial_view["azimuth"])
            if "elevation" in initial_view:
                plotter.camera.Elevation(initial_view["elevation"])
            if "roll" in initial_view:
                plotter.camera.Roll(initial_view["roll"])

        plotter.show(
            title=plot_title,
            window_size=window_size,
            interactive=interactive,
            jupyter_backend=jupyter_backend,
        )

        # Print the current camera view after interaction
        current_camera = plotter.camera
        print(
            f"Current Camera View - Azimuth: {current_camera.azimuth}, "
            f"Elevation: {current_camera.elevation}, "
            f"Roll: {current_camera.roll}"
        )

    @staticmethod
    def save_rotating_contours(
        scalar_field: np.ndarray,
        isosurfaces: list[float] | np.ndarray,
        save_directory: str,
        scalar_field_name: str = "Values",
        rotation_axis: str = "z",
        n_frames: int = 18,
        initial_view: dict[str, float] = None,
        kwargs_mesh: dict[str, float | str | bool] = None,
        window_size: list[int] = None,
    ) -> None:
        """
        Generate and save rotating 3D contour plot images.

        Args:
            scalar_field (np.ndarray): 3D array to visualize.
            isosurfaces (list[float] | np.ndarray): List of isosurface
                values.
            save_directory (str): Directory to save images.
            scalar_field_name (str, optional): Name for the scalar field.
                Defaults to "Values".
            rotation_axis (str, optional): Axis of rotation ("x", "y",
                or "z"). Defaults to "z".
            n_frames (int, optional): Number of rotation frames.
                Defaults to 18.
            initial_view (dict[str, float], optional): Initial camera
                position. Defaults to None.
            kwargs_mesh (dict, optional): PyVista mesh customization.
                Defaults to None.
            window_size (list[int], optional): Window size in pixels.
                Defaults to [1100, 700].

        Returns:
            None: Saves images to disk.

        Raises:
            PyVistaImportError: if PyVista is not installed.
        """
        if not IS_PYVISTA_AVAILABLE:
            raise PyVistaImportError()

        os.makedirs(save_directory, exist_ok=True)

        if window_size is None:
            window_size = [1100, 700]

        if kwargs_mesh is None:
            kwargs_mesh = {
                "cmap": "viridis",
                "opacity": 0.2,
                "show_edges": False,
                "style": "wireframe",
                "log_scale": False,
            }

        # Create the grid and contours
        nx, ny, nz = scalar_field.shape
        x = np.arange(nx, dtype=np.float32)
        y = np.arange(ny, dtype=np.float32)
        z = np.arange(nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data[scalar_field_name] = scalar_field.flatten(order="F")

        contours = grid.contour(isosurfaces=isosurfaces, method="contour")

        plotter = pv.Plotter(window_size=window_size)
        plotter.add_mesh(contours, **kwargs_mesh)

        # Set the initial view if provided
        if initial_view:
            if "azimuth" in initial_view:
                plotter.camera.Azimuth(initial_view["azimuth"])
            if "elevation" in initial_view:
                plotter.camera.Elevation(initial_view["elevation"])
            if "roll" in initial_view:
                plotter.camera.Roll(initial_view["roll"])

        # Determine rotation step
        angle_step = 360 / n_frames

        for i in range(n_frames):
            # Rotate the view
            if rotation_axis == "x":
                plotter.camera.Elevation(angle_step)
            elif rotation_axis == "y":
                plotter.camera.Azimuth(angle_step)
            elif rotation_axis == "z":
                plotter.camera.Roll(angle_step)
            else:
                raise ValueError("rotation_axis must be 'x', 'y', or 'z'")

            plotter.render()

            filename = os.path.join(save_directory, f"frame_{i:03d}.png")
            plotter.screenshot(filename)

        plotter.close()
