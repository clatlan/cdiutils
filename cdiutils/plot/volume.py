import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import numpy as np
import warnings
import os

try:
    import pyvista as pv
    # pv.set_jupyter_backend('client')
    from pyvista.trame.ui.vuetify3 import divider, slider, select
    # from trame.app import get_server
    IS_TRAME_PYVISTA_AVAILABLE = True
except ImportError:
    IS_TRAME_PYVISTA_AVAILABLE = False

from cdiutils.plot.formatting import (
    get_figure_size,
    get_extent,
    save_fig,
    CXI_VIEW_PARAMETERS
)
from cdiutils.utils import (
    find_suitable_array_shape,
    CroppingHandler,
    nan_to_zero
)


class TramePyVistaImportError(ImportError):
    """Custom exception to handle PyVista import error."""
    def __init__(self, msg: str = None) -> None:
        _msg = "PyVista package is not installed."
        if msg is not None:
            _msg += "\n" + msg
        super().__init__(_msg)


class VolumeViewer:
    generic_params = {
        "amplitude": {"cmap": "turbo", "centred_clim": False, "clim": [0, 1]},
        "support": {"cmap": "viridis", "centred_clim": False},
        "phase": {"cmap": "cet_CET_C9s_r", "centred_clim": True},
        "displacement": {"cmap": "cet_CET_D1A", "centred_clim": True},
        "het_strain": {"cmap": "cet_CET_D13", "centred_clim": True},
        "het_strain_from_dspacing": {
            "cmap": "cet_CET_D13", "centred_clim": True
        },
        "lattice_parameter": {"cmap": "turbo", "centred_clim": False},
        "dspacing": {"cmap": "turbo", "centred_clim": False},
        "isosurface": 0.50,
        "cmap": "turbo"
    }

    cmap_options = (
        "turbo", "viridis", "spectral", "inferno", "magma", "plasma",
        "cividis", "RdBu", "coolwarm", "Blues", "Greens", "Greys", "Purples",
        "Oranges", "Reds", "cet_CET_D13", "cet_CET_C9s_r", "cet_CET_D1A"
    )

    @classmethod
    def _generate_toolbar_tools(
        cls,
        initial_scalar: str,
        available_scalars: list[str],
        **kwargs
    ) -> callable:
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
                model=("scalar_field", initial_scalar),  # Correct binding
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
            data_path: str,
            initial_active_scalar: str = "het_strain",
            **data
    ):
        if not IS_TRAME_PYVISTA_AVAILABLE:
            raise TramePyVistaImportError
        if len(data) > 0:
            raise NotImplementedError(
                "Directly parsing numpy.ndarray is not implemented yet."
            )
        if not data_path.endswith(".vti"):
            raise ValueError(
                "The provided data_path should points to a .vti file"
            )
        # server = get_server()
        # state = server.state  # Trame state

        data = pv.read(data_path)  # load volume data
        available_scalars = data.array_names  # the available fields

        plotter = pv.Plotter(notebook=True)

        # generate the initial isosurface
        contours = data.contour(
            [cls.generic_params["isosurface"]], scalars="amplitude"
        )
        if initial_active_scalar not in available_scalars:
            raise ValueError(
                f"initial_active_scalar (={initial_active_scalar}) cannot be"
                "found in the provided data."
            )
        contours.set_active_scalars(initial_active_scalar)
        initial_clim = cls.generic_params[initial_active_scalar].get("clim")
        if cls.generic_params[initial_active_scalar]["centred_clim"]:
            initial_clim = (
                -np.max(data[initial_active_scalar]), 
                np.max(data[initial_active_scalar])
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

        plotter.add_axes()  # the tripod axes

        # get the IPython widget
        widget = plotter.show(
            jupyter_kwargs={"add_menu_items": cls._generate_toolbar_tools(
                initial_active_scalar, available_scalars
            )},
            return_viewer=True
        )

        # connect Trame state with Pyvista
        state = widget.viewer.server.state
        ctrl = widget.viewer.server.controller
        state.isosurface_value = cls.generic_params["isosurface"]
        state.scalar_field = initial_active_scalar
        state.cmap = cls.generic_params[initial_active_scalar]["cmap"]

        ctrl.view_update = widget.viewer.update

        # **Trame Callbacks**
        @state.change("isosurface_value")
        def update_isosurface(isosurface_value, **kwargs):
            """Update isosurface dynamically when user moves the slider."""
            new_contours = data.contour(
                [isosurface_value], scalars="amplitude"
            )
            new_contours.set_active_scalars(state.scalar_field)
            mesh_actor.mapper.dataset = new_contours
            ctrl.view_update()

        @state.change("scalar_field")
        def update_scalar_field(scalar_field, **kwargs):
            """Change the active scalar field dynamically."""
            # Ensure correct field is active
            contours.set_active_scalars(scalar_field)
            mesh_actor.mapper.array_name = scalar_field

            # Get colormap and clim behavior
            cmap = cls.generic_params[scalar_field]["cmap"]

            centred_clim = cls.generic_params[scalar_field]["centred_clim"]
            # Adjust color limits (clim)
            clim_range = list(contours.get_data_range(scalar_field))
            if centred_clim:
                max_abs = max(abs(clim_range[0]), abs(clim_range[1]))
                clim_range = [-max_abs, max_abs]  # Centered around 0
            else:
                clim_range = cls.generic_params[scalar_field].get("clim")

            mesh_actor.mapper.scalar_range = clim_range            

            # Update colormap selection dynamically
            state.cmap = cmap

            # Apply new colormap to PyVista actor
            mesh_actor.mapper.lookup_table = pv.LookupTable(cmap)

            plotter.remove_scalar_bar()  # Remove old colorbar
            plotter.add_scalar_bar(
                title=scalar_field.replace("_", " ").capitalize(), n_labels=5
            )

            ctrl.view_update()

        @state.change("cmap")
        def update_colourmap(cmap, **kwargs):
            """Update the colourmap dynamically."""
            # Apply new cmap
            state.cmap = cmap
            mesh_actor.mapper.lookup_table = pv.LookupTable(cmap)
            plotter.remove_scalar_bar()  # Remove old colourbar
            plotter.add_scalar_bar(
                title=state.scalar_field.replace("_", " ").capitalize(),
                n_labels=5
            )

            ctrl.view_update()  # Refresh the visualization

        return widget  # Display the viewer


# PyVista functions, clément remove this after it's cleaned
"""
The goal is to have the possibility to do multi_isosurfaces
rendering in JupyterLab.

For now this notebook works well in JupyterLab in my own environment.

It is important to properly install pyvista / trame, see
`https://tutorial.pyvista.org/tutorial/09_trame/index.html`,
 `Trame` is the `PyVista` `Jupyter` backend.

Use `pip install 'pyvista[all,trame]' jupyterlab` or
`conda install -c conda-forge pyvista jupyterlab
trame trame-vuetify trame-vtk ipywidgets`.

For now, I still need to figure out how to make it work on
 `JupyterHub`, see
 `https://tutorial.pyvista.org/tutorial/00_jupyter/index.html#remote-jupyterhubs`

 Test the notebook here: /data/id01/inhouse/david/Notebooks/Viewer
"""


def pyvista_mesh(
    scalar_field: np.ndarray,
    isosurfaces: list[float] | np.ndarray,
    initial_view: dict[str, float] = None,
    kwargs_mesh: dict[str, float | str | bool] = {
        "cmap": "viridis",
        "opacity": 0.2,
        "show_edges": False,
        "style": "wireframe",
        "log_scale": False,
    },
    scalar_field_name: str = "Values",
    window_size: list[int] = [1100, 700],
    plot_title: str = "3D view",
    interactive: bool = True,
    jupyter_backend: str = "client",
) -> None:
    """
    Visualise a 3D scalar field using PyVista, with isosurfaces and
    customisable mesh settings.

    This function creates a structured 3D grid from a scalar field and
    generates isosurfaces (contours) for the specified values. The
    isosurfaces are displayed in a 3D interactive plot using PyVista's
    `Plotter`, with options for customization through `kwargs_mesh` and
    other parameters.

    Parameters
    ----------
    scalar_field : np.ndarray
        3D array representing the scalar field to visualize. The shape
        of this array defines the grid dimensions for the mesh.

    isosurfaces : list[float] | np.ndarray
        List of scalar values for which isosurfaces (contours) will be
        generated in the 3D plot.

    initial_view : dict[str, float], optional
        Dictionary specifying the initial camera position, e.g.,
        {"azimuth": 30, "elevation": 20, "roll": 10}.
        Default is None.

    kwargs_mesh : dict[str, float | str | bool], optional
        A dictionary of keyword arguments passed to PyVista's `add_mesh`
        function to customise the appearance of the mesh. Default options
        include:
        - `cmap` (str) : Colormap to use (default is 'viridis').
        - `opacity` (float) : Opacity of the mesh, where 1 is fully
        opaque and 0 is fully transparent.
        - `show_edges` (bool) : Whether to display mesh edges (default
        is False).
        - `style` (str) : Display style of the mesh ('wireframe',
        'surface', etc.).
        - `log_scale` (bool) : Apply logarithmic scaling to the color
        map.

        For more options, see:
        https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh

    scalar_field_name : str, optional
        The name to associate with the scalar field when adding it as
        point data to the grid. This will appear in the plot legend or
        colour bar (default is "Values").

    window_size : list[int], optional
        The size of the rendering window, specified as [width, height]
        in pixels. Default is [700, 500].

    plot_title : str, optional
        Title for the 3D plot window (default is "3D view").

    interactive : bool, optional
        If True, enables interactive mode in the plot (allowing rotation,
        zoom, etc.). Set to False for static views (default is True).

    jupyter_backend : str, optional
        Backend to use for displaying the plot in a Jupyter notebook
        environment. Options include:
            - `'none'` : Do not display the plot in the notebook.
            - `'static'` : Display a static image of the plot.
            - `'trame'` : Use Trame to display an interactive figure.
            - `'html'` : Display an embeddable HTML scene for
            interactive visualisation.
            The default is `'trame'`.

    Returns
    -------
    None
        This function does not return any value. It generates and
        displays the 3D plot.

    Example
    -------
    >>> import numpy as np
    >>> scalar_field = np.random.random((50, 50, 50))
    >>> isosurfaces = [0.3, 0.5, 0.7]
    >>> pyvista_mesh(scalar_field, isosurfaces)

    Notes
    -----
    - This function is useful for visualizing 3D scalar fields in
    scientific and engineering applications, where isosurfaces provide
    insight into spatial distributions of values.
    - For large scalar fields, consider adjusting `kwargs_mesh` to
    balance between visualization quality and rendering performance.
    """
    if not IS_TRAME_PYVISTA_AVAILABLE:
        raise TramePyVistaImportError

    # Create grid for PyVista
    nx, ny, nz = scalar_field.shape
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    z = np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = pv.StructuredGrid(X, Y, Z)

    # Add the scalar field data as point data to the grid, room to play
    # here
    grid.point_data[scalar_field_name] = scalar_field.flatten(order="F")

    # Generate contours for different isosurfaces
    contours = grid.contour(
        isosurfaces=isosurfaces,
        method="contour"  # Other methods do not work
    )

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
        f"Elevation: {current_camera.elevation}, Roll: {current_camera.roll}"
    )


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
):
    """
    Generate and save images of a 3D scalar field contour plot with
    rotations around a specified axis.


    Parameters
    ----------
    scalar_field : np.ndarray
        3D array representing the scalar field to visualize.
    isosurfaces : list[float] | np.ndarray
        List of scalar values for which isosurfaces (contours) will be
        generated.
    save_directory : str
        Directory where the generated images will be saved.
    scalar_field_name : str, optional
        Name to associate with the scalar field for visualization
        (default is "Values").
    rotation_axis : str, optional
        Axis of rotation ("x", "y", or "z"). Default is "z".
    n_frames : int, optional
        Number of frames (rotations) to generate (default is 18).
    initial_view : dict[str, float], optional
        Dictionary specifying the initial camera position, e.g.:
        {"azimuth": 30, "elevation": 20, "roll": 10}. Default is None.
    kwargs_mesh : dict[str, float | str | bool], optional
        Keyword arguments for customizing the PyVista mesh (default
        settings provided).
    window_size : list[int], optional
        Size of the rendering window in pixels (default is [1100, 700]).

    Returns
    -------
    None
    """

    # Ensure the save directory exists
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
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data[scalar_field_name] = scalar_field.flatten(order="F")

    contours = grid.contour(isosurfaces=isosurfaces, method="contour")

    # Initialize the PyVista plotter
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
            plotter.camera.Elevation(angle_step)  # Rotate around the x-axis
        elif rotation_axis == "y":
            plotter.camera.Azimuth(angle_step)  # Rotate around the y-axis
        elif rotation_axis == "z":
            plotter.camera.Roll(angle_step)  # Rotate around the z-axis
        else:
            raise ValueError("rotation_axis must be 'x', 'y', or 'z'")

        # Render the updated view
        plotter.render()

        # Save the current frame
        filename = os.path.join(save_directory, f"frame_{i:03d}.png")
        plotter.screenshot(filename)

    plotter.close()


def plot_3d_voxels(
        data: np.ndarray,
        support: np.ndarray,
        view: str = "y+",
        convention: str = "cxi",
        **plot_params
) -> plt.Figure:
    """
    Plot a 3D volumetric representation of data. Voxel are plotted as
    voxels! No triangulation/interpolation. The voxels to plot are based
    on the provided support, while the colouring is generated from the
    data variable.

    Args:
        data (np.ndarray): the quantity to plot.
        support (np.ndarray): the support of the data to plot.
        view (str, optional): the initial view of the 3D plot. Can be
        "x+-/y+-/z+-". Defaults to "y+".
        convention (str, optional): The convention the provided data
        follow, eitheir XU or CXI. Defaults to "cxi".

    Raises:
        ValueError: if convention in invalid.

    Returns:
        plt.Figure: the matpltolib figure the data were drawn in.
    """
    _plot_params = {
        "cmap": plt.get_cmap("turbo"),
        "norm": Normalize(data.min(), data.max()),
        "figsize": (6, 2)
    }
    if plot_params is not None:
        _plot_params.update(plot_params)

    if convention.lower() == "cxi":
        data = np.swapaxes(data, axis1=0, axis2=2)
        support = np.swapaxes(support, axis1=0, axis2=2)
        views = {
            "x+": (180, 0, 90),
            "y+": (0, -90, 0),
            "z+": (-90, 90, 0),
            "x-": (-180, -180, -90),
            "y-": (0, 90, 0),
            "z-": (90, -90, 0),
        }
    elif convention.lower() == "xu":
        raise ValueError("'XU' not implemented yet.")
    else:
        raise ValueError("Invalid convention, can be 'CXI' or 'XU'.")

    if isinstance(_plot_params["cmap"], str):
        _plot_params["cmap"] = plt.get_cmap(_plot_params["cmap"])

    colors = _plot_params["cmap"](_plot_params["norm"](data))

    figure = plt.figure(layout="tight", figsize=_plot_params["figsize"])
    ax = figure.add_subplot(projection="3d")

    ax.set_xlabel(r"$x_{\text{cxi}}$")
    ax.set_ylabel(r"$y_{cxi}$")
    ax.set_zlabel(r"$z_{cxi}$")
    ax.view_init(elev=views[view][0], azim=views[view][1], roll=views[view][2])

    ax.voxels(
        support,
        facecolors=colors,
        edgecolors=np.clip(2*colors-0.85, 0, 1)
    )
    # ax.set_box_aspect(None, zoom=1.25)
    return figure


def hemisphere_projection(
        data: np.ndarray,
        support: np.ndarray,
        axis: int,
        looking_from_downstream: bool = True
) -> np.ndarray:
    """Compute the hemisphere projection of a volume along one axis.

    Args:
        data (np.ndarray): the volume data to project.
        support (np.ndarray): the support of the reconstructed data.
        axis (int): the axis along which to project.
        looking_from_downstream (bool, optional): The direction along
            axis, positive-going (True) or negative-going (False).
            Defaults to True.

    Returns:
        np.ndarray: the 2D array corresponding to the projection.
    """
    # Make sure we have 0 values instead of nan
    support = nan_to_zero(support)

    # Find the support surface
    if looking_from_downstream:
        support_surface = np.cumsum(support, axis=axis)
    else:
        slices = tuple(
            [np.s_[:]] * axis + [np.s_[::-1]] + [np.s_[:]] * (2 - axis)
        )
        support_surface = np.cumsum(support[slices], axis=axis)[slices]

    support_surface = np.where(support_surface > 1, 0, support_surface)
    half_shell_strain = np.where(support_surface == 0, np.nan, data)

    # Some warning is expecting here as mean of empty slices may occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # project the half shell strain along the direction provided
        # by axis
        return np.nanmean(half_shell_strain, axis=axis)


def plot_3d_surface_projections(
        data: np.ndarray,
        support: np.ndarray,
        voxel_size: tuple | list | np.ndarray,
        view_parameters: dict = None,
        figsize: tuple = None,
        title: str = None,
        cbar_title: str = None,
        save: str = None,
        **plot_params
) -> plt.Figure:
    """Plot 3D projected views from a 3D object.

    Args:
        data (np.ndarray): the data to plot.
        support (np.ndarray): the support of the reconstructed object.
        voxel_size (tuple | list | np.ndarray): the voxel size of
            the data to plot.
        view_parameters (dict, optional): some parameters required for
            setting the plot views. Defaults to CXI_VIEW_PARAMETERS.
        figsize (tuple, optional): the size of the figure. Defaults to
            None.
        title (str, optional): the title of the figure. Defaults to
            None.
        cbar_title (str, optional): the title of the colour bar.
            Defaults to None.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    if view_parameters is None:
        view_parameters = CXI_VIEW_PARAMETERS.copy()

    if figsize is None:
        figsize = get_figure_size(subplots=(3, 3))

    cbar_size, cbar_pad = 0.07, 0.4
    figure, axes = plt.subplots(
        2, 3,
        layout="tight",
        figsize=figsize,
        gridspec_kw={'height_ratios': [1/(1-(cbar_pad+cbar_size)), 1]}
    )
    shape = find_suitable_array_shape(support, symmetrical=False)

    cropped_support,  _, _, roi = CroppingHandler.chain_centring(
        support,
        output_shape=shape,
        methods=["com"],
    )

    cropped_data = data[CroppingHandler.roi_list_to_slices(roi)]

    for v in view_parameters:
        looking_from_downstream = False
        row = 0
        if v.endswith("+"):
            looking_from_downstream = True
            row = 1

        ax = axes[row, view_parameters[v]["axis"]]

        projection = hemisphere_projection(
            cropped_data,
            cropped_support,
            axis=view_parameters[v]["axis"],
            looking_from_downstream=looking_from_downstream
        )

        # Swap axes for matshow if the first plane axis is less than the
        # second, ensuring correct orientation where the first plane
        # corresponds to the y-axis and the seconde plane to the x-axis.
        # If first plane axis > second plane axis, the default orientation is
        # correct, and no swapping is needed.
        if view_parameters[v]["plane"] != sorted(view_parameters[v]["plane"]):
            projection = np.swapaxes(projection, axis1=0, axis2=1)

        # to handle extent and origin please refer to
        # https://matplotlib.org/stable/users/explain/artists/imshow_extent.html#imshow-extent
        extent = get_extent(
            shape,
            voxel_size,
            view_parameters[v]["plane"]
        )

        if view_parameters[v]["xaxis_points_left"]:
            # flip the horizontal extent, and the image horizontally
            extent = (extent[1], extent[0], *extent[2:])
            projection = projection[np.s_[:, ::-1]]

        image = ax.imshow(
            projection,
            extent=extent,
            origin="lower",
            **plot_params
        )
        ax.set_title(v, y=0.95)

        # Set a new boolean for whether y-axis should be right or left
        yaxis_left = view_parameters[v]["xaxis_points_left"]

        # Remove the useless spines
        ax.spines[
            ["top", "left" if yaxis_left else "right"]].set_visible(False)

        # Set the position of the spines
        ax.spines["right" if yaxis_left else "left"].set_position(
                ("axes", yaxis_left)
        )

        # Customize ticks and tick labels
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("right" if yaxis_left else "left")
        ax.yaxis.set_label_position("right" if yaxis_left else "left")

        # Plot the shaft of the axis
        ax.plot(
            yaxis_left,
            1,
            "^k",
            transform=ax.transAxes,
            clip_on=False
        )
        ax.plot(
            1-yaxis_left, 0,
            "<k" if yaxis_left else ">k",
            transform=ax.transAxes,
            clip_on=False
        )
        xlabel = view_parameters[v]["xlabel"]
        ylabel = view_parameters[v]["ylabel"]

        ax.set_xlabel(xlabel, labelpad=1)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.tick_params(axis='both', which='major', pad=1.5)

        ax.locator_params(nbins=5)

    ax_divider = make_axes_locatable(axes[0, 1])
    cax = ax_divider.append_axes("top", size=cbar_size, pad=cbar_pad)
    figure.colorbar(
        image,
        cax=cax,
        extend="both",
        orientation="horizontal",
    )
    cax.set_title(cbar_title)

    figure.suptitle(title)
    if save:
        save_fig(figure, path=save, transparent=False)
    return figure


def plot_3d_object(
        data,
        support=None,
        cmap="turbo",
        title="",
        vmin=None,
        vmax=None,
        show=True,
        marker="H",
        alpha=1
):

    """
    Plot a 3D object.

    :param data: the 3D array (np.array) to plot.
    :param support: 3D array (np.array) with the same shape as data.
    Support is the shape of the 3D data to plot, coordinates whose
    values <= 0 won't be plotted. Coordinates whose values > 0 are
    considred to be part of the object to plot.
    :param cmap: the matplotlib colormap (str) used for the colorbar
    (default: "jet").
    :param title: title (str) of the figure. Default is empty string.
    :param vmin: the minimum value (float) for the color scale
    (default: None).
    :param vmax: the maximum value (float) for the color scale
    (default: None).
    :param show: whether or not to show the figure (bool). If False, the
    figure is not displayed but returned.
    :return: None if show is True, otherwise the figure.
    """

    if support is None:
        support = np.ones(shape=data.shape)

    data_of_interest = np.where(support > 0, data, 0)
    nonzero_coordinates = data_of_interest.nonzero()
    nonzero_data = data_of_interest[(nonzero_coordinates[0],
                                     nonzero_coordinates[1],
                                     nonzero_coordinates[2])]
    if vmin is None:
        vmin = np.min(nonzero_data)
    if vmax is None:
        vmax = np.max(nonzero_data)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    p = ax.scatter(
        nonzero_coordinates[0],
        nonzero_coordinates[1],
        nonzero_coordinates[2],
        c=nonzero_data,
        cmap=cmap,
        marker=marker,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha
    )
    fig.colorbar(p)
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_3d_vector_field(
        data,
        support,
        arrow=True,
        scale=5,
        cmap="jet",
        title="",
        vmin=None,
        vmax=None,
        verbose=False
):
    """
    Plot a 3D vector field represented by arrows.

    :param data: the (4-)3D data (np.array). If the object to plot has
    a shape m * n * l, then the data must be 3 * m * n * l. Each voxel
    must contain 3 chanels that describe the vector to plot.
    :param support: 3D array (np.array) with the same shape as data but
    without the chanels (therefore m * n * l). Support is the shape of
    the 3D data to plot, coordinates whose values <= 0 won't be plotted.
    Coordinates whose values > 0 are considred to be part of the object
    to plot.
    :param arrow: whether or not to used arrows for field representation
    (bool). If False, marker "o" is plotted instead and color represents
    norm of the arrow.
    :param cmap:ScalarMappable the matplotlib colormap (str) used for
    the colorbar (default: "turbo").
    :param title: title (str) of the figure. Default is empty string.
    :param vmin: the minimum value (float) for the color scale
    (default: None).
    :param vmax: the maximum value (float) for the color scale
    (default: None).
    :param verbose: whether or not to print out the min and max values
    of the absolute vector field (bool).
    """

    nonzero_coordinates = np.where(support > 0)
    data_of_interest = data[nonzero_coordinates[0],
                            nonzero_coordinates[1],
                            nonzero_coordinates[2],
                            ...]

    norm = np.empty(data_of_interest.shape[0])

    for i in range(data_of_interest.shape[0]):
        norm[i] = np.linalg.norm(data_of_interest[i, ...])
    if vmin is None:
        vmin = np.min(norm)
    if vmax is None:
        vmax = np.max(norm)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    if arrow:
        colormap = plt.get_cmap(cmap)
        colors = (norm.ravel() - norm.min()) / np.ptp(norm)
        colors = np.concatenate((colors, np.repeat(colors, 2)))
        colors = colormap(colors)

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=None)

        q = ax.quiver(
            nonzero_coordinates[0],
            nonzero_coordinates[1],
            nonzero_coordinates[2],
            data_of_interest[..., 0],
            data_of_interest[..., 1],
            data_of_interest[..., 2],
            arrow_length_ratio=0.2,
            normalize=True,
            length=scale,
            colors=colors
            )

        sm.set_array(np.linspace(vmin, vmax))
        fig.colorbar(sm, ax=ax, orientation='vertical')
        q.set_edgecolor(colors)
        q.set_facecolor(colors)

    else:
        p = ax.scatter(
            nonzero_coordinates[0],
            nonzero_coordinates[1],
            nonzero_coordinates[2],
            c=norm,
            cmap=cmap,
            marker='o',
            vmin=vmin,
            vmax=vmax
            )

        fig.colorbar(p)

    fig.suptitle(title)
    fig.tight_layout()

    if verbose:
        print("Minimum value is {}".format(vmin))
        print("Maximum value is {}".format(vmax))
