"""
Interactive plotting functions for visualizing BCDI data.

This module provides functions for plotting 1D, 2D, and 3D data arrays
with interactive controls using ipywidgets, bokeh, and panel.

Note: This module requires ipywidgets, bokeh, and panel. Dependency checking
is handled at the package level (cdiutils.interactive.__init__).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from ipywidgets import interact, fixed
import ipywidgets as widgets
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper
from bokeh.models import Slider, RadioButtonGroup
import bokeh.palettes as bp
import panel as pn
from IPython.display import display

pn.extension()


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

    Args:
        data_array: The data to be plotted, contained in a numpy array.
        figsize: The size of the figure, by default (10, 10).
        fontsize: The font size to use for plot labels and titles, by default 15.
        log: If True, plot the data using a logarithmic scale. If False, plot
            using a linear scale.
            If "interact", create an interactive toggle button to switch
            between linear and logarithmic scales.
        cmap: The colour map to use for 2D data, by default "turbo".
        title: The title(s) to be used for the plot. If data_array is 2D, title
            can be either a string or a list of strings with one title per
            subplot. If data_array is 1D, title should be a string.
    """
    # Get dimensions
    data_dimensions = data_array.ndim

    if data_dimensions == 1:
        plt.close()
        fig, ax = plt.subplots(figsize=figsize)

        # Depends on log scale
        if log:
            ax.plot(np.log(data_array))
            plt.title(title, fontsize=fontsize + 2)
            plt.tight_layout()
            plt.show()

        elif log is False:
            ax.plot(data_array)
            plt.title(title, fontsize=fontsize + 2)
            plt.tight_layout()
            plt.show()

        elif log == "interact":

            @interact(
                scale=widgets.ToggleButtons(
                    options=["linear", "logarithmic"],
                    value="linear",
                    description="Scale",
                    disabled=False,
                    style={"description_width": "initial"},
                ),
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

                plt.title(title, fontsize=fontsize + 2)
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
                data_array,
                log=log,
                fig=fig,
                ax=ax,
                cmap=cmap,
                title=title,
                fontsize=fontsize,
            )

            # Create axis for colorbar
            cbar_ax = make_axes_locatable(ax).append_axes(
                position="right", size="5%", pad=0.1
            )

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
                    description="Scale",
                    disabled=False,
                    style={"description_width": "initial"},
                ),
                figsize=fixed(figsize),
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
                    position="right", size="5%", pad=0.1
                )

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
        bokey_cmaps = [p for p in bp.__palettes__ if p.endswith("256")]

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
            tooltips=TOOLTIPS,
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
            end=data_array.shape[0] - 1,
            value=0,
            step=1,
            title="Position",
            orientation="horizontal",
        )
        slider_index.on_change("value", callback_change_index)

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
            slider_index.end = slider_range - 1

        select_axis = RadioButtonGroup(
            labels=["x", "y", "z"],
            active=0,
        )
        select_axis.on_change("active", callback_change_axis)

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
        select_data_type.on_change("active", callback_change_data_type)

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
        select_cbar.on_change("active", callback_change_cbar)

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

        fig.add_layout(lin_color_bar, "right")

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
            ),
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
            print("Using complex data, switching to array module for plot.")
            two_d_array = np.abs(two_d_array)
        if contour:
            img = ax.contourf(
                two_d_array,
                norm={"linear": None, "logarithmic": LogNorm()}[scale],
                cmap=cmap,
                origin="lower",
            )
        else:
            img = ax.imshow(
                two_d_array,
                norm={"linear": None, "logarithmic": LogNorm()}[scale],
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
        elif (
            isinstance(title, tuple)
            and len(title) == 3
            or isinstance(title, list)
            and len(title) == 3
        ):
            titles = title
        else:
            titles = [None, None, None]

        # 3D array shape
        shape = data_array.shape

        # Plot first image
        if sum_over_axis:  # Compute sum
            two_d_array = np.sum(np.nan_to_num(data_array), axis=(0))

        else:  # Get middle slice
            two_d_array = data_array[shape[0] // 2, :, :]
        img_x = plot_2d_image(
            two_d_array,
            fig=fig,
            title=titles[0],
            ax=axs[0],
            log=log,
            cmap=cmap,
            fontsize=fontsize,
            x_label="z",
            y_label="y",
            contour=contour,
        )

        # Create axis for colorbar
        cbar_ax = make_axes_locatable(axs[0]).append_axes(
            position="right", size="5%", pad=0.1
        )

        # Create colorbar
        fig.colorbar(mappable=img_x, cax=cbar_ax)

        # Plot second image
        if sum_over_axis:  # Compute sum
            two_d_array = np.sum(np.nan_to_num(data_array), axis=(1))

        else:  # Get middle slice
            two_d_array = data_array[:, shape[1] // 2, :]
        img_y = plot_2d_image(
            two_d_array,
            fig=fig,
            title=titles[1],
            ax=axs[1],
            log=log,
            cmap=cmap,
            fontsize=fontsize,
            x_label="z",
            y_label="x",
            contour=contour,
        )

        # Create axis for colorbar
        cbar_ax = make_axes_locatable(axs[1]).append_axes(
            position="right", size="5%", pad=0.1
        )

        # Create colorbar
        fig.colorbar(mappable=img_y, cax=cbar_ax)

        # Plot third image
        if sum_over_axis:  # Compute sum
            two_d_array = np.sum(np.nan_to_num(data_array), axis=(2))

        else:  # Get middle slice
            two_d_array = data_array[:, :, shape[2] // 2]
        img_z = plot_2d_image(
            two_d_array,
            fig=fig,
            title=titles[2],
            ax=axs[2],
            log=log,
            cmap=cmap,
            fontsize=fontsize,
            x_label="y",
            y_label="x",
            contour=contour,
        )

        # Create axis for colorbar
        cbar_ax = make_axes_locatable(axs[2]).append_axes(
            position="right", size="5%", pad=0.1
        )

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
                description="Scale",
                disabled=False,
                style={"description_width": "initial"},
            ),
            figsize=fixed(figsize),
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
