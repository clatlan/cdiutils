"""
MultiVolumeViewer — Interactive 3D volume visualization widget (ipywidgets + Plotly).

This module provides an interactive Jupyter widget for exploring and comparing
multiple 3D scalar fields (e.g. amplitude, phase, density, masks) defined on the
same voxel grid. The viewer is designed as a lightweight, ParaView-like tool
directly usable inside notebooks, supporting slicing, arbitrary planes,
and half-space clipping.

Main features
-------------
• Multiple raw volume layers
  - Display several 3D scalar fields simultaneously
  - Toggle visibility per layer
  - Independent colormap, opacity, lighting, and color scaling
  - Optional mask-mode (binary surface extraction)

• Derived layers
  - Slice layers:
      * Axis-aligned slicing with adjustable position and thickness
      * Treated as independent layers (not tied to the source after creation)
  - Plane layers:
      * Arbitrary plane defined by normal + origin in physical coordinates
      * Optional thickness (slab averaging) and finite extent
  - Clip layers:
      * Half-space clipping of a raw volume using a plane (normal + origin)
      * Select which side of the plane is kept (up / down)

• Flexible coloring
  - Color by:
      * the layer’s own scalar field
      * another raw volume
      * spatial coordinates (x, y, z)
      * a constant value
  - Independent colorbars per visible layer

• Robust NaN handling
  - Per-layer NaN policy (do nothing, replace by mean/zero/min/max)
  - For slice and plane layers:
      * true “holes” when NaN policy is set to “none”
      * NaN regions are hidden (opacity = 0), similar to ParaView behavior

• Physical coordinate consistency
  - Full support for anisotropic voxel sizes
  - All planes, slices, and clips are defined in physical (x, y, z) space

• Interactive UI
  - Layer creation and editing panels
  - Live updates of geometry and coloring
  - Dark / light theme toggle
  - Optional automatic camera rotation

• Animation export (Rotation + Layer parameter sweeps)
  - Master mode selector:
      * Rotation: export a camera orbit animation
      * Layer: export an animation by sweeping a layer parameter (opacity/pos/offset)
  - Output formats:
      * MP4 (H.264 via imageio/ffmpeg)
      * GIF (either imageio GIF, or optional ffmpeg palette workflow for better quality)
  - Non-blocking export in notebooks:
      * Export runs in an asyncio task (tracked in self._export_task)
      * A Stop button sets a shared cancellation flag (self._anim_cancel) and cancels the task
  - Bounded parallelism during frame rendering:
      * render_workers: size of the rendering worker pool
      * render_in_flight: maximum number of frames concurrently in progress (<= workers)
      * Rotation export streams frames directly to the writer (no giant frames list)
      * Layer export supports preview updates and restores layer state at the end
  - Rendering safety modes (Plotly/Kaleido):
      * rendering_mode="safe": serialize kaleido calls with a lock (stable)
      * rendering_mode="fast": allow concurrent to_image calls (faster, may be less stable)

Typical use cases
-----------------
• Visualization of BCDI reconstructions (amplitude, phase, strain)
• Inspection of defect structures via slices, planes, and clipped volumes
• Rapid, notebook-based exploration without exporting to external viewers
• Exporting publication-ready rotation animations or layer-sweep animations

Dependencies
------------
- numpy
- scipy (RegularGridInterpolator)
- scikit-image (marching_cubes)
- plotly (+ kaleido for static image export)
- ipywidgets
- matplotlib (for colormaps fallback)
- imageio (video writing; ffmpeg backend recommended for mp4)
- ffmpeg (optional; used for high-quality palette-based GIF export)

The widget is intended for interactive use inside Jupyter environments.
"""

# =========================
# Imports
# =========================
import shutil
import asyncio
import concurrent.futures
import io
import math
import os
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Literal

import imageio.v2 as iio2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

# =========================
# Optional dependencies / fallbacks
# =========================
try:
    from .volume import _extract_isosurface_with_values, colorcet_to_plotly

    HAS_VOLUME_UTILS = True
except Exception:
    HAS_VOLUME_UTILS = False

    def colorcet_to_plotly(cmap_name: str, n_colors: int = 256):
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


# =========================
# Constants
# =========================
CBAR_X0 = 1.02  # start just outside the scene
CBAR_DX = 0.2  # horizontal spacing per colorbar

# =========================
# Pure helpers (no class state)
# =========================


def _render_fig_json_to_png(fig_json: dict, w: int, h: int, s: int) -> bytes:
    fig = go.Figure(fig_json)
    return pio.to_image(
        fig, format="png", width=int(w), height=int(h), scale=int(s)
    )


# =========================
# MultiVolumeViewer
# =========================
class MultiVolumeViewer(widgets.Box):
    """Interactive 3D viewer for comparing volumetric scalar fields."""

    # =========================
    # Static / class helpers
    # =========================
    @staticmethod
    def get_colorscale(name: str):
        # Solid colors
        if name in {
            "red",
            "black",
            "white",
            "gray",
            "blue",
            "green",
            "orange",
            "purple",
            "yellow",
            "cyan",
            "magenta",
        }:
            return [[0.0, name], [1.0, name]]

        # Colormaps
        return colorcet_to_plotly(name)

    def _get_all_supported_cmaps(sels) -> list[str]:
        cmaps = list(plt.colormaps())

        # Optional: filter out reversed duplicates (_r)
        cmaps = sorted(set(cmaps))
        FIXED_COLORS = (
            "red",
            "black",
            "white",
            "gray",
            "blue",
            "green",
            "orange",
            "purple",
            "yellow",
            "cyan",
            "magenta",
        )
        return list(FIXED_COLORS) + cmaps

    # ----------------------------
    # public API
    # ----------------------------
    def __init__(
        self,
        dict_data=None,
        voxel_size=(1, 1, 1),
        figsize=(6, 6),
        fontsize=14,
        PLOT_ORDER: Literal["xyz", "zyx"] = "zyx",
        CBAR_LEN=0.7,
        render_workers: int | None = None,
        render_in_flight: int | None = None,
        rendering_mode: Literal["safe", "fast", "process"] = "safe",
    ):
        super().__init__()

        self.cmap_options = self._get_all_supported_cmaps()

        # Global rendering safety switch
        self.force_fixed_color_range = (
            False  # default: disable auto-range everywhere (export + UI)
        )

        mode = str(rendering_mode or "safe").lower().strip()
        allowed = {"safe", "fast", "process"}
        if mode not in allowed:
            raise ValueError(
                f"rendering_mode must be one of {sorted(allowed)}; got {rendering_mode!r}"
            )
        self.rendering_mode = mode
        self._export_proc_pool = None

        self._suspend_rename_autofill_once = False
        self._EDIT_PLACEHOLDER = "__choose_layer__"

        self._ANIM_REGISTRY = {
            "plane": ["offset", "opacity"],
            "slice": ["pos", "opacity"],
            "clip": ["offset", "opacity"],
            "raw": ["opacity"],  # keep raw simple first
        }

        cpu = os.cpu_count() or 1

        self.render_workers = (
            int(render_workers) if render_workers is not None else min(4, cpu)
        )
        self.render_in_flight = (
            int(render_in_flight)
            if render_in_flight is not None
            else self.render_workers
        )

        if self.render_workers < 1:
            raise ValueError("render_workers must be >= 1")
        if self.render_in_flight < 1:
            raise ValueError("render_in_flight must be >= 1")
        if self.render_in_flight > self.render_workers:
            # optional, but usually what you want for bounded memory
            self.render_in_flight = self.render_workers

        # ----------------------------
        # Layout constants
        # ----------------------------
        self._DESC_W = "80px"
        self._CREATE_DESC_W = "60px"
        self._common_style = {"description_width": self._DESC_W}
        self.create_style = {"description_width": self._CREATE_DESC_W}
        """
        INTERNAL_ORDER = 'zyx'  (fixed, do not change)
        PLOT_ORDER     = 'xyz' or 'zyx'
        Controls how coordinates are passed to Plotly ONLY.
        """
        # Canonical internal order is always ZYX
        self.INTERNAL_ORDER = "zyx"

        # Configurable PLOT order (only affects visualization)
        if PLOT_ORDER not in ("xyz", "zyx"):
            raise ValueError("PLOT_ORDER must be 'xyz' or 'zyx'")
        self.PLOT_ORDER = PLOT_ORDER

        self.fontsize = int(fontsize)
        self.CBAR_LEN = CBAR_LEN
        self._DARK_OUTLINE = "#e6eefc"
        self._LIGHT_OUTLINE = "#000000"
        self._DARK_TEXT = "#e6eefc"
        self._LIGHT_TEXT = "#000000"

        self._DARK_AXIS_LINE = "#e6eefc"
        self._LIGHT_AXIS_LINE = "#000000"

        self._DARK_AXIS_GRID = "rgba(200,200,200,0.35)"
        self._LIGHT_AXIS_GRID = "rgba(0,0,0,0.2)"
        self._export_task = None
        self._anim_cancel = False

        self._to_image_lock = threading.Lock()  # serialize rendering
        self._LIGHT_DEFAULTS = dict(
            ambient=0.6,
            diffuse=0.4,
            specular=0.5,
            roughness=0.2,
            fresnel=0.1,
        )

        # ----------------------------
        # Layout constants /Internal data
        # ----------------------------
        self.voxel_size = np.array(voxel_size, dtype=float)
        self.dict_data = {}
        self._rgi = {}
        self._layer_widgets = {}
        self._visible_cb = {}
        self._layers = {}  # name -> spec dict. raw: {"type":"raw","source":None}; derived later
        self._shape0 = None

        # ----------------------------
        # rotation state
        # ----------------------------
        self._rotation_angle = 0
        self._rotation_task = None
        # ----------------------------
        # build view + UI
        # ----------------------------
        # ---- create theme toggle EARLY ----
        self.theme_toggle = widgets.ToggleButton(
            value=True,
            description="Dark Theme",
            tooltip="Toggle dark/light theme",
        )

        self.rotate_toggle = widgets.ToggleButton(
            value=False,
            description="Rotate",
            tooltip="Toggle continuous 3D rotation",
        )

        self.grid_toggle = widgets.Checkbox(
            value=True,
            description="Grid",
            indent=False,
            tooltip="Show/hide 3D grid",
        )

        self._build_figure(figsize)

        self._build_bottom_anim_panel()

        self._build_global_widgets()
        self._build_right_panel(figsize)
        self._build_create_widgets()
        self._compose_layout(figsize)
        self._wire_global_callbacks()

        # ----------------------------
        # Load data if provided
        # ----------------------------
        if dict_data is not None:
            self.set_data(dict_data)

    def set_data(self, dict_data: dict):
        """
        Register raw 3D volumes and rebuild the viewer state.
        """
        self._validate_dict_data(dict_data)
        self._register_raw_layers(dict_data)
        self._build_interpolators()
        self._rebuild_layer_widgets()
        self._rebuild_visible_checkboxes()
        self._rebuild_edit_dropdown()
        self._post_set_data_refresh()

    def close(self):
        self._stop_rotation()
        super().close()

    def show(self):
        display(self)

        loop = asyncio.get_event_loop()
        loop.call_later(
            0.0125, self._reset_view_and_fit
        )  # immediate next loop tick
        loop.call_later(
            0.025, self._reset_view_and_fit
        )  # second pass for slow frontends

    # =========================
    # Validation / registration
    # =========================
    def _validate_dict_data(self, dict_data: dict):
        if dict_data is None or not isinstance(dict_data, dict):
            raise TypeError("dict_data must be a dict {name: np.ndarray}.")
        keys_raw = list(dict_data.keys())
        if not keys_raw:
            raise ValueError("dict_data is empty.")

        shape0 = None
        for k, v in dict_data.items():
            if not isinstance(k, str):
                raise TypeError(f"Layer key {k!r} is not a str.")
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Layer '{k}' is not a numpy array.")
            if v.ndim != 3:
                raise ValueError(
                    f"Layer '{k}' must be a 3D array, got ndim={v.ndim}."
                )
            if np.iscomplexobj(v):
                raise TypeError(
                    f"Layer '{k}' is complex-valued. "
                    "Pass real-valued scalar fields (amp/phase/etc.)."
                )
            if v.dtype.kind not in ("b", "i", "u", "f"):
                raise TypeError(
                    f"Layer '{k}' must be real numeric. Got dtype={v.dtype}."
                )
            if shape0 is None:
                shape0 = v.shape
            elif v.shape != shape0:
                raise ValueError(
                    f"Shape mismatch: '{k}' has {v.shape}, expected {shape0}."
                )
        self._shape0 = shape0

    def _register_raw_layers(self, dict_data: dict):
        self.dict_data = dict_data
        keys_raw = list(dict_data.keys())
        self._layers = {k: {"type": "raw", "source": None} for k in keys_raw}

        raws = [k for k, s in self._layers.items() if s.get("type") == "raw"]
        self.add_clip_source.options = raws
        self.add_clip_source.value = raws[0] if raws else None

    def _build_interpolators(self):
        nz, ny, nx = self._shape0
        grid_z = np.arange(nz, dtype=float)
        grid_y = np.arange(ny, dtype=float)
        grid_x = np.arange(nx, dtype=float)

        self._rgi = {}
        for k, arr in self.dict_data.items():
            arr_f = np.asarray(arr, dtype=float)
            self._rgi[k] = RegularGridInterpolator(
                (grid_z, grid_y, grid_x),
                arr_f,
                bounds_error=False,
                fill_value=np.nan,
            )

    # =========================
    # Figure / layout construction
    # =========================
    def _build_figure(self, figsize):
        axis_common = dict(
            title=dict(font=self._bold_font(self.fontsize * 1.5)),
            tickfont=self._bold_font(self.fontsize),
            showline=True,
            linewidth=3,
            linecolor=self._axis_line_color(),
            showgrid=bool(getattr(self, "grid_toggle", None).value)
            if hasattr(self, "grid_toggle")
            else True,
            gridwidth=2,
            gridcolor=self._axis_grid_color(),
            zeroline=False,
        )
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            template="plotly_white",
            scene=dict(
                xaxis=axis_common | dict(title_text="x"),
                yaxis=axis_common | dict(title_text="y"),
                zaxis=axis_common | dict(title_text="z"),
            ),
            autosize=True,
            height=int(figsize[1] * 90),
            dragmode="orbit",
            margin=dict(l=0, r=0, t=0, b=0),
            font=self._bold_font(),
        )
        self.fig.layout.width = None

    def _compose_layout(self, figsize):
        VIEW_H = int(figsize[1] * 96)
        VIEW_H_css = f"{VIEW_H}px"  # CSS string

        pad_tb = 18 * 2
        anim_min = 170
        anim_margin_top = 10
        reserved = pad_tb + anim_min + anim_margin_top

        fig_h = max(220, VIEW_H - reserved)  # keep your min

        self.layout = widgets.Layout(
            display="flex",
            flex_flow="row",
            align_items="stretch",
            width="100%",
            height=VIEW_H_css,
        )

        # Plotly sizing: keep in plotly layout (NOT widgets.Layout)
        self.fig.update_layout(
            autosize=True,
            height=int(fig_h),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Do NOT touch self.fig.layout / self.fig._layout with widgets.Layout.

        fig_column = widgets.VBox(
            [self.css_rightpanel, self.fig, self.anim_panel],
            layout=widgets.Layout(
                flex="1 1 0%",
                width="100%",
                min_width="0px",
                height="100%",
                overflow_y="auto",
                overflow_x="hidden",
                padding="18px",
                box_sizing="border-box",
            ),
        )

        self.children = [fig_column, self.right_panel]

    def _build_right_panel(self, figsize):
        self.layers_box = widgets.VBox([])
        self.derived_box = widgets.VBox(
            [], layout=widgets.Layout(width="95%", align_items="stretch")
        )

        toggles_row = widgets.HBox(
            [self.theme_toggle, self.rotate_toggle, self.grid_toggle],
            layout=widgets.Layout(
                width="95%", justify_content="flex-start", align_items="center"
            ),
        )

        panel_inner = widgets.VBox(
            [
                widgets.HTML("<b>Visible layers</b>"),
                self.visible_keys_box,
                toggles_row,
                widgets.HTML("<b>Layer controls</b>"),
                self.edit_name_row,
                self.rename_delete_row,
                self.layers_box,
                self.derived_box,
            ],
            layout=widgets.Layout(
                padding="0",
                width="100%",
                overflow="visible",
            ),
        )
        self.css_rightpanel = widgets.HTML("""
        <style>
        /* Panel background */
        .mv-right-panel {
        background: #0b1f3a !important;
        }
        .mv-right-panel-inner {
        background: transparent !important;
        }

        /* Default text */
        .mv-right-panel, .mv-right-panel * {
        color: #e6eefc !important;
        }

        /* ---- Text inputs / textarea (cover JLab + ipywidgets wrappers) ---- */
        .mv-right-panel input,
        .mv-right-panel textarea,
        .mv-right-panel select,
        .mv-right-panel .widget-text input,
        .mv-right-panel .widget-textarea textarea,
        .mv-right-panel .jp-InputWidget input,
        .mv-right-panel .jp-InputWidget textarea,
        .mv-right-panel .jp-HTMLSelect select {
        background: #1b2431 !important;
        color: #e6eefc !important;
        caret-color: #e6eefc !important;
        border: 1px solid #2f3b4d !important;
        }

        /* Placeholders */
        .mv-right-panel input::placeholder,
        .mv-right-panel textarea::placeholder,
        .mv-right-panel .widget-text input::placeholder,
        .mv-right-panel .widget-textarea textarea::placeholder,
        .mv-right-panel .jp-InputWidget input::placeholder,
        .mv-right-panel .jp-InputWidget textarea::placeholder {
        color: rgba(230,238,252,0.70) !important;
        opacity: 1 !important;
        }

        /* Dropdown options */
        .mv-right-panel select option,
        .mv-right-panel .jp-HTMLSelect select option {
        background: #1b2431 !important;
        color: #e6eefc !important;
        }

        /* ---- Buttons (including ToggleButtons) ---- */
        .mv-right-panel button,
        .mv-right-panel .widget-button button,
        .mv-right-panel .widget-toggle-buttons button,
        .mv-right-panel .jupyter-button {
        background: #1b2431 !important;
        color: #e6eefc !important;
        border: 1px solid #2f3b4d !important;
        }

        .mv-right-panel button:hover,
        .mv-right-panel .widget-toggle-buttons button:hover {
        background: #223046 !important;
        }

        .mv-right-panel .widget-toggle-buttons button.mod-active,
        .mv-right-panel button.mod-active,
        .mv-right-panel button:active {
        background: #2a3a55 !important;
        border-color: #3a4c6a !important;
        }

        /* Slider labels/readouts */
        .mv-right-panel .widget-label,
        .mv-right-panel .widget-readout,
        .mv-right-panel output {
        color: #e6eefc !important;
        }
        </style>
        """)

        self.right_panel = widgets.VBox(
            [self.css_rightpanel, panel_inner],
            layout=widgets.Layout(
                flex="0 0 38%",
                height="100%",
                overflow_y="auto",
                overflow_x="hidden",
                padding="20px",
                margin="0px",
                border="3px solid red",
                box_sizing="border-box",
            ),
        )
        self.right_panel.add_class("mv-right-panel")
        panel_inner.add_class("mv-right-panel-inner")

    # =========================
    # Widget construction (global / create / per-layer)
    # =========================
    def _build_global_widgets(self):
        self.visible_keys_box = widgets.Box(
            [],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                justify_content="flex-start",
                align_items="center",
                gap="10px",
                width="95%",
            ),
        )

        self.edit_key = widgets.Dropdown(
            options=[],
            description="Edit",
            tooltip="Select a layer to edit",
            style=self._common_style,
            layout=widgets.Layout(width="33%"),
        )

        self._on_theme(None)
        # --- layer rename / delete controls ---
        self.rename_layer_text = widgets.Text(
            value="",
            placeholder="New layer name",
            description="Name",
            layout=widgets.Layout(width="33%"),
            tooltip="Enter a new name for the selected layer",
        )

        self.rename_layer_btn = widgets.Button(
            description="Rename",
            style=self.create_style,
            layout=widgets.Layout(width="33%"),
            tooltip="Rename the selected layer",
        )

        self.delete_layer = widgets.Button(
            description="Delete ⚠️",
            button_style="danger",
            layout=widgets.Layout(width="100%"),
            tooltip="Delete the selected layer and all dependent derived layers",
        )
        self.delete_layer_btn = widgets.Box(
            [self.delete_layer],
            layout=widgets.Layout(
                width="33%",
                border="3px solid #c62828",
                padding="0px",
                box_sizing="border-box",
            ),
        )

        self.rename_delete_row = widgets.HBox(
            [self.rename_layer_btn, self.delete_layer_btn],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row",
                justify_content="center",
                align_items="center",
                gap="12px",
                width="95%",
            ),
        )

        # --- put Edit + Name on one row ---
        self.edit_name_row = widgets.HBox(
            [self.edit_key, self.rename_layer_text],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row",
                justify_content="center",
                align_items="center",
                gap="12px",
                width="95%",
            ),
        )

    def _build_create_widgets(self):
        self.add_layer_kind = widgets.Dropdown(
            options=[
                ("— choose —", None),
                ("Slice layer", "slice"),
                ("Plane layer", "plane"),
                ("Clip layer", "clip"),
            ],
            value=None,
            description="Add",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Select the type of layer to create",
        )

        self.add_layer_name = widgets.Text(
            value="",
            description="Name",
            placeholder="e.g. phase_zmid",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Name of the new layer (must be unique)",
        )

        # slice create
        self.add_slice_axis = widgets.Dropdown(
            options=[("x", "x"), ("y", "y"), ("z", "z")],
            value="z",
            description="Axis",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Axis along which the slice is extracted",
        )

        self.add_slice_pos = widgets.IntSlider(
            value=0,
            min=0,
            max=1,
            step=1,
            description="Pos",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Slice position along the selected axis (voxel index)",
        )

        self.add_slice_thickness = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description="Thick",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Slice thickness (number of voxels averaged)",
        )

        self.add_slice_center_btn = widgets.Button(
            description="Center pos",
            layout=widgets.Layout(width="auto"),
            tooltip="Center the slice at the middle of the volume",
        )

        # plane create
        self.add_plane_nx = widgets.FloatText(
            value=0.0,
            description="n x",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane normal X component",
        )

        self.add_plane_ny = widgets.FloatText(
            value=0.0,
            description="n y",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane normal Y component",
        )

        self.add_plane_nz = widgets.FloatText(
            value=1.0,
            description="n z",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane normal Z component",
        )

        self.add_plane_ox = widgets.FloatText(
            value=0.0,
            description="o x",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane origin X coordinate (physical units)",
        )

        self.add_plane_oy = widgets.FloatText(
            value=0.0,
            description="o y",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane origin Y coordinate (physical units)",
        )

        self.add_plane_oz = widgets.FloatText(
            value=0.0,
            description="o z",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane origin Z coordinate (physical units)",
        )

        self.add_plane_center_btn = widgets.Button(
            description="Center origin",
            layout=widgets.Layout(width="auto"),
            tooltip="Center the plane origin at the volume center",
        )

        self.add_plane_thickness = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=10.0,
            step=0.1,
            description="Thick",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Plane thickness (slab averaging along normal)",
        )
        self.add_plane_extent = widgets.FloatSlider(
            value=50.0,
            min=1.0,
            max=500.0,
            step=1.0,
            description="Extent",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="In-plane extent of the extracted region",
        )

        self.add_plane_offset = widgets.FloatSlider(
            value=0.0,
            min=-10.0,
            max=10.0,
            step=0.1,
            description="Offset",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Shift plane along its normal direction",
        )

        # clip create
        self.add_clip_source = widgets.Dropdown(
            options=[],
            description="Src",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Source volume used for clipping",
        )

        self.add_clip_side = widgets.Dropdown(
            options=[("Up (+n)", "up"), ("Down (-n)", "down")],
            value="up",
            description="Side",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Select which side of the clipping plane is kept",
        )

        # Final action
        self.add_layer_btn = widgets.Button(
            description="Create layer",
            layout=widgets.Layout(width="auto"),
            tooltip="Create the new layer with the specified parameters",
        )

        self._sync_create_layer_ui()

    def _rebuild_layer_widgets(self):
        self._layer_widgets = {}
        for k in self._layers.keys():
            self._build_widgets_for_layer(k)

    def _rebuild_visible_checkboxes(self):
        self._visible_cb = {}
        cbs = []
        for i, k in enumerate(self._layers.keys()):
            cb = widgets.Checkbox(value=False, description=k, indent=False)
            # cb = widgets.Checkbox(value=(i == 0), description=k, indent=False)
            cb.observe(self._on_visible_changed, names="value")
            self._visible_cb[k] = cb
            cbs.append(cb)
        self.visible_keys_box.children = cbs
        for cb in self.visible_keys_box.children:
            cb.layout.flex = "0 0 140px"
            cb.layout.width = "140px"

    def _rebuild_edit_dropdown(self):
        layer_names = list(self._layers.keys())

        # like "Add:" dropdown
        opts = [("— choose layer —", None)] + [(k, k) for k in layer_names]

        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            self.edit_key.options = opts
            self.edit_key.value = None  # default: nothing selected/edited
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

    def _post_set_data_refresh(self):
        self._refresh_layer_lists()
        if hasattr(self, "anim_layer_key"):
            self._anim_refresh_layers_dropdown()
        self._sync_add_slice_pos_range()
        self._sync_layer_panels_visibility()
        self._on_center_origin_clicked(None)
        self._sync_plane_slider_ranges()
        try:
            self._sync_create_offset_bounds()
        except Exception:
            pass

        self._update_all_traces()
        self._refresh_anim_normal_sources()

    def _make_lighting_box(self, w: dict) -> widgets.Widget:
        """Single-column lighting controls + reset button."""
        # Make sliders take full available width + readable labels
        for key in (
            "light_ambient",
            "light_diffuse",
            "light_specular",
            "light_roughness",
            "light_fresnel",
        ):
            w[key].layout = widgets.Layout(width="95%")
            w[key].style = {"description_width": "110px"}  # readable labels

        reset_btn = widgets.Button(
            description="Reset lighting",
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
            tooltip="Restore default lighting values for this layer",
        )

        def _reset(_):
            d = self._LIGHT_DEFAULTS
            w["light_ambient"].value = float(d["ambient"])
            w["light_diffuse"].value = float(d["diffuse"])
            w["light_specular"].value = float(d["specular"])
            w["light_roughness"].value = float(d["roughness"])
            w["light_fresnel"].value = float(d["fresnel"])

        reset_btn.on_click(_reset)
        box = widgets.VBox(
            [
                w["light_ambient"],
                w["light_diffuse"],
                w["light_specular"],
                w["light_roughness"],
                w["light_fresnel"],
                widgets.HBox(
                    [reset_btn],
                    layout=widgets.Layout(justify_content="flex-start"),
                ),
            ],
            layout=widgets.Layout(
                width="95%",
                gap="8px",
                align_items="stretch",  # keep sliders stretched
            ),
        )
        return box

    def _build_bottom_anim_panel(self):
        self._sync_anim_frames = False
        self._sync_anim_fps = False

        self._bottom_css = widgets.HTML("""
        <style>
        /* ===============================
        Animation panel – force VS Code
        =============================== */

        .mv-anim-panel {
        background: #0b1f3a ;
        border: 1px solid #2f3b4d ;
        border-radius: 5px ;
        padding: 6px ;
        box-sizing: border-box ;
        color: #e6eefc ;
        }

        /* Text containers */
        .mv-anim-panel,
        .mv-anim-panel span,
        .mv-anim-panel div,
        .mv-anim-panel p,
        .mv-anim-panel label,
        .mv-anim-panel b {
        color: #e6eefc
        }

        /* Widget labels (sliders, checkboxes, text inputs) */
        .mv-anim-panel .widget-label,
        .mv-anim-panel .widget-readout,
        .mv-anim-panel .widget-inline-hbox,
        .mv-anim-panel .widget-inline-vbox {
        color: #e6eefc
        }

        /* Text inputs */
        .mv-anim-panel input,
        .mv-anim-panel textarea,
        .mv-anim-panel select {
        background: #1b2431 ;
        color: #e6eefc ;
        border: 1px solid #2f3b4d ;
        }

        /* Slider track + readout */
        .mv-anim-panel input[type=range] {
        background: #1b2431;
        }
        .mv-anim-panel .noUi-value,
        .mv-anim-panel .noUi-tooltip {
        color: #e6eefc
        }

        /* Buttons */
        .mv-anim-panel button {
        background: #1b2431 ;
        color: #e6eefc ;
        border: 1px solid #2f3b4d ;
        }
        .mv-anim-panel button:hover {
        background: #223046
        }

        /* Checkbox / radio text */
        .mv-anim-panel input[type=checkbox] + label,
        .mv-anim-panel input[type=radio] + label {
        color: #e6eefc ;
        }
        .mv-anim-panel .jp-HTMLSelect,
        .mv-anim-panel .jp-InputGroup,
        .mv-anim-panel .jp-InputGroup input,
        .mv-anim-panel .jp-InputGroup select {
        color: #e6eefc ;
        }
        .mv-log-box textarea {
            cursor: default;
        }

        </style>
        """)
        self.anim_panel_title = widgets.HTML("<b>Export animation</b>")

        # --- animation save controls (under figure) ---
        self.anim_name = widgets.Text(
            value="animation",
            description="Save as",
            placeholder="filename (no extension)",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="360px"),
        )

        self.anim_format = widgets.Dropdown(
            options=[("MP4", "mp4"), ("GIF", "gif")],
            value="mp4",  # default
            description="Type",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="180px"),
        )

        # a master dropdown: Rotation vs Layer
        self.anim_master_mode = widgets.Dropdown(
            options=[
                ("— Select —", None),
                ("Rotation", "rotation"),
                ("Layer", "layer"),
            ],
            value=None,
            description="Animate",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="220px"),
        )

        # --- Create the layer-animation widgets (new block, separate) ---
        self.anim_layer_key = widgets.Dropdown(
            options=[("— Select —", None)],  # will be replaced later
            value=None,
            description="Layer",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="320px"),
        )
        self.anim_layer_param = widgets.Dropdown(
            options=[("— Select —", None)],
            value=None,
            description="Param",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="320px"),
        )
        self.anim_points = widgets.IntSlider(
            value=120,
            min=2,
            max=2000,
            step=1,
            description="Points",
            continuous_update=False,
            style={"description_width": "70px"},
            layout=widgets.Layout(width="360px"),
        )
        self.anim_auto_trim = widgets.Checkbox(
            value=False,
            description="Auto-trim empty",
            indent=False,
            layout=widgets.Layout(width="220px"),
        )
        self.anim_min_valid = widgets.FloatSlider(
            value=0.01,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Min valid",
            continuous_update=False,
            style={"description_width": "70px"},
            layout=widgets.Layout(width="360px"),
        )
        self.anim_range = widgets.FloatRangeSlider(
            value=(0.0, 1.0),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Range",
            continuous_update=False,
            style={"description_width": "70px"},
            layout=widgets.Layout(width="560px"),
        )

        # --- rotation mode controls ---
        self.anim_rot_type = widgets.Dropdown(
            options=[
                ("— Select —", None),
                ("Orbit Z", "orbit_z"),
                ("Orbit X", "orbit_x"),
                ("Orbit Y", "orbit_y"),
                ("Around custom axis", "axis"),
            ],
            value=None,
            description="Mode",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="320px"),
        )

        self.anim_axis_x = widgets.FloatText(
            value=0.0,
            description="ax",
            style={"description_width": "30px"},
            layout=widgets.Layout(width="110px"),
        )
        self.anim_axis_y = widgets.FloatText(
            value=0.0,
            description="ay",
            style={"description_width": "30px"},
            layout=widgets.Layout(width="110px"),
        )
        self.anim_axis_z = widgets.FloatText(
            value=1.0,
            description="az",
            style={"description_width": "30px"},
            layout=widgets.Layout(width="110px"),
        )
        self.anim_axis_box = widgets.HBox(
            [self.anim_axis_x, self.anim_axis_y, self.anim_axis_z],
            layout=widgets.Layout(width="100%", gap="8px"),
        )
        self.anim_normal_src = widgets.Dropdown(
            options=[],
            description="Normal",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="320px"),
        )

        self.anim_frames = widgets.IntSlider(
            value=180,
            min=12,
            max=720,
            step=6,
            description="Frames",
            continuous_update=False,
            style={"description_width": "70px"},
            layout=widgets.Layout(width="360px"),
        )
        self.anim_n_frames_text = widgets.IntText(
            value=120,
            description="",
            layout=widgets.Layout(width="120px"),
        )

        def _on_frames_slider_changed(change):
            if self._sync_anim_frames:
                return
            self._sync_anim_frames = True
            self.anim_n_frames_text.value = int(change["new"])
            self._sync_anim_frames = False

        def _on_frames_text_changed(change):
            if self._sync_anim_frames:
                return
            v = int(change["new"])
            # clamp to slider range
            v = max(self.anim_frames.min, min(self.anim_frames.max, v))
            self._sync_anim_frames = True
            self.anim_frames.value = v
            self._sync_anim_frames = False

        self.anim_frames.observe(_on_frames_slider_changed, names="value")
        self.anim_n_frames_text.observe(_on_frames_text_changed, names="value")

        self.anim_fps = widgets.IntSlider(
            value=30,
            min=1,
            max=60,
            step=1,
            description="FPS",
            continuous_update=False,
            style={"description_width": "70px"},
            layout=widgets.Layout(width="260px"),
        )

        self.anim_fps_text = widgets.IntText(
            value=self.anim_fps.value,
            description="",
            layout=widgets.Layout(width="120px"),
        )

        def _on_fps_slider_changed(change):
            if self._sync_anim_fps:
                return
            self._sync_anim_fps = True
            self.anim_fps_text.value = int(change["new"])
            self._sync_anim_fps = False

        def _on_fps_text_changed(change):
            if self._sync_anim_fps:
                return
            v = int(change["new"])
            v = max(self.anim_fps.min, min(self.anim_fps.max, v))
            self._sync_anim_fps = True
            self.anim_fps.value = v
            self._sync_anim_fps = False

        self.anim_fps.observe(_on_fps_slider_changed, names="value")
        self.anim_fps_text.observe(_on_fps_text_changed, names="value")

        self.anim_use_current_camera = widgets.Checkbox(
            value=True,
            description="Use current camera as start",
            indent=False,
            layout=widgets.Layout(width="320px"),
        )

        self.anim_save_btn = widgets.Button(
            description="Export animation",
            layout=widgets.Layout(width="180px"),
        )

        # --- progress UI ---
        self.anim_status = widgets.HTML(
            value="", layout=widgets.Layout(width="100%")
        )

        self.anim_progress = widgets.IntProgress(
            value=0,
            min=0,
            max=1,
            description="",
            bar_style="",  # "", "info", "success", "warning", "danger"
            layout=widgets.Layout(width="100%"),
        )

        # --- stop/cancel export ---
        self.anim_stop_btn = widgets.Button(
            description="Stop",
            layout=widgets.Layout(width="140px"),
        )

        # --- rows ---
        row1 = widgets.HBox(
            [
                widgets.HBox(
                    [self.anim_master_mode],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", gap="12px"
                    ),
                ),
                widgets.HBox(
                    [
                        self.anim_name,
                        self.anim_format,
                        self.anim_save_btn,
                        self.anim_stop_btn,
                    ],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", gap="12px"
                    ),
                ),
            ],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                align_items="center",
                gap="12px",
                justify_content="flex-start",
            ),
        )
        row_mode = widgets.HBox(
            [self.anim_rot_type, self.anim_normal_src],
            layout=widgets.Layout(
                width="100%", flex_flow="row wrap", gap="12px"
            ),
        )
        row2 = widgets.HBox(
            [self.anim_frames, self.anim_fps, self.anim_use_current_camera],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                align_items="center",
                gap="12px",
                justify_content="flex-start",
            ),
        )

        self.anim_layer_box = widgets.VBox(
            [
                widgets.HBox(
                    [self.anim_layer_key, self.anim_layer_param],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", gap="12px"
                    ),
                ),
                widgets.HBox(
                    [self.anim_range, self.anim_auto_trim],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", gap="12px"
                    ),
                ),
                widgets.HBox(
                    [
                        self.anim_points,
                        self.anim_min_valid,
                        self.anim_fps,
                    ],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", gap="12px"
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

        self.anim_rotation_box = widgets.VBox(
            [row_mode, self.anim_axis_box, row2],
            layout=widgets.Layout(width="100%"),
        )

        self.anim_panel_title.value = "<b>Export animation</b>"
        self.anim_rotation_box.layout.display = "none"
        self.anim_layer_box.layout.display = "none"

        self.anim_panel = widgets.VBox(
            [
                self.anim_panel_title,  # make title widget instead of static HTML (see below)
                row1,
                self.anim_rotation_box,
                self.anim_layer_box,
                self.anim_status,
                self.anim_progress,
            ],
            layout=widgets.Layout(
                width="100%",
                padding="10px 12px",
                margin="10px 0 0 0",
                border="1px solid rgba(180,180,180,0.35)",
                border_radius="10px",
                box_sizing="border-box",
                overflow="visible",
            ),
        )

        self.anim_master_mode.observe(self._sync_anim_master_ui, names="value")
        self._sync_anim_master_ui()

        self.anim_panel.add_class("mv-anim-panel")
        self.anim_panel.children = (
            self._bottom_css,
        ) + self.anim_panel.children

        # --- callbacks ( after final widget objects exist) ---
        self.anim_save_btn.on_click(self._on_save_animation_clicked)
        self.anim_stop_btn.on_click(self._on_stop_rotation_export_clicked)
        self.anim_rot_type.observe(self._sync_anim_mode_ui, names="value")
        self.anim_rot_type.observe(self._sync_anim_master_ui, names="value")

        self.anim_layer_key.observe(
            self._sync_anim_layer_param_ui, names="value"
        )
        self.anim_layer_key.observe(
            self._sync_anim_layer_bounds_ui, names="value"
        )
        self.anim_layer_key.observe(self._sync_anim_master_ui, names="value")

        self.anim_layer_param.observe(
            self._sync_anim_layer_bounds_ui, names="value"
        )
        self.anim_layer_param.observe(self._sync_anim_master_ui, names="value")

        self._sync_anim_mode_ui()

    # =========================
    # UI syncing (create + edit panels)
    # =========================
    def _refresh_layer_lists(self):
        layer_names = list(self._layers.keys())

        # visible checkboxes
        existing = set(self._visible_cb.keys())
        if existing != set(layer_names):
            prev = {k: cb.value for k, cb in self._visible_cb.items()}
            # rebuild checkboxes (simple + robust)
            self._visible_cb = {}
            cbs = []

            for i, k in enumerate(layer_names):
                cb = widgets.Checkbox(
                    value=prev.get(k, i == 0), description=k, indent=False
                )
                cb.observe(self._on_visible_changed, names="value")
                self._visible_cb[k] = cb
                cbs.append(cb)
            self.visible_keys_box.children = cbs
            for cb in self.visible_keys_box.children:
                cb.layout.flex = "0 0 140px"
                cb.layout.width = "140px"

        # edit dropdown (with placeholder like "Add:")
        opts = [("— choose layer —", None)] + [(k, k) for k in layer_names]

        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            # IMPORTANT: read current value BEFORE mutating options
            prev = getattr(self.edit_key, "value", None)

            self.edit_key.options = opts

            # Preserve exactly what user had:
            # - if prev is None => keep None
            # - if prev still exists => keep it
            # - else => clear to None
            if prev is None:
                self.edit_key.value = None
            elif prev in self._layers:
                self.edit_key.value = prev
            else:
                self.edit_key.value = None
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

        self._sync_layer_panels_visibility()

        # NEW: keep per-layer dropdowns consistent (rename/delete safe)
        self._refresh_color_by_options()
        # keep rotation-normal sources in sync with current layers
        try:
            self._refresh_anim_normal_sources()
        except Exception:
            pass

    def _sync_create_layer_ui(self, change=None):
        """
        Show only the minimal "Create" header + layer type selector by default.
        Reveal the parameter widgets only after a type is selected.
        """
        kind = self.add_layer_kind.value  # can be None / "slice" / "plane"

        # Always show only the dropdown + name at the top (same row)
        row_add_name = widgets.HBox(
            [self.add_layer_kind, self.add_layer_name],
            layout=widgets.Layout(
                width="100%", gap="12px", align_items="center"
            ),
        )
        # optional: make both share the row nicely
        self.add_layer_kind.layout = widgets.Layout(width="48%")
        self.add_layer_name.layout = widgets.Layout(width="48%")

        base_children = [
            widgets.HTML("<b>Create layer</b>"),
            row_add_name,
        ]

        # Hide everything until a type is chosen
        if kind in (None, "", "none"):
            self.derived_box.children = base_children
            return

        if kind == "slice":
            # ensure slider bounds are correct for the currently selected axis
            self._sync_add_slice_pos_range()

            # auto-center ONLY when entering "slice" mode (so it doesn't overwrite user edits later)
            if getattr(self, "_create_last_kind", None) != "slice":
                self._on_center_slice_pos_clicked(None)

            self._create_last_kind = "slice"

            self.derived_box.children = base_children + [
                widgets.HTML("<b>Slice</b>"),
                widgets.HBox(
                    [self.add_slice_axis, self.add_slice_pos],
                    layout=widgets.Layout(width="100%", gap="12px"),
                ),
                self.add_slice_center_btn,
                self.add_slice_thickness,
                self.add_layer_btn,
            ]
            self._create_last_kind = kind
            return

        if kind == "plane":
            normal_col = widgets.VBox(
                [
                    widgets.HTML("<b>Normal</b>"),
                    self.add_plane_nx,
                    self.add_plane_ny,
                    self.add_plane_nz,
                ],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )

            origin_col = widgets.VBox(
                [
                    widgets.HTML("<b>Origin</b>"),
                    self.add_plane_ox,
                    self.add_plane_oy,
                    self.add_plane_oz,
                    self.add_plane_center_btn,
                ],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )

            # 2 columns
            plane_grid = widgets.HBox(
                [normal_col, origin_col],
                layout=widgets.Layout(
                    display="flex",
                    width="100%",
                    gap="16px",
                    align_items="flex-start",
                ),
            )
            # equal-width columns
            normal_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )
            origin_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )

            self.derived_box.children = base_children + [
                widgets.HTML("<b>Plane</b>"),
                plane_grid,
                self.add_plane_offset,
                self.add_plane_thickness,
                self.add_plane_extent,
                self.add_layer_btn,
            ]
            self._create_last_kind = kind
            return
        if kind == "clip":
            # reuse plane definition (normal + origin) + side
            normal_col = widgets.VBox(
                [
                    widgets.HTML("<b>Normal</b>"),
                    self.add_plane_nx,
                    self.add_plane_ny,
                    self.add_plane_nz,
                ],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )
            origin_col = widgets.VBox(
                [
                    widgets.HTML("<b>Origin</b>"),
                    self.add_plane_ox,
                    self.add_plane_oy,
                    self.add_plane_oz,
                    self.add_plane_center_btn,
                ],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )
            plane_grid = widgets.HBox(
                [normal_col, origin_col],
                layout=widgets.Layout(
                    display="flex",
                    width="100%",
                    gap="16px",
                    align_items="flex-start",
                ),
            )
            normal_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )
            origin_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )

            self.derived_box.children = base_children + [
                widgets.HTML("<b>Clip</b>"),
                self.add_clip_source,  # optional, but recommended
                plane_grid,
                self.add_plane_offset,
                self.add_clip_side,
                self.add_layer_btn,
            ]
            self._create_last_kind = kind
            return

    def _sync_layer_panels_visibility(self):
        k = self.edit_key.value
        if not k or k not in self._layer_widgets:
            self.layers_box.children = []
            return

        spec = self._layers.get(k, {"type": "raw"})
        w = self._layer_widgets[k]

        children = []
        if spec["type"] == "raw":
            children.append(w["thr_op_row"])
        else:
            w["op"].layout = widgets.Layout(width="95%")
            children.append(w["op"])

        if spec["type"] == "slice":
            # --- BEFORE attaching new observers: detach previous ones if they exist ---
            old = w.get("_slice_obs", None)
            if old is not None:
                try:
                    old["ax_dd"].unobserve(old["on_axis"], names="value")
                    old["pos_sl"].unobserve(old["commit"], names="value")
                    old["thick_sl"].unobserve(old["commit"], names="value")
                    w["nan_policy"].unobserve(old["commit"], names="value")
                except Exception:
                    pass
            # editable slice params (update spec + redraw)
            ax_plot0 = self._phys_axis_to_plot_axis(spec["axis_phys"])
            ax_dd = widgets.Dropdown(
                options=[("x", "x"), ("y", "y"), ("z", "z")],
                value=ax_plot0,
                description="axis",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            nz, ny, nx = self._shape0
            ax_plot = ax_dd.value
            ax_phys = self._plot_axis_to_phys_axis(ax_plot)

            axis_max = {"x": nx - 1, "y": ny - 1, "z": nz - 1}[ax_phys]
            axis_len = {"x": nx, "y": ny, "z": nz}[ax_phys]
            # default only if spec has no explicit pos (do not overwrite user-chosen pos)
            pos0 = spec.get("pos", None)
            if pos0 is None:
                mid = {
                    "x": (nx - 1) // 2,
                    "y": (ny - 1) // 2,
                    "z": (nz - 1) // 2,
                }[ax_phys]
                pos0 = mid

            pos_sl = widgets.IntSlider(
                value=int(np.clip(pos0, 0, axis_max)),
                min=0,
                max=int(axis_max),
                step=1,
                description="pos",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            center_btn = widgets.Button(
                description="Center pos", layout=widgets.Layout(width="auto")
            )
            thick_sl = widgets.IntSlider(
                value=int(
                    np.clip(spec["thickness"], 0, max(0, axis_len // 4))
                ),
                min=0,
                max=int(max(0, axis_len // 4)),
                step=1,
                description="thick",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            # after ax_dd, pos_sl, thick_sl are created
            w["axis"] = ax_dd
            w["pos"] = pos_sl
            w["thick"] = thick_sl
            w["center_btn"] = center_btn

            def _sync_pos_max(_=None):
                nz, ny, nx = self._shape0
                ax_phys = self._plot_axis_to_phys_axis(ax_dd.value)

                axis_max = {"x": nx - 1, "y": ny - 1, "z": nz - 1}[ax_phys]
                axis_len = {"x": nx, "y": ny, "z": nz}[ax_phys]

                pos_sl.max = int(axis_max)
                pos_sl.value = int(min(pos_sl.value, pos_sl.max))

                thick_sl.max = int(max(0, axis_len // 4))
                thick_sl.value = int(min(thick_sl.value, thick_sl.max))

            def _center_pos(_=None):
                nz, ny, nx = self._shape0
                ax_phys = self._plot_axis_to_phys_axis(ax_dd.value)

                mid = {
                    "x": (nx - 1) // 2,
                    "y": (ny - 1) // 2,
                    "z": (nz - 1) // 2,
                }[ax_phys]
                pos_sl.value = int(mid)

            center_btn.on_click(lambda _b: (_center_pos(), _commit(None)))

            def _commit(_):
                ax_plot = ax_dd.value
                ax_phys = self._plot_axis_to_phys_axis(ax_plot)
                spec["axis_phys"] = ax_phys
                spec["pos"] = int(pos_sl.value)
                spec["thickness"] = int(thick_sl.value)

                # Re-materialize from the stored source volume (slice layer remains independent
                # from later changes to dict_data because we store vol3d at creation time).
                vol3d = spec["vol3d"]
                policy = w["nan_policy"].value

                # base slice: defines holes when nan_policy == "none"
                data2d_base, Xc, Yc, Zc = self._materialize_slice_rgi(
                    vol3d,
                    spec["axis_phys"],
                    spec["pos"],
                    spec["thickness"],
                    nan_policy="none",
                )
                # displayed slice: respects current policy
                data2d_pol, _, _, _ = self._materialize_slice_rgi(
                    vol3d,
                    spec["axis_phys"],
                    spec["pos"],
                    spec["thickness"],
                    nan_policy=policy,
                )

                spec["data2d_base"] = data2d_base
                spec["data2d"] = data2d_pol
                spec["X"], spec["Y"], spec["Z"] = Xc, Yc, Zc

                self._update_all_traces()

            def _on_axis(ch):
                _sync_pos_max()
                _commit(ch)

            # --- attach observers ONCE for this panel instance ---
            ax_dd.observe(_on_axis, names="value")
            pos_sl.observe(_commit, names="value")
            thick_sl.observe(_commit, names="value")
            w["nan_policy"].observe(_commit, names="value")

            # --- store for later unobserve ---
            w["_slice_obs"] = dict(
                ax_dd=ax_dd,
                pos_sl=pos_sl,
                thick_sl=thick_sl,
                on_axis=_on_axis,
                commit=_commit,
            )

            _sync_pos_max()

            children += [
                widgets.HTML("<b>Slice</b>"),
                ax_dd,
                pos_sl,
                center_btn,
                thick_sl,
            ]
        elif spec["type"] == "plane":
            # --- BEFORE: detach previous observers if they exist ---
            old = w.get("_plane_obs", None)
            if old is not None:
                try:
                    for ww in old["watched"]:
                        ww.unobserve(old["commit"], names="value")
                    w["nan_policy"].unobserve(old["commit"], names="value")
                except Exception:
                    pass
            n_plot0 = self._phys_to_plot_vec(spec["normal"])
            o_plot0 = self._phys_to_plot_point(spec["origin"])

            nx_t = widgets.FloatText(
                value=float(n_plot0[0]),
                description="n x",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            ny_t = widgets.FloatText(
                value=float(n_plot0[1]),
                description="n y",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            nz_t = widgets.FloatText(
                value=float(n_plot0[2]),
                description="n z",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            ox_t = widgets.FloatText(
                value=float(o_plot0[0]),
                description="o x",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oy_t = widgets.FloatText(
                value=float(o_plot0[1]),
                description="o y",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oz_t = widgets.FloatText(
                value=float(o_plot0[2]),
                description="o z",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            thick_sl = widgets.FloatSlider(
                value=float(spec["thickness"]),
                min=0.0,
                max=self.add_plane_thickness.max,
                step=0.1,
                description="thick",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            ext_sl = widgets.FloatSlider(
                value=float(spec["extent"]),
                min=1.0,
                max=self.add_plane_extent.max,
                step=1.0,
                description="extent",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            off_sl = widgets.FloatSlider(
                value=float(spec.get("offset", 0.0)),
                min=float(getattr(self.add_plane_offset, "min", -100.0)),
                max=float(getattr(self.add_plane_offset, "max", 100.0)),
                step=0.1,
                description="offset",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            def _commit(_):
                # read current UI values (PLOT coordinates)
                n_plot = np.array(
                    [nx_t.value, ny_t.value, nz_t.value], dtype=float
                )
                o_plot = np.array(
                    [ox_t.value, oy_t.value, oz_t.value], dtype=float
                )

                # convert to PHYSICAL coordinates
                n_phys = self._plot_to_phys_vec(n_plot)
                o0_phys = self._plot_to_phys_point(o_plot)

                n_norm = float(np.linalg.norm(n_phys))
                if n_norm == 0.0:
                    return
                n_hat = n_phys / n_norm

                offset, off_min, off_max, n_hat = self._clamp_offset_in_bounds(
                    o0_phys, n_phys, off_sl.value
                )
                self._set_offset_slider_bounds(off_sl, off_min, off_max)
                if abs(off_sl.value - offset) > 1e-12:
                    off_sl.value = float(offset)

                o_eff_phys = o0_phys + n_hat * offset

                spec["normal"] = n_phys
                spec["origin0"] = o0_phys
                spec["offset"] = offset
                spec["origin"] = o_eff_phys

                spec["thickness"] = float(thick_sl.value)
                spec["extent"] = float(ext_sl.value)

                policy = w["nan_policy"].value

                data2d_base, Xc, Yc, Zc = self._materialize_plane(
                    spec["vol3d"],
                    spec["normal"],
                    spec["origin"],
                    spec["thickness"],
                    spec["extent"],
                    nan_policy="none",
                )
                data2d_pol, _, _, _ = self._materialize_plane(
                    spec["vol3d"],
                    spec["normal"],
                    spec["origin"],
                    spec["thickness"],
                    spec["extent"],
                    nan_policy=policy,
                )

                spec["data2d_base"] = data2d_base
                spec["data2d"] = data2d_pol
                spec["X"], spec["Y"], spec["Z"] = Xc, Yc, Zc

                self._update_all_traces()

            watched = (
                nx_t,
                ny_t,
                nz_t,
                ox_t,
                oy_t,
                oz_t,
                off_sl,
                thick_sl,
                ext_sl,
            )

            for ww in watched:
                ww.observe(_commit, names="value")
            w["nan_policy"].observe(_commit, names="value")

            w["_plane_obs"] = dict(commit=_commit, watched=watched)

            normal_col = widgets.VBox(
                [widgets.HTML("<b>Normal</b>"), nx_t, ny_t, nz_t],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )
            origin_col = widgets.VBox(
                [widgets.HTML("<b>Origin</b>"), ox_t, oy_t, oz_t],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )

            plane_grid = widgets.HBox(
                [normal_col, origin_col],
                layout=widgets.Layout(
                    display="flex",
                    width="100%",
                    gap="16px",
                    align_items="flex-start",
                ),
            )
            normal_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )
            origin_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )

            children += [
                widgets.HTML("<b>Plane</b>"),
                plane_grid,
                off_sl,
                thick_sl,
                ext_sl,
            ]

        elif spec["type"] == "clip":
            # --- editable clip params ---
            old = w.get("_clip_obs", None)
            if old is not None:
                try:
                    for ww in old["watched"]:
                        ww.unobserve(old["commit"], names="value")
                    w["nan_policy"].unobserve(old["commit"], names="value")
                except Exception:
                    pass
            # source raw
            raw_keys = [
                kk
                for kk, ss in self._layers.items()
                if ss.get("type") == "raw"
            ]
            src_dd = widgets.Dropdown(
                options=raw_keys,
                value=spec.get("source", raw_keys[0] if raw_keys else None),
                description="src",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            n_plot0 = self._phys_to_plot_vec(spec["normal"])
            o_plot0 = self._phys_to_plot_point(
                spec.get("origin0", spec["origin"])
            )

            # normal + origin
            nx_t = widgets.FloatText(
                value=float(n_plot0[0]),
                description="n x",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            ny_t = widgets.FloatText(
                value=float(n_plot0[1]),
                description="n y",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            nz_t = widgets.FloatText(
                value=float(n_plot0[2]),
                description="n z",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            ox_t = widgets.FloatText(
                value=float(o_plot0[0]),
                description="o x",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oy_t = widgets.FloatText(
                value=float(o_plot0[1]),
                description="o y",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oz_t = widgets.FloatText(
                value=float(o_plot0[2]),
                description="o z",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            side_dd = widgets.Dropdown(
                options=[("Up (+n)", "up"), ("Down (-n)", "down")],
                value=spec.get("side", "up"),
                description="side",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            off_sl = widgets.FloatSlider(
                value=float(spec.get("offset", 0.0)),
                min=float(getattr(self.add_plane_offset, "min", -100.0)),
                max=float(getattr(self.add_plane_offset, "max", 100.0)),
                step=0.1,
                description="offset",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            def _commit(_):
                if src_dd.value:
                    spec["source"] = src_dd.value

                n_plot = np.array(
                    [nx_t.value, ny_t.value, nz_t.value], dtype=float
                )
                o_plot = np.array(
                    [ox_t.value, oy_t.value, oz_t.value], dtype=float
                )

                n_phys = self._plot_to_phys_vec(n_plot)
                o0_phys = self._plot_to_phys_point(o_plot)

                n_norm = float(np.linalg.norm(n_phys))
                if n_norm == 0.0:
                    return
                n_hat = n_phys / n_norm

                offset, off_min, off_max, n_hat = self._clamp_offset_in_bounds(
                    o0_phys, n_phys, off_sl.value
                )
                self._set_offset_slider_bounds(off_sl, off_min, off_max)
                if abs(off_sl.value - offset) > 1e-12:
                    off_sl.value = float(offset)

                o_eff_phys = o0_phys + n_hat * offset

                spec["normal"] = n_phys
                spec["origin0"] = o0_phys
                spec["offset"] = offset
                spec["origin"] = o_eff_phys

                spec["side"] = side_dd.value
                self._update_all_traces()

            watched = (
                src_dd,
                nx_t,
                ny_t,
                nz_t,
                ox_t,
                oy_t,
                oz_t,
                off_sl,
                side_dd,
            )

            for ww in watched:
                ww.observe(_commit, names="value")
            w["nan_policy"].observe(_commit, names="value")
            w["_clip_obs"] = dict(commit=_commit, watched=watched)

            normal_col = widgets.VBox(
                [widgets.HTML("<b>Normal</b>"), nx_t, ny_t, nz_t],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )
            origin_col = widgets.VBox(
                [widgets.HTML("<b>Origin</b>"), ox_t, oy_t, oz_t],
                layout=widgets.Layout(width="100%", align_items="stretch"),
            )
            grid = widgets.HBox(
                [normal_col, origin_col],
                layout=widgets.Layout(
                    display="flex",
                    width="100%",
                    gap="16px",
                    align_items="flex-start",
                ),
            )
            normal_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )
            origin_col.layout = widgets.Layout(
                flex="1 1 0%", min_width="0px", align_items="stretch"
            )

            children += [
                widgets.HTML("<b>Clip</b>"),
                src_dd,
                grid,
                off_sl,
                side_dd,
                w["thr"],
            ]

        # common controls
        children += [
            w["nan_color_mask"],
            w["cmap_show_autorange"],
            w["range_slider"],
            widgets.HTML("<b>Lighting</b>"),
            self._make_lighting_box(w),
        ]

        self.layers_box.children = [widgets.VBox(children)]

    def _select_edit_layer(self, name: str):
        """Force the edit dropdown + panel to switch to `name` (robust in VS Code)."""
        if not name or name not in self._layers:
            return

        layer_names = list(self._layers.keys())
        if not layer_names:
            return

        # Keep placeholder option (same as `_refresh_layer_lists`)
        opts = [("— choose layer —", None)] + [(k, k) for k in layer_names]

        # pick a temporary different valid value (never None)
        tmp = next((k for k in layer_names if k != name), name)

        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            with self.edit_key.hold_trait_notifications():
                self.edit_key.options = opts
                if self.edit_key.value != tmp:
                    self.edit_key.value = tmp
                self.edit_key.value = name
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

        self._sync_layer_panels_visibility()

    def _sync_add_slice_pos_range(self, change=None):
        src = self._default_source_key()
        if not src or src not in self.dict_data:
            return

        nz, ny, nx = self.dict_data[src].shape
        ax_plot = self.add_slice_axis.value
        ax_phys = self._plot_axis_to_phys_axis(ax_plot)

        axis_max = {"x": nx - 1, "y": ny - 1, "z": nz - 1}
        axis_len = {"x": nx, "y": ny, "z": nz}[ax_phys]

        self.add_slice_pos.max = int(axis_max[ax_phys])
        self.add_slice_pos.value = int(
            min(self.add_slice_pos.value, self.add_slice_pos.max)
        )

        self.add_slice_thickness.max = int(max(0, axis_len // 4))
        self.add_slice_thickness.value = int(
            min(self.add_slice_thickness.value, self.add_slice_thickness.max)
        )

    def _reset_create_layer_ui(self):
        """Collapse the Create-layer parameters after a layer is created."""
        self.add_layer_name.value = ""  # optional
        self._create_last_kind = None
        # Temporarily disable observer to avoid double-refresh loops (optional but clean)
        self.add_layer_kind.unobserve(
            self._sync_create_layer_ui, names="value"
        )
        try:
            self.add_layer_kind.value = None
        finally:
            self.add_layer_kind.observe(
                self._sync_create_layer_ui, names="value"
            )
        self._sync_create_layer_ui()

    def _sync_anim_master_ui(self, *args):
        mode = (
            None
            if getattr(self, "anim_master_mode", None) is None
            else self.anim_master_mode.value
        )

        if mode is None:
            self.anim_rotation_box.layout.display = "none"
            self.anim_layer_box.layout.display = "none"
            self.anim_panel_title.value = "<b>Export animation</b>"
            self.anim_save_btn.disabled = True
            self.anim_stop_btn.disabled = True
            self.anim_format.disabled = True
            return

        # enable common buttons once a master mode is selected
        self.anim_stop_btn.disabled = False
        self.anim_format.disabled = False

        if mode == "rotation":
            self.anim_rotation_box.layout.display = ""
            self.anim_layer_box.layout.display = "none"
            self.anim_panel_title.value = "<b>Export rotation</b>"

            rot_selected = self.anim_rot_type.value is not None

            # children: [row_mode, axis_box, row2]
            self.anim_rotation_box.children[
                0
            ].layout.display = ""  # SHOW MODE ROW ALWAYS
            self.anim_rotation_box.children[1].layout.display = (
                "" if (self.anim_rot_type.value == "axis") else "none"
            )
            self.anim_rotation_box.children[2].layout.display = (
                "" if rot_selected else "none"
            )

            self.anim_save_btn.disabled = not rot_selected
            return

        if mode == "layer":
            self.anim_rotation_box.layout.display = "none"
            self.anim_layer_box.layout.display = ""
            self.anim_panel_title.value = "<b>Export layer animation</b>"

            if mode == "layer" and not getattr(
                self, "_anim_layer_ui_inited", False
            ):
                self._anim_refresh_layers_dropdown()
                self._anim_layer_ui_inited = True

            layer_selected = self.anim_layer_key.value is not None
            param_selected = self.anim_layer_param.value is not None
            ready = layer_selected and param_selected

            # children: [layer/param row, range row, points row]
            self.anim_layer_box.children[0].layout.display = ""
            self.anim_layer_box.children[1].layout.display = (
                "" if ready else "none"
            )
            self.anim_layer_box.children[2].layout.display = (
                "" if ready else "none"
            )

            self.anim_save_btn.disabled = not ready
            return

    # =========================
    # Create layer callback
    # =========================
    def _on_create_layer_clicked(self, btn):
        kind = self.add_layer_kind.value
        src = self._default_source_key()
        if not src:
            return
        raw_name = self.add_layer_name.value
        base_label = str(raw_name or "").strip() or str(kind or "").strip()
        if not base_label:
            base_label = "layer"
        vol3d = np.asarray(self.dict_data[src], dtype=float).copy()

        policy = self._layer_widgets[src]["nan_policy"].value
        if kind == "slice":
            ax_plot = self.add_slice_axis.value
            ax_phys = self._plot_axis_to_phys_axis(ax_plot)
            pos = int(self.add_slice_pos.value)
            thick = int(self.add_slice_thickness.value)
            # base slice: defines holes when nan_policy == "none"
            data2d_base, Xc, Yc, Zc = self._materialize_slice_rgi(
                vol3d,
                ax_phys,
                pos,
                thick,
                nan_policy="none",
            )
            # displayed slice: respects current policy
            data2d_pol, _, _, _ = self._materialize_slice_rgi(
                vol3d,
                ax_phys,
                pos,
                thick,
                nan_policy=policy,
            )
            new_name = f"{base_label}:slice"
            base = new_name
            kk = 2
            while new_name in self._layers:
                new_name = f"{base}#{kk}"
                kk += 1
            self._layers[new_name] = dict(
                type="slice",
                axis_phys=ax_phys,
                pos=pos,
                thickness=thick,
                vol3d=vol3d,  # store original volume for editing re-slice
                data2d_base=data2d_base,  # defines holes when nan_policy == "none"
                data2d=data2d_pol,
                X=Xc,
                Y=Yc,
                Z=Zc,
            )
            self._build_widgets_for_layer(new_name)
            self._refresh_layer_lists()
            self._refresh_anim_layer_options(keep_selection=True)
            if hasattr(self, "anim_layer_key"):
                self._anim_refresh_layers_dropdown()
            self._update_all_traces()
            self._reset_create_layer_ui()
            return

        if kind == "plane":
            n_plot = np.array(
                [
                    self.add_plane_nx.value,
                    self.add_plane_ny.value,
                    self.add_plane_nz.value,
                ],
                dtype=float,
            )
            o_plot = np.array(
                [
                    self.add_plane_ox.value,
                    self.add_plane_oy.value,
                    self.add_plane_oz.value,
                ],
                dtype=float,
            )

            n_phys = self._plot_to_phys_vec(n_plot)
            o0_phys = self._plot_to_phys_point(o_plot)  # base origin

            # normalize n_phys for offset application
            n_norm = float(np.linalg.norm(n_phys))
            if n_norm == 0.0:
                return
            n_hat = n_phys / n_norm

            offset, off_min, off_max, n_hat = self._clamp_offset_in_bounds(
                o0_phys, n_phys, self.add_plane_offset.value
            )
            # keep create slider consistent with actual feasible range
            self._set_offset_slider_bounds(
                self.add_plane_offset, off_min, off_max
            )
            self.add_plane_offset.value = float(offset)

            o_eff_phys = o0_phys + n_hat * offset

            thick = float(self.add_plane_thickness.value)
            extent = float(self.add_plane_extent.value)

            data2d_base, Xc, Yc, Zc = self._materialize_plane(
                vol3d, n_phys, o_eff_phys, thick, extent, nan_policy="none"
            )
            data2d_pol, _, _, _ = self._materialize_plane(
                vol3d, n_phys, o_eff_phys, thick, extent, nan_policy=policy
            )

            new_name = f"{base_label}:plane"
            base = new_name
            k = 2
            while new_name in self._layers:
                new_name = f"{base}#{k}"
                k += 1
            self._layers[new_name] = dict(
                type="plane",
                normal=n_phys,
                origin0=o0_phys,
                offset=offset,
                origin=o_eff_phys,
                thickness=thick,
                extent=extent,
                vol3d=vol3d,
                data2d_base=data2d_base,
                data2d=data2d_pol,
                X=Xc,
                Y=Yc,
                Z=Zc,
            )
            self._build_widgets_for_layer(new_name)
            self._refresh_layer_lists()
            self._refresh_anim_layer_options(keep_selection=True)
            if hasattr(self, "anim_layer_key"):
                self._anim_refresh_layers_dropdown()
            self._update_all_traces()
            self._reset_create_layer_ui()
            return
        if kind == "clip":
            # which raw to clip
            src_key = (
                self.add_clip_source.value
                if hasattr(self, "add_clip_source")
                else self._default_source_key()
            )
            if not src_key:
                return

            n_plot = np.array(
                [
                    self.add_plane_nx.value,
                    self.add_plane_ny.value,
                    self.add_plane_nz.value,
                ],
                dtype=float,
            )
            o_plot = np.array(
                [
                    self.add_plane_ox.value,
                    self.add_plane_oy.value,
                    self.add_plane_oz.value,
                ],
                dtype=float,
            )

            n_phys = self._plot_to_phys_vec(n_plot)
            o0_phys = self._plot_to_phys_point(o_plot)  # base origin

            n_norm = float(np.linalg.norm(n_phys))
            if n_norm == 0.0:
                return
            n_hat = n_phys / n_norm

            offset, off_min, off_max, n_hat = self._clamp_offset_in_bounds(
                o0_phys, n_phys, self.add_plane_offset.value
            )
            # keep create slider consistent with actual feasible range
            self._set_offset_slider_bounds(
                self.add_plane_offset, off_min, off_max
            )
            self.add_plane_offset.value = float(offset)

            o_eff_phys = o0_phys + n_hat * offset

            side = self.add_clip_side.value  # "up" or "down"

            new_name = f"{base_label}:clip"
            base = new_name
            k = 2
            while new_name in self._layers:
                new_name = f"{base}#{k}"
                k += 1
            self._layers[new_name] = dict(
                type="clip",
                source=src_key,
                normal=n_phys,
                origin0=o0_phys,
                offset=offset,
                origin=o_eff_phys,  # effective origin used for clipping
                side=side,
            )
            self._build_widgets_for_layer(new_name)
            self._refresh_layer_lists()
            self._refresh_anim_layer_options(keep_selection=True)
            if hasattr(self, "anim_layer_key"):
                self._anim_refresh_layers_dropdown()
            self._update_all_traces()
            self._reset_create_layer_ui()

            return

    def _materialize_plane(
        self,
        vol3d,
        normal_xyz,
        origin_xyz,
        thickness,
        extent,
        nu=200,
        nv=200,
        n_thick_samples=5,
        nan_policy: str = "none",
    ):
        """
        Sample a 3D scalar field on an arbitrary plane in physical space.
        Uses RegularGridInterpolator; returns NaN outside bounds.
        """
        vol = np.asarray(vol3d, dtype=float)
        nz, ny, nx = vol.shape

        n = np.asarray(normal_xyz, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm == 0:
            raise ValueError("Plane normal must be non-zero.")
        n = n / n_norm

        r0 = np.asarray(origin_xyz, dtype=float)

        # in-plane basis (u,v)
        a = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, n))) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        su = np.linspace(-extent, extent, int(nu))
        sv = np.linspace(-extent, extent, int(nv))
        UU, VV = np.meshgrid(su, sv)  # (nv, nu)

        rgi = RegularGridInterpolator(
            (
                np.arange(nz, dtype=float),
                np.arange(ny, dtype=float),
                np.arange(nx, dtype=float),
            ),
            vol,
            bounds_error=False,
            fill_value=np.nan,
        )

        if thickness <= 0:
            tt = np.array([0.0])
        else:
            m = max(1, int(n_thick_samples))
            tt = np.linspace(-thickness, thickness, m)

        vals_list = []
        for t in tt:
            X = r0[0] + u[0] * UU + v[0] * VV + n[0] * t
            Y = r0[1] + u[1] * UU + v[1] * VV + n[1] * t
            Z = r0[2] + u[2] * UU + v[2] * VV + n[2] * t

            xi = X / self.voxel_size[2]
            yi = Y / self.voxel_size[1]
            zi = Z / self.voxel_size[0]
            pts = np.stack([zi, yi, xi], axis=-1)

            vals = rgi(pts)
            vals_list.append(vals)

        data2d = self._apply_nan_policy(
            np.nanmean(np.stack(vals_list, axis=0), axis=0),
            nan_policy,
        )
        # surface coords (physical)
        Xc = r0[0] + u[0] * UU + v[0] * VV
        Yc = r0[1] + u[1] * UU + v[1] * VV
        Zc = r0[2] + u[2] * UU + v[2] * VV

        return data2d, Xc, Yc, Zc

    def _materialize_slice_rgi(
        self,
        vol3d: np.ndarray,
        ax: str,
        pos: int,
        thick: int,
        nan_policy: str = "none",
    ):
        vol0 = np.asarray(vol3d, dtype=float)
        vol = self._apply_nan_policy(vol0, nan_policy)

        nz, ny, nx = vol.shape
        rgi = RegularGridInterpolator(
            (
                np.arange(nz, dtype=float),
                np.arange(ny, dtype=float),
                np.arange(nx, dtype=float),
            ),
            vol,
            bounds_error=False,
            fill_value=np.nan,
        )

        if ax == "z":
            # output grid (ny, nx)
            yy, xx = np.meshgrid(
                np.arange(ny, dtype=float),
                np.arange(nx, dtype=float),
                indexing="ij",
            )
            # thickness samples in z
            z_samples = np.arange(
                max(0, pos - thick), min(nz - 1, pos + thick) + 1, dtype=float
            )
            vals_list = []
            for zz in z_samples:
                pts = np.stack([np.full_like(xx, zz), yy, xx], axis=-1)
                vals_list.append(rgi(pts))
            data2d = np.nanmean(np.stack(vals_list, axis=0), axis=0)

            X = np.arange(nx) * self.voxel_size[2]
            Y = np.arange(ny) * self.voxel_size[1]
            Z = np.ones((ny, nx)) * (pos * self.voxel_size[0])
            XX, YY = np.meshgrid(X, Y)
            return data2d, XX, YY, Z

        if ax == "y":
            zz, xx = np.meshgrid(
                np.arange(nz, dtype=float),
                np.arange(nx, dtype=float),
                indexing="ij",
            )
            y_samples = np.arange(
                max(0, pos - thick), min(ny - 1, pos + thick) + 1, dtype=float
            )
            vals_list = []
            for yy in y_samples:
                pts = np.stack([zz, np.full_like(xx, yy), xx], axis=-1)
                vals_list.append(rgi(pts))
            data2d = np.nanmean(np.stack(vals_list, axis=0), axis=0)

            X = np.arange(nx) * self.voxel_size[2]
            Z = np.arange(nz) * self.voxel_size[0]
            Y = np.ones((nz, nx)) * (pos * self.voxel_size[1])
            XX, ZZ = np.meshgrid(X, Z)
            return data2d, XX, Y, ZZ

        # ax == "x"
        zz, yy = np.meshgrid(
            np.arange(nz, dtype=float),
            np.arange(ny, dtype=float),
            indexing="ij",
        )
        x_samples = np.arange(
            max(0, pos - thick), min(nx - 1, pos + thick) + 1, dtype=float
        )

        vals_list = []
        for xx in x_samples:
            pts = np.stack([zz, yy, np.full_like(yy, xx)], axis=-1)
            vals_list.append(rgi(pts))
        data2d = np.nanmean(np.stack(vals_list, axis=0), axis=0)

        Yc = yy * self.voxel_size[1]
        Zc = zz * self.voxel_size[0]
        Xc = np.full((nz, ny), pos * self.voxel_size[2], dtype=float)

        return data2d, Xc, Yc, Zc

    def _slice_mean_2d(
        self,
        vol3d: np.ndarray,
        ax: str,
        pos: int,
        thick: int,
        nan_policy: str = "none",
    ) -> np.ndarray:
        vol = np.asarray(vol3d, dtype=float)
        vol = self._apply_nan_policy(vol, nan_policy)
        nz, ny, nx = vol.shape

        if ax == "z":
            z0 = max(0, pos - thick)
            z1 = min(nz - 1, pos + thick)
            return np.nanmean(vol[z0 : z1 + 1, :, :], axis=0)  # (ny, nx)

        if ax == "y":
            y0 = max(0, pos - thick)
            y1 = min(ny - 1, pos + thick)
            return np.nanmean(vol[:, y0 : y1 + 1, :], axis=1)  # (nz, nx)

        # ax == "x"
        x0 = max(0, pos - thick)
        x1 = min(nx - 1, pos + thick)
        return np.nanmean(vol[:, :, x0 : x1 + 1], axis=2)  # (nz, ny)

    def _sample_raw_on_surface(
        self, raw_key: str, Xc, Yc, Zc, nan_policy="none"
    ) -> np.ndarray:
        """Sample raw 3D field raw_key on a surface defined by physical coords Xc,Yc,Zc."""
        if raw_key not in self._rgi:
            return np.zeros_like(Xc, dtype=float)

        # convert physical -> index space (z,y,x)
        xi = np.asarray(Xc, dtype=float) / self.voxel_size[2]
        yi = np.asarray(Yc, dtype=float) / self.voxel_size[1]
        zi = np.asarray(Zc, dtype=float) / self.voxel_size[0]

        pts = np.stack([zi, yi, xi], axis=-1)  # (..., 3)
        vals = self._rgi[raw_key](pts)
        return self._apply_nan_policy(
            np.asarray(vals, dtype=float), nan_policy
        )

    def _clip_volume_halfspace(
        self, vol3d: np.ndarray, normal_xyz, origin_xyz, side: str
    ):
        vol = np.asarray(vol3d, dtype=float)

        n = np.asarray(normal_xyz, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm == 0.0:
            raise ValueError("Clip normal must be non-zero.")
        n = n / n_norm

        o = np.asarray(origin_xyz, dtype=float)

        nz, ny, nx = vol.shape

        # physical coordinate axes
        x = np.arange(nx, dtype=float) * self.voxel_size[2]  # (nx,)
        y = np.arange(ny, dtype=float) * self.voxel_size[1]  # (ny,)
        z = np.arange(nz, dtype=float) * self.voxel_size[0]  # (nz,)

        # signed distance field d = (r - o)·n, with broadcasting to (nz, ny, nx)
        dx = (x - o[0])[None, None, :] * n[0]
        dy = (y - o[1])[None, :, None] * n[1]
        dz = (z - o[2])[:, None, None] * n[2]
        d = dx + dy + dz

        if side == "up":
            keep = d >= 0.0
        else:
            keep = d <= 0.0

        return keep

    def _center_origin_xyz(self):
        nz, ny, nx = self._shape0
        return np.array(
            [
                (nx - 1) * self.voxel_size[2] / 2.0,
                (ny - 1) * self.voxel_size[1] / 2.0,
                (nz - 1) * self.voxel_size[0] / 2.0,
            ],
            dtype=float,
        )

    def _sync_plane_slider_ranges(self):
        nz, ny, nx = self._shape0
        Lx = (nx - 1) * self.voxel_size[2]
        Ly = (ny - 1) * self.voxel_size[1]
        Lz = (nz - 1) * self.voxel_size[0]
        diag = float(np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz))

        self.add_plane_extent.max = max(self.add_plane_extent.max, diag)
        self.add_plane_extent.value = min(
            self.add_plane_extent.value, self.add_plane_extent.max
        )

        # thickness: reasonable upper bound
        self.add_plane_thickness.max = max(
            self.add_plane_thickness.max, diag / 5.0
        )
        self.add_plane_thickness.value = min(
            self.add_plane_thickness.value, self.add_plane_thickness.max
        )
        self.add_plane_offset.max = max(self.add_plane_offset.max, diag)
        self.add_plane_offset.min = min(self.add_plane_offset.min, -diag)
        self.add_plane_offset.value = float(
            np.clip(
                self.add_plane_offset.value,
                self.add_plane_offset.min,
                self.add_plane_offset.max,
            )
        )

    def _sync_anim_mode_ui(self, *_):
        mode = self.anim_rot_type.value
        self.anim_axis_box.layout.display = (
            "flex" if mode == "axis" else "none"
        )
        self.anim_normal_src.layout.display = (
            "flex" if mode == "normal" else "none"
        )

    def _refresh_anim_normal_sources(self):
        """Refresh the dropdown listing layers that can provide a rotation normal."""
        if not hasattr(self, "anim_normal_src"):
            return

        opts = [("— Select —", None)]
        for k, s in getattr(self, "_layers", {}).items():
            if not isinstance(s, dict):
                continue
            t = str(s.get("type") or s.get("layer_type") or "raw").lower()

            if t in ("plane", "clip"):
                opts.append((k, k))
                continue

            if t == "slice":
                # Slice normal is derived from its axis (in PHYSICAL coords)
                ax = str(s.get("axis_phys", s.get("ax", "z"))).lower()
                opts.append((f"{k} (slice n={ax})", k))
                continue

        cur = self.anim_normal_src.value
        keys = {v for _, v in opts}

        self.anim_normal_src.options = opts

        # keep selection if still valid; else pick first real option (or None)
        if cur in keys:
            return
        self.anim_normal_src.value = opts[1][1] if len(opts) > 1 else None

    # edit name and delete layers
    def _on_rename_layer_clicked(self, _btn):
        old = self.edit_key.value
        new = (self.rename_layer_text.value or "").strip()
        if not old or old not in self._layers:
            return
        if not new or new == old:
            return
        if new in self._layers:
            return

        self._rename_layer(old, new)

        # clear after rename (and prevent immediate autofill once)
        self._suspend_rename_autofill_once = True
        self.rename_layer_text.value = ""

    def _on_delete_layer_clicked(self, _btn):
        name = self.edit_key.value
        if not name or name not in self._layers:
            return

        self._delete_layer(name, cascade=True)

    def _rename_layer(self, old: str, new: str):
        # --- preserve insertion order of self._layers ---
        old_items = list(self._layers.items())
        new_layers = {}
        old_spec = None
        for k, v in old_items:
            if k == old:
                old_spec = v
                new_layers[new] = v
            else:
                new_layers[k] = v
        if old_spec is None:
            return
        self._layers = new_layers

        # --- move per-layer widgets / visibility checkbox ---
        if old in self._layer_widgets:
            self._layer_widgets[new] = self._layer_widgets.pop(old)

        if old in self._visible_cb:
            cb = self._visible_cb.pop(old)
            cb.description = new
            self._visible_cb[new] = cb

        # --- if this is a RAW layer, rename dict_data + interpolator key ---
        if (
            old_spec.get("type") or old_spec.get("layer_type") or "raw"
        ) == "raw":
            if old in getattr(self, "dict_data", {}):
                self.dict_data[new] = self.dict_data.pop(old)
            if old in getattr(self, "_rgi", {}):
                self._rgi[new] = self._rgi.pop(old)

        # --- update references in derived layers that point to old ---
        for k, spec in self._layers.items():
            if not isinstance(spec, dict):
                continue
            for ref_key in ("source", "source_key", "src"):
                if spec.get(ref_key) == old:
                    spec[ref_key] = new

        # --- update any create-panel dropdown that lists raw sources ---
        if hasattr(self, "add_clip_source"):
            try:
                self.add_clip_source.options = list(
                    getattr(self, "dict_data", {}).keys()
                )
            except Exception:
                pass

        # --- refresh UI + traces + keep selection on renamed layer ---
        self._refresh_layer_lists()  # rebuild edit dropdown / visible list:contentReference[oaicite:6]{index=6}
        if hasattr(self, "anim_layer_key"):
            self._anim_refresh_layers_dropdown()
        self._select_edit_layer(
            new
        )  # robust dropdown switching (already implemented):contentReference[oaicite:7]{index=7}
        self.rename_layer_text.value = new
        self._update_all_traces()

    def _delete_layer(self, name: str, cascade: bool = True):
        if name not in self._layers:
            return

        # --- cascade delete derived layers that reference this one ---
        if cascade:
            dependents = []
            for k, spec in list(self._layers.items()):
                if not isinstance(spec, dict):
                    continue
                if (
                    spec.get("source") == name
                    or spec.get("source_key") == name
                    or spec.get("src") == name
                ):
                    dependents.append(k)
            for dk in dependents:
                if dk != name and dk in self._layers:
                    self._delete_layer(dk, cascade=True)

        # --- drop from containers ---
        spec = self._layers.pop(name, None)
        _ = self._layer_widgets.pop(name, None)
        cb = self._visible_cb.get(name, None)
        if cb is not None:
            try:
                cb.unobserve(self._on_visible_changed, names="value")
            except Exception:
                pass

        # --- if raw, remove underlying volume too ---
        if (
            isinstance(spec, dict)
            and (spec.get("type") or spec.get("layer_type") or "raw") == "raw"
        ):
            if name in getattr(self, "dict_data", {}):
                self.dict_data.pop(name, None)
            if name in getattr(self, "_rgi", {}):
                self._rgi.pop(name, None)

        # --- refresh source dropdown used by clip creation ---
        if hasattr(self, "add_clip_source"):
            try:
                self.add_clip_source.options = list(
                    getattr(self, "dict_data", {}).keys()
                )
            except Exception:
                pass

        # --- refresh UI ---
        self._refresh_layer_lists()  # keeps edit dropdown valid:contentReference[oaicite:8]{index=8}
        if hasattr(self, "anim_layer_key"):
            self._anim_refresh_layers_dropdown()

        # pick a new selection (or clear panel)
        remaining = list(self._layers.keys())
        if remaining:
            self._select_edit_layer(remaining[0])
            self.rename_layer_text.value = remaining[0]
        else:
            self.layers_box.children = []
            self.rename_layer_text.value = ""

        self._update_all_traces()

    def _refresh_color_by_options(self):
        """Keep all per-layer 'color_by' dropdown options in sync with current RAW layer names."""
        raw_keys = [
            k
            for k, s in self._layers.items()
            if isinstance(s, dict) and s.get("type") == "raw"
        ]

        base_opts = (
            [("(self)", "__self__")]
            + [(k, k) for k in raw_keys]
            + [
                ("(constant)", "__constant__"),
                ("(x coord)", "__x__"),
                ("(y coord)", "__y__"),
                ("(z coord)", "__z__"),
            ]
        )

        for layer_key, w in getattr(self, "_layer_widgets", {}).items():
            dd = w.get("color_by", None) if isinstance(w, dict) else None
            if dd is None:
                continue

            old_val = dd.value
            dd.options = base_opts

            # preserve selection if still valid, else fallback
            valid_values = {v for (_lbl, v) in base_opts}
            dd.value = old_val if old_val in valid_values else "__self__"

    # =========================
    # Layer widget factory
    # =========================
    def _build_widgets_for_layer(self, k: str):
        spec = self._layers[k]
        if spec["type"] == "raw":
            arr_stats = np.asarray(self.dict_data[k], dtype=float)

        elif spec["type"] == "clip":
            src = spec.get("source", None)
            if (src is None) or (src not in self.dict_data):
                # fallback: use the first raw layer if available
                raw_keys = [
                    kk
                    for kk, ss in self._layers.items()
                    if ss.get("type") == "raw"
                ]
                src = raw_keys[0] if raw_keys else None

            arr_stats = (
                np.asarray(self.dict_data[src], dtype=float)
                if src is not None
                else np.zeros((2, 2, 2), dtype=float)
            )

        else:
            # slice / plane
            arr_stats = np.asarray(spec["data2d"], dtype=float)

        options_layers = [
            kk for kk, ss in self._layers.items() if ss.get("type") == "raw"
        ]

        thr = widgets.FloatSlider(
            value=float(np.nanpercentile(arr_stats, 30)),
            min=float(np.nanmin(arr_stats)),
            max=float(np.nanmax(arr_stats)),
            step=float((np.nanmax(arr_stats) - np.nanmin(arr_stats)) / 300)
            if np.nanmax(arr_stats) > np.nanmin(arr_stats)
            else 0.01,
            description="iso",
            continuous_update=False,
            readout_format=".3g",
            # style=self._common_style,
            layout=widgets.Layout(width="100%"),
            indent=False,
        )
        op = widgets.FloatSlider(
            value=1.0,
            min=0.0,
            max=1.0,
            step=0.01,
            description="α",
            continuous_update=False,
            # style=self._common_style,
            layout=widgets.Layout(width="100%"),
            indent=False,
        )
        for w in (thr, op):
            w.layout.width = "48%"
            w.layout.flex = "1 1 0"

        thr_op_row = widgets.HBox(
            [thr, op],
            layout=widgets.Layout(
                display="flex",
                align_items="center",
                gap="2px",
                width="100%",
                overflow="hidden",
            ),
            # style=self._common_style,
        )

        cmap = widgets.Dropdown(
            options=self.cmap_options,
            value="gray",
            description="cmap",
            style=self._common_style,
        )
        nan_policy = widgets.Dropdown(
            options=[
                ("Do nothing", "none"),
                ("Replace with mean", "mean"),
                ("Replace with zero", "zero"),
                ("Replace with min", "min"),
                ("Replace with max", "max"),
            ],
            value="zero",
            description="NaN",
            style=self._common_style,
        )
        nan_policy.observe(self._on_layer_param_changed, names="value")

        color_by = widgets.Dropdown(
            options=[("(self)", "__self__")]
            + [(kk, kk) for kk in options_layers]
            + [
                ("(constant)", "__constant__"),
                ("(x coord)", "__x__"),
                ("(y coord)", "__y__"),
                ("(z coord)", "__z__"),
            ],
            value="__self__",
            description="color by",
            style=self._common_style,
        )
        color_by.observe(self._on_layer_param_changed, names="value")

        as_mask = widgets.Checkbox(
            value=False,
            description="mask-mode",
            tooltip="If True: threshold to a binary mask then extract its surface.",
            indent=False,
        )

        show_colorbar = widgets.Checkbox(
            value=True, description="show colorbar", indent=False
        )
        auto_range = widgets.Checkbox(
            value=True, description="auto range", indent=False
        )

        nan_color_mask = widgets.HBox(
            [nan_policy, color_by, as_mask],
            layout=widgets.Layout(
                display="flex",
                align_items="center",
                gap="12px",
                width="100%",
                overflow="hidden",
            ),
        )
        cmap_show_autorange = widgets.Box(
            [cmap, show_colorbar, auto_range],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row",
                justify_content="center",
                align_items="center",
                gap="12px",
                width="95%",
            ),
        )
        for cb in (
            as_mask,
            show_colorbar,
            auto_range,
            cmap,
            nan_policy,
            color_by,
        ):
            w.layout.width = "33%"
            cb.layout.flex = "1 1 0%"

        vmin0 = float(np.nanmin(arr_stats))
        vmax0 = float(np.nanmax(arr_stats))
        if not np.isfinite(vmin0):
            vmin0 = 0.0
        if not np.isfinite(vmax0):
            vmax0 = 1.0
        if vmax0 == vmin0:
            vmax0 = vmin0 + 1e-12

        range_slider = widgets.FloatRangeSlider(
            value=(vmin0, vmax0),
            min=vmin0,
            max=vmax0,
            step=(vmax0 - vmin0) / 300 if vmax0 > vmin0 else 0.01,
            description="cmin/cmax",
            continuous_update=False,
            readout_format=".3g",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
            indent=False,
        )
        range_slider.disabled = True

        def _toggle_range_slider(change, _rs=range_slider):
            _rs.disabled = bool(change["new"])

        auto_range.observe(_toggle_range_slider, names="value")

        for wdg in (
            thr,
            op,
            cmap,
            as_mask,
            show_colorbar,
            auto_range,
            range_slider,
        ):
            wdg.observe(self._on_layer_param_changed, names="value")

        # lighting (used for iso; slice ignores but ok to keep)
        light_ambient = widgets.FloatSlider(
            value=self._LIGHT_DEFAULTS["ambient"],
            min=0.0,
            max=1.0,
            step=0.01,
            description="ambient",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
        )
        light_diffuse = widgets.FloatSlider(
            value=self._LIGHT_DEFAULTS["diffuse"],
            min=0.0,
            max=1.0,
            step=0.01,
            description="diffuse",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
        )
        light_specular = widgets.FloatSlider(
            value=self._LIGHT_DEFAULTS["specular"],
            min=0.0,
            max=1.0,
            step=0.01,
            description="specular",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
        )
        light_roughness = widgets.FloatSlider(
            value=self._LIGHT_DEFAULTS["roughness"],
            min=0.01,
            max=1.0,
            step=0.01,
            description="roughness",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
        )
        light_fresnel = widgets.FloatSlider(
            value=self._LIGHT_DEFAULTS["fresnel"],
            min=0.0,
            max=1.0,
            step=0.01,
            description="fresnel",
            continuous_update=False,
            style=self.create_style,
            layout=widgets.Layout(width="95%"),
        )

        for wdg in (
            light_ambient,
            light_diffuse,
            light_specular,
            light_roughness,
            light_fresnel,
        ):
            wdg.observe(self._on_layer_param_changed, names="value")

        self._layer_widgets[k] = dict(
            thr=thr,
            op=op,
            thr_op_row=thr_op_row,
            cmap=cmap,
            cmap_show_autorange=cmap_show_autorange,
            as_mask=as_mask,
            show_colorbar=show_colorbar,
            auto_range=auto_range,
            nan_color_mask=nan_color_mask,
            color_by=color_by,
            range_slider=range_slider,
            light_ambient=light_ambient,
            light_diffuse=light_diffuse,
            light_specular=light_specular,
            light_roughness=light_roughness,
            light_fresnel=light_fresnel,
            nan_policy=nan_policy,
        )

    # =========================
    # Rendering
    # =========================
    def _get_visible_keys(self):
        return [k for k, cb in self._visible_cb.items() if cb.value]

    def _update_all_traces(self):
        selected = self._get_visible_keys()
        with self.fig.batch_update():
            self.fig.data = tuple()
            if not selected:
                return

            cb_idx = 0  # increments ONLY for traces that actually display a colorbar

            for k in selected:
                trace, did_use_cbar = self._make_trace_for_layer(
                    k, cbar_index=cb_idx
                )

                self.fig.add_trace(trace)

                if did_use_cbar:
                    cb_idx += 1

    def _make_trace_for_layer(self, layer_name: str, cbar_index: int = 0):
        spec = self._layers[layer_name]

        if spec["type"] == "raw":
            return self._make_mesh_trace_for_key(
                layer_name, cbar_index=cbar_index
            )

        if spec["type"] in ("slice", "plane"):
            return self._make_slice_trace(layer_name, cbar_index=cbar_index)

        if spec["type"] == "clip":
            return self._make_clip_mesh_trace(
                layer_name, cbar_index=cbar_index
            )

        return go.Scatter3d(), False

    def _make_mesh_trace_for_key(self, key: str, cbar_index: int = 0):
        arr = self.dict_data[key]
        w = self._layer_widgets[key]
        policy = w["nan_policy"].value

        iso = float(w["thr"].value)
        opacity = float(w["op"].value)
        colorscale = self.get_colorscale(w["cmap"].value)

        # ----------------------------
        # Build volume for marching cubes
        # ----------------------------
        if w["as_mask"].value:
            # 1) build mask from a scalar (typically |arr|)
            vol = np.asarray(arr, dtype=float)
            vol = self._apply_nan_policy(vol, policy)
            mask = (vol >= iso).astype(np.float32)
            # mask must contain both 0 and 1 to have an isosurface at 0.5
            if mask.min() == mask.max():
                return (
                    go.Mesh3d(
                        name=key,
                        x=[],
                        y=[],
                        z=[],
                        i=[],
                        j=[],
                        k=[],
                        opacity=opacity,
                        showscale=False,
                    ),
                    False,
                )
            # 2) extract geometry from mask
            verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=1)
            verts_scaled = verts * self.voxel_size

            # 3) : sample original arr at those vertices for coloring (same as normal)
            vals = self._rgi[key](verts)  # verts are in (z,y,x) index space
            vals = self._apply_nan_policy(vals, policy)

        else:
            vol = np.asarray(arr, dtype=float)
            vol = self._apply_nan_policy(vol, policy)
            vmin = float(np.nanmin(vol))
            vmax = float(np.nanmax(vol))
            if (
                not np.isfinite(vmin)
                or not np.isfinite(vmax)
                or not (vmin < iso < vmax)
            ):
                return (
                    go.Mesh3d(
                        name=key,
                        x=[],
                        y=[],
                        z=[],
                        i=[],
                        j=[],
                        k=[],
                        opacity=opacity,
                        showscale=False,
                    ),
                    False,
                )

            if HAS_VOLUME_UTILS:
                # # extract isosurface using marching cubes
                # vol = np.abs(amplitude)

                # # robust finite range
                # finite = np.isfinite(vol)
                # if not finite.any():
                #     # nothing to extract
                #     return np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty((0,), float)

                # vmin = float(vol[finite].min())
                # vmax = float(vol[finite].max())

                # # degenerate: flat volume
                # if vmax <= vmin:
                #     return np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty((0,), float)

                # level = float(isosurface_level)

                # # clamp level to valid range (avoid exact endpoints)
                # eps = 1e-12 * (vmax - vmin)
                # level = min(max(level, vmin + eps), vmax - eps)

                # verts, faces, _, _ = measure.marching_cubes(
                #     vol,
                #     level=level,
                #     step_size=1,
                # )

                verts_scaled, faces, vals = _extract_isosurface_with_values(
                    vol,
                    vol,
                    iso,
                    self.voxel_size,
                    use_interpolator=True,
                )
                verts_scaled = np.asarray(verts_scaled, float)
            else:
                verts, faces, _, _ = marching_cubes(
                    vol, level=iso, step_size=1
                )
                verts_scaled = verts * self.voxel_size
                vals = self._rgi[key](verts)
        vals = np.asarray(vals, dtype=float)
        if policy == "none":
            valid_i = np.isfinite(vals)
            if not np.all(valid_i):
                keep = valid_i[faces].all(axis=1)
                faces = faces[keep]
                if faces.size == 0:
                    return (
                        go.Mesh3d(
                            name=key,
                            x=[],
                            y=[],
                            z=[],
                            i=[],
                            j=[],
                            k=[],
                            opacity=opacity,
                            showscale=False,
                        ),
                        False,
                    )
        verts_zyx_idx = verts_scaled / self.voxel_size  # (z,y,x) index coords
        verts_plot = self._map_zyx_to_plot(verts_scaled)

        # ----------------------------
        # Coloring
        # ----------------------------
        show_colorbar = bool(w["show_colorbar"].value)
        auto_range = bool(w["auto_range"].value)
        rmin, rmax = w["range_slider"].value

        color_sel = w["color_by"].value  # dropdown value

        if color_sel == "__constant__":
            intensity = np.zeros(len(verts_plot), dtype=float)
        elif color_sel in ("__x__", "__y__", "__z__"):
            axis = {"__x__": 0, "__y__": 1, "__z__": 2}[color_sel]
            intensity = verts_plot[:, axis].astype(float)
        elif color_sel in ("__self__", key):
            intensity = np.asarray(vals, dtype=float)
        else:
            intensity = self._rgi[color_sel](verts_zyx_idx)

        intensity = self._apply_nan_policy(intensity, policy)
        # avoid Plotly issues if user chose a field with NaNs/infs
        if policy == "none":
            intensity = np.nan_to_num(
                intensity, nan=0.0, posinf=0.0, neginf=0.0
            )
        data_min = float(np.nanmin(intensity))
        data_max = float(np.nanmax(intensity))
        if not np.isfinite(data_min):
            data_min = 0.0
        if not np.isfinite(data_max):
            data_max = 1.0
        if data_max == data_min:
            data_max = data_min + 1e-12

        cmin, cmax = (
            (data_min, data_max) if auto_range else (float(rmin), float(rmax))
        )
        if cmax <= cmin:
            cmax = cmin + 1e-12

        colorbar = None
        if (color_sel != "__constant__") and show_colorbar:
            if color_sel == "__self__":
                title = key
            elif color_sel in ("__x__", "__y__", "__z__"):
                title = color_sel.strip("_")
            else:
                title = color_sel
            colorbar = self._colorbar_dict(title, cbar_index=cbar_index)

        # Sync slider bounds safely: ONLY EXPAND (never shrink)
        rs = w["range_slider"]
        if color_sel != "__constant__":
            new_min = float(np.nanmin(intensity))
            new_max = float(np.nanmax(intensity))

            if not np.isfinite(new_min):
                new_min = 0.0
            if not np.isfinite(new_max):
                new_max = 1.0
            if new_max <= new_min:
                new_max = new_min + 1e-12

            target_min = min(float(rs.min), new_min)
            target_max = max(float(rs.max), new_max)
            if target_max <= target_min:
                target_max = target_min + 1e-12

            rs.unobserve(self._on_layer_param_changed, names="value")
            try:
                v0, v1 = rs.value
                v0 = float(np.clip(v0, target_min, target_max))
                v1 = float(np.clip(v1, target_min, target_max))
                if v1 < v0:
                    v0, v1 = v1, v0
                rs.value = (v0, v1)

                rs.min = target_min
                rs.max = target_max
                step = (target_max - target_min) / 300
                rs.step = step if step > 0 else 0.01

                if bool(w["auto_range"].value):
                    rs.value = (new_min, new_max)
            finally:
                rs.observe(self._on_layer_param_changed, names="value")
        if cmax <= cmin:
            cmax = cmin + 1e-12
        trace = go.Mesh3d(
            name=key,
            x=verts_plot[:, 0],  # x
            y=verts_plot[:, 1],  # y
            z=verts_plot[:, 2],  # z
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=intensity,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=colorbar,
            showscale=(colorbar is not None),
            cmin=cmin,
            cmax=cmax,
            flatshading=False,
            lighting=dict(
                ambient=w["light_ambient"].value,
                diffuse=w["light_diffuse"].value,
                specular=w["light_specular"].value,
                roughness=w["light_roughness"].value,
                fresnel=w["light_fresnel"].value,
            ),
        )
        return trace, (colorbar is not None)

    def _make_slice_trace(self, layer_name: str, cbar_index: int = 0):
        spec = self._layers[layer_name]
        w = self._layer_widgets[layer_name]
        policy = w["nan_policy"].value

        opacity = float(w["op"].value)
        colorscale = self.get_colorscale(w["cmap"].value)

        # base defines where holes exist
        base_raw = np.asarray(
            spec.get("data2d_base", spec["data2d"]), dtype=float
        )

        # displayed values follow current nan policy
        data2d = (
            self._apply_nan_policy(base_raw, policy)
            if "data2d_base" in spec
            else self._apply_nan_policy(spec["data2d"], policy)
        )

        Xc, Yc, Zc = spec["X"], spec["Y"], spec["Z"]

        color_sel = w["color_by"].value

        if color_sel == "__constant__":
            intensity2d = np.zeros_like(data2d, dtype=float)
            show_cb = False

        elif color_sel in ("__x__", "__y__", "__z__"):
            show_cb = True
            intensity2d = {"__x__": Xc, "__y__": Yc, "__z__": Zc}[color_sel]

        elif color_sel == "__self__":
            intensity2d = np.asarray(data2d, dtype=float)
            show_cb = True

        else:
            show_cb = True
            if color_sel not in self.dict_data:
                intensity2d = np.asarray(data2d, dtype=float)
            else:
                if spec["type"] == "plane":
                    # plane: sample raw on surface coordinates
                    intensity2d = self._sample_raw_on_surface(
                        color_sel, Xc, Yc, Zc, nan_policy=policy
                    )
                else:
                    # slice: reslice raw using same (axis,pos,thick)
                    intensity2d = self._slice_mean_2d(
                        self.dict_data[color_sel],
                        spec["axis_phys"],
                        int(spec["pos"]),
                        int(spec["thickness"]),
                        nan_policy=policy,
                    )
        intensity2d = self._apply_nan_policy(intensity2d, policy)
        # -----------------------------
        # HOLES: only when nan_policy == "none"
        # -----------------------------
        if policy == "none":
            # define invisibility from the ORIGINAL base slice/plane values
            # (so the “outside active volume” stays hidden even if you color by another field)
            mask_invalid = ~np.isfinite(
                base_raw
            )  # True where you want opacity=0

            Xplot = np.array(Xc, dtype=float, copy=True)
            Yplot = np.array(Yc, dtype=float, copy=True)
            Zplot = np.array(Zc, dtype=float, copy=True)
            Cplot = np.array(intensity2d, dtype=float, copy=True)

            Xplot[mask_invalid] = np.nan
            Yplot[mask_invalid] = np.nan
            Zplot[mask_invalid] = np.nan
            Cplot[mask_invalid] = np.nan
        else:
            Xplot, Yplot, Zplot, Cplot = Xc, Yc, Zc, intensity2d

        # ---- Sync slider bounds safely: ONLY EXPAND (never shrink) ----
        rs = w["range_slider"]

        if color_sel != "__constant__":
            new_min = float(np.nanmin(Cplot))
            new_max = float(np.nanmax(Cplot))

            if not np.isfinite(new_min):
                new_min = 0.0
            if not np.isfinite(new_max):
                new_max = 1.0
            if new_max <= new_min:
                new_max = new_min + 1e-12

            target_min = min(float(rs.min), new_min)
            target_max = max(float(rs.max), new_max)
            if target_max <= target_min:
                target_max = target_min + 1e-12

            rs.unobserve(self._on_layer_param_changed, names="value")
            try:
                v0, v1 = rs.value
                v0 = float(np.clip(v0, target_min, target_max))
                v1 = float(np.clip(v1, target_min, target_max))
                if v1 < v0:
                    v0, v1 = v1, v0
                rs.value = (v0, v1)

                rs.min = target_min
                rs.max = target_max
                step = (target_max - target_min) / 300
                rs.step = step if step > 0 else 0.01

                if bool(w["auto_range"].value):
                    rs.value = (new_min, new_max)
            finally:
                rs.observe(self._on_layer_param_changed, names="value")

        auto_range = bool(w["auto_range"].value)
        rmin, rmax = w["range_slider"].value
        data_min = float(np.nanmin(Cplot))
        data_max = float(np.nanmax(Cplot))
        if not np.isfinite(data_min):
            data_min = 0.0
        if not np.isfinite(data_max):
            data_max = 1.0
        if data_max == data_min:
            data_max = data_min + 1e-12

        cmin, cmax = (
            (data_min, data_max) if auto_range else (float(rmin), float(rmax))
        )
        if cmax <= cmin:
            cmax = cmin + 1e-12

        colorbar = None
        if show_cb and bool(w["show_colorbar"].value):
            if color_sel == "__self__":
                title = layer_name
            elif color_sel in ("__x__", "__y__", "__z__"):
                title = color_sel.strip("_")
            else:
                title = color_sel
            colorbar = self._colorbar_dict(title, cbar_index=cbar_index)

        Xp, Yp, Zp = self._map_xyz_surface_to_plot(Xplot, Yplot, Zplot)
        trace = go.Surface(
            name=layer_name,
            x=Xp,
            y=Yp,
            z=Zp,
            surfacecolor=Cplot,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            showscale=(colorbar is not None),
            connectgaps=False,
            colorbar=colorbar,
        )
        return trace, (colorbar is not None)

    def _make_clip_mesh_trace(self, layer_name: str, cbar_index: int = 0):
        spec = self._layers[layer_name]

        # --- ensure clip offset fields exist (backward-compatible) ---
        if "origin0" not in spec:
            spec["origin0"] = np.asarray(
                spec.get("origin", self._center_origin_xyz()), dtype=float
            )
        if "offset" not in spec:
            spec["offset"] = 0.0
        if "origin" not in spec:
            # effective origin = origin0 + n_hat * offset
            n = np.asarray(spec.get("normal", [0, 0, 1]), dtype=float)
            nn = float(np.linalg.norm(n))
            n_hat = n / nn if nn > 0 else np.array([0.0, 0.0, 1.0], float)
            spec["origin"] = np.asarray(
                spec["origin0"], float
            ) + n_hat * float(spec["offset"])

        src = spec["source"]
        if src not in self.dict_data:
            trace = go.Mesh3d(
                name=layer_name, x=[], y=[], z=[], i=[], j=[], k=[]
            )
            return (trace, False)

        w = self._layer_widgets[layer_name]
        policy = w["nan_policy"].value

        # ------------------------------------------------------------
        # 1) Geometry volume: NEVER apply nan_policy here.
        #    Treat NaNs as "outside" by pushing them below iso, same as clip outside.
        # ------------------------------------------------------------
        base_raw = np.asarray(self.dict_data[src], dtype=float)

        # half-space keep mask (depends only on geometry/grid)
        keep = self._clip_volume_halfspace(
            base_raw, spec["normal"], spec["origin"], spec["side"]
        )

        finite = np.isfinite(base_raw)
        if not np.any(finite):
            return (
                go.Mesh3d(
                    name=layer_name,
                    x=[],
                    y=[],
                    z=[],
                    i=[],
                    j=[],
                    k=[],
                    opacity=float(w["op"].value),
                ),
                False,
            )

        vmin = float(np.nanmin(base_raw))  # ignores NaNs
        vmax = float(np.nanmax(base_raw))
        span = vmax - vmin
        pad = 10.0 * (span if np.isfinite(span) and span > 0 else 1.0)

        vol = base_raw.copy()
        # push NaNs below iso
        vol[~finite] = vmin - pad
        # push clipped-away half-space below iso
        vol[~keep] = vmin - pad
        # temporarily render as if it was a raw key, but using this custom vol
        # easiest: inline a slightly modified copy of _make_mesh_trace_for_key
        # (below is the minimal adjustment: replace arr with vol, and vals sampling uses self._rgi[src])

        iso = float(w["thr"].value)
        opacity = float(w["op"].value)
        colorscale = self.get_colorscale(w["cmap"].value)

        vmin2 = float(np.nanmin(vol))
        vmax2 = float(np.nanmax(vol))
        if (
            (not np.isfinite(vmin2))
            or (not np.isfinite(vmax2))
            or not (vmin2 < iso < vmax2)
        ):
            return (
                go.Mesh3d(
                    name=layer_name,
                    x=[],
                    y=[],
                    z=[],
                    i=[],
                    j=[],
                    k=[],
                    opacity=opacity,
                    showscale=False,
                ),
                False,
            )

        # geometry
        verts, faces, _, _ = marching_cubes(vol, level=iso, step_size=1)
        verts_scaled = verts * self.voxel_size

        # ------------------------------------------------------------
        # 2) Sample ORIGINAL source for coloring + for "hole" detection
        # ------------------------------------------------------------
        vals_raw = np.asarray(
            self._rgi[src](verts), dtype=float
        )  # may contain NaNs
        bad_v = ~np.isfinite(vals_raw)
        # verts from marching_cubes are already (z,y,x) in index coordinates
        verts_zyx_idx = verts
        vals_raw = np.asarray(self._rgi[src](verts_zyx_idx), dtype=float)

        # Optional: preserve true holes when nan_policy == "none"
        if str(policy).lower().strip() == "none":
            face_keep = ~np.any(bad_v[faces], axis=1)
            faces = faces[face_keep]
            if faces.size == 0:
                return (
                    go.Mesh3d(
                        name=layer_name,
                        x=[],
                        y=[],
                        z=[],
                        i=[],
                        j=[],
                        k=[],
                        opacity=float(w["op"].value),
                    ),
                    False,
                )

        # Apply policy only for colors/intensity
        vals = self._apply_nan_policy(vals_raw, policy)

        verts_plot = self._map_zyx_to_plot(verts_scaled)

        # coloring selection identical to your raw path
        show_colorbar = bool(w["show_colorbar"].value)
        auto_range = bool(w["auto_range"].value)
        rmin, rmax = w["range_slider"].value
        color_sel = w["color_by"].value

        if color_sel == "__constant__":
            intensity = np.zeros(len(verts_plot), dtype=float)
        elif color_sel in ("__x__", "__y__", "__z__"):
            axis = {"__x__": 0, "__y__": 1, "__z__": 2}[color_sel]
            intensity = verts_plot[:, axis].astype(float)
        elif color_sel in ("__self__", layer_name, src):
            intensity = np.asarray(vals, dtype=float)
        else:
            # allow coloring by other raw layers
            if color_sel in self._rgi:
                intensity = self._rgi[color_sel](verts_zyx_idx)
            else:
                intensity = np.asarray(vals, dtype=float)

        intensity = self._apply_nan_policy(intensity, policy)

        new_min = float(np.nanmin(intensity))
        new_max = float(np.nanmax(intensity))
        if not np.isfinite(new_min):
            new_min = 0.0
        if not np.isfinite(new_max):
            new_max = 1.0
        if new_max <= new_min:
            new_max = new_min + 1e-12

        rs = w["range_slider"]

        # RESET bounds when color_by changes (this is the key difference vs "only expand")
        prev_cb = w.get("_range_ref_color_by", None)
        cb_changed = prev_cb != color_sel
        w["_range_ref_color_by"] = color_sel

        rs.unobserve(self._on_layer_param_changed, names="value")
        try:
            # set bounds to the *current* color_by field range
            rs.min = new_min
            rs.max = new_max
            step = (new_max - new_min) / 300
            rs.step = step if step > 0 else 0.01

            if bool(w["auto_range"].value) or cb_changed:
                # auto-range OR color-by changed -> reset slider window to full range
                rs.value = (new_min, new_max)
            else:
                # keep user's manual window but clamp to new bounds
                v0, v1 = rs.value
                v0 = float(np.clip(v0, new_min, new_max))
                v1 = float(np.clip(v1, new_min, new_max))
                if v1 < v0:
                    v0, v1 = v1, v0
                rs.value = (v0, v1)
        finally:
            rs.observe(self._on_layer_param_changed, names="value")

        # IMPORTANT: read rmin/rmax AFTER bounds update
        rmin, rmax = rs.value

        # now set cmin/cmax
        auto_range = bool(w["auto_range"].value)
        cmin, cmax = (
            (new_min, new_max) if auto_range else (float(rmin), float(rmax))
        )
        if cmax <= cmin:
            cmax = cmin + 1e-12

        colorbar = None
        if (color_sel != "__constant__") and show_colorbar:
            title = (
                src
                if color_sel == "__self__"
                else (
                    color_sel.strip("_")
                    if color_sel in ("__x__", "__y__", "__z__")
                    else color_sel
                )
            )
            colorbar = self._colorbar_dict(title, cbar_index=cbar_index)

        trace = go.Mesh3d(
            name=layer_name,
            x=verts_plot[:, 0],
            y=verts_plot[:, 1],
            z=verts_plot[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=intensity,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=colorbar,
            showscale=(colorbar is not None),
            cmin=cmin,
            cmax=cmax,
            flatshading=False,
            lighting=dict(
                ambient=w["light_ambient"].value,
                diffuse=w["light_diffuse"].value,
                specular=w["light_specular"].value,
                roughness=w["light_roughness"].value,
                fresnel=w["light_fresnel"].value,
            ),
        )

        return trace, (colorbar is not None)

    def _map_zyx_to_plot(self, ZYX):
        """
        ZYX: array (..., 3) with columns (z, y, x) in physical units.
        Returns (..., 3) ordered according to self.PLOT_ORDER.
        """
        ZYX = np.asarray(ZYX, dtype=float)
        if ZYX.shape[-1] != 3:
            raise ValueError(f"Expected (...,3), got {ZYX.shape}")

        if self.PLOT_ORDER == "xyz":
            # (z,y,x) -> (x,y,z)
            return ZYX[..., [2, 1, 0]]
        if self.PLOT_ORDER == "zyx":
            return ZYX
        raise ValueError(f"Unsupported PLOT_ORDER={self.PLOT_ORDER}")

    def _map_xyz_surface_to_plot(self, X, Y, Z):
        if self.PLOT_ORDER == "xyz":
            return X, Y, Z
        if self.PLOT_ORDER == "zyx":
            # Plotly expects (x,y,z). If your "plot order" is zyx, swap axes.
            # meaning: plotly-x <- z, plotly-y <- y, plotly-z <- x
            return Z, Y, X
        raise ValueError(f"Unsupported PLOT_ORDER={self.PLOT_ORDER}")

    # =========================
    # Callbacks
    # =========================
    def _wire_global_callbacks(self):
        self.theme_toggle.observe(self._on_theme, names="value")
        self.rotate_toggle.observe(self._on_rotate_toggle, names="value")
        self.grid_toggle.observe(self._on_grid_toggle, names="value")
        self.edit_key.observe(self._on_edit_key_changed, names="value")

        self.add_layer_kind.observe(self._sync_create_layer_ui, names="value")
        self.add_slice_axis.observe(
            self._sync_add_slice_pos_range, names="value"
        )
        self.add_plane_center_btn.on_click(self._on_center_origin_clicked)
        self.add_layer_btn.on_click(self._on_create_layer_clicked)
        self.add_slice_center_btn.on_click(self._on_center_slice_pos_clicked)

        for ww in (
            self.add_plane_nx,
            self.add_plane_ny,
            self.add_plane_nz,
            self.add_plane_ox,
            self.add_plane_oy,
            self.add_plane_oz,
        ):
            ww.observe(self._sync_create_offset_bounds, names="value")
        self.add_plane_offset.observe(
            self._sync_create_offset_bounds, names="value"
        )
        # --- rename / delete layer callbacks ---
        self.rename_layer_btn.on_click(self._on_rename_layer_clicked)
        self.delete_layer.on_click(self._on_delete_layer_clicked)

    def _on_visible_changed(self, change):
        self._update_all_traces()

    def _on_edit_key_changed(self, change):
        new_k = change["new"]  # selected layer key
        old_k = change.get("old") or ""
        if new_k is None:
            self._sync_layer_panels_visibility()
            return
        # Only auto-fill if the user hasn't started editing:
        # - textbox empty, or
        # - textbox still equals the previous layer name (default autofill)
        if hasattr(self, "rename_layer_text"):
            cur = self.rename_layer_text.value or ""
            if (cur.strip() == "") or (cur == old_k):
                self.rename_layer_text.value = new_k or ""
        if getattr(self, "_suspend_rename_autofill_once", False):
            self._suspend_rename_autofill_once = False
            self._sync_layer_panels_visibility()
            return
        self._sync_layer_panels_visibility()

    def _on_layer_param_changed(self, change):
        self._update_all_traces()

    def _on_theme(self, change):
        self.fig.update_layout(
            template="plotly_dark"
            if self.theme_toggle.value
            else "plotly_white",
            scene=dict(
                xaxis=dict(
                    linecolor=self._axis_line_color(),
                    gridcolor=self._axis_grid_color(),
                    showgrid=bool(self.grid_toggle.value),
                ),
                yaxis=dict(
                    linecolor=self._axis_line_color(),
                    gridcolor=self._axis_grid_color(),
                    showgrid=bool(self.grid_toggle.value),
                ),
                zaxis=dict(
                    linecolor=self._axis_line_color(),
                    gridcolor=self._axis_grid_color(),
                    showgrid=bool(self.grid_toggle.value),
                ),
            ),
        )

        self._update_all_traces()

    def _on_rotate_toggle(self, change):
        if self.rotate_toggle.value:
            self._start_rotation()
        else:
            self._stop_rotation()

    def _on_center_origin_clicked(self, _):
        c_phys = self._center_origin_xyz()  # physical XYZ
        c_plot = self._phys_to_plot_point(c_phys)  # plot XYZ
        self.add_plane_ox.value = float(c_plot[0])
        self.add_plane_oy.value = float(c_plot[1])
        self.add_plane_oz.value = float(c_plot[2])
        if hasattr(self, "add_plane_offset"):
            self.add_plane_offset.value = 0.0

    def _on_center_slice_pos_clicked(self, _):
        src = self._default_source_key()
        if not src or src not in self.dict_data:
            return

        nz, ny, nx = self.dict_data[src].shape
        ax_plot = self.add_slice_axis.value
        ax_phys = self._plot_axis_to_phys_axis(ax_plot)

        mid = {"x": (nx - 1) // 2, "y": (ny - 1) // 2, "z": (nz - 1) // 2}[
            ax_phys
        ]
        self.add_slice_pos.value = int(mid)

    def _on_grid_toggle(self, change):
        show = bool(self.grid_toggle.value)
        with self.fig.batch_update():
            self.fig.update_layout(
                scene=dict(
                    xaxis=dict(showgrid=show),
                    yaxis=dict(showgrid=show),
                    zaxis=dict(showgrid=show),
                )
            )

    # =========================
    # plot↔physical conversion helpers
    # =========================
    def _C_plot_from_phys(self) -> np.ndarray:
        """3x3 permutation matrix: plot = C @ phys."""
        if self.PLOT_ORDER == "xyz":
            return np.eye(3)
        if self.PLOT_ORDER == "zyx":
            return np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        raise ValueError(f"Unsupported PLOT_ORDER={self.PLOT_ORDER}")

    def _phys_to_plot_point(self, p_phys):
        p = np.asarray(p_phys, dtype=float).reshape(
            3,
        )
        C = self._C_plot_from_phys()
        return C @ p

    def _plot_to_phys_point(self, p_plot):
        p = np.asarray(p_plot, dtype=float).reshape(
            3,
        )
        C = self._C_plot_from_phys()
        # inverse of permutation is transpose (same here)
        return C.T @ p

    def _phys_to_plot_vec(self, v_phys):
        return self._phys_to_plot_point(v_phys)

    def _plot_to_phys_vec(self, v_plot):
        return self._plot_to_phys_point(v_plot)

    def _plot_axis_to_phys_axis(self, ax_plot: str) -> str:
        if self.PLOT_ORDER == "xyz":
            return ax_plot
        if self.PLOT_ORDER == "zyx":
            return {"x": "z", "y": "y", "z": "x"}[ax_plot]
        raise ValueError(f"Unsupported PLOT_ORDER={self.PLOT_ORDER}")

    def _phys_axis_to_plot_axis(self, ax_phys: str) -> str:
        if self.PLOT_ORDER == "xyz":
            return ax_phys
        if self.PLOT_ORDER == "zyx":
            return {"x": "z", "y": "y", "z": "x"}[ax_phys]
        raise ValueError(f"Unsupported PLOT_ORDER={self.PLOT_ORDER}")

    # ----------------------------
    # font helper
    # ----------------------------
    def _bold_font(self, size=None):
        if size is None:
            size = self.fontsize
        return dict(family="DejaVu Sans Bold", size=int(size))

    def _colorbar_outline_color(self) -> str:
        return (
            self._DARK_OUTLINE
            if self.theme_toggle.value
            else self._LIGHT_OUTLINE
        )

    def _colorbar_tick_color(self) -> str:
        return self._DARK_TEXT if self.theme_toggle.value else self._LIGHT_TEXT

    def _axis_line_color(self):
        return (
            self._DARK_AXIS_LINE
            if self.theme_toggle.value
            else self._LIGHT_AXIS_LINE
        )

    def _axis_grid_color(self):
        return (
            self._DARK_AXIS_GRID
            if self.theme_toggle.value
            else self._LIGHT_AXIS_GRID
        )

    def _colorbar_dict(self, title: str, cbar_index: int = 0) -> dict:
        return dict(
            title=dict(
                text=title,
                font=self._bold_font(self.fontsize * 1.5),
                side="top",
            ),
            tickfont=self._bold_font(self.fontsize)
            | dict(color=self._colorbar_tick_color()),
            thickness=28,
            # theme-aware outline
            outlinewidth=2,
            outlinecolor=self._colorbar_outline_color(),
            # control length explicitly
            len=float(self.CBAR_LEN),
            lenmode="fraction",
            # OPTIONAL: stable placement for multiple visible layers
            x=CBAR_X0 + CBAR_DX * cbar_index,
            xanchor="left",
            y=0.5,
            yanchor="middle",
        )

    # ----------------------------
    # NaN policy
    # ----------------------------
    def _apply_nan_policy(
        self, arr: np.ndarray, policy: str = "none"
    ) -> np.ndarray:
        """
        Apply NaN replacement policy.

        Note: when policy == 'none', NaNs are preserved and treated as
        holes (opacity = 0) in slice and plane rendering.
        """
        arr = np.asarray(arr, dtype=float)
        bad = ~np.isfinite(arr)  # catches NaN and ±inf
        if policy == "none" or not bad.any():
            return arr

        tmp = arr.copy()
        tmp[~np.isfinite(tmp)] = np.nan  # so nan* reducers work

        if policy == "mean":
            v = np.nanmean(tmp)
        elif policy == "zero":
            v = 0.0
        elif policy == "min":
            v = np.nanmin(tmp)
        elif policy == "max":
            v = np.nanmax(tmp)
        else:
            return arr

        if not np.isfinite(v):
            v = 0.0

        out = arr.copy()
        out[bad] = v
        return out

    # ----------------------------
    # layers anim helper
    # ----------------------------
    def _anim_param_bounds(self, layer_key, param):
        """
        Return (vmin, vmax) for the animation parameter.
        MUST always return a 2-tuple: (None, None) on failure.
        """

        try:
            layers = self._get_layers_container()
            spec = layers.get(layer_key)
            if not isinstance(spec, dict):
                return (None, None)

            # robust layer type
            t_raw = (
                spec.get("type")
                or spec.get("layer_type")
                or spec.get("kind")
                or "raw"
            )
            t = str(t_raw).lower().strip()
            if "slice" in t:
                t = "slice"
            elif "plane" in t:
                t = "plane"
            elif "clip" in t:
                t = "clip"
            else:
                t = "raw"

            # robust param (some UIs might pass "pos" or "position")
            p = str(param or "").lower().strip()
            if p.startswith("pos"):
                p = "pos"
            elif p.startswith("opa"):
                p = "opacity"
            elif p.startswith("off"):
                p = "offset"

            if p == "opacity":
                return (0.0, 1.0)

            if p == "offset":
                if t == "plane":
                    out = self._plane_offset_bounds(layer_key)
                    return (
                        out
                        if (isinstance(out, tuple) and len(out) == 2)
                        else (None, None)
                    )
                if t == "clip":
                    out = self._clip_offset_bounds(layer_key)
                    return (
                        out
                        if (isinstance(out, tuple) and len(out) == 2)
                        else (None, None)
                    )
                return (None, None)

            if p == "pos":
                if t != "slice":
                    return (None, None)

                # Prefer reading the EXISTING slice pos widget bounds
                w = (
                    getattr(self, "_layer_widgets", {}).get(layer_key, {})
                    or {}
                )
                pos_w = (
                    w.get("pos")
                    or w.get("slice_pos")
                    or w.get("pos_slider")
                    or w.get("position")
                )
                if (
                    pos_w is not None
                    and getattr(pos_w, "min", None) is not None
                    and getattr(pos_w, "max", None) is not None
                ):
                    try:
                        return (float(pos_w.min), float(pos_w.max))
                    except Exception:
                        pass

                # Fallback to shape from stored vol3d or source
                vol = spec.get("vol3d", None)
                if vol is None:
                    src = spec.get("source", None)
                    vol = self.dict_data.get(src) if src else None

                if vol is None or np.ndim(vol) != 3:
                    return (None, None)

                ax = str(spec.get("axis_phys", spec.get("axis", "z"))).lower()
                dim = {"z": 0, "y": 1, "x": 2}.get(ax, 0)  # vol is (z,y,x)
                n = int(vol.shape[dim])
                return (0.0, float(max(0, n - 1)))

            return (None, None)

        except Exception:
            return (None, None)

    def _plane_offset_bounds(self, layer_key: str):
        layers = self._get_layers_container()
        spec = layers[layer_key]

        origin0 = np.asarray(spec.get("origin0", spec.get("origin")), float)
        normal = np.asarray(spec.get("normal", (0, 0, 1)), float)

        # must exist in your file; if named differently, adapt the call here
        off_min, off_max = self._offset_bounds_for_origin_normal(
            origin0, normal
        )
        return float(off_min), float(off_max)

    def _clip_offset_bounds(self, layer_key: str):
        layers = self._get_layers_container()
        spec = layers[layer_key]

        if "origin0" not in spec:
            spec["origin0"] = np.asarray(
                spec.get("origin", self._center_origin_xyz()), dtype=float
            )

        origin0 = np.asarray(spec["origin0"], float)
        normal = np.asarray(spec.get("normal", [0, 0, 1]), float)

        return self._offset_bounds_for_origin_normal(origin0, normal)

    def _volume_bounds_phys(self):
        """Axis-aligned bounds of the volume in PHYSICAL xyz coordinates."""
        nz, ny, nx = self._shape0
        Lx = (nx - 1) * float(self.voxel_size[2])
        Ly = (ny - 1) * float(self.voxel_size[1])
        Lz = (nz - 1) * float(self.voxel_size[0])
        lo = np.array([0.0, 0.0, 0.0], dtype=float)
        hi = np.array([Lx, Ly, Lz], dtype=float)
        return lo, hi

    def _offset_bounds_for_origin_normal(
        self, origin0_phys, n_hat_phys, eps=1e-12
    ):
        """
        Compute [off_min, off_max] such that origin0 + n_hat*off stays inside volume bounds.
        """
        origin0 = np.asarray(origin0_phys, dtype=float).reshape(
            3,
        )
        n_hat = np.asarray(n_hat_phys, dtype=float).reshape(
            3,
        )

        lo, hi = self._volume_bounds_phys()

        off_min = -np.inf
        off_max = +np.inf

        for i in range(3):
            ni = float(n_hat[i])
            o0 = float(origin0[i])

            if abs(ni) < eps:
                # No freedom along this axis; origin must already be in-bounds
                if (o0 < lo[i] - 1e-9) or (o0 > hi[i] + 1e-9):
                    # Empty feasible interval: return a degenerate range that forces 0
                    return 0.0, 0.0
                continue

            # Solve lo <= o0 + ni*off <= hi
            a = (lo[i] - o0) / ni
            b = (hi[i] - o0) / ni
            ai, bi = (a, b) if a <= b else (b, a)

            off_min = max(off_min, ai)
            off_max = min(off_max, bi)

        if not np.isfinite(off_min):
            off_min = 0.0
        if not np.isfinite(off_max):
            off_max = 0.0

        if off_max < off_min:
            # No feasible solution
            return 0.0, 0.0

        return float(off_min), float(off_max)

    def _clamp_offset_in_bounds(self, origin0_phys, n_phys, offset):
        """
        Clamp offset so origin0 + n_hat*offset stays inside volume bounds.
        Returns (offset_clamped, off_min, off_max, n_hat)
        """
        n = np.asarray(n_phys, dtype=float).reshape(
            3,
        )
        nn = float(np.linalg.norm(n))
        if nn == 0.0:
            return 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0], float)

        n_hat = n / nn
        off_min, off_max = self._offset_bounds_for_origin_normal(
            origin0_phys, n_hat
        )
        off = float(np.clip(float(offset), off_min, off_max))
        return off, off_min, off_max, n_hat

    def _set_offset_slider_bounds(self, off_slider, off_min, off_max):
        """Safely update a FloatSlider bounds + value without leaving it out-of-range."""
        off_slider.min = float(off_min)
        off_slider.max = float(off_max)
        off_slider.value = float(
            np.clip(float(off_slider.value), off_slider.min, off_slider.max)
        )

    def _sync_create_offset_bounds(self, _=None):
        # read current CREATE values in PLOT coords
        n_plot = np.array(
            [
                self.add_plane_nx.value,
                self.add_plane_ny.value,
                self.add_plane_nz.value,
            ],
            dtype=float,
        )
        o_plot = np.array(
            [
                self.add_plane_ox.value,
                self.add_plane_oy.value,
                self.add_plane_oz.value,
            ],
            dtype=float,
        )

        n_phys = self._plot_to_phys_vec(n_plot)
        o0_phys = self._plot_to_phys_point(o_plot)

        off_clamped, off_min, off_max, _ = self._clamp_offset_in_bounds(
            o0_phys, n_phys, self.add_plane_offset.value
        )
        self._set_offset_slider_bounds(self.add_plane_offset, off_min, off_max)

        # avoid recursive observe loops: only assign if changed
        if abs(self.add_plane_offset.value - off_clamped) > 1e-12:
            self.add_plane_offset.value = float(off_clamped)

    def _sync_anim_layer_param_ui(self, *args):
        if getattr(self, "_anim_refreshing_layers", False):
            return

        k = self.anim_layer_key.value
        if not k:
            self.anim_layer_param.options = [("— Select —", None)]
            self.anim_layer_param.value = None
            return

        layers = self._get_layers_container()
        layer = layers[k]
        t = layer.get("type") or layer.get("layer_type") or "raw"

        params = self._ANIM_REGISTRY.get(t, [])
        opts = [("— Select —", None)] + [(p, p) for p in params]

        self.anim_layer_param.options = opts
        self.anim_layer_param.value = None  # always require explicit choice

    def _sync_anim_layer_bounds_ui(self, *args):
        """
        Update anim_range bounds when (layer,param) changes.
        Critical: do NOT run when the event comes from anim_range itself,
        otherwise dragging will be overwritten and the slider "freezes".
        """
        # 1) ignore events coming from anim_range (prevents self-reset while dragging)
        if args:
            ch = args[0]
            owner = getattr(ch, "owner", None)
            if owner is None and isinstance(ch, dict):
                owner = ch.get("owner", None)
            if owner is self.anim_range:
                return

        # 2) hard recursion guard
        if getattr(self, "_anim_ui_busy", False):
            return

        k = getattr(self.anim_layer_key, "value", None)
        p = getattr(self.anim_layer_param, "value", None)
        if not k or not p:
            self.anim_range.disabled = True
            return
        out = self._anim_param_bounds(k, p)
        if not (isinstance(out, tuple) and len(out) == 2):
            self.anim_range.disabled = True
            return
        vmin, vmax = out
        if vmin is None or vmax is None:
            self.anim_range.disabled = True
            return

        # normalize param for step
        pnorm = str(p).lower().strip()
        if pnorm.startswith("pos"):
            pnorm = "pos"

        self._anim_ui_busy = True
        try:
            self.anim_range.disabled = False

            vmin = float(vmin)
            vmax = float(vmax)
            if vmax <= vmin:
                vmax = vmin + 1e-12

            # For slice pos: integer stepping helps and avoids jitter
            if pnorm == "pos":
                try:
                    self.anim_range.step = 1.0
                except Exception:
                    pass

            # Use hold_trait_notifications to avoid repeated front-end churn
            try:
                with self.anim_range.hold_trait_notifications():
                    self.anim_range.min = vmin
                    self.anim_range.max = vmax

                    lo, hi = self.anim_range.value
                    lo = max(float(lo), vmin)
                    hi = min(float(hi), vmax)
                    if hi <= lo:
                        lo, hi = vmin, vmax

                    if pnorm == "pos":
                        lo = float(int(round(lo)))
                        hi = float(int(round(hi)))
                        if hi <= lo:
                            lo, hi = (
                                float(int(round(vmin))),
                                float(int(round(vmax))),
                            )

                    self.anim_range.value = (lo, hi)
            except Exception:
                # fallback without context manager
                self.anim_range.min = vmin
                self.anim_range.max = vmax
        finally:
            self._anim_ui_busy = False

    def _get_layers_container(self):
        """
        Return the existing layers container without changing architecture.
        Tries common attribute names used in this codebase.
        """
        for name in (
            "layers",
            "_layers",
            "layer_specs",
            "_layer_specs",
            "_layer_store",
        ):
            obj = getattr(self, name, None)
            if isinstance(obj, dict):
                return obj
        raise AttributeError(
            "Could not find layers container. "
            "Search for where you store layers (dict) and add its attribute name to _get_layers_container()."
        )

    def _anim_list_layers(self):
        layers = self._get_layers_container()
        out = []
        for k, layer in layers.items():
            # 'type' key name depends on your existing structure; keep it permissive
            t = None
            if isinstance(layer, dict):
                t = (
                    layer.get("type")
                    or layer.get("layer_type")
                    or layer.get("kind")
                )
            if t is None:
                t = "raw"
            out.append((f"{k} [{t}]", k))
        return out

    def _refresh_anim_layer_options(self, keep_selection: bool = True):
        """Refresh Layer dropdown options (for Animate=Layer), without changing layer architecture."""
        if not hasattr(self, "anim_layer_key"):
            return  # panel not built yet

        prev = self.anim_layer_key.value if keep_selection else None
        opts = self._anim_list_layers()

        self.anim_layer_key.options = opts

        if keep_selection and prev is not None:
            # restore if still present
            keys = {k for _, k in opts}
            if prev in keys:
                self.anim_layer_key.value = prev
                return

        # otherwise pick first if available
        if opts:
            self.anim_layer_key.value = opts[0][1]

    def _anim_refresh_layers_dropdown(self):
        if not hasattr(self, "anim_layer_key"):
            return
        if getattr(self, "_anim_refreshing_layers", False):
            return

        self._anim_refreshing_layers = True
        try:
            opts = [("— Select —", None)] + self._anim_list_layers()

            # IMPORTANT: do NOT “prefer” current value here if it might trigger loops.
            # Just set options and keep value if still valid.
            cur = self.anim_layer_key.value
            keys = {v for _, v in opts}

            # set options first, then set value only if needed
            self.anim_layer_key.options = opts
            if cur in keys:
                # keep current without forcing a write
                pass
            else:
                self.anim_layer_key.value = None

        finally:
            self._anim_refreshing_layers = False

    # ===========================================================================
    # Rotation
    # ===========================================================================
    def _start_rotation(self):
        if self._rotation_task is not None and not self._rotation_task.done():
            return

        async def rotate():
            # pick rotation mode (reuse anim_rot_type if you don't have a separate widget)
            def _live_mode():
                if hasattr(
                    self, "rot_type"
                ):  # if you have a widget for live rotation
                    return self.rot_type.value
                if hasattr(
                    self, "anim_rot_type"
                ):  # fallback to the export dropdown
                    return self.anim_rot_type.value
                return "orbit_z"

            # capture starting camera vectors once
            try:
                cam0 = self.fig.layout.scene.camera.to_plotly_json()
            except Exception:
                cam0 = None

            if cam0 and "eye" in cam0:
                eye0 = np.array(
                    [
                        cam0["eye"].get("x", 1.5),
                        cam0["eye"].get("y", 0.0),
                        cam0["eye"].get("z", 1.5),
                    ],
                    float,
                )
                up0 = np.array(
                    [
                        cam0.get("up", {}).get("x", 0.0),
                        cam0.get("up", {}).get("y", 0.0),
                        cam0.get("up", {}).get("z", 1.0),
                    ],
                    float,
                )
                center0 = cam0.get("center", {"x": 0.0, "y": 0.0, "z": 0.0})
            else:
                eye0 = np.array([1.5, 0.0, 1.5], float)
                up0 = np.array([0.0, 0.0, 1.0], float)
                center0 = {"x": 0.0, "y": 0.0, "z": 0.0}

            ang = 0.0
            dtheta = np.deg2rad(2.0)

            while self.rotate_toggle.value:
                ang += dtheta

                mode = _live_mode()
                axis = np.asarray(self._get_rotation_axis_plot(mode), float)
                n = float(np.linalg.norm(axis))
                axis = axis / n if n > 0 else np.array([0.0, 0.0, 1.0], float)

                eye = self._rotvec_axis_angle(eye0, axis, ang)
                upv = self._rotvec_axis_angle(up0, axis, ang)

                with self.fig.batch_update():
                    self.fig.layout.scene.camera = dict(
                        eye=dict(
                            x=float(eye[0]), y=float(eye[1]), z=float(eye[2])
                        ),
                        up=dict(
                            x=float(upv[0]), y=float(upv[1]), z=float(upv[2])
                        ),
                        center=center0,
                    )

                await asyncio.sleep(0.05)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._rotation_task = loop.create_task(rotate())

    def _stop_rotation(self):
        if self._rotation_task is not None:
            self._rotation_task.cancel()
            self._rotation_task = None

    def _default_source_key(self):
        for k, s in self._layers.items():
            if s.get("type") == "raw":
                return k
        return None

    # ------------------------------------------------------------------------------------
    # animation general
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _png_bytes_list_to_gif_ffmpeg_palette(
        png_frames: list[bytes],
        out_path: str,
        fps: int,
    ) -> None:
        """
        Robust GIF writer using FFmpeg on a PNG FILE SEQUENCE (no stdin piping).

        - Writes frames to a temp dir as frame_000000.png, frame_000001.png, ...
        - Generates a palette (palettegen), then encodes (paletteuse)
        - Raises a RuntimeError with FFmpeg stderr on failure
        """
        if not png_frames:
            raise RuntimeError("No frames to write.")

        # Basic validation: ensure frames look like PNG
        sig = b"\x89PNG\r\n\x1a\n"
        for i, b in enumerate(png_frames[:5]):
            if not isinstance(b, (bytes, bytearray)) or not b.startswith(sig):
                raise RuntimeError(
                    f"Frame {i} is not valid PNG bytes (missing PNG signature)."
                )

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fps = int(fps)
        if fps <= 0:
            raise ValueError("fps must be >= 1")

        with tempfile.TemporaryDirectory() as td:
            # Write PNG frames to disk
            for i, b in enumerate(png_frames):
                fn = os.path.join(td, f"frame_{i:06d}.png")
                with open(fn, "wb") as f:
                    f.write(b)

            pattern = os.path.join(td, "frame_%06d.png")
            palette = os.path.join(td, "palette.png")

            # 1) palettegen
            cmd1 = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-vf",
                "palettegen",
                palette,
            ]
            p1 = subprocess.run(cmd1, capture_output=True)
            if p1.returncode != 0 or not os.path.exists(palette):
                err = (p1.stderr or b"").decode("utf-8", "ignore").strip()
                out = (p1.stdout or b"").decode("utf-8", "ignore").strip()
                raise RuntimeError(
                    "FFmpeg palettegen failed.\n"
                    f"cmd: {' '.join(cmd1)}\n"
                    f"returncode={p1.returncode}\n"
                    f"stderr:\n{err}\n"
                    f"stdout:\n{out}"
                )

            # 2) paletteuse
            cmd2 = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-i",
                palette,
                "-lavfi",
                "paletteuse",
                out_path,
            ]
            p2 = subprocess.run(cmd2, capture_output=True)
            if p2.returncode != 0:
                err = (p2.stderr or b"").decode("utf-8", "ignore").strip()
                out = (p2.stdout or b"").decode("utf-8", "ignore").strip()
                raise RuntimeError(
                    "FFmpeg paletteuse failed.\n"
                    f"cmd: {' '.join(cmd2)}\n"
                    f"returncode={p2.returncode}\n"
                    f"stderr:\n{err}\n"
                    f"stdout:\n{out}"
                )

            if not os.path.exists(out_path):
                raise RuntimeError("FFmpeg reported success but output GIF was not created.")

    # ============================================================
    # Animation export — writers
    # ============================================================
    def _on_save_animation_clicked(self, btn):
        if self.anim_master_mode.value == "rotation":
            if self.anim_rot_type.value is None:
                self.anim_status.value = (
                    "<b style='color:#ffb3b3'>Select a rotation mode.</b>"
                )
                return
            else:
                return self._on_save_rotation_clicked(btn)

        if self.anim_master_mode.value == "layer":
            if (
                self.anim_layer_key.value is None
                or self.anim_layer_param.value is None
            ):
                self.anim_status.value = "<b style='color:#ffb3b3'>Select a layer and parameter.</b>"
                return
            else:
                return self._on_save_layer_clicked(btn)

    # ------------------------------------------------------------------------------------
    # saving rotation animation
    # ------------------------------------------------------------------------------------
    # ============================================================
    # Animation export — UI callbacks
    # ============================================================
    def _on_save_rotation_clicked(self, _):
        raw = (self.anim_name.value or "rotation").strip()
        fmt = (self.anim_format.value or "mp4").lower()

        p = Path(raw)
        if raw.endswith(("/", "\\")) or (p.exists() and p.is_dir()):
            out_path = str((p / "rotation").with_suffix(f".{fmt}"))
        else:
            if p.suffix:
                p = p.with_suffix("")
            out_path = str(p.with_suffix(f".{fmt}"))

        # stop live rotation while exporting
        self.rotate_toggle.value = False

        n_frames = int(self.anim_frames.value)
        fps = int(self.anim_fps.value)
        use_cam = bool(self.anim_use_current_camera.value)

        # width/height fallback
        w = self.fig.layout.width
        h = self.fig.layout.height
        w = 900 if w is None else int(str(w).replace("px", ""))
        h = 900 if h is None else int(str(h).replace("px", ""))

        # cancel previous export if still running
        t = getattr(self, "_export_task", None)
        if t is not None and not t.done():
            self._anim_cancel = True
            t.cancel()

        self._anim_cancel = False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        coro = self._save_rotation_animation_async(
            out_path=out_path,
            fmt=fmt,
            n_frames=n_frames,
            fps=fps,
            width=w,
            height=h,
            scale=2,
            use_current_camera_as_start=use_cam,
            max_workers=self.render_workers,
            max_in_flight=self.render_in_flight,
        )
        try:
            loop = asyncio.get_running_loop()
            self._export_task = loop.create_task(coro)
            self._export_task.add_done_callback(lambda t: self._on_done(t))
        except RuntimeError:
            asyncio.run(coro)

    def _on_stop_rotation_export_clicked(self, _):
        # request cancellation (shared by rotation + layer)
        self._anim_cancel = True

        # if you also created an Event-based flag, set it too (optional)
        ev = getattr(self, "_stop_anim_export", None)
        if ev is not None:
            ev.set()

        # cancel asyncio task (if any)
        t = getattr(self, "_export_task", None)
        if t is not None and not t.done():
            t.cancel()

        # optional: stop live rotation toggle
        self.rotate_toggle.value = False

        if hasattr(self, "anim_status"):
            self.anim_status.value = "Stopping…"
        if hasattr(self, "anim_progress"):
            self.anim_progress.bar_style = "warning"

    def _on_done(self, task: asyncio.Task):
        # This callback runs in the event loop thread (safe for widgets)
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    # ============================================================
    # Animation export — camera / rotation math
    # ============================================================
    def _get_rotation_axis_plot(self, mode: str):
        """Return rotation axis in PLOT (scene) coordinates."""
        # orbit_* are in PLOT coords (scene axes), not PHYSICAL coords
        if mode == "orbit_x":
            return np.array([1.0, 0.0, 0.0], float)
        if mode == "orbit_y":
            return np.array([0.0, 1.0, 0.0], float)
        if mode == "orbit_z":
            return np.array([0.0, 0.0, 1.0], float)

        if mode == "axis":
            # axis typed by user is in PLOT coords already
            return np.array(
                [
                    self.anim_axis_x.value,
                    self.anim_axis_y.value,
                    self.anim_axis_z.value,
                ],
                float,
            )

        if mode == "normal":
            src = (
                getattr(self, "anim_normal_src", None).value
                if hasattr(self, "anim_normal_src")
                else None
            )
            if not src or src not in getattr(self, "_layers", {}):
                return np.array([0.0, 0.0, 1.0], float)

            spec = self._layers.get(src, {})
            t = str(
                spec.get("type") or spec.get("layer_type") or "raw"
            ).lower()

            n_phys = None
            if t in ("plane", "clip"):
                n_phys = spec.get("normal", None)
            elif t == "slice":
                ax = str(spec.get("axis_phys", spec.get("ax", "z"))).lower()
                if ax == "x":
                    n_phys = np.array([1.0, 0.0, 0.0], float)
                elif ax == "y":
                    n_phys = np.array([0.0, 1.0, 0.0], float)
                else:
                    n_phys = np.array([0.0, 0.0, 1.0], float)

            if n_phys is None:
                return np.array([0.0, 0.0, 1.0], float)

            n_phys = np.asarray(n_phys, float).reshape(
                3,
            )
            nn = float(np.linalg.norm(n_phys))
            if nn == 0.0:
                return np.array([0.0, 0.0, 1.0], float)

            # layer normals are stored in PHYSICAL -> convert to PLOT
            return np.asarray(self._phys_to_plot_vec(n_phys / nn), float)

        return np.array([0.0, 0.0, 1.0], float)

    def _rotvec_axis_angle(self, v, axis, theta_rad):
        v = np.asarray(v, float)
        k = np.asarray(axis, float)
        kn = np.linalg.norm(k)
        if kn == 0:
            raise ValueError("Rotation axis must be non-zero.")
        k = k / kn
        ct = np.cos(theta_rad)
        st = np.sin(theta_rad)
        return v * ct + np.cross(k, v) * st + k * (np.dot(k, v)) * (1 - ct)

    # ============================================================
    # Animation export — Export core
    # ============================================================
    @staticmethod
    def _require_ffmpeg():
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg is required for GIF export.\n"
                "Install with:\n"
                "  conda install -c conda-forge ffmpeg\n"
            )
    async def _save_rotation_animation_async(
        self,
        out_path: str,
        fmt: str,
        n_frames: int,
        fps: int,
        r: float = 1.5,
        z: float = 1.5,
        width: int = 900,
        height: int = 900,
        scale: int = 2,
        use_current_camera_as_start: bool = True,
        max_workers: int | None = None,
        max_in_flight: int | None = None,
        preview_every: int = 1,
    ):
        """
        Rotation animation export with TRUE parallelism:
        - Camera update is sequential (main task) to build fig_json per frame.
        - Rendering (fig_json -> PNG) is process-parallel via _render_fig_json_to_png (module-level).
        - Frames are written in order using a pending buffer.

        Uses your rotation mode selector via self._get_rotation_axis_plot(mode).
        """

        def _fmt_seconds(sec: float) -> str:
            if sec < 60:
                return f"{sec:.2f} s"
            m, s = divmod(sec, 60.0)
            if m < 60:
                return f"{int(m)} min {s:04.1f} s"
            h, m = divmod(m, 60.0)
            return f"{int(h)} h {int(m)} min {s:04.1f} s"

        def _rot_matrix(axis_vec: np.ndarray, ang: float) -> np.ndarray:
            """Rodrigues rotation matrix for axis (plot coords) and angle (rad)."""
            ax = np.asarray(axis_vec, float).reshape(
                3,
            )
            n = float(np.linalg.norm(ax))
            if n <= 0:
                ax = np.array([0.0, 0.0, 1.0], float)
            else:
                ax = ax / n

            x, y, z_ = ax
            c = math.cos(ang)
            s = math.sin(ang)
            C = 1.0 - c
            return np.array(
                [
                    [c + x * x * C, x * y * C - z_ * s, x * z_ * C + y * s],
                    [y * x * C + z_ * s, c + y * y * C, y * z_ * C - x * s],
                    [z_ * x * C - y * s, z_ * y * C + x * s, c + z_ * z_ * C],
                ],
                dtype=float,
            )

        t_start = time.perf_counter()
        self._anim_cancel = False

        fmt = (fmt or "mp4").lower().strip()
        n_frames = int(n_frames)
        fps = int(fps)

        # ---- UI init ----
        try:
            if hasattr(self, "anim_progress"):
                self.anim_progress.value = 0
                self.anim_progress.min = 0
                self.anim_progress.max = int(n_frames)
                self.anim_progress.bar_style = "info"
        except Exception:
            pass
        try:
            if hasattr(self, "anim_status"):
                self.anim_status.value = "Preparing rotation export…"
        except Exception:
            pass

        # ---- base (offscreen) figure ----
        base_fig = go.Figure(self.fig.to_plotly_json())

        # ---- starting camera ----
        cam0 = None
        try:
            if use_current_camera_as_start:
                cam0 = self.fig.layout.scene.camera
        except Exception:
            cam0 = None

        if cam0 is None:
            cam0 = dict(
                eye=dict(x=float(r), y=float(r), z=float(z)),
                up=dict(x=0.0, y=0.0, z=1.0),
                center=dict(x=0.0, y=0.0, z=0.0),
            )
        else:
            cam0 = (
                cam0.to_plotly_json()
                if hasattr(cam0, "to_plotly_json")
                else dict(cam0)
            )

        eye0 = np.array(
            [
                cam0.get("eye", {}).get("x", float(r)),
                cam0.get("eye", {}).get("y", float(r)),
                cam0.get("eye", {}).get("z", float(z)),
            ],
            dtype=float,
        )
        up0 = np.array(
            [
                cam0.get("up", {}).get("x", 0.0),
                cam0.get("up", {}).get("y", 0.0),
                cam0.get("up", {}).get("z", 1.0),
            ],
            dtype=float,
        )
        center0 = cam0.get("center", {"x": 0.0, "y": 0.0, "z": 0.0})

        # ---- rotation axis from your mode selector ----
        mode = (
            getattr(self, "anim_rot_type", None).value
            if hasattr(self, "anim_rot_type")
            else "orbit_z"
        )
        axis = np.asarray(
            self._get_rotation_axis_plot(str(mode)), float
        ).reshape(
            3,
        )
        nax = float(np.linalg.norm(axis))
        axis = axis / nax if nax > 0 else np.array([0.0, 0.0, 1.0], float)

        # ---- concurrency knobs ----
        if max_workers is None:
            max_workers = int(getattr(self, "render_workers", 4))
        if max_in_flight is None:
            max_in_flight = int(getattr(self, "render_in_flight", max_workers))

        max_workers = max(1, int(max_workers))
        max_in_flight = max(1, int(max_in_flight))

        render_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )
        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(max_in_flight)

        writer = None
        preview_widget = getattr(self, "anim_preview", None)

        gif_enabled = False
        try:
            if hasattr(self, "anim_status"):
                self.anim_status.value = "Rendering rotation…"

            if fmt == "mp4":
                try:
                    import imageio_ffmpeg  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "MP4 export requires imageio-ffmpeg. Install with: pip install imageio-ffmpeg"
                    ) from e
                writer = iio2.get_writer(
                    out_path, fps=int(fps), codec="libx264"
                )
            elif fmt == "gif":
                gif_png_frames = []
                self._require_ffmpeg()
                gif_enabled = True
                writer = None
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            async def _schedule_render(fig_json: dict, idx: int):
                async with sem:
                    if self._anim_cancel:
                        return idx, None
                    png = await loop.run_in_executor(
                        render_pool,
                        _render_fig_json_to_png,  # must be module-level
                        fig_json,
                        int(width),
                        int(height),
                        int(scale),
                    )
                    return idx, png

            in_flight: dict[asyncio.Task, int] = {}
            pending: dict[int, bytes | None] = {}
            next_to_write = 0
            frames_done = 0

            # ---- producer loop: sequential camera update -> submit render ----
            for k in range(n_frames):
                if self._anim_cancel:
                    break

                ang = 2.0 * math.pi * (k / max(1, n_frames))
                Rm = _rot_matrix(axis, ang)

                eye = (Rm @ eye0).astype(float)
                upv = (Rm @ up0).astype(float)

                base_fig.update_layout(
                    scene=dict(
                        camera=dict(
                            eye=dict(
                                x=float(eye[0]),
                                y=float(eye[1]),
                                z=float(eye[2]),
                            ),
                            up=dict(
                                x=float(upv[0]),
                                y=float(upv[1]),
                                z=float(upv[2]),
                            ),
                            center=center0,
                        )
                    )
                )

                fig_json = base_fig.to_plotly_json()

                task = asyncio.create_task(_schedule_render(fig_json, k))
                in_flight[task] = k

                # throttle
                while len(in_flight) >= max_in_flight:
                    done_set, _ = await asyncio.wait(
                        in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done_set:
                        _ = in_flight.pop(t)
                        i0, png_bytes = t.result()
                        pending[i0] = png_bytes

                    # write in order
                    while next_to_write in pending:
                        png_bytes = pending.pop(next_to_write)

                        if png_bytes is None:
                            next_to_write += 1
                            continue
                        if fmt == "gif":
                            gif_png_frames.append(png_bytes)
                        else:
                            writer.append_data(
                                iio2.imread(io.BytesIO(png_bytes))
                            )
                        frames_done += 1

                        if (
                            preview_widget is not None
                            and preview_every
                            and ((next_to_write + 1) % int(preview_every) == 0)
                        ):
                            try:
                                preview_widget.value = png_bytes
                            except Exception:
                                pass

                        try:
                            if hasattr(self, "anim_progress"):
                                self.anim_progress.value = next_to_write + 1
                            if hasattr(self, "anim_status"):
                                self.anim_status.value = f"Rendering {next_to_write + 1}/{n_frames}…"
                        except Exception:
                            pass

                        next_to_write += 1

                await asyncio.sleep(0)

            # ---- drain ----
            while in_flight:
                done_set, _ = await asyncio.wait(
                    in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for t in done_set:
                    _ = in_flight.pop(t)
                    i0, png_bytes = t.result()
                    pending[i0] = png_bytes

                while next_to_write in pending:
                    png_bytes = pending.pop(next_to_write)

                    if png_bytes is None:
                        next_to_write += 1
                        continue
                    if fmt == "gif":
                        gif_png_frames.append(png_bytes)
                    else:
                        writer.append_data(iio2.imread(io.BytesIO(png_bytes)))
                    frames_done += 1

                    if (
                        preview_widget is not None
                        and preview_every
                        and ((next_to_write + 1) % int(preview_every) == 0)
                    ):
                        try:
                            preview_widget.value = png_bytes
                        except Exception:
                            pass

                    try:
                        if hasattr(self, "anim_progress"):
                            self.anim_progress.value = next_to_write + 1
                        if hasattr(self, "anim_status"):
                            self.anim_status.value = (
                                f"Rendering {next_to_write + 1}/{n_frames}…"
                            )
                    except Exception:
                        pass

                    next_to_write += 1

            # ---- finalize ----
            t_end = time.perf_counter()
            dt = t_end - t_start
            fps_eff = (frames_done / dt) if dt > 0 else 0.0

            try:
                if hasattr(self, "anim_status"):
                    self.anim_status.value = (
                        f"Saved: {out_path} | "
                        f"{frames_done} frames in {_fmt_seconds(dt)} "
                        f"({fps_eff:.2f} fps)"
                    )
                if hasattr(self, "anim_progress"):
                    self.anim_progress.bar_style = "success"
            except Exception:
                pass

        except Exception as e:
            tb = traceback.format_exc()
            try:
                if hasattr(self, "anim_debug"):
                    self.anim_debug.value = tb[-12000:]
            except Exception:
                pass
            try:
                if hasattr(self, "anim_progress"):
                    self.anim_progress.bar_style = "danger"
                if hasattr(self, "anim_status"):
                    msg = str(e).strip()
                    self.anim_status.value = f"Error: {type(e).__name__}: {msg}"

            except Exception:
                pass
            raise
        finally:
            gif_err = None
            try:
                if fmt == "gif" and gif_enabled:
                    # IMPORTANT: run ffmpeg work off the UI thread
                    await asyncio.to_thread(
                        self._png_bytes_list_to_gif_ffmpeg_palette,
                        gif_png_frames,
                        out_path,
                        int(fps),
                    )
                elif writer is not None:
                    # writer.close() can also block; keep it off-thread too
                    await asyncio.to_thread(writer.close)
            except Exception as e:
                gif_err = e

            try:
                if render_pool is not None:
                    # shutdown can block; off-thread avoids UI freeze
                    await asyncio.to_thread(render_pool.shutdown, True, False)
            except Exception:
                pass

            if gif_err is not None:
                try:
                    if hasattr(self, "anim_status"):
                        self.anim_status.value = (
                            f"GIF export failed: {type(gif_err).__name__}: {gif_err}"
                        )
                    if hasattr(self, "anim_progress"):
                        self.anim_progress.bar_style = "danger"
                except Exception:
                    pass
                raise gif_err

    # ============================================================
    # Animation export — Export settings / control
    # ============================================================
    def _reset_view_and_fit(self):
        # camera reset (optional)
        cam = getattr(self, "_initial_camera", None)
        if cam is None:
            cam = dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=0, z=1))

        # force plotly to recompute size from container
        with self.fig.batch_update():
            #  do not keep a fixed width if you want auto-fit
            self.fig.layout.width = None

            # keep autosize on
            self.fig.layout.autosize = True

            # “nudge” relayout: re-set height (or set to None then back)
            h = self.fig.layout.height
            self.fig.layout.height = None
            self.fig.layout.height = h

            # reset camera
            self.fig.layout.scene.camera = cam

            # optional: force UI refresh (changes the internal revision)
            self.fig.layout.uirevision = str(np.random.rand())

    # ------------------------------------------------------------------------------------
    # saving layers animation
    # ------------------------------------------------------------------------------------
    def _on_save_layer_clicked(self, btn):
        self._anim_cancel = False

        layer_key = self.anim_layer_key.value
        param = self.anim_layer_param.value

        if not layer_key or not param:
            self.anim_status.value = (
                "<b style='color:#ffb3b3'>Select a layer and parameter.</b>"
            )
            return

        fmt = (self.anim_format.value or "mp4").lower()
        name = (self.anim_name.value or "layer_anim").strip()
        if not name:
            name = "layer_anim"

        out_path = Path(os.getcwd()) / f"{name}.{fmt}"

        # values to animate
        lo, hi = map(float, self.anim_range.value)
        v0, v1 = float(lo), float(hi)
        n_frames = int(self.anim_points.value)
        fps = int(self.anim_fps.value)

        if n_frames < 2:
            self.anim_status.value = (
                "<b style='color:#ffb3b3'>Points must be ≥ 2.</b>"
            )
            return

        # optional: clamp to geometric bounds (important for plane offset)
        vmin, vmax = self._anim_param_bounds(layer_key, param)
        if vmin is not None and vmax is not None:
            a, b = sorted((v0, v1))
            a = max(a, float(vmin))
            b = min(b, float(vmax))
            if b <= a:
                self.anim_status.value = "<b style='color:#ffb3b3'>Invalid range after clamping.</b>"
                return
            v0, v1 = a, b

        # reset progress + stop flag
        self.anim_progress.value = 0
        self.anim_progress.max = n_frames
        self.anim_status.value = (
            f"Exporting layer animation → <code>{out_path.name}</code>"
        )
        # run async if your environment supports it; else run sync fallback
        try:
            coro = self._save_layer_animation_async(
                out_path=str(out_path),
                fmt=fmt,
                layer_key=layer_key,
                param=param,
                v0=v0,
                v1=v1,
                n_frames=n_frames,
                fps=fps,
                max_workers=getattr(self, "render_workers", None),
                max_in_flight=getattr(self, "render_in_flight", None),
            )

            # if an event loop is running (notebook), schedule task
            try:
                loop = asyncio.get_running_loop()
                self._export_task = loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)

        except Exception as e:
            self.anim_status.value = (
                f"<b style='color:#ffb3b3'>Export failed:</b> {e}"
            )

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # ============================================================
    # Animation layers — layers
    # ============================================================
    def _to_png(self, fig, width: int, height: int, scale: int) -> bytes:
        mode = getattr(self, "rendering_mode", "safe")

        if mode == "fast":
            # true parallelism in threads: each thread uses its own scope
            return pio.to_image(
                fig,
                format="png",
                width=int(width),
                height=int(height),
                scale=int(scale),
            )

        if mode == "safe":
            # serialize pio usage
            with self._to_image_lock:
                return pio.to_image(
                    fig,
                    format="png",
                    width=int(width),
                    height=int(height),
                    scale=int(scale),
                )

        # optional: keep "process" separate (see Option 4)
        with self._to_image_lock:
            return pio.to_image(
                fig,
                format="png",
                width=int(width),
                height=int(height),
                scale=int(scale),
            )

    async def _save_layer_animation_async(
        self,
        out_path: str,
        fmt: str,
        layer_key: str,
        param: str,
        v0: float,
        v1: float,
        n_frames: int,
        fps: int,
        width: int = 900,
        height: int = 900,
        scale: int = 2,
        preview_every: int = 1,
        **kwargs,
    ):
        """
        Process-parallel renderer for layer animation export.

        Key change vs your current version:
        - Geometry/materialization stays SEQUENTIAL in the main process (so we never mutate `spec` concurrently).
        - Rendering (fig_json -> png bytes) runs in a ProcessPool with bounded in-flight tasks.
        - Frames are written in order as results complete (streaming; no giant `frames` list).
        """

        def _fmt_seconds(sec: float) -> str:
            if sec < 60:
                return f"{sec:.2f} s"
            m, s = divmod(sec, 60.0)
            if m < 60:
                return f"{int(m)} min {s:04.1f} s"
            h, m = divmod(m, 60.0)
            return f"{int(h)} h {int(m)} min {s:04.1f} s"

        # -----------------------------
        # module-level worker (defined inside for copy/paste convenience)
        # NOTE: if you hit pickling issues on Windows, move this helper to top-level in the file.
        # -----------------------------

        t_start = time.perf_counter()

        self._anim_cancel = False
        fmt = (fmt or "mp4").lower().strip()
        param = (param or "").lower().strip()

        # ---- UI init ----
        try:
            if hasattr(self, "anim_progress"):
                self.anim_progress.value = 0
                self.anim_progress.min = 0
                self.anim_progress.max = int(n_frames)
                self.anim_progress.bar_style = "info"
        except Exception:
            pass
        try:
            if hasattr(self, "anim_status"):
                self.anim_status.value = "Preparing export…"
        except Exception:
            pass

        # ---- offscreen clone (we mutate base_fig sequentially) ----
        base_fig = go.Figure(self.fig.to_plotly_json())

        # locate trace idx
        trace_idx = None
        for i, tr in enumerate(base_fig.data):
            if getattr(tr, "name", None) == layer_key:
                trace_idx = i
                break
        if trace_idx is None:
            for i, tr in enumerate(base_fig.data):
                if getattr(tr, "legendgroup", None) == layer_key:
                    trace_idx = i
                    break
        if trace_idx is None:
            raise RuntimeError(
                f"Could not locate a trace for layer '{layer_key}' in the current figure. "
                "Ensure the layer is visible before exporting."
            )

        # layer spec (READ ONLY during export, but we restore afterwards)
        layers = self._get_layers_container()
        if layer_key not in layers:
            raise KeyError(f"Unknown layer_key: {layer_key}")
        spec = layers[layer_key]

        # snapshot initial state (restore in finally)
        _initial_spec = {}
        for k in (
            "opacity",
            "pos",
            "offset",
            "origin",
            "data2d",
            "data2d_base",
            "X",
            "Y",
            "Z",
        ):
            if k in spec:
                _initial_spec[k] = spec[k]

        layer_type = (
            spec.get("type") or spec.get("layer_type") or "raw"
        ).lower()
        w = getattr(self, "_layer_widgets", {}).get(layer_key, {}) or {}
        nan_policy = str(
            getattr(w.get("nan_policy", None), "value", "none") or "none"
        ).lower()

        # ---- helpers (same as your version) ----
        def _update_surface_from_spec():
            wloc = getattr(self, "_layer_widgets", {}).get(layer_key, {}) or {}
            policy = str(
                getattr(wloc.get("nan_policy", None), "value", "none")
                or "none"
            ).lower()

            opacity = float(
                getattr(
                    wloc.get("op", None), "value", spec.get("opacity", 1.0)
                )
            )
            colorscale = self.get_colorscale(
                getattr(wloc.get("cmap", None), "value", "Viridis")
            )
            color_sel = str(
                getattr(wloc.get("color_by", None), "value", "__self__")
            )

            base_raw = np.asarray(
                spec.get("data2d_base", spec["data2d"]), float
            )

            data2d = (
                self._apply_nan_policy(base_raw, policy)
                if "data2d_base" in spec
                else self._apply_nan_policy(
                    np.asarray(spec["data2d"], float), policy
                )
            )

            Xc, Yc, Zc = spec["X"], spec["Y"], spec["Z"]

            if color_sel == "__constant__":
                intensity2d = np.zeros_like(data2d, dtype=float)
            elif color_sel in ("__x__", "__y__", "__z__"):
                intensity2d = {"__x__": Xc, "__y__": Yc, "__z__": Zc}[
                    color_sel
                ]
            elif color_sel == "__self__":
                intensity2d = np.asarray(data2d, dtype=float)
            else:
                if color_sel not in self.dict_data:
                    intensity2d = np.asarray(data2d, dtype=float)
                else:
                    if spec["type"] == "plane":
                        intensity2d = self._sample_raw_on_surface(
                            color_sel, Xc, Yc, Zc, nan_policy=policy
                        )
                    else:
                        intensity2d = self._slice_mean_2d(
                            self.dict_data[color_sel],
                            spec["axis_phys"],
                            int(spec["pos"]),
                            int(spec["thickness"]),
                            nan_policy=policy,
                        )

            intensity2d = self._apply_nan_policy(
                np.asarray(intensity2d, float), policy
            )

            if policy == "none":
                mask_invalid = ~np.isfinite(base_raw)
                Xplot = np.array(Xc, float, copy=True)
                Yplot = np.array(Yc, float, copy=True)
                Zplot = np.array(Zc, float, copy=True)
                Cplot = np.array(intensity2d, float, copy=True)
                Xplot[mask_invalid] = np.nan
                Yplot[mask_invalid] = np.nan
                Zplot[mask_invalid] = np.nan
                Cplot[mask_invalid] = np.nan
            else:
                Xplot, Yplot, Zplot, Cplot = Xc, Yc, Zc, intensity2d

            Xp, Yp, Zp = self._map_xyz_surface_to_plot(Xplot, Yplot, Zplot)

            auto_range = bool(
                getattr(wloc.get("auto_range", None), "value", True)
            )
            rs = wloc.get("range_slider", None)

            data_min = (
                float(np.nanmin(Cplot))
                if np.isfinite(np.nanmin(Cplot))
                else 0.0
            )
            data_max = (
                float(np.nanmax(Cplot))
                if np.isfinite(np.nanmax(Cplot))
                else 1.0
            )
            if data_max <= data_min:
                data_max = data_min + 1e-12

            if (not auto_range) and (rs is not None):
                rmin, rmax = rs.value
                cmin, cmax = float(rmin), float(rmax)
                if cmax <= cmin:
                    cmax = cmin + 1e-12
            else:
                cmin, cmax = data_min, data_max

            tr = base_fig.data[trace_idx]
            tr.update(
                x=Xp,
                y=Yp,
                z=Zp,
                surfacecolor=Cplot,
                colorscale=colorscale,
                opacity=opacity,
                cmin=cmin,
                cmax=cmax,
                cauto=False,
                connectgaps=False,
            )

        def _remat_slice(pos_i: int):
            vol3d = np.asarray(spec["vol3d"], float)
            ax_phys = str(spec["axis_phys"]).lower()
            thick = int(spec.get("thickness", 0))

            data2d_base, Xc, Yc, Zc = self._materialize_slice_rgi(
                vol3d, ax_phys, int(pos_i), int(thick), nan_policy="none"
            )
            data2d_pol, _, _, _ = self._materialize_slice_rgi(
                vol3d, ax_phys, int(pos_i), int(thick), nan_policy=nan_policy
            )

            spec["pos"] = int(pos_i)
            spec["data2d_base"] = data2d_base
            spec["data2d"] = data2d_pol
            spec["X"], spec["Y"], spec["Z"] = Xc, Yc, Zc
            _update_surface_from_spec()

        def _remat_plane(offset_val: float):
            vol3d = np.asarray(spec["vol3d"], float)
            n_phys = np.asarray(spec["normal"], float)
            o0_phys = np.asarray(spec["origin0"], float)

            offset, _, _, n_hat = self._clamp_offset_in_bounds(
                o0_phys, n_phys, float(offset_val)
            )
            o_eff = o0_phys + np.asarray(n_hat, float) * float(offset)

            thick = float(spec.get("thickness", 0.0))
            extent = float(spec.get("extent", 1.0))

            data2d_base, Xc, Yc, Zc = self._materialize_plane(
                vol3d, n_phys, o_eff, thick, extent, nan_policy="none"
            )
            data2d_pol, _, _, _ = self._materialize_plane(
                vol3d, n_phys, o_eff, thick, extent, nan_policy=nan_policy
            )

            spec["offset"] = float(offset)
            spec["origin"] = o_eff
            spec["data2d_base"] = data2d_base
            spec["data2d"] = data2d_pol
            spec["X"], spec["Y"], spec["Z"] = Xc, Yc, Zc
            _update_surface_from_spec()

        def _remat_clip(offset_val: float):
            # ---- validate source ----
            src = spec.get("source", None)
            if not src or src not in self.dict_data:
                base_fig.data[trace_idx].update(
                    x=[], y=[], z=[], i=[], j=[], k=[], intensity=[]
                )
                return

            # ---- compute effective origin from origin0 + n_hat*offset (with clamping) ----
            o0 = np.asarray(
                spec.get("origin0", self._center_origin_xyz()), float
            )
            n = np.asarray(spec.get("normal", [0, 0, 1]), float)
            nn = float(np.linalg.norm(n))
            n_hat = n / nn if nn > 0 else np.array([0.0, 0.0, 1.0], float)

            offset, *_ = self._clamp_offset_in_bounds(o0, n, float(offset_val))
            origin = o0 + n_hat * float(offset)
            spec["offset"] = float(offset)
            spec["origin"] = origin

            side = str(spec.get("side", "up")).lower()

            # ---- base volume + clipping ----
            base = np.asarray(self.dict_data[src], float)
            base = self._apply_nan_policy(base, nan_policy)

            keep = self._clip_volume_halfspace(base, n, origin, side)

            vmin = float(np.nanmin(base))
            vmax = float(np.nanmax(base))
            span = vmax - vmin
            pad = 10.0 * (span if np.isfinite(span) and span > 0 else 1.0)

            vol = base.copy()
            vol[~keep] = vmin - pad

            # ---- iso threshold ----
            iso = float(getattr(w.get("thr", None), "value", 0.0))

            vmin2 = float(np.nanmin(vol))
            vmax2 = float(np.nanmax(vol))
            if (
                (not np.isfinite(vmin2))
                or (not np.isfinite(vmax2))
                or not (vmin2 < iso < vmax2)
            ):
                base_fig.data[trace_idx].update(
                    x=[], y=[], z=[], i=[], j=[], k=[], intensity=[]
                )
                return

            # ---- surface extraction ----
            verts, faces, _, _ = marching_cubes(vol, level=iso, step_size=1)

            # verts are in zyx index coordinates; convert to physical coords for plotting
            verts_scaled = verts * self.voxel_size
            verts_plot = self._map_zyx_to_plot(verts_scaled)

            # ---- color_by (respect user choice) ----
            color_sel = str(
                getattr(w.get("color_by", None), "value", "__self__")
                or "__self__"
            )

            if color_sel == "__self__":
                # sample the geometry source at isosurface vertices (zyx index coords)
                vals = self._rgi[src](verts)

            elif color_sel in self.dict_data:
                # sample the user-selected dataset at isosurface vertices (zyx index coords)
                vals = self._rgi[color_sel](verts)

            elif color_sel == "__x__":
                vals = verts_plot[:, 0]
            elif color_sel == "__y__":
                vals = verts_plot[:, 1]
            elif color_sel == "__z__":
                vals = verts_plot[:, 2]
            elif color_sel == "__constant__":
                vals = np.zeros((verts_plot.shape[0],), dtype=float)
            else:
                # safe fallback
                vals = self._rgi[src](verts)

            vals = np.asarray(vals, float)
            vals = self._apply_nan_policy(vals, nan_policy)

            # ---- update trace ----
            tr = base_fig.data[trace_idx]
            tr.update(
                x=verts_plot[:, 0],
                y=verts_plot[:, 1],
                z=verts_plot[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=vals,
            )

        # ---- values ----
        values = np.linspace(float(v0), float(v1), int(n_frames)).astype(float)

        preview_widget = getattr(self, "anim_preview", None)

        # ---- concurrency knobs ----
        max_workers = int(getattr(self, "render_workers", 4))
        max_in_flight = int(getattr(self, "render_in_flight", max_workers))
        max_workers = max(1, max_workers)
        max_in_flight = max(1, max_in_flight)

        # fast/process => process pool rendering
        mode = getattr(self, "rendering_mode", "safe")
        use_process = mode in ("fast", "process")

        render_pool = (
            concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            if use_process
            else None
        )
        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(max_in_flight)

        writer = None
        gif_enabled = False
        try:
            if hasattr(self, "anim_status"):
                self.anim_status.value = "Rendering…"

            if fmt == "mp4":
                try:
                    import imageio_ffmpeg  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "MP4 export requires imageio-ffmpeg. Install with: pip install imageio-ffmpeg"
                    ) from e
                writer = iio2.get_writer(
                    out_path, fps=int(fps), codec="libx264"
                )
            elif fmt == "gif":
                gif_png_frames = []
                self._require_ffmpeg()
                if_enabled = True
                writer = None
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            async def _schedule_render(fig_json: dict, idx: int):
                async with sem:
                    if self._anim_cancel:
                        return idx, None
                    if use_process:
                        png = await loop.run_in_executor(
                            render_pool,
                            _render_fig_json_to_png,
                            fig_json,
                            int(width),
                            int(height),
                            int(scale),
                        )
                    else:
                        # safe fallback: serialize with lock in main process
                        png = await asyncio.to_thread(
                            self._to_png,
                            go.Figure(fig_json),
                            int(width),
                            int(height),
                            int(scale),
                        )
                    return idx, png

            in_flight = {}
            pending = {}
            next_to_write = 0
            frames_done = 0
            # producer loop: sequentially update geometry -> fig_json, then submit render
            for k, val in enumerate(values):
                if self._anim_cancel:
                    break

                # update for this frame (SEQUENTIAL, touches spec/base_fig)
                if param == "opacity":
                    base_fig.data[trace_idx].update(opacity=float(val))
                    spec["opacity"] = float(val)
                elif param == "pos":
                    if layer_type != "slice":
                        raise ValueError(
                            f"param='pos' only valid for slice layers (got {layer_type})"
                        )
                    _remat_slice(int(round(float(val))))
                elif param == "offset":
                    if layer_type == "plane":
                        _remat_plane(float(val))
                    elif layer_type == "clip":
                        _remat_clip(float(val))
                    else:
                        raise ValueError(
                            f"param='offset' only valid for plane/clip layers (got {layer_type})"
                        )
                else:
                    raise ValueError(f"Unsupported param: {param}")

                # snapshot JSON for render workers (cheap)
                fig_json = base_fig.to_plotly_json()

                # submit render task (bounded by max_in_flight)
                task = asyncio.create_task(_schedule_render(fig_json, k))
                in_flight[task] = k

                # if too many in flight, wait for at least one completion
                while len(in_flight) >= max_in_flight:
                    done_set, _ = await asyncio.wait(
                        in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done_set:
                        _ = in_flight.pop(t)
                        i0, png_bytes = t.result()
                        pending[i0] = png_bytes

                    # write in order
                    while next_to_write in pending:
                        png_bytes = pending.pop(next_to_write)
                        if png_bytes is None:
                            # empty frame: skip writing, but DO advance the sequence
                            next_to_write += 1
                            continue
                        if fmt == "gif":
                            gif_png_frames.append(png_bytes)
                        else:
                            writer.append_data(
                                iio2.imread(io.BytesIO(png_bytes))
                            )
                        frames_done += 1

                        # preview
                        if (
                            preview_widget is not None
                            and preview_every
                            and ((next_to_write + 1) % int(preview_every) == 0)
                        ):
                            try:
                                preview_widget.value = png_bytes
                            except Exception:
                                pass

                        # progress
                        try:
                            if hasattr(self, "anim_progress"):
                                self.anim_progress.value = next_to_write + 1
                        except Exception:
                            pass
                        try:
                            if hasattr(self, "anim_status"):
                                self.anim_status.value = f"Rendering {next_to_write + 1}/{int(n_frames)}…"
                        except Exception:
                            pass

                        next_to_write += 1

                await asyncio.sleep(0)

            # drain remaining tasks
            while in_flight:
                done_set, _ = await asyncio.wait(
                    in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for t in done_set:
                    _ = in_flight.pop(t)
                    i0, png_bytes = t.result()
                    pending[i0] = png_bytes

                while next_to_write in pending:
                    png_bytes = pending.pop(next_to_write)
                    if png_bytes is None:
                        # empty frame: skip writing, but DO advance the sequence
                        next_to_write += 1
                        continue
                    if fmt == "gif":
                        gif_png_frames.append(png_bytes)
                    else:
                        writer.append_data(iio2.imread(io.BytesIO(png_bytes)))

                    if (
                        preview_widget is not None
                        and preview_every
                        and ((next_to_write + 1) % int(preview_every) == 0)
                    ):
                        try:
                            preview_widget.value = png_bytes
                        except Exception:
                            pass

                    try:
                        if hasattr(self, "anim_progress"):
                            self.anim_progress.value = next_to_write + 1
                    except Exception:
                        pass
                    try:
                        if hasattr(self, "anim_status"):
                            self.anim_status.value = f"Rendering {next_to_write + 1}/{int(n_frames)}…"
                    except Exception:
                        pass
                    frames_done += 1
                    next_to_write += 1

            # finalize
            try:
                t_end = time.perf_counter()
                dt = t_end - t_start
                fps_eff = next_to_write / dt if dt > 0 else 0.0
                if hasattr(self, "anim_status"):
                    self.anim_status.value = (
                        f"Saved: {out_path} | "
                        f"{next_to_write} frames in {_fmt_seconds(dt)} "
                        f"({fps_eff:.2f} fps)"
                    )
                if hasattr(self, "anim_progress"):
                    self.anim_progress.bar_style = "success"
            except Exception:
                pass

        except Exception as e:
            tb = traceback.format_exc()
            try:
                if hasattr(self, "anim_debug"):
                    max_chars = 12000
                    self.anim_debug.value = (
                        tb[-max_chars:] if len(tb) > max_chars else tb
                    )
            except Exception:
                pass
            try:
                if hasattr(self, "anim_progress"):
                    self.anim_progress.bar_style = "danger"
                if hasattr(self, "anim_status"):
                    self.anim_status.value = f"Error: {type(e).__name__}"
            except Exception:
                pass
            raise

        finally:
            gif_err = None
            try:
                if fmt == "gif" and gif_enabled:
                    # IMPORTANT: run ffmpeg work off the UI thread
                    await asyncio.to_thread(
                        self._png_bytes_list_to_gif_ffmpeg_palette,
                        gif_png_frames,
                        out_path,
                        int(fps),
                    )
                elif writer is not None:
                    # writer.close() can also block; keep it off-thread too
                    await asyncio.to_thread(writer.close)
            except Exception as e:
                gif_err = e

            try:
                if render_pool is not None:
                    # shutdown can block; off-thread avoids UI freeze
                    await asyncio.to_thread(render_pool.shutdown, True, False)
            except Exception:
                pass

            if gif_err is not None:
                try:
                    if hasattr(self, "anim_status"):
                        self.anim_status.value = (
                            f"GIF export failed: {type(gif_err).__name__}: {gif_err}"
                        )
                    if hasattr(self, "anim_progress"):
                        self.anim_progress.bar_style = "danger"
                except Exception:
                    pass
                raise gif_err

            # cleanup spec keys created during export
            try:
                for k in (
                    "opacity",
                    "pos",
                    "offset",
                    "origin",
                    "data2d",
                    "data2d_base",
                    "X",
                    "Y",
                    "Z",
                ):
                    if k in spec and k not in _initial_spec:
                        del spec[k]
            except Exception:
                pass

            # restore initial spec values
            try:
                for k, v in _initial_spec.items():
                    spec[k] = v
            except Exception:
                pass

            # restore UI slider values
            try:
                if (
                    param == "pos"
                    and "pos" in _initial_spec
                    and w.get("pos", None) is not None
                ):
                    w["pos"].value = int(_initial_spec["pos"])
                if (
                    param == "offset"
                    and "offset" in _initial_spec
                    and w.get("offset", None) is not None
                ):
                    w["offset"].value = float(_initial_spec["offset"])
                if (
                    param == "opacity"
                    and "opacity" in _initial_spec
                    and w.get("op", None) is not None
                ):
                    w["op"].value = float(_initial_spec["opacity"])
            except Exception:
                pass

            # re-render live fig from restored spec
            try:
                self._update_all_traces()
            except Exception:
                pass

            # clear preview
            try:
                if preview_widget is not None:
                    preview_widget.value = b""
            except Exception:
                pass
