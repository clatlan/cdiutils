"""
MultiVolumeViewer — Interactive 3D volume visualization widget (ipywidgets + Plotly).

This module provides an interactive Jupyter widget for exploring and comparing
multiple 3D scalar fields (e.g. amplitude, phase, density, masks) defined on the
same voxel grid. The viewer is designed as a lightweight, ParaView-like tool
directly usable inside notebooks, with support for slicing, arbitrary planes,
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

Typical use cases
-----------------
• Visualization of BCDI reconstructions (amplitude, phase, strain)
• Inspection of defect structures via slices, planes, and clipped volumes
• Rapid, notebook-based exploration without exporting to external viewers

Dependencies
------------
- numpy
- scipy (RegularGridInterpolator)
- scikit-image (marching_cubes)
- plotly
- ipywidgets
- matplotlib (for colormaps fallback)

The widget is intended for interactive use inside Jupyter environments.
"""

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

try:
    from .volume import _extract_isosurface_with_values, colorcet_to_plotly

    HAS_VOLUME_UTILS = True
except Exception:
    HAS_VOLUME_UTILS = False
    import matplotlib.pyplot as plt

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


CBAR_X0 = 1.02  # start just outside the scene
CBAR_DX = 0.2  # horizontal spacing per colorbar


class MultiVolumeViewer(widgets.Box):
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

    @staticmethod
    def get_all_supported_cmaps():
        import matplotlib.pyplot as plt

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

        return FIXED_COLORS + tuple(cmaps)

    cmap_options = get_all_supported_cmaps.__func__()

    # ----------------------------
    # init setup
    # ----------------------------
    def __init__(self, dict_data=None, voxel_size=(1, 1, 1), figsize=(6, 6)):
        super().__init__()

        # ----------------------------
        # Layout constants
        # ----------------------------
        self._DESC_W = "160px"
        self._common_style = {"description_width": self._DESC_W}

        self.voxel_size = np.array(voxel_size, dtype=float)

        # ----------------------------
        # Figure
        # ----------------------------
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
                    projection=dict(type="perspective"),
                ),
            ),
            autosize=True,
            height=figsize[1] * 90,
            width=figsize[0] * 90,
            dragmode="orbit",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        # ----------------------------
        # Internal data
        # ----------------------------
        self.dict_data = {}
        self._rgi = {}
        self._layer_widgets = {}
        self._visible_cb = {}
        self._layers = {}  # name -> spec dict. raw: {"type":"raw","source":None}; derived later

        # ----------------------------
        # Global controls
        # ----------------------------
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

        self.theme_toggle = widgets.ToggleButton(
            value=True,
            description="Dark Theme",
        )
        self._on_theme(None)  # apply initial theme state
        self.rotate_toggle = widgets.ToggleButton(
            value=False,
            description="Rotate",
        )

        toggles_row = widgets.HBox(
            [self.theme_toggle, self.rotate_toggle],
            layout=widgets.Layout(
                width="95%",
                justify_content="flex-start",
                align_items="center",
            ),
        )
        self.edit_key = widgets.Dropdown(
            options=[],
            description="Edit:",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )

        # 1) define the atomic widgets FIRST
        self.add_layer_kind = widgets.Dropdown(
            options=[
                ("— choose —", None),
                ("Slice layer", "slice"),
                ("Plane layer", "plane"),
                ("Clip layer", "clip"),  # NEW
            ],
            value=None,
            description="Add:",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        self.add_layer_name = widgets.Text(
            value="",
            description="Name:",
            placeholder="e.g. phase_zmid",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        self.add_slice_axis = widgets.Dropdown(
            options=[("x", "x"), ("y", "y"), ("z", "z")],
            value="z",
            description="Axis:",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        self.add_slice_pos = widgets.IntSlider(
            value=0,
            min=0,
            max=1,
            step=1,
            description="Pos:",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        self.add_slice_thickness = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description="Thick:",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )

        # --- Plane controls (creation) ---
        self.add_plane_nx = widgets.FloatText(
            value=0.0,
            description="n x:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_ny = widgets.FloatText(
            value=0.0,
            description="n y:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_nz = widgets.FloatText(
            value=1.0,
            description="n z:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_ox = widgets.FloatText(
            value=0.0,
            description="o x:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_oy = widgets.FloatText(
            value=0.0,
            description="o y:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_oz = widgets.FloatText(
            value=0.0,
            description="o z:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_center_btn = widgets.Button(
            description="Center origin", layout=widgets.Layout(width="auto")
        )
        self.add_plane_thickness = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=10.0,
            step=0.1,
            description="Thick:",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_plane_extent = widgets.FloatSlider(
            value=50.0,
            min=1.0,
            max=500.0,
            step=1.0,
            description="Extent:",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_layer_btn = widgets.Button(description="Create layer")

        # --- clip controls (creation) ---
        self.add_clip_side = widgets.Dropdown(
            options=[("Up (+n)", "up"), ("Down (-n)", "down")],
            value="up",
            description="Side:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )
        self.add_clip_source = widgets.Dropdown(
            options=[],
            description="Src:",
            style=self._common_style,
            layout=widgets.Layout(width="100%"),
        )

        # 2) wire callbacks
        self.theme_toggle.observe(self._on_theme, names="value")
        self.rotate_toggle.observe(self._on_rotate_toggle, names="value")
        self.edit_key.observe(self._on_edit_key_changed, names="value")

        self.add_layer_kind.observe(self._sync_create_layer_ui, names="value")
        self.add_slice_axis.observe(
            self._sync_add_slice_pos_range, names="value"
        )
        self.add_plane_center_btn.on_click(self._on_center_origin_clicked)
        self.add_layer_btn.on_click(self._on_create_layer_clicked)

        # Ensure consistent label width
        self._CREATE_DESC_W = "60px"
        _create_style = {"description_width": self._CREATE_DESC_W}

        for w in (
            self.add_layer_kind,
            self.add_layer_name,
            self.add_slice_axis,
            self.add_slice_pos,
            self.add_slice_thickness,
            self.add_plane_nx,
            self.add_plane_ny,
            self.add_plane_nz,
            self.add_plane_ox,
            self.add_plane_oy,
            self.add_plane_oz,
            self.add_clip_source,
            self.add_clip_side,
        ):
            w.style = _create_style
            w.layout = widgets.Layout(width="95%")  # IMPORTANT inside grid

        left_col = widgets.VBox(
            [self.add_layer_kind, self.add_layer_name, self.add_slice_axis],
            layout=widgets.Layout(width="100%"),
        )

        right_col = widgets.VBox(
            [self.add_slice_pos, self.add_slice_thickness],
            layout=widgets.Layout(width="100%"),
        )

        grid = widgets.HBox(
            [left_col, right_col],
            layout=widgets.Layout(
                display="flex",
                width="100%",
                gap="16px",
                align_items="flex-start",
            ),
        )

        # enforce equal columns (must be set on the HBox children)
        left_col.layout = widgets.Layout(
            flex="1 1 0%", min_width="0px", align_items="stretch"
        )
        right_col.layout = widgets.Layout(
            flex="1 1 0%", min_width="0px", align_items="stretch"
        )
        grid.layout = widgets.Layout(
            display="flex",
            width="100%",
            gap="16px",
            align_items="flex-start",
        )
        # centered button on its own line
        self.add_layer_btn.layout = widgets.Layout(width="auto")

        self.derived_box = widgets.VBox(
            [], layout=widgets.Layout(width="95%", align_items="stretch")
        )

        self.derived_box.layout = widgets.Layout(
            width="95%", align_items="stretch"
        )

        self._sync_create_layer_ui()

        # Container for per-layer controls
        self.layers_box = widgets.VBox([])

        # ----------------------------
        # Final layout
        # ----------------------------
        # self.layout = widgets.Layout(align_items="flex-start")
        VIEW_H = f"{int(figsize[1] * 96)}px"
        self.layout = widgets.Layout(
            display="flex",
            flex_flow="row",
            align_items="stretch",
            width="100%",
            height=VIEW_H,
        )
        # ----------------------------
        # Figure container (flex-grow)
        # ----------------------------
        fig_box = widgets.Box(
            [self.fig],
            layout=widgets.Layout(
                width="100%",
                flex="1 1 0%",  # take remaining space
                min_width="0px",  # IMPORTANT: allow shrinking instead of pushing/overflowing
                height="100%",
                overflow="hidden",  # optional, prevents horizontal spill
                padding="28px 28px 28px 28px",  # reserve scrollbar gutter
            ),
        )
        # Important: ipywidgets needs this to actually expand to full available width
        self.add_class("widget-stretch")
        # ----------------------------
        # Right control panel
        # ----------------------------
        panel_inner = widgets.VBox(
            [
                widgets.HTML("<b>Visible layers</b>"),
                self.visible_keys_box,
                toggles_row,
                widgets.HTML("<b>Layer controls</b>"),
                self.edit_key,
                self.layers_box,
                self.derived_box,
            ],
            layout=widgets.Layout(padding="0"),
        )

        right_panel = widgets.VBox(
            [panel_inner],
            layout=widgets.Layout(
                flex="0 0 35%",
                height="100%",
                overflow_y="auto",
                overflow_x="hidden",
                padding="20px",
                margin="0px",
                border="3px solid red",
                box_sizing="border-box",
            ),
        )

        # inner never scrolls
        panel_inner.layout = widgets.Layout(width="100%", overflow="visible")
        css = widgets.HTML("""
        <style>
        /* Panel background */
        .mv-right-panel {
        background: #0b1f3a !important;
        }
        .mv-right-panel-inner {
        background: transparent !important;
        }

        /* Text color */
        .mv-right-panel, .mv-right-panel * {
        color: #e6eefc !important;
        }

        /* Inputs */
        .mv-right-panel select,
        .mv-right-panel input {
        background: #1b2431 !important;
        color: #e6eefc !important;
        border: 1px solid #2f3b4d !important;
        }

        /* --- FIX: ToggleButtons (they are <button>) --- */
        .mv-right-panel button {
        background: #1b2431 !important;
        color: #e6eefc !important;
        border: 1px solid #2f3b4d !important;
        }

        /* Hover */
        .mv-right-panel button:hover {
        background: #223046 !important;
        }

        /* “Pressed” / active toggle (class can differ by frontend, cover both) */
        .mv-right-panel button:active,
        .mv-right-panel button.mod-active,
        .mv-right-panel .jupyter-button.mod-active,
        .mv-right-panel .jupyter-button:active {
        background: #2a3a55 !important;
        border-color: #3a4c6a !important;
        }
        </style>
        """)

        right_panel.add_class("mv-right-panel")
        panel_inner.add_class("mv-right-panel-inner")

        right_panel.children = (css, panel_inner)

        self.children = [fig_box, right_panel]

        # ----------------------------
        # Rotation state
        # ----------------------------
        self._rotation_angle = 0
        self._rotation_task = None

        # ----------------------------
        # Load data if provided
        # ----------------------------
        if dict_data is not None:
            self.set_data(dict_data)

    # ----------------------------
    # Data setup
    # ----------------------------
    def set_data(self, dict_data: dict):
        """
        Register/validate layers and build interpolators + UI controls.

        Contract (strict):
        - dict_data maps layer name -> 3D numpy array
        - all arrays must be real-valued (float/int/bool), same shape
        - complex arrays are rejected (users must pass amp/phase explicitly)
        """
        if dict_data is None or not isinstance(dict_data, dict):
            raise TypeError("dict_data must be a dict {name: np.ndarray}.")

        keys_raw = list(dict_data.keys())
        if not keys_raw:
            raise ValueError("dict_data is empty.")

        # ---- validate arrays ----
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
                    "This viewer expects real-valued scalar fields "
                    "(e.g., amplitude, phase (radians), density, mask). "
                    "Convert explicitly before passing."
                )
            if v.dtype.kind not in ("b", "i", "u", "f"):
                raise TypeError(
                    f"Layer '{k}' must be a real numeric array "
                    f"(bool/int/float). Got dtype={v.dtype}."
                )

            if shape0 is None:
                shape0 = v.shape
            elif v.shape != shape0:
                raise ValueError(
                    f"Shape mismatch: '{k}' has {v.shape}, expected {shape0}."
                )
        self._shape0 = shape0  # (nz, ny, nx) common shape for all raw layers
        # ---- store raw arrays ----
        self.dict_data = dict_data

        # ---- initialize layer registry (raw only for Step A) ----
        # IMPORTANT: preserve ordering from dict_data insertion order
        self._layers = {k: {"type": "raw", "source": None} for k in keys_raw}

        raws = [k for k, s in self._layers.items() if s.get("type") == "raw"]
        self.add_clip_source.options = raws
        self.add_clip_source.value = raws[0] if raws else None

        # ---- build interpolators (always real) ----
        nz, ny, nx = shape0
        grid_z = np.arange(nz, dtype=float)
        grid_y = np.arange(ny, dtype=float)
        grid_x = np.arange(nx, dtype=float)

        self._rgi = {}
        for k, arr in dict_data.items():
            arr_f = np.asarray(arr, dtype=float)
            self._rgi[k] = RegularGridInterpolator(
                (grid_z, grid_y, grid_x),
                arr_f,
                bounds_error=False,
                fill_value=0.0,
            )

        # ---- build per-layer controls (RAW layers only for Step A) ----
        self._layer_widgets = {}
        self._rebuild_layer_widgets()

        # ---- visible keys checkboxes now follow self._layers ----
        self._visible_cb = {}
        cbs = []
        for i, k in enumerate(self._layers.keys()):
            cb = widgets.Checkbox(value=(i == 0), description=k, indent=False)
            cb.observe(self._on_visible_changed, names="value")
            self._visible_cb[k] = cb
            cbs.append(cb)

        self.visible_keys_box.children = cbs
        for cb in self.visible_keys_box.children:
            cb.layout.flex = "0 0 140px"
            cb.layout.width = "140px"

        # ---- edit key dropdown now follows self._layers ----
        layer_names = list(self._layers.keys())
        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            self.edit_key.options = layer_names
            self.edit_key.value = layer_names[0]
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

        # ---- render ----
        self._refresh_layer_lists()
        self._sync_add_slice_pos_range()
        self._sync_layer_panels_visibility()
        self._on_center_origin_clicked(None)
        self._sync_plane_slider_ranges()
        self._update_all_traces()

    # ----------------------------
    # UI building
    # ----------------------------
    def _rebuild_layer_widgets(self):
        for k in self._layers.keys():
            self._build_widgets_for_layer(k)

    def _sync_layer_panels_visibility(self):
        k = self.edit_key.value
        if not k or k not in self._layer_widgets:
            self.layers_box.children = []
            return

        spec = self._layers.get(k, {"type": "raw"})
        w = self._layer_widgets[k]
        policy = self._layer_widgets[k]["nan_policy"].value

        children = []
        if spec["type"] == "raw":
            children.append(w["thr"])
        elif spec["type"] == "slice":
            # editable slice params (update spec + redraw)
            ax_dd = widgets.Dropdown(
                options=[("x", "x"), ("y", "y"), ("z", "z")],
                value=spec["axis"],
                description="axis:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            pos_sl = widgets.IntSlider(
                value=int(spec["pos"]),
                min=0,
                max=1,
                step=1,
                description="pos:",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            thick_sl = widgets.IntSlider(
                value=int(spec["thickness"]),
                min=0,
                max=0,
                step=1,
                description="thick:",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            def _sync_pos_max(_=None):
                nz, ny, nx = self._shape0
                axis_max = {"x": nx - 1, "y": ny - 1, "z": nz - 1}
                pos_sl.max = int(axis_max[ax_dd.value])
                pos_sl.value = int(min(pos_sl.value, pos_sl.max))

                axis_len = {"x": nx, "y": ny, "z": nz}[ax_dd.value]
                thick_sl.max = int(max(0, axis_len // 4))
                thick_sl.value = int(min(thick_sl.value, thick_sl.max))

            def _commit(_):
                spec["axis"] = ax_dd.value
                spec["pos"] = int(pos_sl.value)
                spec["thickness"] = int(thick_sl.value)

                # Re-materialize the stored slice (independent layer)
                # Choose a volume to re-slice: use the one used at creation time (store it),
                # OR simplest: store the original 3D volume in spec at creation.
                vol3d = spec["vol3d"]  # see step 4 below

                data2d, Xc, Yc, Zc = self._materialize_slice(
                    vol3d,
                    spec["axis"],
                    spec["pos"],
                    spec["thickness"],
                    nan_policy=policy,
                )
                spec["data2d"] = data2d
                spec["X"], spec["Y"], spec["Z"] = Xc, Yc, Zc

                self._update_all_traces()

            ax_dd.observe(
                lambda ch: (_sync_pos_max(), _commit(ch)), names="value"
            )
            pos_sl.observe(_commit, names="value")
            thick_sl.observe(_commit, names="value")
            w["nan_policy"].observe(_commit, names="value")

            _sync_pos_max()

            children += [widgets.HTML("<b>Slice</b>"), ax_dd, pos_sl, thick_sl]

        elif spec["type"] == "plane":
            nx_t = widgets.FloatText(
                value=float(spec["normal"][0]),
                description="n x:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            ny_t = widgets.FloatText(
                value=float(spec["normal"][1]),
                description="n y:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            nz_t = widgets.FloatText(
                value=float(spec["normal"][2]),
                description="n z:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            ox_t = widgets.FloatText(
                value=float(spec["origin"][0]),
                description="o x:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oy_t = widgets.FloatText(
                value=float(spec["origin"][1]),
                description="o y:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oz_t = widgets.FloatText(
                value=float(spec["origin"][2]),
                description="o z:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            thick_sl = widgets.FloatSlider(
                value=float(spec["thickness"]),
                min=0.0,
                max=self.add_plane_thickness.max,
                step=0.1,
                description="thick:",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            ext_sl = widgets.FloatSlider(
                value=float(spec["extent"]),
                min=1.0,
                max=self.add_plane_extent.max,
                step=1.0,
                description="extent:",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            def _commit(_):
                n = np.array([nx_t.value, ny_t.value, nz_t.value], dtype=float)
                o = np.array([ox_t.value, oy_t.value, oz_t.value], dtype=float)
                spec["normal"] = n
                spec["origin"] = o
                spec["thickness"] = float(thick_sl.value)
                spec["extent"] = float(ext_sl.value)

                data2d, Xc, Yc, Zc = self._materialize_plane(
                    spec["vol3d"],
                    spec["normal"],
                    spec["origin"],
                    spec["thickness"],
                    spec["extent"],
                )
                spec["data2d"] = data2d
                spec["X"], spec["Y"], spec["Z"] = Xc, Yc, Zc
                self._update_all_traces()

            for ww in (nx_t, ny_t, nz_t, ox_t, oy_t, oz_t, thick_sl, ext_sl):
                ww.observe(_commit, names="value")
            w["nan_policy"].observe(_commit, names="value")
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
                thick_sl,
                ext_sl,
            ]

        elif spec["type"] == "clip":
            # --- editable clip params ---
            # source raw
            raw_keys = [
                kk
                for kk, ss in self._layers.items()
                if ss.get("type") == "raw"
            ]
            src_dd = widgets.Dropdown(
                options=raw_keys,
                value=spec.get("source", raw_keys[0] if raw_keys else None),
                description="src:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            # normal + origin
            nx_t = widgets.FloatText(
                value=float(spec["normal"][0]),
                description="n x:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            ny_t = widgets.FloatText(
                value=float(spec["normal"][1]),
                description="n y:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            nz_t = widgets.FloatText(
                value=float(spec["normal"][2]),
                description="n z:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            ox_t = widgets.FloatText(
                value=float(spec["origin"][0]),
                description="o x:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oy_t = widgets.FloatText(
                value=float(spec["origin"][1]),
                description="o y:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            oz_t = widgets.FloatText(
                value=float(spec["origin"][2]),
                description="o z:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            side_dd = widgets.Dropdown(
                options=[("Up (+n)", "up"), ("Down (-n)", "down")],
                value=spec.get("side", "up"),
                description="side:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )

            def _commit(_):
                if src_dd.value:
                    spec["source"] = src_dd.value
                spec["normal"] = np.array(
                    [nx_t.value, ny_t.value, nz_t.value], dtype=float
                )
                spec["origin"] = np.array(
                    [ox_t.value, oy_t.value, oz_t.value], dtype=float
                )
                spec["side"] = side_dd.value
                self._update_all_traces()

            for ww in (src_dd, nx_t, ny_t, nz_t, ox_t, oy_t, oz_t, side_dd):
                ww.observe(_commit, names="value")
            w["nan_policy"].observe(_commit, names="value")

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
                side_dd,
                w["thr"],
            ]

        # common controls
        children += [
            w["op"],
            w["cmap"],
            w["row"],
            w["nan_policy"],
            w["color_by"],
            w["range_slider"],
            widgets.HTML("<b>Lighting</b>"),
            w["light_ambient"],
            w["light_diffuse"],
            w["light_specular"],
            w["light_roughness"],
            w["light_fresnel"],
        ]

        self.layers_box.children = [widgets.VBox(children)]

    # --- helper to refresh layer lists after set_data / new layers ---
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

        # edit dropdown
        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            self.edit_key.options = layer_names
            if self.edit_key.value not in layer_names:
                self.edit_key.value = layer_names[0] if layer_names else None
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

    # --- range sync for the "Add slice" UI sliders ---
    def _default_source_key(self):
        for k, s in self._layers.items():
            if s.get("type") == "raw":
                return k
        return None

    def _sync_add_slice_pos_range(self, change=None):
        src = self._default_source_key()
        if not src or src not in self.dict_data:
            return

        nz, ny, nx = self.dict_data[src].shape
        ax = self.add_slice_axis.value

        axis_max = {"x": nx - 1, "y": ny - 1, "z": nz - 1}
        axis_len = {"x": nx, "y": ny, "z": nz}[ax]

        self.add_slice_pos.max = int(axis_max[ax])
        self.add_slice_pos.value = int(
            min(self.add_slice_pos.value, self.add_slice_pos.max)
        )

        self.add_slice_thickness.max = int(max(0, axis_len // 4))
        self.add_slice_thickness.value = int(
            min(self.add_slice_thickness.value, self.add_slice_thickness.max)
        )

    # --- "Create slice layer" callback ---
    def _on_create_layer_clicked(self, btn):
        kind = self.add_layer_kind.value
        src = self._default_source_key()
        if not src:
            return

        base_label = (self.add_layer_name.value or kind).strip() or kind
        vol3d = np.asarray(self.dict_data[src], dtype=float).copy()

        policy = self._layer_widgets[src]["nan_policy"].value
        if kind == "slice":
            ax = self.add_slice_axis.value
            pos = int(self.add_slice_pos.value)
            thick = int(self.add_slice_thickness.value)
            data2d, Xc, Yc, Zc = self._materialize_slice(
                vol3d, ax, pos, thick, nan_policy=policy
            )

            new_name = f"{base_label}:slice"
            base = new_name
            kk = 2
            while new_name in self._layers:
                new_name = f"{base}#{kk}"
                kk += 1

            self._layers[new_name] = dict(
                type="slice",
                axis=ax,
                pos=pos,
                thickness=thick,
                vol3d=vol3d,
                data2d=data2d,
                X=Xc,
                Y=Yc,
                Z=Zc,
            )

            self._build_widgets_for_layer(new_name)
            self._refresh_layer_lists()
            self._visible_cb[new_name].value = True
            self.edit_key.value = new_name
            self._sync_layer_panels_visibility()
            self._update_all_traces()
            return

        if kind == "plane":
            n = np.array(
                [
                    self.add_plane_nx.value,
                    self.add_plane_ny.value,
                    self.add_plane_nz.value,
                ],
                dtype=float,
            )
            o = np.array(
                [
                    self.add_plane_ox.value,
                    self.add_plane_oy.value,
                    self.add_plane_oz.value,
                ],
                dtype=float,
            )
            thick = float(self.add_plane_thickness.value)
            extent = float(self.add_plane_extent.value)

            data2d, Xc, Yc, Zc = self._materialize_plane(
                vol3d, n, o, thick, extent
            )

            new_name = f"{base_label}:plane"
            base = new_name
            k = 2
            while new_name in self._layers:
                new_name = f"{base}#{k}"
                k += 1

            self._layers[new_name] = dict(
                type="plane",
                normal=n,
                origin=o,
                thickness=thick,
                extent=extent,
                vol3d=vol3d,
                data2d=data2d,
                X=Xc,
                Y=Yc,
                Z=Zc,
            )

            self._build_widgets_for_layer(new_name)
            self._refresh_layer_lists()
            if new_name in self._visible_cb:
                self._visible_cb[new_name].value = True
            self.edit_key.value = new_name
            self._sync_layer_panels_visibility()
            self._update_all_traces()
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

            n = np.array(
                [
                    self.add_plane_nx.value,
                    self.add_plane_ny.value,
                    self.add_plane_nz.value,
                ],
                dtype=float,
            )
            o = np.array(
                [
                    self.add_plane_ox.value,
                    self.add_plane_oy.value,
                    self.add_plane_oz.value,
                ],
                dtype=float,
            )
            side = self.add_clip_side.value  # "up" or "down"

            new_name = f"{base_label}:clip"
            base = new_name
            k = 2
            while new_name in self._layers:
                new_name = f"{base}#{k}"
                k += 1

            self._layers[new_name] = dict(
                type="clip",
                source=src_key,  # raw source key
                normal=n,
                origin=o,
                side=side,
            )

            self._build_widgets_for_layer(new_name)
            self._refresh_layer_lists()
            self._visible_cb[new_name].value = True
            self.edit_key.value = new_name
            self._sync_layer_panels_visibility()
            self._update_all_traces()
            return

    def _materialize_slice(
        self,
        vol3d: np.ndarray,
        ax: str,
        pos: int,
        thick: int,
        nan_policy: str = "none",
    ):
        vol = np.asarray(vol3d, dtype=float)
        vol = self._apply_nan_policy(vol, nan_policy)
        nz, ny, nx = vol.shape

        if ax == "z":
            z0 = max(0, pos - thick)
            z1 = min(nz - 1, pos + thick)
            slab = vol[z0 : z1 + 1, :, :]
            data2d = np.nanmean(slab, axis=0)  # (ny, nx)

            X = np.arange(nx) * self.voxel_size[2]
            Y = np.arange(ny) * self.voxel_size[1]
            Z = np.ones((ny, nx)) * (pos * self.voxel_size[0])
            XX, YY = np.meshgrid(X, Y)
            return data2d, XX, YY, Z

        if ax == "y":
            y0 = max(0, pos - thick)
            y1 = min(ny - 1, pos + thick)
            slab = vol[:, y0 : y1 + 1, :]
            data2d = np.nanmean(slab, axis=1)  # (nz, nx)

            X = np.arange(nx) * self.voxel_size[2]
            Z = np.arange(nz) * self.voxel_size[0]
            Y = np.ones((nz, nx)) * (pos * self.voxel_size[1])
            XX, ZZ = np.meshgrid(X, Z)
            return data2d, XX, Y, ZZ

        # ax == "x"
        x0 = max(0, pos - thick)
        x1 = min(nx - 1, pos + thick)
        slab = vol[:, :, x0 : x1 + 1]
        data2d = np.nanmean(slab, axis=2)  # (nz, ny)

        Y = np.arange(ny) * self.voxel_size[1]
        Z = np.arange(nz) * self.voxel_size[0]
        X = np.ones((nz, ny)) * (pos * self.voxel_size[2])
        YY, ZZ = np.meshgrid(Y, Z)
        return data2d, X, YY, ZZ

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
        vol = np.asarray(vol3d, dtype=float)
        vol = self._apply_nan_policy(vol, nan_policy)
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

        acc = 0.0
        cnt = 0
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
            acc += np.nan_to_num(vals, nan=0.0)
            if nan_policy != "none":
                vals = self._apply_nan_policy(vals, nan_policy)
            vals_list.append(vals)
            cnt += 1

        data2d = np.nanmean(np.stack(vals_list, axis=0), axis=0)

        # surface coords (physical)
        Xc = r0[0] + u[0] * UU + v[0] * VV
        Yc = r0[1] + u[1] * UU + v[1] * VV
        Zc = r0[2] + u[2] * UU + v[2] * VV

        return data2d, Xc, Yc, Zc

    # --- Factor widget building: raw vs derived share most controls ---
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
            description="iso:",
            continuous_update=False,
            readout_format=".3g",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        op = widgets.FloatSlider(
            value=1.0,
            min=0.0,
            max=1.0,
            step=0.01,
            description="α:",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        cmap = widgets.Dropdown(
            options=self.cmap_options,
            value="gray",
            description="cmap:",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
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
            description="NaN:",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
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
            description="color:",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
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

        row = widgets.Box(
            [as_mask, show_colorbar, auto_range],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row",
                justify_content="space-between",
                align_items="center",
                gap="12px",
                width="95%",
            ),
        )
        for cb in (as_mask, show_colorbar, auto_range):
            cb.layout.width = "auto"
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
            description="cmin/cmax:",
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
            value=0.6,
            min=0.0,
            max=1.0,
            step=0.01,
            description="ambient",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        light_diffuse = widgets.FloatSlider(
            value=0.4,
            min=0.0,
            max=1.0,
            step=0.01,
            description="diffuse",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        light_specular = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description="specular",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        light_roughness = widgets.FloatSlider(
            value=0.2,
            min=0.01,
            max=1.0,
            step=0.01,
            description="roughness",
            continuous_update=False,
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        light_fresnel = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description="fresnel",
            continuous_update=False,
            style=self._common_style,
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
            cmap=cmap,
            row=row,
            as_mask=as_mask,
            show_colorbar=show_colorbar,
            auto_range=auto_range,
            color_by=color_by,
            range_slider=range_slider,
            light_ambient=light_ambient,
            light_diffuse=light_diffuse,
            light_specular=light_specular,
            light_roughness=light_roughness,
            light_fresnel=light_fresnel,
            nan_policy=nan_policy,
        )

    # --- dispatcher in rendering: raw -> iso mesh, slice -> surface ---
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
        return go.Scatter3d()  # fallback

    def _get_visible_keys(self):
        return [k for k, cb in self._visible_cb.items() if cb.value]

    def _update_all_traces(self):
        selected = self._get_visible_keys()
        with self.fig.batch_update():
            self.fig.data = tuple()
            if not selected:
                return
            for idx, k in enumerate(selected):
                self.fig.add_trace(
                    self._make_trace_for_layer(k, cbar_index=idx)
                )

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
                return go.Mesh3d(
                    name=key,
                    x=[],
                    y=[],
                    z=[],
                    i=[],
                    j=[],
                    k=[],
                    opacity=opacity,
                    showscale=False,
                )
            # 2) extract geometry from mask
            verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=1)
            verts_scaled = verts * self.voxel_size

            # 3) IMPORTANT: sample original arr at those vertices for coloring (same as normal)
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
                return go.Mesh3d(
                    name=key,
                    x=[],
                    y=[],
                    z=[],
                    i=[],
                    j=[],
                    k=[],
                    opacity=opacity,
                    showscale=False,
                )

            if HAS_VOLUME_UTILS:
                verts_scaled, faces, vals = _extract_isosurface_with_values(
                    vol,
                    vol,
                    iso,
                    self.voxel_size,
                    use_interpolator=True,
                )
            else:
                verts, faces, _, _ = marching_cubes(
                    vol, level=iso, step_size=1
                )
                verts_scaled = verts * self.voxel_size
                vals = self._rgi[key](verts)
        vals = np.asarray(vals, dtype=float)

        if policy == "none":
            # remove triangles touching NaNs
            valid = np.isfinite(vals)
            if not np.all(valid):
                keep = valid[faces].all(axis=1)
                faces = faces[keep]
        verts_idx = verts_scaled / self.voxel_size  # (z,y,x) index space
        verts_xyz = np.column_stack(
            [
                verts_scaled[:, 2],  # x
                verts_scaled[:, 1],  # y
                verts_scaled[:, 0],
            ]
        )  # z

        # ----------------------------
        # Coloring
        # ----------------------------
        show_colorbar = bool(w["show_colorbar"].value)
        auto_range = bool(w["auto_range"].value)
        rmin, rmax = w["range_slider"].value

        color_sel = w["color_by"].value  # dropdown value

        if color_sel == "__constant__":
            intensity = np.zeros(len(verts_xyz), dtype=float)
        elif color_sel in ("__x__", "__y__", "__z__"):
            axis = {"__x__": 0, "__y__": 1, "__z__": 2}[color_sel]
            intensity = verts_xyz[:, axis].astype(float)
        elif color_sel in ("__self__", key):
            intensity = np.asarray(vals, dtype=float)
        else:
            intensity = np.asarray(
                self._rgi[color_sel](verts_idx), dtype=float
            )

        intensity = self._apply_nan_policy(intensity, policy)

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

            colorbar = dict(
                title=title,
                len=0.7,
                x=CBAR_X0 + cbar_index * CBAR_DX,
                thickness=18,
            )

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

        return go.Mesh3d(
            name=key,
            x=verts_xyz[:, 0],  # x
            y=verts_xyz[:, 1],  # y
            z=verts_xyz[:, 2],  # z
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

    def _make_slice_trace(self, layer_name: str, cbar_index: int = 0):
        spec = self._layers[layer_name]
        w = self._layer_widgets[layer_name]
        policy = w["nan_policy"].value

        opacity = float(w["op"].value)
        colorscale = self.get_colorscale(w["cmap"].value)

        base_raw = np.asarray(spec["data2d"], dtype=float)
        data2d = self._apply_nan_policy(spec["data2d"], policy)
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
                        color_sel, Xc, Yc, Zc
                    )
                else:
                    # slice: reslice raw using same (axis,pos,thick)
                    intensity2d = self._slice_mean_2d(
                        self.dict_data[color_sel],
                        spec["axis"],
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
            colorbar = dict(
                title=title,
                len=0.7,
                x=CBAR_X0 + cbar_index * CBAR_DX,
                thickness=18,
            )

        return go.Surface(
            name=layer_name,
            x=Xplot,
            y=Yplot,
            z=Zplot,
            surfacecolor=Cplot,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            showscale=(colorbar is not None),
            colorbar=colorbar,
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

    def _make_clip_mesh_trace(self, layer_name: str, cbar_index: int = 0):
        spec = self._layers[layer_name]
        src = spec["source"]
        if src not in self.dict_data:
            return go.Mesh3d(
                name=layer_name, x=[], y=[], z=[], i=[], j=[], k=[]
            )

        w = self._layer_widgets[layer_name]
        policy = w["nan_policy"].value

        # base volume
        base = np.asarray(self.dict_data[src], dtype=float)
        base = self._apply_nan_policy(base, policy)

        # half-space keep mask
        keep = self._clip_volume_halfspace(
            base, spec["normal"], spec["origin"], spec["side"]
        )

        # IMPORTANT: marching_cubes cannot handle NaN reliably.
        # So push "outside" to a value guaranteed below iso (for most typical iso usage),
        # producing a clean cut. You still get proper coloring from sampled values.
        vol = base.copy()
        vmin = float(np.nanmin(base))
        vmax = float(np.nanmax(base))
        pad = 1e6 * max(1.0, abs(vmax - vmin))

        if spec["side"] == "up":
            # keep d>=0, remove d<0 -> push removed side BELOW iso
            vol[~keep] = vmin - pad
        else:
            # keep d<=0, remove d>0 -> also push removed side BELOW iso
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
            return go.Mesh3d(
                name=layer_name,
                x=[],
                y=[],
                z=[],
                i=[],
                j=[],
                k=[],
                opacity=opacity,
                showscale=False,
            )

        # geometry
        verts, faces, _, _ = marching_cubes(vol, level=iso, step_size=1)
        verts_scaled = verts * self.voxel_size

        # sample ORIGINAL source field for coloring, not the clipped volume
        vals = self._rgi[src](verts)  # verts are (z,y,x) index space
        vals = np.asarray(vals, dtype=float)
        vals = self._apply_nan_policy(vals, policy)

        verts_idx = verts_scaled / self.voxel_size
        verts_xyz = np.column_stack(
            [verts_scaled[:, 2], verts_scaled[:, 1], verts_scaled[:, 0]]
        )

        # coloring selection identical to your raw path
        show_colorbar = bool(w["show_colorbar"].value)
        auto_range = bool(w["auto_range"].value)
        rmin, rmax = w["range_slider"].value
        color_sel = w["color_by"].value

        if color_sel == "__constant__":
            intensity = np.zeros(len(verts_xyz), dtype=float)
        elif color_sel in ("__x__", "__y__", "__z__"):
            axis = {"__x__": 0, "__y__": 1, "__z__": 2}[color_sel]
            intensity = verts_xyz[:, axis].astype(float)
        elif color_sel in ("__self__", layer_name, src):
            intensity = np.asarray(vals, dtype=float)
        else:
            # allow coloring by other raw layers
            if color_sel in self._rgi:
                intensity = np.asarray(
                    self._rgi[color_sel](verts_idx), dtype=float
                )
            else:
                intensity = np.asarray(vals, dtype=float)

        intensity = self._apply_nan_policy(intensity, policy)

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
            title = (
                src
                if color_sel == "__self__"
                else (
                    color_sel.strip("_")
                    if color_sel in ("__x__", "__y__", "__z__")
                    else color_sel
                )
            )
            colorbar = dict(
                title=title,
                len=0.7,
                x=CBAR_X0 + cbar_index * CBAR_DX,
                thickness=18,
            )

        return go.Mesh3d(
            name=layer_name,
            x=verts_xyz[:, 0],
            y=verts_xyz[:, 1],
            z=verts_xyz[:, 2],
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

    # ----------------------------
    # Callbacks
    # ----------------------------
    def _on_visible_changed(self, change):
        self._update_all_traces()

    def _on_edit_key_changed(self, change):
        self._sync_layer_panels_visibility()

    def _on_layer_param_changed(self, change):
        self._update_all_traces()

    def _on_theme(self, change):
        self.fig.update_layout(
            template="plotly_dark"
            if self.theme_toggle.value
            else "plotly_white"
        )

    def _on_rotate_toggle(self, change):
        if self.rotate_toggle.value:
            self._start_rotation()
        else:
            self._stop_rotation()

    # ----------------------------
    # Rotation
    # ----------------------------
    def _start_rotation(self):
        import asyncio

        if self._rotation_task is not None and not self._rotation_task.done():
            return

        async def rotate():
            while self.rotate_toggle.value:
                self._rotation_angle += 2
                eye_x = 1.5 * np.cos(np.radians(self._rotation_angle))
                eye_y = 1.5 * np.sin(np.radians(self._rotation_angle))
                with self.fig.batch_update():
                    self.fig.layout.scene.camera.eye = dict(
                        x=eye_x, y=eye_y, z=1.5
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

    # ----------------------------
    # NaN policy
    # ----------------------------
    def _apply_nan_policy(
        self, arr: np.ndarray, policy: str = "none"
    ) -> np.ndarray:
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

    def _on_center_origin_clicked(self, _):
        c = self._center_origin_xyz()
        self.add_plane_ox.value = float(c[0])
        self.add_plane_oy.value = float(c[1])
        self.add_plane_oz.value = float(c[2])

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

    def _sample_raw_on_surface(self, raw_key: str, Xc, Yc, Zc) -> np.ndarray:
        """Sample raw 3D field raw_key on a surface defined by physical coords Xc,Yc,Zc."""
        if raw_key not in self._rgi:
            return np.zeros_like(Xc, dtype=float)

        # convert physical -> index space (z,y,x)
        xi = np.asarray(Xc, dtype=float) / self.voxel_size[2]
        yi = np.asarray(Yc, dtype=float) / self.voxel_size[1]
        zi = np.asarray(Zc, dtype=float) / self.voxel_size[0]

        pts = np.stack([zi, yi, xi], axis=-1)  # (..., 3)
        vals = self._rgi[raw_key](pts)
        return self._apply_nan_policy(np.asarray(vals, dtype=float))

    def _sync_create_layer_ui(self, change=None):
        """
        Show only the minimal "Create" header + layer type selector by default.
        Reveal the parameter widgets only after a type is selected.
        """
        kind = self.add_layer_kind.value  # can be None / "slice" / "plane"

        # Always show only the dropdown (and optionally name) at the top.
        base_children = [
            widgets.HTML("<b>Create layer</b>"),
            self.add_layer_kind,
            self.add_layer_name,
        ]

        # Hide everything until a type is chosen
        if kind in (None, "", "none"):
            self.derived_box.children = base_children
            return

        if kind == "slice":
            self.derived_box.children = base_children + [
                widgets.HTML("<b>Slice</b>"),
                widgets.HBox(
                    [self.add_slice_axis, self.add_slice_pos],
                    layout=widgets.Layout(width="100%", gap="12px"),
                ),
                self.add_slice_thickness,
                self.add_layer_btn,
            ]
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
                self.add_plane_thickness,
                self.add_plane_extent,
                self.add_layer_btn,
            ]
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
                self.add_clip_side,
                self.add_layer_btn,
            ]
            return

    def close(self):
        self._stop_rotation()
        super().close()

    def show(self):
        display(self)
