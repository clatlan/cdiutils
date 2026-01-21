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
            width = figsize[0] * 90,
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
        self._on_theme(None) # apply initial theme state
        self.rotate_toggle = widgets.ToggleButton(
            value=False,
            description="Rotate",
        )

        self.theme_toggle.observe(self._on_theme, names="value")
        self.rotate_toggle.observe(self._on_rotate_toggle, names="value")

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
        self.edit_key.observe(self._on_edit_key_changed, names="value")

        self.nan_policy = widgets.Dropdown(
            options=[
                ("Do nothing", "none"),
                ("Replace with mean", "mean"),
                ("Replace with zero", "zero"),
                ("Replace with min", "min"),
                ("Replace with max", "max"),
            ],
            value="zero",
            description="NaN handling:",
            tooltip="How to replace NaN values before rendering",
            style=self._common_style,
            layout=widgets.Layout(width="95%"),
        )
        self.nan_policy.observe(self._on_layer_param_changed, names="value")

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
                width='100%',
                flex="1 1 0%",   # take remaining space
                min_width="0px",   # IMPORTANT: allow shrinking instead of pushing/overflowing
                height="100%",
                overflow="hidden", # optional, prevents horizontal spill
                padding="28px 28px 28 28px",# reserve scrollbar gutter
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
                self.edit_key,
                self.nan_policy,
                widgets.HTML("<b>Layer controls</b>"),
                self.layers_box,
            ],
            layout=widgets.Layout(padding="0"),  # reserve space for scrollbar
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

    def show(self):
        display(self)

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

        keys = list(dict_data.keys())
        if not keys:
            raise ValueError("dict_data is empty.")

        # ---- validate arrays ----
        shape0 = None
        for k, v in dict_data.items():
            if not isinstance(k, str):
                raise TypeError(f"Layer key {k!r} is not a str.")
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Layer '{k}' is not a numpy array.")
            if v.ndim != 3:
                raise ValueError(f"Layer '{k}' must be a 3D array, got ndim={v.ndim}.")
            if np.iscomplexobj(v):
                raise TypeError(
                    f"Layer '{k}' is complex-valued. "
                    "This viewer expects real-valued scalar fields "
                    "(e.g., amplitude, phase (radians), density, mask). "
                    "Convert explicitly before passing."
                )

            # >>> ADD THIS BLOCK HERE <<<
            if v.dtype.kind not in ("b", "i", "u", "f"):
                raise TypeError(
                    f"Layer '{k}' must be a real numeric array "
                    f"(bool/int/float). Got dtype={v.dtype}."
                )
            # >>> END ADDITION <<<

            if shape0 is None:
                shape0 = v.shape
            elif v.shape != shape0:
                raise ValueError(
                    f"Shape mismatch: '{k}' has {v.shape}, expected {shape0}."
                )

        # Optional: basic sanity check for finite ranges (do not reject NaNs; NaN policy handles them)
        # You can uncomment to reject all-NaN layers if desired.
        # for k, v in dict_data.items():
        #     if not np.isfinite(v).any():
        #         raise ValueError(f"Layer '{k}' contains no finite values (all NaN/inf).")

        # ---- store ----
        self.dict_data = dict_data

        # ---- build interpolators (always real) ----
        nz, ny, nx = shape0
        grid_z = np.arange(nz, dtype=float)
        grid_y = np.arange(ny, dtype=float)
        grid_x = np.arange(nx, dtype=float)

        # Use float arrays for interpolator (RegularGridInterpolator expects numeric types)
        self._rgi = {}
        for k, arr in dict_data.items():
            arr_f = np.asarray(arr, dtype=float)  # safe for bool/int/float
            self._rgi[k] = RegularGridInterpolator(
                (grid_z, grid_y, grid_x),
                arr_f,
                bounds_error=False,
                fill_value=0.0,
            )

        # ---- build per-layer controls ----
        self._layer_widgets = {}
        self._rebuild_layer_widgets()

        # ---- visible keys checkboxes ----
        self._visible_cb = {}
        cbs = []
        for i, k in enumerate(keys):
            cb = widgets.Checkbox(
                value=(i == 0),
                description=k,
                indent=False,
            )
            cb.observe(self._on_visible_changed, names="value")
            self._visible_cb[k] = cb
            cbs.append(cb)
        self.visible_keys_box.children = cbs
        # apply per-child layout AFTER assigning children
        for cb in self.visible_keys_box.children:
            cb.layout.flex = "0 0 140px"   # fixed column width
            cb.layout.width = "140px"      # helps in some frontends

        # ---- edit key dropdown ----
        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            self.edit_key.options = keys
            self.edit_key.value = keys[0]
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

        # ---- render ----
        self._sync_layer_panels_visibility()
        self._update_all_traces()


    # ----------------------------
    # UI building
    # ----------------------------
    def _rebuild_layer_widgets(self):
        for k in self.dict_data.keys():
            amp = np.asarray(self.dict_data[k], dtype=float)

            thr = widgets.FloatSlider(
                value=float(np.nanpercentile(amp, 30)),
                min=float(np.nanmin(amp)),
                max=float(np.nanmax(amp)),
                step=float((np.nanmax(amp) - np.nanmin(amp)) / 300)
                if np.nanmax(amp) > np.nanmin(amp)
                else 0.01,
                description=f"{k} iso:",
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
                description=f"{k} α:",
                continuous_update=False,
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            cmap = widgets.Dropdown(
                options=self.cmap_options,
                value="grey",
                description=f"{k} cmap:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            color_by = widgets.Dropdown(
                options=[("(self)", k)] + [(kk, kk) for kk in self.dict_data.keys() if kk != k] + [
                    ("(constant)", "__constant__"),
                    ("(x coord)", "__x__"),
                    ("(y coord)", "__y__"),
                    ("(z coord)", "__z__"),
                ],
                value=k,
                description=f"{k} color:",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),
            )
            color_by.observe(self._on_layer_param_changed, names="value")


            as_mask = widgets.Checkbox(
                value=False,
                description=f"{k} mask-mode",
                tooltip="If True: threshold to a binary mask then extract its surface.",
                indent=False,
            )

            show_colorbar = widgets.Checkbox(
                value=True,
                description=f"{k} show colorbar",
                indent=False,
            )

            auto_range = widgets.Checkbox(
                value=True,
                description=f"{k} auto range",
                indent=False,
            )

            row = widgets.Box(
                [as_mask, show_colorbar, auto_range],
                layout=widgets.Layout(
                    display="flex",
                    flex_flow="row",
                    justify_content="space-between",  # or "flex-start"
                    align_items="center",
                    gap="12px",
                    width="95%",
                ),
            )
            for cb in (as_mask, show_colorbar, auto_range):
                cb.layout.width = "auto"
                cb.layout.flex = "1 1 0%"


            vmin0 = float(np.nanmin(amp))
            vmax0 = float(np.nanmax(amp))
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
                description=f"{k} cmin/cmax:",
                continuous_update=False,
                readout_format=".3g",
                style=self._common_style,
                layout=widgets.Layout(width="95%"),indent=False,
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
            # ---- Lighting controls (ParaView-style) ----
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
            )

    def _sync_layer_panels_visibility(self):
        k = self.edit_key.value
        if not k or k not in self._layer_widgets:
            self.layers_box.children = []
            return
        w = self._layer_widgets[k]

        self.layers_box.children = [
            widgets.VBox([
                w["thr"],
                w["op"],
                w["cmap"],
                w["row"],
                w["color_by"],
                w["range_slider"],
                widgets.HTML("<b>Lighting</b>"),
                w["light_ambient"],
                w["light_diffuse"],
                w["light_specular"],
                w["light_roughness"],
                w["light_fresnel"],
            ])
        ]

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
    # Rendering
    # ----------------------------
    def _get_visible_keys(self):
        return [k for k, cb in self._visible_cb.items() if cb.value]

    def _update_all_traces(self):
        selected = self._get_visible_keys()

        with self.fig.batch_update():
            self.fig.data = tuple()
            for idx, k in enumerate(selected):
                self.fig.add_trace(
                    self._make_mesh_trace_for_key(k, cbar_index=idx)
                )

    def _make_mesh_trace_for_key(self, key: str, cbar_index: int = 0):
        arr = self.dict_data[key]
        w = self._layer_widgets[key]

        iso = float(w["thr"].value)
        opacity = float(w["op"].value)
        colorscale = self.get_colorscale(w["cmap"].value)

        # ----------------------------
        # Build volume for marching cubes
        # ----------------------------
        if w["as_mask"].value:
            # 1) build mask from a scalar (typically |arr|)
            vol = np.asarray(arr, dtype=float)
            vol = self._apply_nan_policy(vol)

            mask = (vol >= iso).astype(np.float32)
            if not (mask.min() < 0.5 < mask.max()):
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
            vals = self._apply_nan_policy(vals)

        else:
            vol = np.asarray(arr, dtype=float)
            vol = self._apply_nan_policy(vol)

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

        verts_idx = verts_scaled / self.voxel_size                 # (z,y,x) index space
        verts_xyz = np.column_stack([verts_scaled[:, 2],           # x
                                    verts_scaled[:, 1],           # y
                                    verts_scaled[:, 0]])          # z

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
        elif color_sel == key:
            intensity = np.asarray(vals, dtype=float)
        else:
            intensity = np.asarray(self._rgi[color_sel](verts_idx), dtype=float)

        intensity = self._apply_nan_policy(intensity)

        data_min = float(np.nanmin(intensity))
        data_max = float(np.nanmax(intensity))
        if not np.isfinite(data_min):
            data_min = 0.0
        if not np.isfinite(data_max):
            data_max = 1.0
        if data_max == data_min:
            data_max = data_min + 1e-12

        cmin, cmax = (
            (data_min, data_max)
            if auto_range
            else (float(rmin), float(rmax))
        )
        if cmax <= cmin:
            cmax = cmin + 1e-12

        colorbar = None
        if (color_sel != "__constant__") and show_colorbar:
            title = color_sel if not color_sel.startswith("__") else color_sel.strip("_")
            colorbar = dict(title=title, len=0.7, x=CBAR_X0 + cbar_index * CBAR_DX, thickness=18)

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
            x=verts_xyz[:, 0], # x
            y=verts_xyz[:, 1], # y
            z=verts_xyz[:, 2], # z
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
    # Rotation
    # ----------------------------
    def _start_rotation(self):
        import asyncio

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

        loop = asyncio.get_event_loop()
        self._rotation_task = loop.create_task(rotate())

    def _stop_rotation(self):
        if self._rotation_task is not None:
            self._rotation_task.cancel()
            self._rotation_task = None

    # ----------------------------
    # NaN policy
    # ----------------------------
    def _apply_nan_policy(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)

        policy = self.nan_policy.value
        if policy == "none" or not np.isnan(arr).any():
            return arr

        if policy == "mean":
            v = np.nanmean(arr)
        elif policy == "zero":
            v = 0.0
        elif policy == "min":
            v = np.nanmin(arr)
        elif policy == "max":
            v = np.nanmax(arr)
        else:
            return arr

        if not np.isfinite(v):
            v = 0.0

        return np.where(np.isnan(arr), v, arr)


