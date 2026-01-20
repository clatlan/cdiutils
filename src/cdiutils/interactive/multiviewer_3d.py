import ipywidgets as widgets
import numpy as np
from IPython.display import display

import plotly.graph_objects as go
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
                f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})",
            ]
            for i, c in enumerate(colors)
        ]
CBAR_X0 = 1.02       # start just outside the scene
CBAR_DX = 0.2       # horizontal spacing per colorbar


class MultiVolumeViewer(widgets.Box):
    @staticmethod
    def get_colorscale(name: str):
        # Solid colors
        if name in {
            "red", "black", "white", "gray",
            "blue", "green", "orange", "purple",
            "yellow", "cyan", "magenta",
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

    cmap_options = get_all_supported_cmaps()


    def __init__(self, dict_data=None, voxel_size=(1, 1, 1), figsize=(9, 6)):
        super().__init__()

        self.voxel_size = np.array(voxel_size, dtype=float)

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
            width=figsize[0] * 96,
            height=figsize[1] * 96,
            dragmode="orbit",
        )
        self.fig.update_layout(margin=dict(r=220))

        # Data / interpolators
        self.dict_data = {}
        self._rgi = {}  # key -> RegularGridInterpolator
        self._layer_widgets = {}  # key -> widgets bundle
        self._visible_cb = {}  # key -> Checkbox

        # Top controls
        self.theme_toggle = widgets.ToggleButton(value=False, description="Dark Theme")
        self.rotate_toggle = widgets.ToggleButton(value=False, description="Rotate")

        self.theme_toggle.observe(self._on_theme, names="value")
        self.rotate_toggle.observe(self._on_rotate_toggle, names="value")

        self.nan_policy = widgets.Dropdown(
            options=[
                ("Do nothing", "none"),
                ("Replace with mean", "mean"),
                ("Replace with zero", "zero"),
                ("Replace with min", "min"),
                ("Replace with max", "max"),
            ],
            value="mean",
            description="NaN handling:",
            tooltip="How to replace NaN values before rendering",
        )
        self.nan_policy.observe(self._on_layer_param_changed, names="value")

        # Visible keys checkboxes + "edit key" dropdown
        self.visible_keys_box = widgets.VBox([])
        self.edit_key = widgets.Dropdown(options=[], description="Edit:")
        self.edit_key.observe(self._on_edit_key_changed, names="value")

        # Container for per-layer controls (only for edit_key)
        self.layers_box = widgets.VBox([])

        left = widgets.VBox(
            [
                widgets.HTML("<b>Visible layers</b>"),
                self.visible_keys_box,
                self.edit_key,
                self.theme_toggle,
                self.rotate_toggle,
                self.nan_policy,
                widgets.HTML("<b>Layer controls</b>"),
                self.layers_box,
            ]
        )
        self.children = [self.fig, left]

        # rotation state
        self._rotation_angle = 0
        self._rotation_task = None

        if dict_data is not None:
            self.set_data(dict_data)

    def show(self):
        display(self)

    # ----------------------------
    # Data setup
    # ----------------------------
    def set_data(self, dict_data: dict):
        keys = list(dict_data.keys())
        if not keys:
            raise ValueError("dict_data is empty.")

        shape0 = dict_data[keys[0]].shape
        for k, v in dict_data.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(f"{k} is not a numpy array.")
            if v.shape != shape0:
                raise ValueError(f"Shape mismatch: {k} has {v.shape}, expected {shape0}.")

        self.dict_data = dict_data

        # Build interpolators
        nz, ny, nx = shape0
        grid_z = np.arange(nz)
        grid_y = np.arange(ny)
        grid_x = np.arange(nx)

        self._rgi = {
            k: RegularGridInterpolator(
                (grid_z, grid_y, grid_x), arr, bounds_error=False, fill_value=0
            )
            for k, arr in dict_data.items()
        }

        # Build per-layer controls (widgets) + visible checkboxes
        self._layer_widgets = {}
        self._rebuild_layer_widgets()

        self._visible_cb = {}
        cbs = []
        for k in keys:
            cb = widgets.Checkbox(
                value=(k == keys[0]),
                description=k,
                indent=False,
            )
            cb.observe(self._on_visible_changed, names="value")
            self._visible_cb[k] = cb
            cbs.append(cb)
        self.visible_keys_box.children = cbs

        # Edit key dropdown
        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            self.edit_key.options = keys
            self.edit_key.value = keys[0]
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")

        # Render
        self._sync_layer_panels_visibility()
        self._update_all_traces()

    # ----------------------------
    # UI building
    # ----------------------------
    def _rebuild_layer_widgets(self):
        for k in self.dict_data.keys():
            arr = self.dict_data[k]
            amp = np.abs(arr) if np.iscomplexobj(arr) else np.asarray(arr)

            thr = widgets.FloatSlider(
                value=float(np.nanpercentile(amp, 70)),
                min=float(np.nanmin(amp)),
                max=float(np.nanmax(amp)),
                step=float((np.nanmax(amp) - np.nanmin(amp)) / 300)
                if np.nanmax(amp) > np.nanmin(amp)
                else 0.01,
                description=f"{k} iso:",
                continuous_update=False,
                readout_format=".3g",
            )
            op = widgets.FloatSlider(
                value=1.0,
                min=0.0,
                max=1.0,
                step=0.01,
                description=f"{k} α:",
                continuous_update=False,
            )
            cmap = widgets.Dropdown(
                options=self.cmap_options,
                value="cet_CET_C9s_r",
                description=f"{k} cmap:",
            )
            as_mask = widgets.Checkbox(
                value=False,
                description=f"{k} mask-mode",
                tooltip="If True: threshold to a binary mask then extract its surface.",
            )
            show_colorbar = widgets.Checkbox(
                value=True, description=f"{k} show colorbar", indent=False
            )
            auto_range = widgets.Checkbox(
                value=True,
                description=f"{k} auto range",
                indent=False,
                tooltip="If on: cmin/cmax follow data. If off: use the slider range.",
            )

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
            )
            range_slider.disabled = True

            def _toggle_range_slider(change, _rs=range_slider):
                _rs.disabled = bool(change["new"])

            auto_range.observe(_toggle_range_slider, names="value")

            for wdg in (thr, op, cmap, as_mask, show_colorbar, auto_range, range_slider):
                wdg.observe(self._on_layer_param_changed, names="value")

            self._layer_widgets[k] = dict(
                thr=thr,
                op=op,
                cmap=cmap,
                as_mask=as_mask,
                show_colorbar=show_colorbar,
                auto_range=auto_range,
                range_slider=range_slider,
            )

    def _sync_layer_panels_visibility(self):
        k = self.edit_key.value
        if not k or k not in self._layer_widgets:
            self.layers_box.children = []
            return
        w = self._layer_widgets[k]
        self.layers_box.children = [
            widgets.VBox(
                [
                    w["thr"],
                    w["op"],
                    w["cmap"],
                    w["as_mask"],
                    w["show_colorbar"],
                    w["auto_range"],
                    w["range_slider"],
                ]
            )
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
        self.fig.update_layout(template="plotly_dark" if self.theme_toggle.value else "plotly_white")

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
                self.fig.add_trace(self._make_mesh_trace_for_key(k, cbar_index=idx))

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
            vol = np.abs(arr) if np.iscomplexobj(arr) else np.asarray(arr, dtype=float)
            vol = self._apply_nan_policy(vol)

            mask = (vol >= iso).astype(np.float32)
            if not (mask.min() < 0.5 < mask.max()):
                return go.Mesh3d(name=key, x=[], y=[], z=[], i=[], j=[], k=[],
                                opacity=opacity, showscale=False)


            # 2) extract geometry from mask
            verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=1)
            verts_scaled = verts * self.voxel_size

            # 3) IMPORTANT: sample original arr at those vertices for coloring (same as normal)
            vals = self._rgi[key](verts)      # verts are in (z,y,x) index space
            vals = self._apply_nan_policy(vals)
            intensity_mode = "scalar"

        else:
            vol_for_mc = np.abs(arr) if np.iscomplexobj(arr) else np.asarray(arr, dtype=float)
            vol_for_mc = self._apply_nan_policy(vol_for_mc)

            if HAS_VOLUME_UTILS:
                verts_scaled, faces, vals = _extract_isosurface_with_values(
                    vol_for_mc, arr, iso, self.voxel_size, use_interpolator=True
                )
            else:
                verts, faces, _, _ = marching_cubes(vol_for_mc, level=iso, step_size=1)
                verts_scaled = verts * self.voxel_size
                vals = self._rgi[key](verts)

            intensity_mode = "scalar"

        # ----------------------------
        # Coloring
        # ----------------------------
        show_colorbar = bool(w["show_colorbar"].value)
        auto_range = bool(w["auto_range"].value)
        rmin, rmax = w["range_slider"].value

        if intensity_mode == "constant":
            intensity = np.zeros(len(verts_scaled), dtype=float)
            cmin, cmax = 0.0, 1.0
            colorbar = None
        else:
            if np.iscomplexobj(arr):
                intensity = np.angle(vals)
                intensity = self._apply_nan_policy(intensity)

                data_min, data_max = -np.pi, np.pi
                cmin, cmax = (data_min, data_max) if auto_range else (float(rmin), float(rmax))

                colorbar = dict(
                    title=f"{key} phase (rad)",
                    len=0.7,
                    x=CBAR_X0 + cbar_index * CBAR_DX,
                    thickness=18,
                    tickmode="array",
                    tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                    ticktext=["-π", "-π/2", "0", "π/2", "π"],
                )
            else:
                intensity = np.asarray(vals, dtype=float)
                intensity = self._apply_nan_policy(intensity)

                data_min = float(np.nanmin(intensity))
                data_max = float(np.nanmax(intensity))
                if not np.isfinite(data_min):
                    data_min = 0.0
                if not np.isfinite(data_max):
                    data_max = 1.0
                if data_max == data_min:
                    data_max = data_min + 1e-12

                cmin, cmax = (data_min, data_max) if auto_range else (float(rmin), float(rmax))

                colorbar = dict(
                    title=f"{key}",
                    len=0.7,
                    x=CBAR_X0 + cbar_index * CBAR_DX,
                    thickness=18,
                )
            if not show_colorbar:
                colorbar = None

        # Sync slider bounds safely: ONLY EXPAND (never shrink)
        rs = w["range_slider"]
        if intensity_mode != "constant":
            if np.iscomplexobj(arr):
                new_min, new_max = -np.pi, np.pi
            else:
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
        if (cmax <= cmin):
            cmax = cmin + 1e-12
        return go.Mesh3d(
            name=key,
            x=verts_scaled[:, 0],
            y=verts_scaled[:, 1],
            z=verts_scaled[:, 2],
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
                ambient=0.85,
                diffuse=0.1,
                specular=0.5,
                roughness=0.2,
                fresnel=0.5,
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
                    self.fig.layout.scene.camera.eye = dict(x=eye_x, y=eye_y, z=1.5)
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
        arr = np.asarray(arr)
        if arr.dtype.kind not in ("f", "c"):
            return arr

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
