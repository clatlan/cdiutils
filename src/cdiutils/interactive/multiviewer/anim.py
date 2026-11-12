"""Animation export, still images and live rotation for :class:`MultiVolumeViewer` (mixin)."""

import asyncio
import math
import time
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import ipywidgets as widgets
import numpy as np

from . import geometry as geo
from .export import REPEAT, FrameSink, output_path, run_ordered_export
from .rendering import decode_plotly_array, render_png

ANIM_PARAMS = {
    "raw": ["opacity", "thr"],
    "diff": ["opacity", "thr"],
    "region": ["opacity"],
    "slice": ["pos", "opacity"],
    "plane": ["offset", "opacity"],
    "clip": ["offset", "opacity", "thr"],
    "lines": ["opacity"],
    "arrows": ["opacity"],
}
PARAM_LABELS = {
    "opacity": "opacity",
    "thr": "iso level",
    "pos": "slice position",
    "offset": "offset along normal",
}


def _trace_bounds(traces):
    """(lo, hi) plot-coordinate bounds of trace JSONs (None if empty)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for t in traces:
        if t.get("visible") is False:
            continue
        for k, a in enumerate("xyz"):
            v = t.get(a)
            if v is None:
                continue
            v = decode_plotly_array(v, float).ravel()
            v = v[np.isfinite(v)]
            if v.size:
                lo[k], hi[k] = min(lo[k], v.min()), max(hi[k], v.max())
    if not np.all(np.isfinite(lo)):
        return None, None
    return lo, np.maximum(hi, lo + 1e-12)


def _fmt_seconds(sec):
    if sec < 60:
        return f"{sec:.1f} s"
    m, s = divmod(sec, 60.0)
    return (
        f"{int(m)} min {s:04.1f} s"
        if m < 60
        else f"{int(m // 60)} h {int(m % 60)} min"
    )


class AnimMixin:
    _ANIM_REGISTRY = ANIM_PARAMS

    # =========================================================================
    # UI
    # =========================================================================
    def _build_anim_ui(self):
        L = widgets.Layout
        D = {"description_width": "70px"}
        self._export_token = {"cancel": False}
        self._export_tokens = []
        self.anim_master_mode = widgets.Dropdown(
            options=[
                ("— Select —", None),
                ("Rotation", "rotation"),
                ("Layer", "layer"),
                ("Conditions", "conditions"),
            ],
            value=None,
            description="Animate",
            style=D,
            layout=L(width="220px"),
        )
        self.anim_name = widgets.Text(
            value="animation",
            description="Save as",
            placeholder="name or path, e.g. figures/spin",
            tooltip="File name or path (relative to the notebook folder, "
            "absolute, or ~/...). Missing folders are created.",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_format = widgets.Dropdown(
            options=[("MP4", "mp4"), ("GIF", "gif"), ("PNG frames", "png")],
            value="mp4",
            description="Type",
            style={"description_width": "40px"},
            layout=L(width="170px"),
        )
        self.anim_save_btn = widgets.Button(
            description="Export animation",
            button_style="primary",
            layout=L(width="160px"),
        )
        self.anim_stop_btn = widgets.Button(
            description="Stop", layout=L(width="90px")
        )
        self.anim_html_btn = widgets.Button(
            description="Save HTML",
            tooltip="Interactive page of the current view (with a slider "
            "over the conditions when there are several)",
            layout=L(width="110px"),
        )
        self.anim_fps = widgets.IntSlider(
            value=30,
            min=1,
            max=60,
            description="FPS",
            continuous_update=False,
            style=D,
            layout=L(width="260px"),
        )
        self.anim_width = widgets.IntText(
            value=self.export_width,
            description="width",
            style={"description_width": "40px"},
            layout=L(width="130px"),
        )
        self.anim_height = widgets.IntText(
            value=self.export_height,
            description="height",
            style={"description_width": "45px"},
            layout=L(width="135px"),
        )
        self.anim_scale = widgets.IntSlider(
            value=2,
            min=1,
            max=4,
            description="scale",
            style={"description_width": "40px"},
            layout=L(width="190px"),
        )

        self.anim_smooth = widgets.Checkbox(
            value=False,
            description="Smooth edges",
            indent=False,
            tooltip="Render at 2x and downsample: smoother outlines (data values are not changed)",
            layout=L(width="auto"),
        )
        self.anim_flat = widgets.Checkbox(
            value=False,
            description="Flat colours",
            indent=False,
            tooltip="One colour per triangle (no colour blending inside triangles)",
            layout=L(width="auto"),
        )

        self.anim_backend = widgets.Dropdown(
            options=[
                ("Built-in", "software"),
                ("Kaleido", "kaleido"),
                ("VTK", "vtk"),
            ],
            value=str(self._requested_backend or "software").replace(
                "auto", "software"
            ),
            description="Renderer",
            style={"description_width": "60px"},
            layout=L(width="190px"),
            tooltip="Built-in: numpy + matplotlib, works everywhere. "
            "Kaleido (Chrome) and VTK are optional.",
        )
        self.anim_backend.observe(
            lambda ch: setattr(self, "rendering_backend", ch["new"]),
            names="value",
        )

        self.anim_rot_type = widgets.Dropdown(
            options=[
                ("— Select —", None),
                ("Orbit Z", "orbit_z"),
                ("Orbit X", "orbit_x"),
                ("Orbit Y", "orbit_y"),
                ("Custom axis", "axis"),
                ("Around a layer normal", "normal"),
            ],
            value=None,
            description="Mode",
            style=D,
            layout=L(width="280px"),
        )
        self.anim_axis_x, self.anim_axis_y, self.anim_axis_z = (
            widgets.FloatText(
                value=v,
                description=f"a{a}",
                style={"description_width": "25px"},
                layout=L(width="110px"),
            )
            for a, v in zip("xyz", (0.0, 0.0, 1.0))
        )
        self.anim_axis_box = widgets.HBox(
            [self.anim_axis_x, self.anim_axis_y, self.anim_axis_z]
        )
        self.anim_normal_src = widgets.Dropdown(
            options=[],
            description="Normal of",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_frames = widgets.IntSlider(
            value=180,
            min=12,
            max=1440,
            step=6,
            description="Frames",
            continuous_update=False,
            style=D,
            layout=L(width="320px"),
        )
        self.anim_degrees = widgets.FloatText(
            value=360.0,
            description="Degrees",
            style=D,
            layout=L(width="160px"),
        )
        self.anim_use_current_camera = widgets.Checkbox(
            value=True, description="Start from current camera", indent=False
        )

        self.anim_layer_key = widgets.Dropdown(
            options=[("— Select —", None)],
            value=None,
            description="Layer",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_layer_param = widgets.Dropdown(
            options=[("— Select —", None)],
            value=None,
            description="Param",
            style=D,
            layout=L(width="300px"),
        )
        self.anim_range = widgets.FloatRangeSlider(
            value=(0.0, 1.0),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Range",
            continuous_update=False,
            readout_format=".4g",
            style=D,
            layout=L(width="520px"),
        )
        self.anim_points = widgets.IntSlider(
            value=120,
            min=2,
            max=2000,
            description="Frames",
            continuous_update=False,
            style=D,
            layout=L(width="320px"),
        )
        self.anim_fix_range = widgets.Checkbox(
            value=True,
            description="Fix colour range to current view",
            indent=False,
        )
        self.anim_auto_trim = widgets.Checkbox(
            value=False, description="Skip empty frames", indent=False
        )
        self.anim_min_valid = widgets.FloatSlider(
            value=0.01,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Min valid",
            continuous_update=False,
            style=D,
            layout=L(width="260px"),
            tooltip="Slices/planes: minimum fraction of defined pixels",
        )

        # --- condition label (all exports) and condition animation ---
        self.anim_label = widgets.Checkbox(
            value=True,
            description="Condition label",
            indent=False,
            tooltip="Write the condition label on images and animations "
            "(when there are several conditions)",
            layout=L(width="auto"),
        )
        self.anim_label_fmt = widgets.Text(
            value="{label}",
            description="text",
            tooltip="{label}, {index} (from 1) and {n} are replaced",
            style={"description_width": "30px"},
            layout=L(width="200px"),
        )
        self.anim_cond_hold = widgets.IntSlider(
            value=15,
            min=1,
            max=240,
            description="Frames each",
            tooltip="Frames per condition (with FPS: how long each is shown)",
            continuous_update=False,
            style={"description_width": "85px"},
            layout=L(width="320px"),
        )
        self.anim_cond_degrees = widgets.FloatText(
            value=0.0,
            description="Rotate (°)",
            tooltip="Camera rotation (orbit Z) over the whole animation",
            style=D,
            layout=L(width="160px"),
        )
        self.anim_cond_shared = widgets.Checkbox(
            value=True,
            description="Same colour range for all conditions",
            indent=False,
        )
        self.anim_cond_fix_axes = widgets.Checkbox(
            value=True,
            description="Same axes for all conditions",
            indent=False,
        )

        self.anim_status = widgets.HTML("")
        self.anim_progress = widgets.IntProgress(
            value=0, min=0, max=1, layout=L(width="100%")
        )

        def row(*c):
            return widgets.HBox(
                list(c),
                layout=L(
                    flex_flow="row wrap", grid_gap="8px", align_items="center"
                ),
            )

        self.anim_rotation_box = widgets.VBox(
            [
                row(self.anim_rot_type, self.anim_normal_src),
                self.anim_axis_box,
                row(
                    self.anim_frames,
                    self.anim_degrees,
                    self.anim_use_current_camera,
                ),
            ]
        )
        self.anim_layer_box = widgets.VBox(
            [
                row(self.anim_layer_key, self.anim_layer_param),
                row(self.anim_range),
                row(self.anim_points, self.anim_fix_range),
                row(self.anim_auto_trim, self.anim_min_valid),
            ]
        )
        self.anim_cond_box = widgets.VBox(
            [
                row(self.anim_cond_hold, self.anim_cond_degrees),
                row(self.anim_cond_shared, self.anim_cond_fix_axes),
            ]
        )
        self.anim_label_row = row(self.anim_label, self.anim_label_fmt)
        self.anim_panel = widgets.VBox(
            [
                widgets.HTML("<b>Export animation</b>"),
                row(
                    self.anim_master_mode,
                    self.anim_name,
                    self.anim_format,
                    self.anim_save_btn,
                    self.anim_stop_btn,
                    self.anim_html_btn,
                ),
                row(
                    self.anim_fps,
                    self.anim_width,
                    self.anim_height,
                    self.anim_scale,
                    self.anim_backend,
                    self.anim_smooth,
                    self.anim_flat,
                ),
                self.anim_label_row,
                self.anim_rotation_box,
                self.anim_layer_box,
                self.anim_cond_box,
                self.anim_status,
                self.anim_progress,
            ],
            layout=L(width="100%", padding="8px 10px", margin="10px 0 0 0"),
        )

        self.anim_save_btn.on_click(self._on_save_animation_clicked)
        self.anim_stop_btn.on_click(lambda _b: self._stop_export())
        self.anim_html_btn.on_click(self._on_save_html_clicked)
        for wd in (
            self.anim_master_mode,
            self.anim_rot_type,
            self.anim_layer_key,
            self.anim_layer_param,
        ):
            wd.observe(lambda ch: self._sync_anim_ui(ch), names="value")
        self._sync_anim_ui()

    def _anim_refresh_options(self):
        names = list(self._layers)
        self._set_options(
            self.anim_layer_key,
            [("— Select —", None)]
            + [(f"{k} [{self._layers[k]['type']}]", k) for k in names],
            fallback=None,
        )
        opts = []
        for k, s in self._layers.items():
            if s["type"] in ("plane", "clip"):
                opts.append((k, k))
            elif s["type"] == "slice":
                opts.append(
                    (
                        f"{k} (axis {self.frame.phys_axis_to_plot(s['axis_phys'])})",
                        k,
                    )
                )
        self._set_options(self.anim_normal_src, opts)

    def _sync_anim_ui(self, ch=None):
        mode = self.anim_master_mode.value
        owner = ch.get("owner") if isinstance(ch, dict) else None
        self.anim_rotation_box.layout.display = (
            "" if mode == "rotation" else "none"
        )
        self.anim_layer_box.layout.display = "" if mode == "layer" else "none"
        multi = len(getattr(self, "_conditions", [])) > 1
        self.anim_cond_box.layout.display = (
            "" if mode == "conditions" else "none"
        )
        self.anim_label_row.layout.display = "" if multi else "none"
        rot = self.anim_rot_type.value
        self.anim_axis_box.layout.display = "" if rot == "axis" else "none"
        self.anim_normal_src.layout.display = "" if rot == "normal" else "none"
        if owner is self.anim_layer_key:
            k = self.anim_layer_key.value
            params = (
                ANIM_PARAMS.get(self._layers[k]["type"], [])
                if k in self._layers
                else []
            )
            self.anim_layer_param.options = [("— Select —", None)] + [
                (PARAM_LABELS[p], p) for p in params
            ]
            self.anim_layer_param.value = None
        if owner is self.anim_layer_param and self.anim_layer_param.value:
            lo, hi = self._anim_param_bounds(
                self.anim_layer_key.value, self.anim_layer_param.value
            )
            if lo is not None:
                with self.anim_range.hold_trait_notifications():
                    if lo > self.anim_range.max:
                        self.anim_range.max, self.anim_range.min = hi, lo
                    else:
                        self.anim_range.min, self.anim_range.max = lo, hi
                    self.anim_range.step = (
                        1.0
                        if self.anim_layer_param.value == "pos"
                        else (hi - lo) / 500
                    )
                    self.anim_range.value = (lo, hi)
        ready = (
            (mode == "rotation" and rot is not None)
            or (
                mode == "layer"
                and self.anim_layer_key.value is not None
                and self.anim_layer_param.value is not None
            )
            or (mode == "conditions" and multi)
        )
        self.anim_save_btn.disabled = not ready

    def _anim_param_bounds(self, layer, param):
        """(min, max) of an animatable parameter, or (None, None)."""
        if layer not in self._layers:
            return None, None
        s = self._layers[layer]
        if param == "opacity":
            return 0.0, 1.0
        if param == "thr" and "thr" in self._layer_widgets.get(layer, {}):
            t = self._layer_widgets[layer]["thr"]
            return float(t.min), float(t.max)
        if param == "pos" and s["type"] == "slice":
            return 0.0, float(self.frame.n_along(s["axis_phys"]) - 1)
        if param == "offset" and s["type"] in ("plane", "clip"):
            return self._offset_range(s)
        return None, None

    # =========================================================================
    # Live rotation
    # =========================================================================
    def _rotation_axis_plot(self, mode, normal_layer=None):
        if mode in ("orbit_x", "orbit_y", "orbit_z"):
            return np.eye(3)["xyz".index(mode[-1])]
        if mode == "axis":
            return geo.unit(
                [
                    self.anim_axis_x.value,
                    self.anim_axis_y.value,
                    self.anim_axis_z.value,
                ]
            )
        if mode == "normal":
            s = self._layers.get(normal_layer or self.anim_normal_src.value)
            if s is not None:
                n = (
                    s.get("normal")
                    if s["type"] in ("plane", "clip")
                    else np.eye(3)["xyz".index(s["axis_phys"])]
                    if s["type"] == "slice"
                    else None
                )
                if n is not None:
                    return geo.unit(self.frame.phys_to_plot(n))
        return np.array([0.0, 0.0, 1.0])

    _get_rotation_axis_plot = _rotation_axis_plot

    def _current_camera(self):
        cam = self.fig.layout.scene.camera
        cam = cam.to_plotly_json() if cam is not None else {}
        eye = cam.get("eye") or {"x": 1.5, "y": 1.5, "z": 1.5}
        up = cam.get("up") or {"x": 0, "y": 0, "z": 1}
        return (
            np.array([eye.get(k, 1.5) for k in "xyz"], float),
            np.array([up.get(k, 0.0) for k in "xyz"], float),
            cam.get("center") or {"x": 0, "y": 0, "z": 0},
            cam.get("projection"),
        )

    @staticmethod
    def _camera_dict(eye, up, center, projection=None):
        d = dict(
            eye=dict(zip("xyz", map(float, eye))),
            up=dict(zip("xyz", map(float, up))),
            center=center,
        )
        if projection:
            d["projection"] = projection
        return d

    def _start_rotation(self):
        if self._rotation_task is not None and not self._rotation_task.done():
            return

        async def rotate():
            eye0, up0, center, proj = self._current_camera()
            ang = 0.0
            while self.rotate_toggle.value:
                ang += math.radians(2.0)
                R = geo.rodrigues(
                    self._rotation_axis_plot(
                        self.anim_rot_type.value or "orbit_z"
                    ),
                    ang,
                )
                self.fig.layout.scene.camera = self._camera_dict(
                    R @ eye0, R @ up0, center, proj
                )
                await asyncio.sleep(0.05)

        try:
            self._rotation_task = asyncio.get_running_loop().create_task(
                rotate()
            )
        except RuntimeError:
            self._rotation_task = None

    def _stop_rotation(self):
        if self._rotation_task is not None:
            self._rotation_task.cancel()
            self._rotation_task = None

    # =========================================================================
    # Still image
    # =========================================================================
    def _export_base_json(self, width, height, label=None, label_format=None):
        """Figure JSON of the current view for exports (helper traces
        dropped); with several conditions, the condition label is added
        unless ``label=False`` (default: the panel's *Condition label*)."""
        d = self.fig.to_plotly_json()
        d["data"] = [
            t for t in d["data"] if not str(t.get("name", "")).startswith("__")
        ]
        d["layout"] = dict(
            d.get("layout") or {},
            autosize=False,
            width=int(width),
            height=int(height),
        )
        ann = self._label_annotation(self._cond_index, label, label_format)
        if ann is not None:
            d["layout"]["annotations"] = list(
                d["layout"].get("annotations") or []
            ) + [ann]
        return d

    def _label_annotation(self, index, label=None, label_format=None):
        """Paper annotation with the label of condition ``index`` (or None)."""
        show = self.anim_label.value if label is None else bool(label)
        if not show or len(self._conditions) < 2:
            return None
        fmt = label_format or self.anim_label_fmt.value or "{label}"
        try:
            text = fmt.format(
                label=self._cond_labels[index],
                index=index + 1,
                n=len(self._conditions),
            )
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(
                f"Bad label text {fmt!r}: use {{label}}, {{index}}, {{n}}"
            ) from e
        dark = self._dark()
        return dict(
            text=f"<b>{text}</b>",
            x=0.02,
            y=0.97,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(
                family="DejaVu Sans Bold",
                size=int(self.fontsize * 1.8),
                color="#e6eefc" if dark else "#000000",
            ),
            name="__condition_label",
        )

    def _resolve_conditions(self, conditions):
        """Indices from None/'all', ints, labels or a mix."""
        n = len(self._conditions)
        if conditions is None or conditions == "all":
            return list(range(n))
        if isinstance(conditions, (int, str)):
            conditions = [conditions]
        out = []
        for c in conditions:
            if isinstance(c, str) and c in self._cond_labels:
                out.append(self._cond_labels.index(c))
            else:
                i = int(c)
                if not 0 <= i < n:
                    raise IndexError(
                        f"Condition {c!r} out of range (0..{n - 1})."
                    )
                out.append(i)
        if not out:
            raise ValueError("No condition to animate.")
        return out

    def _render_options(self, smooth=None, flat=None):
        return {
            "smooth": bool(
                self.anim_smooth.value if smooth is None else smooth
            ),
            "flat": bool(self.anim_flat.value if flat is None else flat),
        }

    def write_image(
        self,
        out_path,
        width=None,
        height=None,
        scale=1,
        rendering_backend=None,
        smooth=None,
        flat=None,
        label=None,
    ):
        """Save the current view as PNG.

        ``smooth`` / ``flat`` (software renderer) default to the export
        panel's *Smooth edges* / *Flat colours* checkboxes; ``label`` (several
        conditions) to its *Condition label* checkbox.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = (
            int(width or self.export_width),
            int(height or self.export_height),
        )
        backend = str(rendering_backend or self.rendering_backend).lower()
        self._rebuild()
        png = render_png(
            self._export_base_json(width, height, label=label),
            width,
            height,
            int(scale),
            backend,
            self.fontsize,
            self._render_options(smooth, flat),
        )
        if not png.startswith(b"\x89PNG"):
            raise RuntimeError(
                f"The {backend!r} renderer did not return PNG data."
            )
        out_path.write_bytes(png)
        return out_path

    def export_html(
        self,
        path,
        conditions=False,
        plotlyjs="embed",
        shared_range=True,
        label=None,
        label_format=None,
    ):
        """Save the current view as an interactive HTML page (Plotly).

        The page opens in any browser without Python: rotate, zoom, hover,
        toggle traces. ``conditions=True`` (or a list of indices/labels)
        adds a slider and a play button over the conditions, with the same
        colour range (``shared_range``) and axes in every condition.
        ``plotlyjs="embed"`` (default) works offline (~3.5 MB larger);
        ``"cdn"`` makes a small file that needs internet to open.
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        path = Path(path).expanduser()
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_name(path.name + ".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        js = {"embed": True, "cdn": "cdn"}.get(plotlyjs, plotlyjs)
        self._rebuild()
        multi = len(self._conditions) > 1 and conditions not in (False, None)
        if not multi:
            d = self._export_base_json(
                self.export_width, self.export_height, label, label_format
            )
            d["layout"].update(autosize=True, width=None, height=None)
            fig = go.Figure(d)
        else:
            conds = self._resolve_conditions(
                None if conditions is True else conditions
            )
            base, frames = self._condition_frames(
                conds, bool(shared_range), True, 800, 600
            )
            idx = sorted(frames[conds[0]])
            data = list(base["data"])
            for i in idx:
                data[i] = frames[conds[0]][i]
            layout = dict(base["layout"], autosize=True)
            layout.pop("width", None)
            layout.pop("height", None)
            anns = list(layout.get("annotations") or [])

            def ann(c):
                a = self._label_annotation(c, label, label_format)
                return anns + ([a] if a else [])

            layout["annotations"] = ann(conds[0])
            names = [self._cond_labels[c] for c in conds]
            fr = [
                go.Frame(
                    name=nm,
                    data=[frames[c][i] for i in idx],
                    traces=idx,
                    layout={"annotations": ann(c)},
                )
                for c, nm in zip(conds, names)
            ]
            step = {
                "frame": {"duration": 0, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0},
            }
            layout["sliders"] = [
                {
                    "active": 0,
                    "currentvalue": {"prefix": "condition: "},
                    "pad": {"t": 30},
                    "steps": [
                        {
                            "label": nm,
                            "method": "animate",
                            "args": [[nm], step],
                        }
                        for nm in names
                    ],
                }
            ]
            play = dict(step, frame={"duration": 800, "redraw": True})
            layout["updatemenus"] = [
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.0,
                    "y": 0.0,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 60},
                    "buttons": [
                        {
                            "label": "▶ play",
                            "method": "animate",
                            "args": [None, dict(play, fromcurrent=True)],
                        },
                        {
                            "label": "❚❚",
                            "method": "animate",
                            "args": [[None], step],
                        },
                    ],
                }
            ]
            fig = go.Figure(data=data, layout=layout, frames=fr)
        pio.write_html(
            fig,
            str(path),
            include_plotlyjs=js,
            full_html=True,
            auto_play=False,
            config={"displaylogo": False, "responsive": True},
        )
        return path

    # =========================================================================
    # Animation export
    # =========================================================================
    def _stop_export(self):
        """Stop the running export and drop the queued ones."""
        for tok in self._export_tokens:
            tok["cancel"] = True
        self._anim_cancel = True
        self.rotate_toggle.value = False
        self.anim_status.value = "Stopping…"
        self.anim_progress.bar_style = "warning"

    def _on_stop_rotation_export_clicked(self, _b=None):
        return self._stop_export()

    def _on_save_animation_clicked(self, _b=None):
        try:
            if self.anim_master_mode.value == "rotation":
                self.start_rotation_export()
            elif self.anim_master_mode.value == "layer":
                self.start_layer_export(
                    self.anim_layer_key.value,
                    self.anim_layer_param.value,
                    self.anim_range.value,
                )
            elif self.anim_master_mode.value == "conditions":
                self.start_condition_export()
        except Exception as e:
            self.anim_status.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    def _on_save_html_clicked(self, _b=None):
        try:
            self.anim_status.value = "Writing HTML…"
            out = output_path(self.anim_name.value, "html", "view")
            out = self.export_html(out, conditions=len(self._conditions) > 1)
            self.anim_status.value = f"Saved: <code>{out.resolve()}</code>"
        except Exception as e:
            self.anim_status.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    def start_rotation_export(
        self,
        mode=None,
        axis=None,
        frames=None,
        fps=None,
        fmt=None,
        name=None,
        normal_layer=None,
        degrees=None,
        queue=False,
        smooth=None,
        flat=None,
    ):
        """Export a camera rotation (MP4 / GIF / PNG frames).

        Returns the asyncio task (in a notebook) or the output path.
        ``queue=True`` waits for a running export instead of stopping it.
        """
        mode = mode or self.anim_rot_type.value or "orbit_z"
        if axis is not None:
            (
                self.anim_axis_x.value,
                self.anim_axis_y.value,
                self.anim_axis_z.value,
            ) = map(float, axis)
        n = int(frames or self.anim_frames.value)
        deg = float(self.anim_degrees.value if degrees is None else degrees)
        ax = self._rotation_axis_plot(mode, normal_layer)
        W, H = int(self.anim_width.value), int(self.anim_height.value)
        self.rotate_toggle.value = False
        base = self._export_base_json(W, H)
        if self.anim_use_current_camera.value:
            eye0, up0, center, proj = self._current_camera()
        else:
            eye0, up0, center, proj = (
                np.array([1.5, 1.5, 1.5]),
                np.array([0, 0, 1.0]),
                {"x": 0, "y": 0, "z": 0},
                None,
            )

        def produce(k):
            R = geo.rodrigues(ax, math.radians(deg) * k / max(1, n))
            return {
                "camera": self._camera_dict(R @ eye0, R @ up0, center, proj)
            }

        return self._launch_export(
            n,
            produce,
            fmt,
            fps,
            name,
            "rotation",
            queue=queue,
            options=self._render_options(smooth, flat),
            base=base,
        )

    def start_layer_export(
        self,
        layer,
        param,
        value_range,
        points=None,
        fps=None,
        fmt=None,
        name=None,
        auto_trim=None,
        min_valid=None,
        queue=False,
        smooth=None,
        flat=None,
    ):
        """Sweep a layer parameter (opacity, thr, pos, offset) and export it.

        Returns the asyncio task (in a notebook) or the output path.
        ``queue=True`` waits for a running export instead of stopping it.
        """
        layer = self._resolve_layer_name(layer)
        param = {"iso": "thr", "op": "opacity", "slice_pos": "pos"}.get(
            param, param
        )
        if param not in ANIM_PARAMS.get(self._layers[layer]["type"], []):
            raise ValueError(
                f"Parameter {param!r} cannot be animated for a {self._layers[layer]['type']} layer."
            )
        if layer not in self._trace_order:
            raise RuntimeError(
                f"Layer '{layer}' must be visible to be animated."
            )
        lo, hi = sorted(map(float, value_range))
        blo, bhi = self._anim_param_bounds(layer, param)
        if blo is not None:
            lo, hi = max(lo, blo), min(hi, bhi)
        if hi < lo:
            raise ValueError("Empty range after clamping to the valid bounds.")
        n = int(points or self.anim_points.value)
        trim = (
            self.anim_auto_trim.value if auto_trim is None else bool(auto_trim)
        )
        min_valid = float(
            self.anim_min_valid.value if min_valid is None else min_valid
        )
        values = np.linspace(lo, hi, max(2, n))
        W, H = int(self.anim_width.value), int(self.anim_height.value)
        base = self._export_base_json(W, H)
        idx = self._trace_order.index(layer)
        spec0, style0 = dict(self._layers[layer]), self._style(layer)
        live = base["data"][idx]
        if self.anim_fix_range.value and "cmin" in live and param != "opacity":
            style0.update(
                auto_range=False,
                range=(float(live["cmin"]), float(live["cmax"])),
            )
        ci = self._cbar_index_of(layer)

        def produce(k):
            spec, style, v = dict(spec0), dict(style0), float(values[k])
            if param == "opacity":
                style["op"] = v
            elif param == "thr":
                style["thr"] = v
            elif param == "pos":
                spec["pos"] = int(round(v))
            else:
                spec["offset"] = v
            tr, info = self._build_trace(
                layer, spec=spec, style=style, cbar_index=ci, sync=False
            )
            if trim and (
                info.get("empty") or info.get("valid", 1.0) < min_valid
            ):
                return None
            return {"traces": {idx: tr.to_plotly_json()}}

        return self._launch_export(
            len(values),
            produce,
            fmt,
            fps,
            name,
            "layer_anim",
            queue=queue,
            options=self._render_options(smooth, flat),
            base=base,
        )

    def start_condition_export(
        self,
        conditions=None,
        hold=None,
        degrees=None,
        fps=None,
        fmt=None,
        name=None,
        shared_range=None,
        fix_axes=None,
        label=None,
        label_format=None,
        queue=False,
        smooth=None,
        flat=None,
    ):
        """Export the visible layers through the conditions (MP4 / GIF / PNG).

        ``conditions``: indices or labels (default: all, in order); ``hold``:
        frames per condition; ``degrees``: optional camera orbit (Z) over the
        whole animation; ``shared_range``: one colour range for all
        conditions (layers with a manual range keep it); ``fix_axes``: same
        axis ranges in every frame, so the camera does not jump; ``label`` /
        ``label_format``: condition label written on each frame (``{label}``,
        ``{index}``, ``{n}``). Frames repeated for ``hold`` are rendered
        once. Returns the asyncio task (in a notebook) or the output path.
        """
        if len(self._conditions) < 2:
            raise RuntimeError("There is only one condition.")
        conds = self._resolve_conditions(conditions)
        hold = max(1, int(self.anim_cond_hold.value if hold is None else hold))
        deg = float(
            self.anim_cond_degrees.value if degrees is None else degrees
        )
        shared = (
            self.anim_cond_shared.value
            if shared_range is None
            else bool(shared_range)
        )
        fix = (
            self.anim_cond_fix_axes.value
            if fix_axes is None
            else bool(fix_axes)
        )
        W, H = int(self.anim_width.value), int(self.anim_height.value)
        self.rotate_toggle.value = False
        base, frames = self._condition_frames(conds, shared, fix, W, H)
        annotations = list(base["layout"].get("annotations") or [])
        eye0, up0, center, proj = self._current_camera()
        n = len(conds) * hold

        def produce(k):
            c = conds[k // hold]
            if k % hold and not deg:
                return REPEAT
            delta = {"traces": frames[c]}
            ann = self._label_annotation(c, label, label_format)
            delta["annotations"] = annotations + ([ann] if ann else [])
            if deg:
                R = geo.rodrigues(
                    np.array([0.0, 0.0, 1.0]), math.radians(deg) * k / n
                )
                delta["camera"] = self._camera_dict(
                    R @ eye0, R @ up0, center, proj
                )
            return delta

        return self._launch_export(
            n,
            produce,
            fmt,
            fps,
            name,
            "conditions",
            queue=queue,
            options=self._render_options(smooth, flat),
            base=base,
        )

    def _condition_frames(self, conds, shared, fix, W, H):
        """Base scene and ``{condition: {trace index: trace json}}`` of the
        visible layers; optional shared colour range and fixed axes."""
        self._rebuild()
        base = self._export_base_json(W, H, label=False)
        names = [
            (i, t.get("name"))
            for i, t in enumerate(base["data"])
            if t.get("name") in self._layers
        ]
        styles = {}
        for _, nm in names:
            st = self._style(nm)
            if shared and self._range_mode(st) != "manual":
                st["range_all"] = True
            styles[nm] = st
        frames = {}  # condition -> {trace index: trace json}
        for c in dict.fromkeys(conds):
            with self._in_condition(c):
                frames[c] = {
                    i: self._build_trace(
                        nm,
                        style=styles[nm],
                        cbar_index=self._cbar_index_of(nm),
                        sync=False,
                    )[0].to_plotly_json()
                    for i, nm in names
                }
        if fix:
            lo, hi = _trace_bounds(
                [tr for f in frames.values() for tr in f.values()]
                + [
                    t
                    for i, t in enumerate(base["data"])
                    if i not in dict(names)
                ]
            )
            if lo is not None:
                pad = (hi - lo) / 32  # Plotly's autorange padding
                scene = dict(base["layout"].get("scene") or {})
                for k, a in enumerate("xyz"):
                    ax = dict(scene.get(f"{a}axis") or {})
                    ax.update(
                        range=[float(lo[k] - pad[k]), float(hi[k] + pad[k])],
                        autorange=False,
                    )
                    scene[f"{a}axis"] = ax
                base["layout"] = dict(base["layout"], scene=scene)
        return base, frames

    def _launch_export(
        self,
        n,
        produce,
        fmt,
        fps,
        name,
        stem,
        queue=False,
        options=None,
        base=None,
    ):
        fmt = (fmt or self.anim_format.value or "mp4").lower()
        fps = int(fps or self.anim_fps.value)
        out = output_path(name or self.anim_name.value, fmt, stem)
        prev = self._export_task
        prev = prev if prev is not None and not prev.done() else None
        if not queue:
            for tok in self._export_tokens:  # stop running/queued exports
                tok["cancel"] = True
            prev = None
        token = {"cancel": False}
        self._export_token = token
        self._export_tokens = [
            t for t in self._export_tokens if not t["cancel"]
        ] + [token]
        self._anim_cancel = False
        coro = self._export_coro(
            n, produce, out, fmt, fps, token, options, base
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        async def after_previous():
            if prev is not None:
                try:
                    await prev
                except BaseException:
                    pass
            if token["cancel"]:
                coro.close()
                return None
            return await coro

        task = loop.create_task(after_previous())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._export_task = task
        return task

    async def _export_coro(
        self, n, produce, out, fmt, fps, token, options=None, base=None
    ):
        t0 = time.perf_counter()
        W, H, S = (
            int(self.anim_width.value),
            int(self.anim_height.value),
            int(self.anim_scale.value),
        )
        (
            self.anim_progress.min,
            self.anim_progress.max,
            self.anim_progress.value,
        ) = 0, int(n), 0
        self.anim_progress.bar_style = "info"
        self.anim_status.value = "Checking the rendering backend…"
        sink = None
        try:
            backend = await asyncio.to_thread(lambda: self.rendering_backend)
            sink = FrameSink(out, fmt, fps)
            self.anim_status.value = (
                f"Rendering with {backend} ({self.rendering_mode})…"
            )

            def progress(i, total):
                self.anim_progress.value = i
                self.anim_status.value = f"Rendering {i}/{total}…"

            def cancelled():
                return token["cancel"] or (
                    self._anim_cancel and token is self._export_token
                )

            async def render(mode):
                return await run_ordered_export(
                    n,
                    produce,
                    render_png,
                    (W, H, S, backend, self.fontsize, options),
                    sink,
                    mode=mode,
                    workers=self.render_workers,
                    in_flight=self.render_in_flight,
                    is_cancelled=cancelled,
                    on_progress=progress,
                    base=base,
                )

            try:
                written = await render(self.rendering_mode)
            except BrokenProcessPool:
                # Typical in a plain script without an
                # ``if __name__ == "__main__":`` guard (spawned workers
                # re-import it). Fall back to sequential rendering.
                warnings.warn(
                    "The render process pool failed; rendering "
                    "sequentially. In a script, put the code under "
                    "'if __name__ == \"__main__\":' or use "
                    "rendering_mode='safe'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sink.close(encode=False)
                sink = FrameSink(out, fmt, fps)
                written = await render("safe")
            await asyncio.to_thread(sink.close, True)
            dt = time.perf_counter() - t0
            stopped = " (stopped)" if token["cancel"] else ""
            self.anim_status.value = (
                f"Saved{stopped}: <code>{out.resolve()}</code> | {written} frames in {_fmt_seconds(dt)} "
                f"({written / dt if dt > 0 else 0:.2f} frames/s)"
            )
            self.anim_progress.bar_style = "warning" if stopped else "success"
            return out
        except BaseException as e:
            if sink is not None:
                try:
                    sink.close(encode=False)
                except Exception:
                    pass
            self.anim_progress.bar_style = "danger"
            self.anim_status.value = f"<b style='color:#d62728'>Export failed: {type(e).__name__}: {e}</b>"
            raise

    async def wait_for_export(self):
        """Wait for the running (and queued) exports; returns the last output."""
        out = None
        while self._export_task is not None:
            task = self._export_task
            try:
                out = await task
            finally:
                if self._export_task is task and task.done():
                    self._export_task = None
        return out

    async def animate_layer_and_wait(self, **spec):
        await self.run_actions_async([{"action": "animate_layer", **spec}])
        return await self.wait_for_export()

    async def animate_rotation_and_wait(self, **spec):
        await self.run_actions_async([{"action": "animate_rotation", **spec}])
        return await self.wait_for_export()
