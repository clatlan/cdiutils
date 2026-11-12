"""Widget construction and callbacks for :class:`MultiVolumeViewer` (mixin)."""

from contextlib import contextmanager

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from . import analysis as mva
from . import geometry as geo

LIGHT_DEFAULTS = dict(
    ambient=0.6, diffuse=0.4, specular=0.5, roughness=0.2, fresnel=0.1
)
SOLID = (
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
NAN_OPTIONS = [
    ("Do nothing (holes)", "none"),
    ("Replace with mean", "mean"),
    ("Replace with zero", "zero"),
    ("Replace with min", "min"),
    ("Replace with max", "max"),
]
CSS = """
<style>
.mv-root { gap: 12px; }
.mv-panel { border: 1px solid rgba(127,127,127,0.35); border-radius: 6px; box-sizing: border-box; }
.mv-section .widget-toggle-button button, .mv-section button.jupyter-button.mv-header { text-align: left; }
.mv-root.mv-dark .mv-panel { background: #0b1f3a; }
.mv-root.mv-dark .mv-panel, .mv-root.mv-dark .mv-panel * { color: #e6eefc; }
.mv-root.mv-dark .mv-panel input, .mv-root.mv-dark .mv-panel textarea, .mv-root.mv-dark .mv-panel select,
.mv-root.mv-dark .mv-panel button, .mv-root.mv-dark .mv-panel select option {
  background: #1b2431 !important; color: #e6eefc !important; border: 1px solid #2f3b4d !important; }
.mv-root.mv-dark .mv-panel button:hover { background: #223046 !important; }
.mv-root.mv-dark .mv-panel button.mod-active { background: #2a3a55 !important; }
.mv-root.mv-dark .mv-panel table td, .mv-root.mv-dark .mv-panel table th { border-color: #2f3b4d; }
.mv-panel { background: var(--jp-layout-color0, transparent); }
.mv-toolbar { gap: 4px; padding: 4px 6px; border: 1px solid rgba(127,127,127,0.35); border-radius: 6px;
  margin-bottom: 6px; align-items: center; }
.mv-toolbar .widget-inline-hbox { margin: 0; }
.mv-status { font-size: 12px; opacity: 0.8; gap: 14px; padding: 2px 8px;
  border-top: 1px solid rgba(127,127,127,0.25); }
.mv-row { border-bottom: 1px solid rgba(127,127,127,0.15); align-items: center; }
.mv-row-sel { background: rgba(47,111,219,0.12); }
.mv-row button { text-align: left; }
.mv-tile button { height: 52px; }
.mv-root.mv-dark .mv-toolbar, .mv-root.mv-dark .mv-status { background: #0b1f3a; color: #e6eefc; }
.mv-root.mv-dark .mv-toolbar *, .mv-root.mv-dark .mv-status * { color: #e6eefc; }
.mv-root.mv-dark .mv-toolbar button, .mv-root.mv-dark .mv-toolbar select {
  background: #1b2431 !important; border: 1px solid #2f3b4d !important; }
.mv-root.mv-dark .mv-panel .widget-tab-contents, .mv-root.mv-dark .mv-panel .lm-TabBar,
.mv-root.mv-dark .mv-panel .lm-TabBar-tab, .mv-root.mv-dark .mv-panel .p-TabBar-tab {
  background: #0b1f3a !important; color: #e6eefc !important; border-color: #2f3b4d !important; }
.mv-root.mv-dark .mv-panel .lm-TabBar-tab.lm-mod-current { background: #13294b !important; }
.mv-table { border-collapse: collapse; font-size: 12px; }
.mv-table td, .mv-table th { border: 1px solid rgba(127,127,127,0.4); padding: 2px 6px; text-align: right;
  white-space: nowrap; }
.mv-table th:first-child, .mv-table td:first-child { text-align: left; }
</style>
"""


TAB_TITLES = ("Layers", "Add", "Analyze", "Export", "Script", "Help")

#: Export size presets: (width, height, scale). Tunable per viewer with
#: ``figure_presets=`` or :meth:`set_figure_preset`.
FIGURE_PRESETS = {
    "screen": None,  # the viewer's export size
    "paper": (1200, 900, 2),
    "slide": (1600, 900, 1),
    "poster": (2400, 1800, 1),
}


def _gradient(colorscale, n=8):
    cs = list(colorscale)
    if len(cs) > n:  # a few stops are enough for a 40-px swatch
        cs = [cs[round(i * (len(cs) - 1) / (n - 1))] for i in range(n)]
    stops = ", ".join(f"{c} {100 * float(p):.0f}%" for p, c in cs)
    return f"linear-gradient(90deg, {stops})"


def _lay(width="95%", **kw):
    return widgets.Layout(width=width, **kw)


class UIMixin:
    _DESC = {"description_width": "80px"}

    # =========================================================================
    # Top-level construction
    # =========================================================================
    def _build_ui(self):
        self.cmap_options = list(SOLID) + sorted(set(plt.colormaps()))
        self.theme_toggle = widgets.ToggleButton(
            value=False, description="Dark theme", layout=_lay("auto")
        )
        self.rotate_toggle = widgets.ToggleButton(
            value=False, description="Rotate", layout=_lay("auto")
        )
        self.grid_toggle = widgets.Checkbox(
            value=True, description="Grid", indent=False, layout=_lay("auto")
        )
        self.reset_cam_btn = widgets.Button(
            description="Reset camera", layout=_lay("auto")
        )
        self._build_figure()

        self.visible_keys_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                width="100%",
                border="1px solid rgba(127,127,127,0.35)",
                margin="4px 0 8px 0",
            ),
        )
        self._rows = {}
        self.cond_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            description="Condition",
            continuous_update=False,
            style=self._DESC,
            layout=_lay("70%"),
        )
        self.cond_label = widgets.HTML("")
        self.cond_row = widgets.HBox(
            [self.cond_slider, self.cond_label], layout=_lay(display="none")
        )

        self.edit_key = widgets.Dropdown(
            options=[("— choose layer —", None)],
            value=None,
            description="Edit",
            style={"description_width": "40px"},
            layout=_lay("48%"),
        )
        self.rename_layer_text = widgets.Text(
            placeholder="New layer name", layout=_lay("48%")
        )
        self.rename_layer_btn = widgets.Button(
            description="Rename", layout=_lay("48%")
        )
        self.delete_layer_btn = widgets.Button(
            description="Delete (+ dependents)",
            button_style="danger",
            layout=_lay("48%"),
        )
        self.layers_box = widgets.VBox([], layout=_lay())

        self.add_layer_kind = widgets.Dropdown(
            options=[
                ("— choose —", None),
                ("Slice", "slice"),
                ("Plane", "plane"),
                ("Clip", "clip"),
                ("Difference", "difference"),
                ("Surface shell / core", "region"),
            ],
            value=None,
            description="Add",
            style={"description_width": "40px"},
            layout=_lay("48%"),
        )
        self.add_layer_name = widgets.Text(
            placeholder="name (optional)", layout=_lay("48%")
        )
        self.add_layer_source = widgets.Dropdown(
            options=[],
            description="Source",
            style={"description_width": "50px"},
            layout=_lay("60%"),
        )
        self.add_layer_btn = widgets.Button(
            description="Create layer",
            button_style="primary",
            layout=_lay("35%"),
        )
        self.add_layer_msg = widgets.HTML("")
        self.add_clip_source = self.add_layer_source  # former name
        CD = {"description_width": "60px"}
        self.add_slice_axis = widgets.Dropdown(
            options=["x", "y", "z"],
            value="z",
            description="Axis",
            style=CD,
            layout=_lay("48%"),
        )
        self.add_slice_pos = widgets.IntSlider(
            value=0,
            min=0,
            max=1,
            description="Pos",
            continuous_update=False,
            style=CD,
            layout=_lay("95%"),
        )
        self.add_slice_thickness = widgets.IntSlider(
            value=0,
            min=0,
            max=1,
            description="± thick",
            continuous_update=False,
            style=CD,
            layout=_lay("95%"),
        )
        self.add_plane_nx, self.add_plane_ny, self.add_plane_nz = (
            widgets.FloatText(
                value=v,
                description=f"n{a}",
                style={"description_width": "25px"},
                layout=_lay("31%"),
            )
            for a, v in zip("xyz", (0.0, 0.0, 1.0))
        )
        self.add_clip_side = widgets.Dropdown(
            options=[("Keep +n side", "up"), ("Keep -n side", "down")],
            value="up",
            description="Side",
            style=CD,
            layout=_lay("60%"),
        )
        self._add_slice_row = widgets.VBox(
            [self.add_slice_axis, self.add_slice_pos, self.add_slice_thickness]
        )
        self._add_normal_row = widgets.VBox(
            [
                widgets.HTML("Normal (plane through the centre):"),
                widgets.HBox(
                    [self.add_plane_nx, self.add_plane_ny, self.add_plane_nz],
                    layout=_lay(),
                ),
            ]
        )
        self.add_params_box = widgets.VBox([])

        create_box = widgets.VBox(
            [
                widgets.HBox(
                    [self.add_layer_kind, self.add_layer_name], layout=_lay()
                ),
                self.add_layer_source,
                self.add_params_box,
                widgets.HBox([self.add_layer_btn], layout=_lay()),
                self.add_layer_msg,
                widgets.HTML(
                    "<i>Fine-tune the new layer in the layer panel above.</i>"
                ),
            ]
        )

        tool_sections = self._build_tools_ui()
        self._build_anim_ui()

        help_sec, probe_sec, facet_sec, reg_sec, ortho_sec, state_sec = (
            tool_sections
        )
        self.anim_panel.add_class("mv-panel")
        self.toolbar = self._build_toolbar()
        self.status_bar = self._build_status_bar()

        # --- tabs of the side panel ---
        self.edit_key.layout.display = "none"  # the layer list selects
        self.rename_layer_text.layout.width = "50%"
        self.rename_layer_btn.layout.width = "auto"
        self.delete_layer_btn.layout.width = "auto"
        self.delete_layer_btn.description = "Delete"
        layers_tab = widgets.VBox(
            [
                self.visible_keys_box,
                widgets.HBox(
                    [
                        self.edit_key,
                        self.rename_layer_text,
                        self.rename_layer_btn,
                        self.delete_layer_btn,
                    ],
                    layout=_lay("100%", align_items="center"),
                ),
                self.layers_box,
            ],
            layout=_lay("100%"),
        )
        add_tab = widgets.VBox(
            [self._build_add_tiles(), create_box], layout=_lay("100%")
        )
        analyze_tab = widgets.VBox(
            [probe_sec, facet_sec, reg_sec, ortho_sec], layout=_lay("100%")
        )
        export_tab = widgets.VBox(
            [self.anim_panel, state_sec], layout=_lay("100%")
        )
        script_tab = self._build_script_tab()
        help_tab = widgets.VBox([help_sec], layout=_lay("100%"))
        self.tabs = widgets.Tab(
            [
                layers_tab,
                add_tab,
                analyze_tab,
                export_tab,
                script_tab,
                help_tab,
            ],
            layout=_lay("100%"),
        )
        for i, t in enumerate(TAB_TITLES):
            self.tabs.set_title(i, t)
        # sections inside tabs start open
        for sec in (help_sec, probe_sec):
            sec.children[0].value = True

        self.right_panel = widgets.VBox(
            [self.tabs],
            layout=widgets.Layout(
                flex=f"0 0 {self.panel_width}",
                width=self.panel_width,
                padding="0",
                overflow="auto",
                max_height=f"{int(self.figsize[1] * 160)}px",
            ),
        )
        self.right_panel.add_class("mv-panel")
        self.busy_label = widgets.HTML("", layout=widgets.Layout(width="auto"))
        self.status_bar.children = (*self.status_bar.children, self.busy_label)
        left = widgets.VBox(
            [self.fig, self.status_bar, self.ortho_box],
            layout=widgets.Layout(
                flex="1 1 0%", min_width="0px", padding="0 6px"
            ),
        )
        side = [left, self.right_panel]
        if self.panel_side == "left":
            side = side[::-1]
        self.body = widgets.HBox(
            side,
            layout=widgets.Layout(
                width="100%", align_items="flex-start", grid_gap="10px"
            ),
        )
        self.children = [widgets.HTML(CSS), self.toolbar, self.body]
        self.layout = widgets.Layout(
            display="flex", flex_flow="column", width="100%"
        )
        self.add_class("mv-root")
        self._wire_global_callbacks()

    def _build_figure(self):
        ax = self._axis_layout()
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            template="plotly_white",
            dragmode="orbit",
            autosize=True,
            height=int(self.figsize[1] * 90),
            margin=dict(l=0, r=0, t=0, b=0),
            font=self._bold_font(),
            scene=dict(
                xaxis=ax | {"title_text": self._axis_label("X")},
                yaxis=ax | {"title_text": self._axis_label("Y")},
                zaxis=ax | {"title_text": self._axis_label("Z")},
            ),
        )

    def _axis_layout(self):
        dark = self._dark() if hasattr(self, "theme_toggle") else False
        return {
            "title": {"font": self._bold_font(self.fontsize * 1.5)},
            "tickfont": self._bold_font(self.fontsize),
            "showline": True,
            "linewidth": 3,
            "linecolor": "#e6eefc" if dark else "#000000",
            "showgrid": bool(self.grid_toggle.value)
            if hasattr(self, "grid_toggle")
            else True,
            "gridwidth": 2,
            "gridcolor": "rgba(200,200,200,0.35)"
            if dark
            else "rgba(0,0,0,0.2)",
            "zeroline": False,
            "ticklen": 18,
            "tickwidth": 2,
            "nticks": 4,
            "showexponent": "all",
            "exponentformat": "power",
        }

    # =========================================================================
    # Toolbar, status bar, layer list, Add tiles, Script tab
    # =========================================================================
    def _build_toolbar(self):
        B = widgets.Button
        self.reset_cam_btn.description = "⟲ Reset"
        self.reset_cam_btn.tooltip = "Reset the camera"
        self.view_btns = {}
        for v in ("iso", "x", "y", "z"):
            b = B(description=v, layout=_lay("auto"), tooltip=f"View {v}")
            b.on_click(
                lambda _b, v=v: self._act_camera(v, None, None, None, 1.0)
            )
            self.view_btns[v] = b
        self.theme_toggle.description = "Dark"
        self.rotate_toggle.description = "⟳ Rotate"
        self.cond_slider.layout.width = "260px"
        self.cond_slider.style = {"description_width": "65px"}
        presets = list(self.figure_presets)
        self.fig_preset = widgets.Dropdown(
            options=presets,
            value=self._figure_preset
            if self._figure_preset in presets
            else presets[0],
            description="Figure",
            style={"description_width": "45px"},
            layout=_lay("150px"),
            tooltip="Export size preset (width x height, scale)",
        )
        self.fig_preset.observe(
            lambda ch: self.set_figure_preset(ch["new"]), names="value"
        )
        self.png_btn = B(
            description="📷 PNG",
            layout=_lay("auto"),
            tooltip="Save the view as PNG ('Save as' name of the Export tab)",
        )
        self.png_btn.on_click(self._on_png_clicked)
        self.html_btn = B(
            description="⇩ HTML",
            layout=_lay("auto"),
            tooltip="Interactive HTML page of the view",
        )
        self.html_btn.on_click(self._on_save_html_clicked)
        self.panel_toggle = widgets.ToggleButton(
            value=True,
            description="Panel",
            icon="columns",
            layout=_lay("auto"),
            tooltip="Show / hide the side panel",
        )
        self.panel_toggle.observe(self._on_panel_toggle, names="value")
        sep = widgets.HTML(
            "<span style='opacity:0.3'>│</span>", layout=_lay("auto")
        )
        bar = widgets.HBox(
            [
                self.reset_cam_btn,
                *self.view_btns.values(),
                widgets.HTML(sep.value),
                self.grid_toggle,
                self.theme_toggle,
                self.rotate_toggle,
                widgets.HTML(sep.value),
                self.cond_row,
                self.fig_preset,
                self.png_btn,
                self.html_btn,
                widgets.HBox([], layout=_lay("auto", flex="1 1 auto")),
                self.panel_toggle,
            ],
            layout=_lay("100%", flex_flow="row wrap"),
        )
        bar.add_class("mv-toolbar")
        self.cond_row.layout.width = "auto"
        return bar

    def _build_status_bar(self):
        self.status_html = widgets.HTML("")
        bar = widgets.HBox([self.status_html], layout=_lay("100%"))
        bar.add_class("mv-status")
        return bar

    def _update_status(self):
        if not hasattr(self, "status_html") or self.frame is None:
            return
        vis = self._visible_names()
        vs = "×".join(f"{v:g}" for v in self.voxel_size)
        parts = [
            f"{len(self._layers)} layers · <b>{len(vis)} visible</b>",
            f"voxel <b>{vs}</b> {self.unit or ''}",
        ]
        if len(self._conditions) > 1:
            parts.append(
                f"condition <b>{self._cond_labels[self._cond_index]}</b>"
            )
        try:
            info = self.cache_info()
            mb = info["isosurfaces_MB"] + info["slices_MB"]
            parts.append(
                f"cache <b>{mb:.1f} MB</b>"
                if mb < 10
                else f"cache <b>{mb:.0f} MB</b>"
            )
        except Exception:
            pass
        self.status_html.value = " &nbsp;·&nbsp; ".join(parts)

    def _on_panel_toggle(self, ch):
        self.right_panel.layout.display = "" if ch["new"] else "none"

    def set_figure_preset(
        self, name=None, width=None, height=None, scale=None
    ):
        """Export size for images and animations.

        ``name``: a key of ``figure_presets`` (``screen``, ``paper``,
        ``slide``, ``poster`` by default); ``width`` / ``height`` /
        ``scale`` override it. Returns ``(width, height, scale)``.
        """
        if name is not None:
            if name not in self.figure_presets:
                raise ValueError(
                    f"Unknown figure preset {name!r}; use one of "
                    f"{list(self.figure_presets)}"
                )
            p = self.figure_presets[name]
            w, h, sc = (
                (self.export_width, self.export_height, 2) if p is None else p
            )
            self._figure_preset = name
            if self.fig_preset.value != name:
                self.fig_preset.value = name
        else:
            w, h, sc = (
                self.anim_width.value,
                self.anim_height.value,
                self.anim_scale.value,
            )
        w, h, sc = (
            int(width or w),
            int(height or h),
            int(scale or sc),
        )
        self.anim_width.value, self.anim_height.value = w, h
        self.anim_scale.value = max(1, min(4, sc))
        return w, h, self.anim_scale.value

    def _on_png_clicked(self, _b=None):
        from .export import output_path

        try:
            out = output_path(self.anim_name.value, "png", "view").with_suffix(
                ".png"
            )
            if out.name == ".png":
                out = out.with_name("view.png")
            self.write_image(
                out,
                self.anim_width.value,
                self.anim_height.value,
                self.anim_scale.value,
            )
            self.anim_status.value = f"Saved: <code>{out.resolve()}</code>"
            self.busy_label.value = f"saved {out.name}"
        except Exception as e:
            self.busy_label.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    # --- layer list ---------------------------------------------------------
    def _layer_row(self, name):
        cb = self._visible_cb[name]
        cb.description = ""
        cb.layout.width = "28px"
        t = self._layers[name]["type"]
        btn = widgets.Button(
            description=name,
            tooltip=f"Edit {name}",
            layout=_lay("auto", flex="1 1 auto"),
        )
        btn.on_click(lambda _b, n=name: self._select_edit_layer(n))
        sw = widgets.HTML(self._swatch_html(name), layout=_lay("44px"))
        badge = widgets.HTML(
            f"<span style='font-size:11px;opacity:0.7;border:1px solid "
            f"rgba(127,127,127,0.5);border-radius:9px;padding:0 6px'>{t}</span>",
            layout=_lay("auto"),
        )
        row = widgets.HBox([cb, sw, btn, badge], layout=_lay("100%"))
        row.add_class("mv-row")
        self._rows[name] = (row, sw)
        return row

    def _swatch_html(self, name):
        st = self._style(name) if name in self._layer_widgets else {}
        cmap = st.get("cmap")
        bg = (
            _gradient(self._colorscale(cmap))
            if cmap
            else st.get("color", "#888")
        )
        return (
            f"<div style='width:38px;height:12px;border-radius:3px;"
            f"border:1px solid rgba(0,0,0,0.25);background:{bg}'></div>"
        )

    def _refresh_layer_list(self):
        self._rows = {}
        self.visible_keys_box.children = [
            self._layer_row(k) for k in self._layers if k in self._visible_cb
        ]
        self._highlight_row(self.edit_key.value)

    def _highlight_row(self, name):
        for n, (row, _) in getattr(self, "_rows", {}).items():
            (row.add_class if n == name else row.remove_class)("mv-row-sel")

    def _update_swatch(self, name):
        r = getattr(self, "_rows", {}).get(name)
        if r is not None:
            r[1].value = self._swatch_html(name)

    # --- Add tiles ----------------------------------------------------------
    def _build_add_tiles(self):
        tiles = []
        for label, kind in (
            ("▤ Slice", "slice"),
            ("◇ Plane", "plane"),
            ("◐ Clip", "clip"),
            ("Δ Difference", "difference"),
            ("◎ Shell / core", "region"),
            ("⊹ Crystal axes", "triad"),
            ("━ Scale bar", "scale_bar"),
            ("∿ Lines", "lines"),
            ("➚ Arrows", "arrows"),
        ):
            b = widgets.Button(description=label, layout=_lay("auto"))
            b.add_class("mv-tile")
            b.on_click(lambda _b, k=kind: self._on_add_tile(k))
            tiles.append(b)
        self.add_tile_msg = widgets.HTML("")
        grid = widgets.GridBox(
            tiles,
            layout=widgets.Layout(
                grid_template_columns="repeat(3, 1fr)",
                grid_gap="6px",
                width="100%",
            ),
        )
        return widgets.VBox([grid, self.add_tile_msg], layout=_lay("100%"))

    def _on_add_tile(self, kind):
        self.add_tile_msg.value = ""
        if kind in ("slice", "plane", "clip", "difference", "region"):
            self.add_layer_kind.value = kind
            return
        try:
            if kind == "triad":
                self._select_edit_layer(self.add_axes_triad())
            elif kind == "scale_bar":
                self._select_edit_layer(self.add_scale_bar())
            else:
                ex = (
                    "v.add_lines('dislo', [points_a, points_b])"
                    if kind == "lines"
                    else "v.add_arrows('b', origins, vectors, labels=['b'])"
                )
                self.add_tile_msg.value = (
                    f"<i>{kind.capitalize()} come from your data; in code:</i> "
                    f"<code>{ex}</code> <i>(see</i> <code>v.help('{kind}')</code><i>)</i>"
                )
        except Exception as e:
            self.add_tile_msg.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    # --- Script tab ---------------------------------------------------------
    def _build_script_tab(self):
        B = widgets.Button
        self.script_text = widgets.Textarea(
            value="",
            placeholder="- {only: amplitude}\n- {edit: amplitude, thr: 0.3}",
            layout=_lay("100%", height="340px"),
        )
        self.script_text.add_class("mv-script")
        self.script_from_view = B(
            description="⇦ From current view", layout=_lay("auto")
        )
        self.script_check = B(description="Check", layout=_lay("auto"))
        self.script_run = B(
            description="Run", button_style="primary", layout=_lay("auto")
        )
        self.script_path = widgets.Text(
            value="view.yaml", placeholder="script.yaml", layout=_lay("40%")
        )
        self.script_open = B(description="Open", layout=_lay("auto"))
        self.script_save = B(description="Save", layout=_lay("auto"))
        self.script_msg = widgets.HTML("")
        self.script_from_view.on_click(lambda _b: self._script_from_view())
        self.script_check.on_click(lambda _b: self._script_check())
        self.script_run.on_click(lambda _b: self._script_run())
        self.script_open.on_click(lambda _b: self._script_io(save=False))
        self.script_save.on_click(lambda _b: self._script_io(save=True))
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.script_from_view,
                        self.script_check,
                        self.script_run,
                    ],
                    layout=_lay("100%"),
                ),
                self.script_text,
                widgets.HBox(
                    [self.script_path, self.script_open, self.script_save],
                    layout=_lay("100%"),
                ),
                self.script_msg,
            ],
            layout=_lay("100%"),
        )

    def _script_steps(self):
        import yaml

        steps = yaml.safe_load(self.script_text.value or "") or []
        return steps if isinstance(steps, list) else [steps]

    def _script_from_view(self):
        import yaml

        self.script_text.value = yaml.safe_dump(
            self.to_actions(),
            sort_keys=False,
            default_flow_style=None,
            allow_unicode=True,
        )
        self.script_msg.value = "<i>The current view as a script.</i>"

    def _script_check(self):
        try:
            steps = self._script_steps()
            problems = self.validate_actions(steps)
        except Exception as e:
            problems = [f"{type(e).__name__}: {e}"]
            steps = []
        self.script_msg.value = (
            "<br>".join(
                f"<span style='color:#d62728'>✗ {p}</span>" for p in problems
            )
            if problems
            else f"✓ No problems found · {len(steps)} steps"
        )
        return not problems

    def _script_run(self):
        if not self._script_check():
            return
        try:
            rep = self.run_actions(self._script_steps())
            self.script_msg.value = (
                f"✓ ran {len(rep)} steps"
                if getattr(rep, "ok", True)
                else "Running in the background (exports)…"
            )
        except Exception as e:
            self.script_msg.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    def _script_io(self, save):
        from pathlib import Path

        try:
            path = Path(self.script_path.value).expanduser()
            if save:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self.script_text.value)
                self.script_msg.value = f"Saved <code>{path.resolve()}</code>"
            else:
                self.script_text.value = path.read_text()
                self.script_msg.value = f"Opened <code>{path.resolve()}</code>"
        except Exception as e:
            self.script_msg.value = (
                f"<b style='color:#d62728'>{type(e).__name__}: {e}</b>"
            )

    def _section(self, title, body, open_by_default=False):
        header = widgets.ToggleButton(
            value=open_by_default,
            description=title,
            layout=_lay("100%"),
            icon="chevron-down" if open_by_default else "chevron-right",
        )
        holder = widgets.VBox(
            [body] if open_by_default else [], layout=_lay("100%")
        )

        def _toggle(ch):
            header.icon = "chevron-down" if ch["new"] else "chevron-right"
            holder.children = [body] if ch["new"] else []

        header.observe(_toggle, names="value")
        box = widgets.VBox([header, holder], layout=_lay("100%"))
        box.add_class("mv-section")
        return box

    _make_collapsible_section = _section

    def _wire_global_callbacks(self):
        self.theme_toggle.observe(
            lambda ch: self._apply_theme(), names="value"
        )
        self.grid_toggle.observe(self._on_grid_toggle, names="value")
        self.rotate_toggle.observe(
            lambda ch: (
                self._start_rotation() if ch["new"] else self._stop_rotation()
            ),
            names="value",
        )
        self.reset_cam_btn.on_click(lambda _: self._reset_camera())
        self.edit_key.observe(self._on_edit_key_changed, names="value")
        self.rename_layer_btn.on_click(self._on_rename_clicked)
        self.delete_layer_btn.on_click(self._on_delete_clicked)
        self.add_layer_btn.on_click(self._on_create_layer_clicked)
        self.add_layer_kind.observe(
            lambda ch: self._sync_create_ui(), names="value"
        )
        self.add_slice_axis.observe(
            lambda ch: self._sync_create_ui(center=True), names="value"
        )
        self.cond_slider.observe(
            lambda ch: (
                (
                    self.set_condition(ch["new"], refresh=False),
                    self._ui_refresh(None),
                )
                if ch["new"] != self._cond_index
                else None
            ),
            names="value",
        )

    # =========================================================================
    # Theme / camera
    # =========================================================================
    def _dark(self):
        return bool(self.theme_toggle.value)

    def _apply_theme(self):
        dark = self._dark()
        (self.add_class if dark else self.remove_class)("mv-dark")
        ax = self._axis_layout()
        self.fig.update_layout(
            template="plotly_dark" if dark else "plotly_white",
            scene=dict(xaxis=ax, yaxis=ax, zaxis=ax),
        )
        self._rebuild()

    def _on_theme(self, change=None):
        return self._apply_theme()

    def _on_grid_toggle(self, change):
        show = bool(self.grid_toggle.value)
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=show),
                yaxis=dict(showgrid=show),
                zaxis=dict(showgrid=show),
            )
        )

    def _reset_camera(self):
        self.fig.layout.scene.camera = dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

    def _fit_plot_to_container(self):
        with self.fig.batch_update():
            self.fig.layout.width = None
            self.fig.layout.autosize = True

    # =========================================================================
    # After data / layer changes
    # =========================================================================
    def _ui_after_data(self):
        for name, s in self._layers.items():
            w = self._layer_widgets.get(name)
            if w is not None and "thr" in w:
                self._set_thr_bounds(name, keep_value=True)
        self._update_condition_ui()
        self._after_layers_changed(select=None)
        self._sync_create_ui(center=True)
        self._tools_after_data()

    def _update_condition_ui(self):
        n = len(self._conditions)
        self.cond_row.layout.display = "" if n > 1 else "none"
        self.cond_slider.max = max(0, n - 1)
        if self.cond_slider.value != self._cond_index:
            self.cond_slider.value = self._cond_index
        lab = self._cond_labels[self._cond_index] if self._cond_labels else ""
        self.cond_label.value = f"<b>{lab}</b>"
        if hasattr(self, "anim_label_row"):
            self._sync_anim_ui()

    def _after_layers_changed(self, select=None):
        """Re-sync every widget that lists layers (keeps current values when still valid)."""
        names = list(self._layers)
        prev = self.edit_key.value
        self.edit_key.unobserve(self._on_edit_key_changed, names="value")
        try:
            self.edit_key.options = [("— choose layer —", None)] + [
                (k, k) for k in names
            ]
            target = (
                select
                if select in self._layers
                else prev
                if prev in self._layers
                else None
            )
            self.edit_key.value = target
        finally:
            self.edit_key.observe(self._on_edit_key_changed, names="value")
        vols = self.volume_keys()
        self._set_options(self.add_layer_source, [(k, k) for k in vols])
        for name, w in self._layer_widgets.items():
            self._refresh_layer_options(name, w)
        self._tools_refresh_options()
        self._anim_refresh_options()
        if target is not None:
            self.rename_layer_text.value = target
        self._refresh_layer_list()
        self._show_layer_panel(self.edit_key.value)

    @staticmethod
    def _set_options(dd, options, fallback="first"):
        vals = [v for _, v in options]
        cur = dd.value
        dd.options = options
        if cur in vals:
            dd.value = cur
        elif vals:
            dd.value = (
                vals[0]
                if fallback == "first"
                else fallback
                if fallback in vals
                else vals[0]
            )

    def _refresh_layer_options(self, name, w):
        vols = self.volume_keys()
        with self._muted(w):
            spec = self._layers[name]
            cb = [("(self)", "__self__")] + [(k, k) for k in vols if k != name]
            cb += [
                ("(x coord)", "__x__"),
                ("(y coord)", "__y__"),
                ("(z coord)", "__z__"),
                ("(constant)", "__constant__"),
            ]
            if spec["type"] in ("raw", "diff", "region", "clip"):
                cb.append(("(facet family)", "__facet__"))
                cb.append(("(facet patch)", "__patch__"))
            if "color_by" in w:
                self._set_options(w["color_by"], cb, fallback="__self__")
            for key in ("source", "a", "field", "support"):
                if f"p_{key}" in w:
                    opts = [(k, k) for k in vols if k != name]
                    self._set_options(w[f"p_{key}"], opts)
                    if spec.get(key) in [v for _, v in opts]:
                        w[f"p_{key}"].value = spec[key]
            if "p_b" in w:
                opts = [("(reference condition)", None)] + [
                    (k, k) for k in vols if k != name
                ]
                w["p_b"].options = opts
                w["p_b"].value = (
                    spec.get("b")
                    if spec.get("b") in [v for _, v in opts]
                    else None
                )
            if "p_ref" in w:
                w["p_ref"].max = max(0, len(self._conditions) - 1)

    # =========================================================================
    # Visible checkboxes
    # =========================================================================
    def _add_visible_checkbox(self, name, visible=True):
        cb = widgets.Checkbox(
            value=bool(visible),
            description=name,
            indent=False,
            layout=widgets.Layout(width="150px"),
        )
        cb.observe(lambda ch: self._ui_refresh(None), names="value")
        self._visible_cb[name] = cb

    def _get_visible_keys(self):
        return self._visible_names()

    # =========================================================================
    # Layer widgets
    # =========================================================================
    @contextmanager
    def _muted(self, w):
        prev = w.get("_mute", False)
        w["_mute"] = True
        try:
            yield
        finally:
            w["_mute"] = prev

    def _default_cmap(self, name, spec):
        if name in self.default_cmaps:
            return self.default_cmaps[name]
        src = (
            name
            if spec["type"] in ("raw", "diff")
            else spec.get("source") or spec.get("field") or name
        )
        if "phase" in str(src).lower():
            return "twilight"
        return (
            "gray" if spec["type"] in ("raw", "clip") else "viridis"
        )  # solid gray isosurfaces, coloured slices

    def _range_source(self, name, spec):
        """Values bounding the initial colour range (display units)."""
        t = spec["type"]
        if t in ("raw", "diff", "region"):
            vals = self.get_volume(name)
        elif t == "clip":
            vals = self.get_volume(spec["source"])
        elif t in ("slice", "plane"):
            vals = self._surface_data(spec)[0]
        else:
            return np.array([0.0, 1.0])
        scale = self._unit_of(self._color_source(name, spec))[1]
        return vals if scale == 1.0 else np.asarray(vals, float) * scale

    # --- units ----------------------------------------------------------------
    def _unit_field_of(self, name):
        w, spec = self._layer_widgets.get(name, {}), self._layers[name]
        sel = w["color_by"].value if "color_by" in w else "__self__"
        if sel in ("__constant__", "__facet__", "__patch__"):
            return None
        return self._color_field(name, spec, sel)

    def _sync_unit_widgets(self, name):
        w = self._layer_widgets.get(name)
        if not w or "u_unit" not in w:
            return
        field = self._unit_field_of(name)
        label, scale = self._field_unit(field) if field else ("", 1.0)
        coord = field is None or str(field).startswith("__")
        with self._muted(w):
            w["u_unit"].value, w["u_scale"].value = label, scale
            w["u_unit"].disabled = w["u_scale"].disabled = coord
            w["u_unit"].description = (
                "unit" if coord else f"unit of {field}"[:24]
            )

    def _on_unit_change(self, name):
        w = self._layer_widgets.get(name)
        if w is None or w.get("_mute"):
            return
        field = self._unit_field_of(name)
        if field is None or str(field).startswith("__"):
            return
        scale = float(w["u_scale"].value) or 1.0
        self.set_units({field: (w["u_unit"].value.strip(), scale)})

    def _set_thr_bounds(self, name, keep_value=True):
        w, spec = self._layer_widgets[name], self._layers[name]
        src = (
            self.get_volume(name)
            if spec["type"] in ("raw", "diff")
            else self.get_volume(spec["source"])
        )
        if "abs_iso" in w and w["abs_iso"].value:
            src = np.abs(np.asarray(src, dtype=float))
        lo, hi = geo.finite_range(src)
        thr = w["thr"]
        old = thr.value
        with self._muted(w):
            if lo > thr.max:
                thr.max = hi
                thr.min = lo
            else:
                thr.min = lo
                thr.max = hi
            thr.step = (hi - lo) / 300
            a = np.asarray(src, dtype=float)
            a = a[np.isfinite(a)]
            if keep_value and lo < old < hi:
                thr.value = float(old)
            else:
                p = float(np.percentile(a, 30)) if a.size else lo
                thr.value = p if lo < p < hi else lo + 0.3 * (hi - lo)

    def _build_layer_widgets(self, name):
        spec = self._layers[name]
        t = spec["type"]
        h = {"name": name}
        w = {"_h": h, "_mute": True}
        D = self._DESC

        def FS(**kw):
            return widgets.FloatSlider(
                continuous_update=False,
                readout_format=".3g",
                style=D,
                layout=_lay(),
                **kw,
            )

        w["op"] = FS(
            value=1.0, min=0.0, max=1.0, step=0.01, description="opacity"
        )
        if t in ("raw", "diff", "clip"):
            w["thr"] = FS(
                value=0.0, min=0.0, max=1.0, step=0.01, description="iso"
            )
        if t in ("raw", "diff"):
            w["as_mask"] = widgets.Checkbox(
                value=False,
                description="mask mode",
                indent=False,
                tooltip="Threshold to a binary mask, then extract its surface",
            )
            w["abs_iso"] = widgets.Checkbox(
                value=True,
                description="iso on |value|",
                indent=False,
                tooltip="Isosurface of |field| (both signs) instead of the signed field",
            )
        if t in ("lines", "arrows"):
            w["color"] = widgets.ColorPicker(
                value="#000000",
                description="color",
                concise=False,
                style=D,
                layout=_lay(),
            )
            w["width"] = FS(
                value=6.0, min=1.0, max=30.0, step=0.5, description="width"
            )
        else:
            w["cmap"] = widgets.Dropdown(
                options=self.cmap_options,
                value=self._default_cmap(name, spec),
                description="cmap",
                style=D,
                layout=_lay("48%"),
            )
            w["nan_policy"] = widgets.Dropdown(
                options=NAN_OPTIONS,
                value="zero" if t != "region" else "none",
                description="NaN",
                style=D,
                layout=_lay("48%"),
            )
            w["color_by"] = widgets.Dropdown(
                options=[("(self)", "__self__")],
                value="__self__",
                description="color by",
                style=D,
                layout=_lay("48%"),
            )
            w["show_colorbar"] = widgets.Checkbox(
                value=True,
                description="colorbar",
                indent=False,
                layout=_lay("auto"),
            )
            w["range_mode"] = widgets.Dropdown(
                options=[(lab, k) for k, lab in mva.RANGE_MODES.items()],
                value="minmax",
                description="range",
                style=D,
                layout=_lay("48%"),
            )
            w["range_all"] = widgets.Checkbox(
                value=False,
                description="same for all conditions",
                tooltip="One colour range for every condition "
                "(union of their ranges)",
                indent=False,
                layout=_lay(
                    "auto",
                    display="" if len(self._conditions) > 1 else "none",
                ),
            )
            w["u_unit"] = widgets.Text(
                value="",
                description="unit",
                placeholder="e.g. rad, %",
                tooltip="Unit of the colour field (all layers using it)",
                continuous_update=False,
                style=D,
                layout=_lay("48%"),
            )
            w["u_scale"] = widgets.FloatText(
                value=1.0,
                description="× scale",
                tooltip="Displayed values = field × scale (100 for %)",
                style=D,
                layout=_lay("48%"),
            )
            w["range_slider"] = widgets.FloatRangeSlider(
                value=(0.0, 1.0),
                min=0.0,
                max=1.0,
                step=0.01,
                description="cmin/cmax",
                continuous_update=False,
                readout_format=".3g",
                style=D,
                layout=_lay(),
                disabled=True,
            )
            w["range_min"] = widgets.FloatText(
                value=0.0, description="bound min", style=D, layout=_lay("47%")
            )
            w["range_max"] = widgets.FloatText(
                value=1.0, description="bound max", style=D, layout=_lay("47%")
            )
        for k, v in LIGHT_DEFAULTS.items():
            w[f"light_{k}"] = FS(
                value=v,
                min=0.01 if k == "roughness" else 0.0,
                max=1.0,
                step=0.01,
                description=k,
            )

        # --- transform ---
        lo, hi = self.frame.bounds_phys()
        ext = self.frame.phys_to_plot(hi - lo) + max(self.voxel_size)
        for i, a in enumerate("xyz"):
            w[f"rot_{a}"] = FS(
                value=0.0, min=-180, max=180, step=1, description=f"R{a} (°)"
            )
            w[f"trans_{a}"] = FS(
                value=0.0,
                min=-2 * ext[i],
                max=2 * ext[i],
                step=float(ext[i]) / 200,
                description=f"T{a}",
            )

        # --- type-specific parameters ---
        self._build_param_widgets(name, spec, w)

        # --- mesh export ---
        if t not in ("lines",):
            w["x_path"] = widgets.Text(
                value=f"{name.replace(':', '_').replace('/', '_')}.ply",
                description="file",
                style=D,
                layout=_lay("60%"),
            )
            w["x_frame"] = widgets.Dropdown(
                options=[
                    ("as displayed", "plot"),
                    ("physical xyz", "physical"),
                ],
                value="plot",
                layout=_lay("35%"),
            )
            w["x_btn"] = widgets.Button(
                description="Export mesh", layout=_lay("auto")
            )
            w["x_msg"] = widgets.HTML("")
            w["x_btn"].on_click(
                lambda _b, h=h: self._on_export_mesh_clicked(h["name"])
            )

        # --- observers ---
        for k, wd in w.items():
            if (
                k.startswith("_")
                or k.startswith("x_")
                or isinstance(wd, widgets.Button)
            ):
                continue
            if k.startswith("u_"):
                wd.observe(
                    lambda ch, h=h: self._on_unit_change(h["name"]),
                    names="value",
                )
                continue
            if k.startswith("p_"):
                wd.observe(
                    lambda ch, h=h: self._on_param_change(h["name"], ch),
                    names="value",
                )
            elif k.startswith(("rot_", "trans_")):
                wd.observe(
                    lambda ch, h=h: self._on_transform_change(h["name"]),
                    names="value",
                )
            else:
                wd.observe(
                    lambda ch, h=h: self._on_style_change(h["name"], ch),
                    names="value",
                )
        self._layer_widgets[name] = w
        self._refresh_layer_options(name, w)
        if "thr" in w:
            self._set_thr_bounds(name, keep_value=False)
        if "range_slider" in w:
            lo_, hi_ = geo.finite_range(self._range_source(name, spec))
            w["range_min"].value, w["range_max"].value = lo_, hi_
            self._sync_range_widget(w, lo_, hi_, True)
            self._sync_unit_widgets(name)
        self._params_to_widgets(name)
        self._load_transform_widgets(name)
        w["_defaults"] = self._style(name)  # reference for to_actions()
        w["_mute"] = False

    def _build_param_widgets(self, name, spec, w):
        t, D, h = spec["type"], self._DESC, w["_h"]

        def FT(d):
            return widgets.FloatText(
                description=d,
                style={"description_width": "30px"},
                layout=_lay("95%"),
            )

        if t in ("slice", "plane", "clip"):
            w["p_source"] = widgets.Dropdown(
                options=[], description="source", style=D, layout=_lay()
            )
        if t == "slice":
            w["p_axis"] = widgets.Dropdown(
                options=["x", "y", "z"],
                value="z",
                description="axis",
                style=D,
                layout=_lay(),
            )
            w["p_pos"] = widgets.IntSlider(
                min=0,
                max=1,
                description="pos",
                continuous_update=False,
                style=D,
                layout=_lay(),
            )
            w["p_thickness"] = widgets.IntSlider(
                min=0,
                max=1,
                description="± thick",
                continuous_update=False,
                style=D,
                layout=_lay(),
                tooltip="Mean over pos ± thick voxels",
            )
        if t in ("plane", "clip"):
            for a in "xyz":
                w[f"p_n{a}"], w[f"p_o{a}"] = FT(f"n{a}"), FT(f"o{a}")
            w["p_offset"] = widgets.FloatSlider(
                min=-1,
                max=1,
                step=0.1,
                description="offset",
                continuous_update=False,
                readout_format=".3g",
                style=D,
                layout=_lay(),
            )
            w["center_btn"] = widgets.Button(
                description="Centre origin", layout=_lay("auto")
            )
            w["center_btn"].on_click(
                lambda _b, h=h: self.set_layer_params(
                    h["name"],
                    origin=self.frame.phys_to_plot(self.frame.center_phys()),
                    offset=0.0,
                )
            )
        if t == "slice":
            w["center_btn"] = widgets.Button(
                description="Centre pos", layout=_lay("auto")
            )
            w["center_btn"].on_click(
                lambda _b, h=h: self.set_layer_params(
                    h["name"],
                    pos=(
                        self.frame.n_along(
                            self._layers[h["name"]]["axis_phys"]
                        )
                        - 1
                    )
                    // 2,
                )
            )
        if t == "plane":
            diag = self.frame.diagonal()
            w["p_thickness"] = widgets.FloatSlider(
                min=0,
                max=diag / 5,
                step=diag / 1000,
                description="± thick",
                continuous_update=False,
                readout_format=".3g",
                style=D,
                layout=_lay(),
            )
            w["p_extent"] = widgets.FloatSlider(
                min=max(self.voxel_size),
                max=diag,
                step=diag / 500,
                description="extent",
                continuous_update=False,
                readout_format=".3g",
                style=D,
                layout=_lay(),
            )
        if t == "clip":
            w["p_side"] = widgets.Dropdown(
                options=[("Keep +n side", "up"), ("Keep -n side", "down")],
                description="side",
                style=D,
                layout=_lay(),
            )
            w["p_smooth"] = widgets.Checkbox(
                value=True, description="smooth planar cap", indent=False
            )
        if t == "diff":
            w["p_a"] = widgets.Dropdown(
                options=[], description="A", style=D, layout=_lay()
            )
            w["p_b"] = widgets.Dropdown(
                options=[("(reference condition)", None)],
                description="B",
                style=D,
                layout=_lay(),
            )
            w["p_ref"] = widgets.IntSlider(
                min=0,
                max=max(0, len(self._conditions) - 1),
                description="ref cond.",
                continuous_update=False,
                style=D,
                layout=_lay(),
            )
        if t == "region":
            w["p_field"] = widgets.Dropdown(
                options=[], description="field", style=D, layout=_lay()
            )
            w["p_support"] = widgets.Dropdown(
                options=[], description="support", style=D, layout=_lay()
            )
            w["p_threshold"] = widgets.FloatText(
                description="support ≥", style=D, layout=_lay()
            )
            w["p_mode"] = widgets.Dropdown(
                options=[("Surface shell", "shell"), ("Core", "core")],
                description="region",
                style=D,
                layout=_lay(),
            )
            w["p_depth"] = widgets.FloatText(
                description=f"depth{f' ({self.unit})' if self.unit else ''}",
                style=D,
                layout=_lay(),
            )

    # --- spec <-> widgets ---------------------------------------------------------
    def _params_to_widgets(self, name):
        w, s = self._layer_widgets.get(name), self._layers[name]
        if w is None:
            return
        with self._muted(w):
            for key in ("source", "a", "field", "support"):
                if f"p_{key}" in w and s.get(key) in [
                    v for _, v in w[f"p_{key}"].options
                ]:
                    w[f"p_{key}"].value = s[key]
            if s["type"] == "slice":
                ax = s["axis_phys"]
                w["p_axis"].value = self.frame.phys_axis_to_plot(ax)
                n = self.frame.n_along(ax)
                w["p_pos"].max = n - 1
                w["p_pos"].value = int(np.clip(s["pos"], 0, n - 1))
                w["p_thickness"].max = max(1, n // 4)
                w["p_thickness"].value = int(
                    np.clip(s["thickness"], 0, w["p_thickness"].max)
                )
            if s["type"] in ("plane", "clip"):
                n_plot = self.frame.phys_to_plot(s["normal"])
                o_plot = self.frame.phys_to_plot(s["origin0"])
                for i, a in enumerate("xyz"):
                    w[f"p_n{a}"].value = float(n_plot[i])
                    w[f"p_o{a}"].value = float(o_plot[i])
                self._sync_offset_widget(name)
            if s["type"] == "plane":
                w["p_thickness"].value = float(s["thickness"])
                w["p_extent"].value = float(s["extent"])
            if s["type"] == "clip":
                w["p_side"].value = s.get("side", "up")
                w["p_smooth"].value = bool(s.get("smooth", True))
            if s["type"] == "diff":
                w["p_b"].value = (
                    s.get("b")
                    if s.get("b") in [v for _, v in w["p_b"].options]
                    else None
                )
                w["p_ref"].value = int(s.get("ref") or 0)
                w["p_ref"].disabled = bool(s.get("b"))
            if s["type"] == "region":
                w["p_threshold"].value = float(s["threshold"])
                w["p_mode"].value = s["mode"]
                w["p_depth"].value = float(s["depth"])

    def _sync_offset_widget(self, name):
        w, s = self._layer_widgets[name], self._layers[name]
        a, b = self._offset_range(s)
        sl = w["p_offset"]
        if b <= a:
            b = a + 1e-9
        with self._muted(w):
            if a > sl.max:
                sl.max, sl.min = b, a
            else:
                sl.min, sl.max = a, b
            sl.step = max((b - a) / 400, 1e-9)
            s["offset"] = float(np.clip(s.get("offset", 0.0), a, b))
            sl.value = s["offset"]

    def _widgets_to_params(self, name, owner):
        w, s = self._layer_widgets[name], self._layers[name]
        for key in ("source", "a", "field", "support"):
            if f"p_{key}" in w and w[f"p_{key}"].value:
                s[key] = w[f"p_{key}"].value
        if s["type"] == "slice":
            if owner is w["p_axis"]:
                s["axis_phys"] = self.frame.plot_axis_to_phys(
                    w["p_axis"].value
                )
                s["pos"] = (self.frame.n_along(s["axis_phys"]) - 1) // 2
            else:
                s["pos"], s["thickness"] = (
                    int(w["p_pos"].value),
                    int(w["p_thickness"].value),
                )
            self._params_to_widgets(name)
        if s["type"] in ("plane", "clip"):
            n_plot = np.array([w[f"p_n{a}"].value for a in "xyz"], dtype=float)
            if np.linalg.norm(n_plot) > 0:
                s["normal"] = self.frame.plot_to_phys(n_plot)
            s["origin0"] = self.frame.plot_to_phys(
                np.array([w[f"p_o{a}"].value for a in "xyz"], dtype=float)
            )
            s["offset"] = float(w["p_offset"].value)
            self._sync_offset_widget(name)
        if s["type"] == "plane":
            s["thickness"], s["extent"] = (
                float(w["p_thickness"].value),
                float(w["p_extent"].value),
            )
        if s["type"] == "clip":
            s["side"], s["smooth"] = (
                w["p_side"].value,
                bool(w["p_smooth"].value),
            )
            if owner is w.get("p_source"):
                self._set_thr_bounds(name, keep_value=False)
        if s["type"] == "diff":
            s["b"], s["ref"] = w["p_b"].value, int(w["p_ref"].value)
            w["p_ref"].disabled = bool(s["b"])
        if s["type"] == "region":
            s["threshold"], s["mode"], s["depth"] = (
                float(w["p_threshold"].value),
                w["p_mode"].value,
                float(w["p_depth"].value),
            )

    def _on_param_change(self, name, ch):
        w = self._layer_widgets.get(name)
        if w is None or w.get("_mute"):
            return
        self._widgets_to_params(name, ch.get("owner"))
        t = self._layers[name]["type"]
        if t in ("diff", "region"):
            self._vol_cache.pop(name, None)
            if "thr" in w:
                self._set_thr_bounds(name, keep_value=True)
            self._ui_refresh(None)
        else:
            self._ui_refresh([name])

    # --- style --------------------------------------------------------------------
    def _style(self, name):
        w = self._layer_widgets.get(name, {})

        def g(k, d):
            return w[k].value if k in w else d

        st = {
            "thr": g("thr", 0.0),
            "op": g("op", 1.0),
            "as_mask": g("as_mask", False),
            "cmap": g("cmap", "gray"),
            "nan_policy": g("nan_policy", "none"),
            "color_by": g("color_by", "__self__"),
            "show_colorbar": g("show_colorbar", False),
            "range_mode": g("range_mode", "minmax"),
            "range_all": g("range_all", False),
            "auto_range": g("range_mode", "minmax") != "manual",
            "abs_iso": g("abs_iso", False),
            "range": tuple(g("range_slider", (0.0, 1.0))),
            "range_min": g("range_min", 0.0),
            "range_max": g("range_max", 1.0),
            "color": g("color", "#000000"),
            "width": g("width", 6.0),
        }
        st.update(
            {
                f"light_{k}": g(f"light_{k}", v)
                for k, v in LIGHT_DEFAULTS.items()
            }
        )
        return st

    def _set_style(self, name, **kw):
        w = self._layer_widgets[name]
        aliases = {
            "opacity": "op",
            "range_slider": "range",
            "range_min_override": "range_min",
            "range_max_override": "range_max",
            "iso": "thr",
        }
        with self._muted(w):
            for k, v in kw.items():
                k = aliases.get(k, k)
                if k == "range" and isinstance(v, str):  # preset name
                    k, v = "range_mode", v
                if k == "range":
                    w["_pending_range"] = tuple(float(x) for x in v)
                    continue
                if k == "auto_range" and "range_mode" in w:
                    if not v:
                        k, v = "range_mode", "manual"
                    elif w["range_mode"].value == "manual":
                        k, v = "range_mode", "minmax"
                    else:
                        continue
                if k == "range_mode":
                    v = mva.range_mode(v)
                if k not in w:
                    if k in (
                        "thr",
                        "as_mask",
                        "abs_iso",
                        "cmap",
                        "color",
                        "width",
                        "nan_policy",
                        "color_by",
                        "show_colorbar",
                        "auto_range",
                        "range_mode",
                        "range_all",
                        "range_min",
                        "range_max",
                    ) or k.startswith("light_"):
                        continue  # not applicable to this layer type
                    raise ValueError(
                        f"Parameter '{k}' not available for layer '{name}'"
                    )
                if k == "thr":
                    sl = w["thr"]
                    sl.min, sl.max = (
                        min(sl.min, float(v)),
                        max(sl.max, float(v)),
                    )
                if k == "color_by" and v not in [
                    o[1] for o in w["color_by"].options
                ]:
                    raise ValueError(f"Cannot colour '{name}' by {v!r}")
                if k == "color_by" and v != w["color_by"].value:
                    w["_reset_bounds"] = True  # new field: new slider bounds
                w[k].value = v
            if "range_mode" in w:
                w["range_slider"].disabled = w["range_mode"].value != "manual"
        if "color_by" in kw:
            self._sync_unit_widgets(name)

    def _on_style_change(self, name, ch):
        w = self._layer_widgets.get(name)
        if w is None or w.get("_mute"):
            return
        owner = ch.get("owner")
        if owner is w.get("range_mode"):
            w["range_slider"].disabled = ch["new"] != "manual"
        if owner is w.get("color_by"):
            w["_reset_bounds"] = True
            self._sync_unit_widgets(name)
        if owner is w.get("abs_iso"):
            self._set_thr_bounds(name, keep_value=True)
        if owner is w.get("op") or any(
            owner is w.get(f"light_{k}") for k in LIGHT_DEFAULTS
        ):
            if name in self._trace_order and not self._batch:
                st = self._style(name)
                self.fig.data[self._trace_order.index(name)].update(
                    opacity=float(st["op"]), lighting=self._lighting(st)
                )
                return
        if owner is w.get("cmap") or owner is w.get("color"):
            self._update_swatch(name)
        if self._restyle(name, owner):
            return
        self._ui_refresh([name])

    _RESTYLE_KEYS = ("cmap", "range_slider", "range_mode")

    def _restyle(self, name, owner):
        """Colour-only change (colormap, colour range): update the trace in
        place from its current values, without recomputing or resending the
        geometry. Returns False when a full refresh is needed."""
        w = self._layer_widgets.get(name, {})
        if (
            self._batch
            or name not in self._trace_order
            or not any(owner is w.get(k) for k in self._RESTYLE_KEYS)
        ):
            return False
        st = self._style(name)
        if (
            st["color_by"] in ("__facet__", "__patch__", "__constant__")
            or st.get("range_all")
            or self._wants_cbar(st) != bool(self._trace_cbar.get(name))
        ):
            return False
        tr = self.fig.data[self._trace_order.index(name)]
        vals = getattr(tr, "intensity", None)
        if vals is None:
            vals = getattr(tr, "surfacecolor", None)
        if vals is None or tr.cmin is None:
            return False
        vals = np.asarray(vals, dtype=float)
        if not vals.size:
            return False
        cmin, cmax = self._color_range(
            name, self._layers[name], st, vals, sync=True
        )
        with self.fig.batch_update():
            tr.update(
                colorscale=self._colorscale(st["cmap"]), cmin=cmin, cmax=cmax
            )
        return True

    def _sync_range_widget(self, w, dmin, dmax, auto, value=None):
        rs = w["range_slider"]
        reset = w.pop("_reset_bounds", False)
        if reset:  # new colour source: previous bounds and manual range are meaningless
            with self._muted(w):
                w["range_min"].value, w["range_max"].value = dmin, dmax
        umin, umax = sorted(
            (float(w["range_min"].value), float(w["range_max"].value))
        )
        pending = w.pop("_pending_range", None)
        if pending is not None and not auto:
            # an explicit manual range widens the slider bounds if needed
            pl, ph = sorted(map(float, pending))
            if pl < umin or ph > umax:
                umin, umax = min(umin, pl), max(umax, ph)
                with self._muted(w):
                    w["range_min"].value, w["range_max"].value = umin, umax
        smin, smax = min(dmin, umin), max(dmax, umax)
        if smax <= smin:
            smax = smin + 1e-12
        lo, hi = (
            pending
            if pending is not None
            else (dmin, dmax)
            if reset
            else rs.value
        )
        with self._muted(w), rs.hold_trait_notifications():
            if smin > rs.max:
                rs.max, rs.min = smax, smin
            else:
                rs.min, rs.max = smin, smax
            rs.step = (smax - smin) / 300
            if auto:
                rs.value = value if value is not None else (dmin, dmax)
            else:
                lo, hi = (
                    float(np.clip(lo, smin, smax)),
                    float(np.clip(hi, smin, smax)),
                )
                rs.value = (lo, hi) if hi > lo else (smin, smax)

    # --- transform ----------------------------------------------------------------
    def _load_transform_widgets(self, name):
        w = self._layer_widgets.get(name)
        if w is None:
            return
        tr = self._layer_transform[name]
        with self._muted(w):
            for i, a in enumerate("xyz"):
                w[f"rot_{a}"].value = float(tr["rot_deg"][i])
                sl = w[f"trans_{a}"]
                v = float(tr["trans"][i])
                sl.min, sl.max = min(sl.min, v), max(sl.max, v)
                sl.value = v

    def _set_transform_widgets(self, name, **kw):
        w = self._layer_widgets[name]
        with self._muted(w):
            for k, v in kw.items():
                if k.startswith("trans_"):
                    w[k].min, w[k].max = (
                        min(w[k].min, float(v)),
                        max(w[k].max, float(v)),
                    )
                w[k].value = float(v)
        self._read_transform(name)

    def _read_transform(self, name):
        w = self._layer_widgets[name]
        self._layer_transform[name] = {
            "rot_deg": [float(w[f"rot_{a}"].value) for a in "xyz"],
            "trans": [float(w[f"trans_{a}"].value) for a in "xyz"],
        }

    def _on_transform_change(self, name):
        w = self._layer_widgets.get(name)
        if w is None or w.get("_mute"):
            return
        self._read_transform(name)
        self._ui_refresh([name])

    # =========================================================================
    # Layer panel
    # =========================================================================
    def _show_layer_panel(self, name):
        if not name or name not in self._layer_widgets:
            self.layers_box.children = []
            return
        w, s = self._layer_widgets[name], self._layers[name]
        t = s["type"]
        rows = [widgets.HTML(f"<b>{name}</b> <i>({t})</i>")]
        if t == "slice":
            rows += [
                w["p_source"],
                w["p_axis"],
                w["p_pos"],
                w["center_btn"],
                w["p_thickness"],
            ]
        elif t in ("plane", "clip"):
            grid = widgets.HBox(
                [
                    widgets.VBox(
                        [widgets.HTML("<b>Normal</b>")]
                        + [w[f"p_n{a}"] for a in "xyz"],
                        layout=_lay("48%"),
                    ),
                    widgets.VBox(
                        [widgets.HTML("<b>Origin</b>")]
                        + [w[f"p_o{a}"] for a in "xyz"],
                        layout=_lay("48%"),
                    ),
                ],
                layout=_lay(),
            )
            rows += [w["p_source"], grid, w["center_btn"], w["p_offset"]]
            rows += (
                [w["p_thickness"], w["p_extent"]]
                if t == "plane"
                else [w["p_side"], w["p_smooth"]]
            )
        elif t == "diff":
            rows += [w["p_a"], w["p_b"], w["p_ref"]]
        elif t == "region":
            rows += [
                w["p_field"],
                w["p_support"],
                w["p_threshold"],
                w["p_mode"],
                w["p_depth"],
            ]
        style = []
        if "thr" in w:
            style.append(w["thr"])
        style.append(w["op"])
        if "as_mask" in w:
            style.append(
                widgets.HBox([w["as_mask"], w["abs_iso"]], layout=_lay())
            )
        if "cmap" in w:
            style += [
                widgets.HBox([w["cmap"], w["nan_policy"]], layout=_lay()),
                widgets.HBox(
                    [w["color_by"], w["show_colorbar"]],
                    layout=_lay(align_items="center"),
                ),
                widgets.HBox(
                    [w["range_mode"], w["range_all"]],
                    layout=_lay(align_items="center"),
                ),
                w["range_slider"],
                widgets.HBox([w["range_min"], w["range_max"]], layout=_lay()),
                widgets.HBox([w["u_unit"], w["u_scale"]], layout=_lay()),
            ]
        else:
            style += [w["color"], w["width"]]
        rows += style
        light = widgets.VBox(
            [w[f"light_{k}"] for k in LIGHT_DEFAULTS]
            + [self._light_reset_btn(name)]
        )
        trans = widgets.VBox(
            [widgets.HTML("<b>Rotation (about the volume centre)</b>")]
            + [w[f"rot_{a}"] for a in "xyz"]
            + [widgets.HTML("<b>Translation</b>")]
            + [w[f"trans_{a}"] for a in "xyz"]
        )
        rows += (
            []
            if t in ("slice", "plane")
            else [self._section("Lighting", light)]
        ) + [self._section("Transform", trans)]
        if "x_btn" in w:
            rows.append(
                self._section(
                    "Export mesh",
                    widgets.VBox(
                        [
                            widgets.HBox(
                                [w["x_path"], w["x_frame"]], layout=_lay()
                            ),
                            w["x_btn"],
                            w["x_msg"],
                            widgets.HTML(
                                "<i>.ply / .vtk keep the colour values; .stl / .obj geometry only.</i>"
                            ),
                        ]
                    ),
                )
            )
        self.layers_box.children = [widgets.VBox(rows, layout=_lay("100%"))]

    def _sync_layer_panels_visibility(self):
        return self._show_layer_panel(self.edit_key.value)

    def _light_reset_btn(self, name):
        b = widgets.Button(description="Reset lighting", layout=_lay("auto"))
        b.on_click(
            lambda _b: self._set_style_and_refresh(
                name, **{f"light_{k}": v for k, v in LIGHT_DEFAULTS.items()}
            )
        )
        return b

    def _set_style_and_refresh(self, name, **kw):
        self._set_style(name, **kw)
        self._refresh([name])

    def _select_edit_layer(self, name):
        if name not in self._layers:
            return
        if self.edit_key.value == name:
            self._show_layer_panel(name)
        else:
            self.edit_key.value = name

    def _on_edit_key_changed(self, ch):
        new, old = ch["new"], ch.get("old") or ""
        cur = self.rename_layer_text.value or ""
        if new and (not cur.strip() or cur == old):
            self.rename_layer_text.value = new
        self._show_layer_panel(new)
        self._highlight_row(new)

    def _on_rename_clicked(self, _b):
        old, new = (
            self.edit_key.value,
            (self.rename_layer_text.value or "").strip(),
        )
        if old and new and new != old and new not in self._layers:
            self.rename_layer(old, new)

    def _on_delete_clicked(self, _b):
        if self.edit_key.value:
            self.delete_layer(self.edit_key.value, cascade=True)

    def _remove_layer_widgets(self, name):
        w = self._layer_widgets.pop(name, None)
        if w:
            for k, wd in w.items():
                if isinstance(wd, widgets.Widget):
                    wd.close()
        cb = self._visible_cb.pop(name, None)
        if cb is not None:
            cb.close()

    def _rename_layer_widgets(self, old, new):
        if old in self._layer_widgets:
            w = self._layer_widgets.pop(old)
            w["_h"]["name"] = new
            self._layer_widgets[new] = w
        if old in self._visible_cb:
            cb = self._visible_cb.pop(old)
            cb.description = new
            self._visible_cb[new] = cb
        for w in self._layer_widgets.values():
            if "color_by" in w and w["color_by"].value == old:
                with self._muted(w):
                    self._refresh_layer_options_value(w, new)

    def _refresh_layer_options_value(self, w, new):
        w["color_by"].options = list(w["color_by"].options) + [(new, new)]
        w["color_by"].value = new

    # =========================================================================
    # Create layer
    # =========================================================================
    def _sync_create_ui(self, center=False):
        kind = self.add_layer_kind.value
        rows = {
            "slice": [self._add_slice_row],
            "plane": [self._add_normal_row],
            "clip": [self._add_normal_row, self.add_clip_side],
        }.get(kind, [])
        self.add_params_box.children = rows
        if self.frame is not None:
            n = self.frame.n_along(
                self.frame.plot_axis_to_phys(self.add_slice_axis.value)
            )
            self.add_slice_pos.max = n - 1
            self.add_slice_thickness.max = max(1, n // 4)
            if (
                center
                or kind == "slice"
                and getattr(self, "_create_last_kind", None) != "slice"
            ):
                self.add_slice_pos.value = (n - 1) // 2
        self._create_last_kind = kind

    _sync_create_layer_ui = _sync_create_ui

    def _on_create_layer_clicked(self, _b):
        kind, src = self.add_layer_kind.value, self.add_layer_source.value
        base = (self.add_layer_name.value or "").strip() or (src or "layer")
        normal = (
            self.add_plane_nx.value,
            self.add_plane_ny.value,
            self.add_plane_nz.value,
        )
        try:
            if kind is None or src is None:
                raise ValueError("Choose a layer type and a source.")
            if kind == "slice":
                n = self.add_slice(
                    f"{base}:slice",
                    src,
                    self.add_slice_axis.value,
                    self.add_slice_pos.value,
                    self.add_slice_thickness.value,
                )
            elif kind == "plane":
                n = self.add_plane(f"{base}:plane", src, normal)
            elif kind == "clip":
                n = self.add_clip(
                    f"{base}:clip", src, normal, side=self.add_clip_side.value
                )
            elif kind == "difference":
                others = [k for k in self.volume_keys() if k != src]
                b = (
                    None
                    if len(self._conditions) > 1 or not others
                    else others[0]
                )
                n = self.add_difference(f"{base}:diff", src, b, 0)
            else:
                n = self.add_region(
                    f"{base}:shell", src, self.raw_keys()[0], mode="shell"
                )
            self.add_layer_msg.value = f"Created <b>{n}</b>."
            self.add_layer_name.value = ""
        except Exception as e:
            self.add_layer_msg.value = (
                f"<span style='color:#d62728'>{type(e).__name__}: {e}</span>"
            )

    def _on_export_mesh_clicked(self, name):
        w = self._layer_widgets[name]
        try:
            out = self.export_mesh(
                name, w["x_path"].value, frame=w["x_frame"].value
            )
            w["x_msg"].value = f"Saved <code>{out}</code>"
        except Exception as e:
            w[
                "x_msg"
            ].value = (
                f"<span style='color:#d62728'>{type(e).__name__}: {e}</span>"
            )
