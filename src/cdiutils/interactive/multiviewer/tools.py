"""Analysis tools for :class:`MultiVolumeViewer` (mixin): probe and line profiles,
facet classification, surface-shell / core statistics, orthoslices and linked
cameras."""

import csv
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display

from . import analysis as mva
from . import geometry as geo


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.4g}"


def _html_table(rows, cols):
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{_fmt(r.get(c)) if isinstance(r.get(c), float) else r.get(c, '')}</td>"
            for c in cols
        )
        + "</tr>"
        for r in rows
    )
    return (
        "<div style='overflow-x:auto;max-width:100%'>"
        f"<table class='mv-table'><tr>{head}</tr>{body}</table></div>"
    )


class ToolsMixin:
    # =========================================================================
    # UI
    # =========================================================================
    def _build_tools_ui(self):
        L = widgets.Layout
        D = {"description_width": "80px"}
        self._probe_points = []
        self._ortho_center = None
        self._camera_group = None
        self._cam_sync = False

        # --- probe & profile ---
        self.probe_toggle = widgets.ToggleButton(
            value=False,
            description="Probe clicks",
            icon="crosshairs",
            tooltip="Click a surface to read all volumes there",
            layout=L(width="auto"),
        )
        self.probe_clear = widgets.Button(
            description="Clear", layout=L(width="auto")
        )
        self.probe_to_origin = widgets.Button(
            description="Last point → origin",
            layout=L(width="auto"),
            tooltip="Move the edited plane/clip origin (or the orthoslice centre) here",
        )
        self.probe_out = widgets.HTML(
            "<i>Turn on 'Probe clicks' and click on a surface.</i>"
        )
        self.profile_fields = widgets.SelectMultiple(
            options=[],
            description="fields",
            rows=4,
            style=D,
            layout=L(width="95%"),
        )
        self.profile_n = widgets.IntText(
            value=200, description="points", style=D, layout=L(width="45%")
        )
        self.profile_btn = widgets.Button(
            description="Profile A → B", layout=L(width="auto")
        )
        self.profile_csv = widgets.Text(
            value="profile.csv",
            description="CSV",
            style=D,
            layout=L(width="60%"),
        )
        self.profile_csv_btn = widgets.Button(
            description="Save CSV", layout=L(width="auto")
        )
        self.profile_plot = widgets.Output()
        self._last_profile = None
        probe_box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.probe_toggle,
                        self.probe_clear,
                        self.probe_to_origin,
                    ],
                    layout=L(flex_flow="row wrap"),
                ),
                self.probe_out,
                self.profile_fields,
                widgets.HBox([self.profile_n, self.profile_btn]),
                self.profile_plot,
                widgets.HBox([self.profile_csv, self.profile_csv_btn]),
            ]
        )
        self.probe_toggle.observe(
            lambda ch: self._update_probe_trace(), names="value"
        )
        self.probe_clear.on_click(lambda _b: self.clear_probe())
        self.probe_to_origin.on_click(self._on_probe_to_origin)
        self.profile_btn.on_click(self._on_profile_clicked)
        self.profile_csv_btn.on_click(self._on_profile_csv)

        # --- facets ---
        self.facet_matrix = widgets.Textarea(
            value="1 0 0; 0 1 0; 0 0 1",
            description="U (rows)",
            rows=2,
            style=D,
            layout=L(width="95%"),
            tooltip="n_crystal = U @ n_lab (lab = physical x, y, z of the volume)",
        )
        self.facet_fams = widgets.Text(
            value="100, 110, 111",
            description="families",
            style=D,
            layout=L(width="95%"),
        )
        self.facet_tol = widgets.FloatSlider(
            value=10.0,
            min=1.0,
            max=30.0,
            step=0.5,
            description="tolerance (°)",
            continuous_update=False,
            style=D,
            layout=L(width="95%"),
        )
        self.facet_apply = widgets.Button(
            description="Apply", layout=L(width="auto")
        )
        self.facet_layer = widgets.Dropdown(
            options=[], description="surface", style=D, layout=L(width="95%")
        )
        self.facet_field = widgets.Dropdown(
            options=[],
            description="statistic of",
            style=D,
            layout=L(width="95%"),
        )
        self.facet_by = widgets.Dropdown(
            options=[
                ("family", "family"),
                ("individual plane", "plane"),
                ("patches (connected facets)", "patch"),
            ],
            description="group by",
            style=D,
            layout=L(width="95%"),
        )
        self.facet_btn = widgets.Button(
            description="Compute facet table", layout=L(width="auto")
        )
        self.facet_out = widgets.HTML(
            "<i>Set 'color by' to (facet family) or (facet patch) on an "
            "isosurface to see the facets.</i>"
        )
        self.patch_min = widgets.IntText(
            value=self.patch_min_faces,
            description="min faces",
            tooltip="Smaller patches are not listed (grey)",
            style=D,
            layout=L(width="48%"),
        )
        self.patch_labels = widgets.Checkbox(
            value=False, description="(hkl) labels", indent=False
        )
        self.patch_show = widgets.Button(
            description="Colour surface by patch", layout=L(width="auto")
        )
        self.facet_csv = widgets.Text(
            value="facets.csv",
            description="CSV",
            style=D,
            layout=L(width="60%"),
        )
        self.facet_csv_btn = widgets.Button(
            description="Save CSV", layout=L(width="auto")
        )
        facet_box = widgets.VBox(
            [
                self.facet_matrix,
                self.facet_fams,
                self.facet_tol,
                self.facet_apply,
                self.facet_layer,
                self.facet_field,
                self.facet_by,
                widgets.HBox([self.patch_min, self.patch_labels]),
                widgets.HBox([self.facet_btn, self.patch_show]),
                widgets.HBox([self.facet_csv, self.facet_csv_btn]),
                self.facet_out,
            ]
        )
        self.facet_apply.on_click(self._on_facet_apply)
        self.facet_btn.on_click(self._on_facet_table)
        self.patch_show.on_click(self._on_patch_show)
        self.facet_csv_btn.on_click(self._on_facet_csv)

        # --- surface shell / core ---
        self.reg_field = widgets.Dropdown(
            options=[], description="field", style=D, layout=L(width="95%")
        )
        self.reg_support = widgets.Dropdown(
            options=[], description="support", style=D, layout=L(width="95%")
        )
        # 0 means automatic (NaN is not allowed in widget messages)
        self.reg_thr = widgets.FloatText(
            value=0.0,
            description="support ≥",
            style=D,
            layout=L(width="95%"),
            tooltip="0 = 30% of the support maximum",
        )
        self.reg_depth = widgets.FloatText(
            value=0.0,
            description="shell depth",
            style=D,
            layout=L(width="95%"),
            tooltip="Physical units; 0 = 2 voxels",
        )
        self.reg_btn = widgets.Button(
            description="Compute", layout=L(width="auto")
        )
        self.reg_add = widgets.Button(
            description="Add shell + core layers", layout=L(width="auto")
        )
        self.reg_out = widgets.HTML("")
        self.reg_plot = widgets.Output()
        reg_box = widgets.VBox(
            [
                self.reg_field,
                self.reg_support,
                self.reg_thr,
                self.reg_depth,
                widgets.HBox([self.reg_btn, self.reg_add]),
                self.reg_out,
                self.reg_plot,
            ]
        )
        self.reg_btn.on_click(self._on_region_stats)
        self.reg_add.on_click(self._on_region_add)

        # --- orthoslices ---
        self.ortho_show = widgets.Checkbox(
            value=False, description="Show orthoslices", indent=False
        )
        self.ortho_field = widgets.Dropdown(
            options=[], description="field", style=D, layout=L(width="95%")
        )
        self.ortho_cmap = widgets.Dropdown(
            options=self.cmap_options,
            value="gray",
            description="cmap",
            style=D,
            layout=L(width="95%"),
        )
        self.ortho_cross3d = widgets.Checkbox(
            value=True, description="Show centre in 3D", indent=False
        )
        self.ortho_center_btn = widgets.Button(
            description="Centre", layout=L(width="auto")
        )
        ortho_ctrl = widgets.VBox(
            [
                self.ortho_show,
                self.ortho_field,
                self.ortho_cmap,
                self.ortho_cross3d,
                self.ortho_center_btn,
                widgets.HTML(
                    "<i>Click inside a slice to move the centre.</i>"
                ),
            ]
        )
        self._ortho_figs = []
        for _ in range(3):
            f = go.FigureWidget(
                layout=dict(
                    height=260,
                    margin=dict(l=40, r=10, t=30, b=35),
                    template="plotly_white",
                )
            )
            f.add_trace(
                go.Heatmap(z=[[0]], showscale=False, hoverinfo="x+y+z")
            )
            f.layout.yaxis.scaleanchor = "x"
            self._ortho_figs.append(f)
        self.ortho_box = widgets.HBox(
            self._ortho_figs, layout=L(display="none", width="100%")
        )
        for i, f in enumerate(self._ortho_figs):
            f.data[0].on_click(
                lambda tr, pts, sel, i=i: self._on_ortho_click(i, pts)
            )
        self.ortho_show.observe(self._on_ortho_toggle, names="value")
        for wd in (self.ortho_field, self.ortho_cmap):
            wd.observe(lambda ch: self._ortho_update(), names="value")
        self.ortho_cross3d.observe(
            lambda ch: self._update_probe_trace(), names="value"
        )
        self.ortho_center_btn.on_click(lambda _b: self.set_ortho_center(None))

        # --- state ---
        self.state_path = widgets.Text(
            value="viewer_state.json",
            description="file",
            style=D,
            layout=L(width="95%"),
        )
        self.state_save = widgets.Button(
            description="Save view", layout=L(width="auto")
        )
        self.state_load = widgets.Button(
            description="Load view", layout=L(width="auto")
        )
        self.state_msg = widgets.HTML("")
        state_box = widgets.VBox(
            [
                self.state_path,
                widgets.HBox([self.state_save, self.state_load]),
                self.state_msg,
                widgets.HTML(
                    "<i>Saves layers, styles, transforms, camera and tool settings "
                    "(not the volumes).</i>"
                ),
            ]
        )
        self.state_save.on_click(lambda _b: self._state_io(save=True))
        self.state_load.on_click(lambda _b: self._state_io(save=False))

        # --- help & tutorial ---
        self.help_out = widgets.Output(layout=L(width="100%"))
        help_btns = []
        for label, topic in (
            ("Overview", None),
            ("Actions", "actions"),
            ("Edit keys", "edit"),
        ):
            b = widgets.Button(description=label, layout=L(width="auto"))
            b.on_click(lambda _b, t=topic: self._show_help_in_panel(t))
            help_btns.append(b)
        tuto_btn = widgets.Button(
            description="Start tutorial",
            button_style="info",
            layout=L(width="auto"),
        )
        tuto_btn.on_click(lambda _b: self._show_tutorial_in_panel())
        help_box = widgets.VBox(
            [
                widgets.HBox(
                    help_btns + [tuto_btn], layout=L(flex_flow="row wrap")
                ),
                widgets.HTML(
                    "<i>In code: <code>v.help()</code>, <code>v.help('slice')</code>, "
                    "<code>v.tutorial()</code>.</i>"
                ),
                self.help_out,
            ]
        )

        return [
            self._section("Help & tutorial", help_box),
            self._section("Probe & line profile", probe_box),
            self._section("Facets", facet_box),
            self._section("Surface shell / core statistics", reg_box),
            self._section("Orthoslices", ortho_ctrl),
            self._section("Save / load view", state_box),
        ]

    def _show_help_in_panel(self, topic):
        self.help_out.clear_output()
        with self.help_out:
            self.help(topic)

    def _show_tutorial_in_panel(self):
        self.help_out.clear_output()
        with self.help_out:
            display(self.tutorial())

    def _tools_after_data(self):
        self.facet_matrix.value = "; ".join(
            " ".join(f"{v:g}" for v in row) for row in self.orientation_matrix
        )
        self.facet_fams.value = ", ".join(
            lab.strip("{}") for lab, _ in self.facet_families
        )
        self.facet_tol.value = self.facet_tolerance
        if self._ortho_center is None and self.frame is not None:
            self._ortho_center = self.frame.center_phys()
        if self.ortho_show.value:
            self._ortho_update()

    def _tools_refresh_options(self):
        vols = self.volume_keys()
        opts = [(k, k) for k in vols]
        cur = set(self.profile_fields.value)
        self.profile_fields.options = vols
        self.profile_fields.value = tuple(
            k for k in vols if k in cur
        ) or tuple(vols[:1])
        for dd in (self.reg_field, self.reg_support, self.ortho_field):
            self._set_options(dd, opts)
        self._set_options(
            self.facet_layer,
            [
                (k, k)
                for k, s in self._layers.items()
                if s["type"] in ("raw", "diff", "region", "clip")
            ],
        )
        self._set_options(
            self.facet_field, [("(area only)", None)] + opts, fallback=None
        )

    # =========================================================================
    # Extra traces (probe markers, orthoslice centre)
    # =========================================================================
    def _extra_traces(self):
        return [self._probe_trace()]

    def _probe_trace(self):
        pts = list(self._probe_points[-2:])
        text = ["A", "B"][: len(pts)] if len(pts) == 2 else ["A"] * len(pts)
        if (
            self.ortho_show.value
            and self.ortho_cross3d.value
            and self._ortho_center is not None
        ):
            pts.append(self._ortho_center)
            text.append("+")
        P = (
            self.frame.phys_to_plot(np.asarray(pts))
            if pts
            else np.zeros((0, 3))
        )
        return go.Scatter3d(
            name="__probe__",
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers+text",
            text=text,
            textposition="top center",
            marker=dict(
                size=6, color="#ff1493", line=dict(width=1, color="black")
            ),
            showlegend=False,
            hoverinfo="text",
        )

    def _update_probe_trace(self):
        idx = [i for i, t in enumerate(self.fig.data) if t.name == "__probe__"]
        if not idx:
            return self._rebuild()
        d = self._probe_trace().to_plotly_json()
        d.pop("type", None)
        self.fig.data[idx[0]].update(d)

    # =========================================================================
    # Probe and profiles
    # =========================================================================
    def _on_trace_click(self, trace, points, selector):
        if (
            not self.probe_toggle.value
            or not points.point_inds
            or str(trace.name).startswith("__")
        ):
            return
        p_plot = np.array(
            [points.xs[0], points.ys[0], points.zs[0]], dtype=float
        )
        if trace.name in self._layers:
            p_plot = self._from_display(p_plot, trace.name)
        self.probe(self.frame.plot_to_phys(p_plot), coords="physical")

    def probe(self, point, coords="plot"):
        """Add a probe point and return the values of every volume there."""
        p = np.asarray(point, dtype=float).reshape(3)
        p = (
            self.frame.plot_to_phys(p)
            if coords == "plot"
            else self.frame.idx_to_phys(p)
            if coords == "index"
            else p
        )
        self._probe_points.append(p)
        vals = {
            k: float(self.sample(k, p[None])[0]) for k in self.volume_keys()
        }
        pp = self.frame.phys_to_plot(p)
        u = f" {self.unit}" if self.unit else ""
        html = f"<b>Point</b> (X, Y, Z) = ({pp[0]:.4g}, {pp[1]:.4g}, {pp[2]:.4g}){u}<br>"
        html += _html_table(
            [{"field": k, "value": v} for k, v in vals.items()],
            ["field", "value"],
        )
        if len(self._probe_points) >= 2:
            d = float(
                np.linalg.norm(self._probe_points[-1] - self._probe_points[-2])
            )
            html += f"<br><b>|AB|</b> = {d:.4g}{u}"
        self.probe_out.value = html
        self._update_probe_trace()
        return vals

    def clear_probe(self):
        self._probe_points = []
        self.probe_out.value = "<i>Probe cleared.</i>"
        self._update_probe_trace()

    def line_profile(
        self, p0=None, p1=None, fields=None, n=200, coords="plot"
    ):
        """Sample volumes along a segment. Defaults to the last two probe points.

        Returns ``(distance, {field: values})``; ``coords`` of ``p0``/``p1``:
        ``"plot"``, ``"physical"`` or ``"index"``.
        """
        if p0 is None or p1 is None:
            if len(self._probe_points) < 2:
                raise ValueError("Probe two points first (or pass p0 and p1).")
            a, b = self._probe_points[-2], self._probe_points[-1]
        else:
            conv = {
                "plot": self.frame.plot_to_phys,
                "index": self.frame.idx_to_phys,
            }.get(coords, lambda x: np.asarray(x, float))
            a, b = conv(np.asarray(p0, float)), conv(np.asarray(p1, float))
        fields = list(fields or self.volume_keys())
        pts, s = mva.line_points(a, b, n)
        return s, {f: self.sample(f, pts) for f in fields}

    def _on_profile_clicked(self, _b):
        self.profile_plot.clear_output()
        with self.profile_plot:
            try:
                s, prof = self.line_profile(
                    fields=self.profile_fields.value,
                    n=max(2, int(self.profile_n.value)),
                )
            except Exception as e:
                print(e)
                return
            self._last_profile = (s, prof)
            fig, axes = plt.subplots(
                len(prof),
                1,
                figsize=(5, 1.8 * len(prof)),
                sharex=True,
                squeeze=False,
            )
            for ax, (k, v) in zip(axes[:, 0], prof.items()):
                ax.plot(s, v, lw=1.5)
                ax.set_ylabel(k)
                ax.grid(alpha=0.3)
            axes[-1, 0].set_xlabel(
                f"distance A → B{f' ({self.unit})' if self.unit else ''}"
            )
            fig.tight_layout()
            plt.show()

    def _on_profile_csv(self, _b):
        if self._last_profile is None:
            self.probe_out.value += "<br><i>Compute a profile first.</i>"
            return
        s, prof = self._last_profile
        path = Path(self.profile_csv.value or "profile.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["distance"] + list(prof))
            for i in range(len(s)):
                wr.writerow([s[i]] + [prof[k][i] for k in prof])
        self.probe_out.value += f"<br>Saved <code>{path}</code>"

    def _on_probe_to_origin(self, _b):
        if not self._probe_points:
            return
        p = self._probe_points[-1]
        name = self.edit_key.value
        if name and self._layers[name]["type"] in ("plane", "clip"):
            self.set_layer_params(
                name, origin=self.frame.phys_to_plot(p), offset=0.0
            )
        else:
            self.set_ortho_center(p, coords="physical")

    # =========================================================================
    # Facets
    # =========================================================================
    def set_facet_settings(
        self, orientation_matrix=None, families=None, tolerance=None
    ):
        """Update the orientation matrix ``U`` (n_crystal = U @ n_lab), families and tolerance."""
        if orientation_matrix is not None:
            self.orientation_matrix = mva.parse_matrix(orientation_matrix)
        if families is not None:
            self.facet_families = mva.parse_families(families)
        if tolerance is not None:
            self.facet_tolerance = float(tolerance)
        self._tools_after_data()
        names = [
            k
            for k in self._layers
            if self._style(k).get("color_by") in ("__facet__", "__patch__")
        ]
        if names:
            self._refresh(None)

    def _on_facet_apply(self, _b):
        try:
            self.set_facet_settings(
                self.facet_matrix.value,
                self.facet_fams.value,
                self.facet_tol.value,
            )
            self.facet_out.value = "Facet settings applied."
        except Exception as e:
            self.facet_out.value = f"<span style='color:#d62728'>{e}</span>"

    def facet_stats(self, layer, field=None, by="family"):
        """Area-weighted statistics of ``field`` per facet family (or individual plane)
        on the isosurface of ``layer`` (normals taken before the layer transform)."""
        layer = self._resolve_layer_name(layer)
        geom = self._mesh_geometry(
            layer, self._layers[layer], self._style(layer)
        )
        if geom is None:
            raise ValueError(
                f"Layer '{layer}' has no surface at the current iso level."
            )
        verts_idx, faces, vn = geom
        vp = self.frame.idx_to_phys(verts_idx)
        fn, areas = geo.face_normals(vp, faces)
        fn = geo.orient_faces_like(fn, vn, faces)
        fam, plane, ang = mva.classify_normals(
            fn,
            self.facet_families,
            self.orientation_matrix,
            self.facet_tolerance,
        )
        vals = None
        if field:
            vals = np.asarray(
                self._interp(field, True)(verts_idx[faces].mean(axis=1)),
                dtype=float,
            )
            scale = self._unit_of(field)[1]
            if scale != 1.0:
                vals = vals * scale
        if by == "patch":
            patch = self._patch_ids(layer, verts_idx, faces, vn)
            U = self.orientation_matrix
            ncry = fn @ np.asarray(U, dtype=float).T
            ncry /= np.maximum(np.linalg.norm(ncry, axis=1), 1e-30)[:, None]
            cen = vp[faces].mean(axis=1)
            return mva.patch_table(
                patch,
                plane,
                fam,
                areas,
                ncry,
                cen,
                [lab for lab, _ in self.facet_families],
                vals,
            )
        if by == "plane":
            labels = sorted(set(plane[fam >= 0]))
            ids = np.array(
                [labels.index(p) if p in labels else -1 for p in plane]
            )
            return mva.facet_table(ids, areas, labels, vals)
        return mva.facet_table(
            fam, areas, [lab for lab, _ in self.facet_families], vals
        )

    PATCH_COLUMNS = (
        "patch",
        "plane",
        "family",
        "faces",
        "area_frac",
        "deviation_deg",
        "hkl_of_mean",
    )

    def facet_patches(
        self,
        layer,
        field=None,
        min_faces=None,
        min_area_frac=None,
        labels=False,
        colour=False,
        label_min_frac=0.01,
    ):
        """Facets of an isosurface as connected patches, with statistics.

        Faces are classified by crystal plane (facet families, orientation
        matrix and tolerance of :meth:`set_facet_settings`); edge-connected
        faces of the same plane form a patch. Returns one row per patch:
        plane, family, faces, area (``unit``²), area fraction, mean normal
        (crystal frame), its deviation from the plane (deg), the (hkl) of the
        mean normal (cdiutils' facet-analysis rule), centroid and, with
        ``field``, the area-weighted mean / std of the field on the patch.

        ``labels``: add a layer ``"<layer>:facets"`` with the plane of each
        patch written at its centre: ``True`` / ``"front"`` for the patches
        facing the current camera (labels are not hidden behind surfaces),
        ``"all"`` for every patch; patches below ``label_min_frac`` of the
        area are not labelled. ``colour``: colour the surface by patch.
        """
        layer = self._resolve_layer_name(layer)
        if min_faces is not None:
            self.patch_min_faces = int(min_faces)
        if min_area_frac is not None:
            self.patch_min_area = float(min_area_frac)
        self._patch_cache.clear()
        rows = self.facet_stats(layer, field, by="patch")
        if colour:
            self.set_layer_params(layer, color_by="__patch__")
        name = f"{layer}:facets"
        if name in self._layers:
            self.delete_layer(name, cascade=False)
        if labels and rows:
            d = 0.03 * self.frame.diagonal()
            eye = self._current_camera()[0]
            labs = []
            for r in rows:
                if r["area_frac"] < label_min_frac:
                    continue
                n = self._lab_normal(r)
                if labels != "all" and self.frame.phys_to_plot(n) @ eye <= 0:
                    continue  # facing away from the camera
                labs.append(
                    (np.array([r["cx"], r["cy"], r["cz"]]) + d * n, r["plane"])
                )
            self.add_lines(name, [], coords="physical", labels=labs)
            self._layer_transform[name] = dict(
                self._layer_transform.get(layer) or {}
            )
        return rows

    def _lab_normal(self, row):
        """Patch normal back in the physical frame (labels sit outside)."""
        n = np.array([row["nx"], row["ny"], row["nz"]])
        n = np.linalg.solve(np.asarray(self.orientation_matrix, float), n)
        return n / (np.linalg.norm(n) or 1.0)

    def save_facet_csv(self, path, layer, field=None, by="patch"):
        """Write :meth:`facet_stats` / :meth:`facet_patches` rows to CSV."""
        import csv
        from pathlib import Path

        rows = self.facet_stats(layer, field, by=by)
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return path

    def _patch_ids(self, layer, verts_idx, faces, vertex_normals):
        key = (
            layer,
            id(faces),
            len(faces),
            tuple(lab for lab, _ in self.facet_families),
            float(self.facet_tolerance),
            tuple(np.round(np.ravel(self.orientation_matrix), 9)),
            self.patch_min_faces,
            self.patch_min_area,
        )
        hit = self._patch_cache.get(key)
        if hit is not None and hit[0] is faces:
            return hit[1]
        vp = self.frame.idx_to_phys(verts_idx)
        fn, areas = geo.face_normals(vp, faces)
        fn = geo.orient_faces_like(fn, vertex_normals, faces)
        fam, plane, _ = mva.classify_normals(
            fn,
            self.facet_families,
            self.orientation_matrix,
            self.facet_tolerance,
        )
        ids = mva.face_patches(
            faces,
            plane,
            fam,
            len(verts_idx),
            areas,
            self.patch_min_faces,
            self.patch_min_area,
        )
        while len(self._patch_cache) >= 8:
            self._patch_cache.pop(next(iter(self._patch_cache)))
        self._patch_cache[key] = (faces, ids)
        return ids

    def _on_facet_table(self, _b):
        try:
            by = self.facet_by.value
            if by == "patch":
                rows = self.facet_patches(
                    self.facet_layer.value,
                    self.facet_field.value,
                    min_faces=self.patch_min.value,
                    labels=self.patch_labels.value,
                )
                cols = list(self.PATCH_COLUMNS)
            else:
                rows = self.facet_stats(
                    self.facet_layer.value, self.facet_field.value, by
                )
                cols = ["family", "faces", "area_frac"]
            if self.facet_field.value:
                cols += ["mean", "std"]
            self.facet_out.value = _html_table(rows, cols)
        except Exception as e:
            self.facet_out.value = f"<span style='color:#d62728'>{e}</span>"

    def _on_patch_show(self, _b):
        try:
            self.patch_min_faces = int(self.patch_min.value)
            self._patch_cache.clear()
            self.set_layer_params(self.facet_layer.value, color_by="__patch__")
            self._select_edit_layer(self.facet_layer.value)
        except Exception as e:
            self.facet_out.value = f"<span style='color:#d62728'>{e}</span>"

    def _on_facet_csv(self, _b):
        try:
            p = self.save_facet_csv(
                self.facet_csv.value,
                self.facet_layer.value,
                self.facet_field.value,
                self.facet_by.value,
            )
            self.facet_out.value += f"<br>Saved <code>{p.resolve()}</code>"
        except Exception as e:
            self.facet_out.value = f"<span style='color:#d62728'>{e}</span>"

    # =========================================================================
    # Surface shell / core
    # =========================================================================
    def region_stats(self, field, support, threshold=None, depth=None):
        """Statistics of ``field`` in the surface shell, the core and the whole support."""
        sup = np.asarray(
            self.get_volume(self._check_volume(support)), dtype=float
        )
        thr = (
            0.3 * float(np.nanmax(sup))
            if threshold is None or not np.isfinite(threshold)
            else float(threshold)
        )
        depth = (
            2 * float(np.max(self.voxel_size))
            if depth is None or not np.isfinite(depth)
            else float(depth)
        )
        support_mask = np.nan_to_num(sup, nan=-np.inf) >= thr
        shell, core, _ = mva.shell_core_masks(
            support_mask, self.voxel_size, depth
        )
        f = self.get_volume(self._check_volume(field))
        return {
            "threshold": thr,
            "depth": depth,
            "shell": mva.masked_stats(f, shell),
            "core": mva.masked_stats(f, core),
            "all": mva.masked_stats(f, support_mask),
            "_masks": (shell, core),
        }

    def _region_inputs(self):
        t = float(self.reg_thr.value or 0)
        thr = t if np.isfinite(t) and t > 0 else None
        dep = (
            float(self.reg_depth.value)
            if np.isfinite(self.reg_depth.value) and self.reg_depth.value > 0
            else None
        )
        return self.reg_field.value, self.reg_support.value, thr, dep

    def _on_region_stats(self, _b):
        field, support, thr, dep = self._region_inputs()
        try:
            r = self.region_stats(field, support, thr, dep)
        except Exception as e:
            self.reg_out.value = f"<span style='color:#d62728'>{e}</span>"
            return
        rows = [{"region": k, **r[k]} for k in ("shell", "core", "all")]
        u = f" {self.unit}" if self.unit else ""
        self.reg_out.value = (
            f"support ≥ {r['threshold']:.4g}, shell depth {r['depth']:.4g}{u}<br>"
            + _html_table(
                rows, ["region", "n", "mean", "std", "median", "p5", "p95"]
            )
        )
        shell, core = r["_masks"]
        f = np.asarray(self.get_volume(field), dtype=float)
        self.reg_plot.clear_output()
        with self.reg_plot:
            fig, ax = plt.subplots(figsize=(5, 2.6))
            for m, lab in ((shell, "shell"), (core, "core")):
                v = f[m]
                v = v[np.isfinite(v)]
                if v.size:
                    ax.hist(
                        v,
                        bins=80,
                        histtype="step",
                        density=True,
                        lw=1.6,
                        label=lab,
                    )
            ax.set_xlabel(field)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            plt.show()

    def _on_region_add(self, _b):
        field, support, thr, dep = self._region_inputs()
        try:
            a = self.add_region(
                f"{field}:shell", field, support, thr, "shell", dep
            )
            b = self.add_region(
                f"{field}:core",
                field,
                support,
                thr,
                "core",
                dep,
                visible=False,
            )
            self.reg_out.value = f"Added <b>{a}</b> and <b>{b}</b>."
        except Exception as e:
            self.reg_out.value = f"<span style='color:#d62728'>{e}</span>"

    # =========================================================================
    # Orthoslices
    # =========================================================================
    def _on_ortho_toggle(self, ch):
        self.ortho_box.layout.display = "" if ch["new"] else "none"
        if ch["new"]:
            self._ortho_update()
        self._update_probe_trace()

    def set_ortho_center(self, point=None, coords="plot"):
        """Move the orthoslice centre (``None`` = volume centre)."""
        if point is None:
            p = self.frame.center_phys()
        else:
            p = np.asarray(point, dtype=float)
            p = (
                self.frame.plot_to_phys(p)
                if coords == "plot"
                else self.frame.idx_to_phys(p)
                if coords == "index"
                else p
            )
        lo, hi = self.frame.bounds_phys()
        self._ortho_center = np.clip(p, lo, hi)
        self._ortho_update()
        self._update_probe_trace()

    def _ortho_panels(self):
        """For each panel: (fixed plot axis, horizontal plot axis, vertical plot axis)."""
        return [(0, 1, 2), (1, 0, 2), (2, 0, 1)]

    def _ortho_update(self):
        if (
            not self.ortho_show.value
            or self.frame is None
            or not self.ortho_field.value
        ):
            return
        vol = np.asarray(self.get_volume(self.ortho_field.value), dtype=float)
        c = (
            self._ortho_center
            if self._ortho_center is not None
            else self.frame.center_phys()
        )
        names = "XYZ"
        lo_v, hi_v = geo.finite_range(vol)
        cs = self._colorscale(self.ortho_cmap.value)
        for fig, (f, h, v) in zip(self._ortho_figs, self._ortho_panels()):
            pf, ph, pv = (
                self.frame.plot_axis_to_phys("xyz"[a]) for a in (f, h, v)
            )
            df, dv = geo.DIM[pf], geo.DIM[pv]
            vox = self.frame.voxel_xyz
            i = int(
                np.clip(
                    round(c["xyz".index(pf)] / vox["xyz".index(pf)]),
                    0,
                    vol.shape[df] - 1,
                )
            )
            img = np.take(vol, i, axis=df)
            rem = [d for d in (0, 1, 2) if d != df]
            if rem.index(dv) != 0:
                img = img.T
            xh, yv = self.frame.coords(ph), self.frame.coords(pv)
            ch_, cv_ = c["xyz".index(ph)], c["xyz".index(pv)]
            u = f" {self.unit}" if self.unit else ""
            with fig.batch_update():
                fig.data[0].update(
                    z=img, x=xh, y=yv, colorscale=cs, zmin=lo_v, zmax=hi_v
                )
                fig.layout.title = dict(
                    text=f"{names[f]} = {i * vox['xyz'.index(pf)]:.4g}{u}",
                    font=dict(size=12),
                )
                fig.layout.xaxis.title = names[h]
                fig.layout.yaxis.title = names[v]
                fig.layout.shapes = [
                    dict(
                        type="line",
                        x0=ch_,
                        x1=ch_,
                        y0=yv[0],
                        y1=yv[-1],
                        line=dict(color="#ff1493", width=1),
                    ),
                    dict(
                        type="line",
                        x0=xh[0],
                        x1=xh[-1],
                        y0=cv_,
                        y1=cv_,
                        line=dict(color="#ff1493", width=1),
                    ),
                ]
                fig.layout.template = (
                    "plotly_dark" if self._dark() else "plotly_white"
                )

    def _on_ortho_click(self, panel, points):
        if not points.xs:
            return
        f, h, v = self._ortho_panels()[panel]
        c = np.array(self._ortho_center, dtype=float)
        c["xyz".index(self.frame.plot_axis_to_phys("xyz"[h]))] = points.xs[0]
        c["xyz".index(self.frame.plot_axis_to_phys("xyz"[v]))] = points.ys[0]
        self.set_ortho_center(c, coords="physical")

    # =========================================================================
    # Linked cameras
    # =========================================================================
    def link_cameras(self, *others):
        """Keep the cameras of this viewer and ``others`` synchronised."""
        group = [self, *others]
        for v in group:
            v._camera_group = group
            if not getattr(v, "_cam_handler", False):
                v.fig.layout.on_change(v._on_camera_change, "scene.camera")
                v._cam_handler = True
        return group

    @staticmethod
    def link(*viewers):
        return viewers[0].link_cameras(*viewers[1:])

    def unlink_cameras(self):
        for v in self._camera_group or []:
            v._camera_group = None

    def _on_camera_change(self, layout, camera):
        if self._cam_sync or not self._camera_group:
            return
        cam = (
            camera.to_plotly_json()
            if hasattr(camera, "to_plotly_json")
            else camera
        )
        for v in self._camera_group:
            if v is self:
                continue
            v._cam_sync = True
            try:
                v.fig.layout.scene.camera = cam
            finally:
                v._cam_sync = False

    # =========================================================================
    # State I/O buttons
    # =========================================================================
    def _state_io(self, save):
        try:
            if save:
                p = self.save_state(self.state_path.value)
                self.state_msg.value = f"Saved <code>{p}</code>"
            else:
                self.load_state(self.state_path.value)
                self.state_msg.value = "View restored."
        except Exception as e:
            self.state_msg.value = (
                f"<span style='color:#d62728'>{type(e).__name__}: {e}</span>"
            )
