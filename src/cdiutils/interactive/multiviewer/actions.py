"""
Scripted actions for :class:`MultiVolumeViewer` (mixin).

A script is a list of steps. Every step can be written in the short form

    {"show": "phase"}
    {"edit": "amplitude", "thr": 0.3, "cmap": "viridis"}
    {"slice": "zmid", "source": "phase", "axis": "z"}

(the action name is the key, its value is the main argument), as a tuple
``("edit", "amplitude", {"thr": 0.3})``, as a string ``"show phase"``, or in
the former long form ``{"action": "edit", "layer": ..., "params": {...}}``.
Scripts can also be YAML/JSON files, or built with ``viewer.script()``.

Every script is checked before anything runs (unknown actions, parameters,
layers and values are reported together, with suggestions). Creating a layer
whose name already exists replaces it, so re-running a cell is safe. Exports
run one after the other.
"""

import asyncio
import difflib
import json
import time
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from . import analysis as mva
from . import geometry as geo

REQ = object()  # marks a required parameter


def _unit_arg(v):
    """YAML/JSON lists -> (label, scale) tuples for set_units."""
    return tuple(v) if isinstance(v, list) else v


STYLE_KEYS = {
    "thr": "isosurface level",
    "iso": "alias of thr",
    "opacity": "0 to 1",
    "op": "alias of opacity",
    "cmap": "colormap name or solid colour (gray, red, ...)",
    "color_by": "'__self__', another volume, '__x__'/'__y__'/'__z__', "
    "'__constant__', '__facet__' or '__patch__'",
    "nan_policy": "none (holes), mean, zero, min or max",
    "as_mask": "isosurface of the thresholded mask",
    "abs_iso": "isosurface of |value| (default True for volumes)",
    "show_colorbar": "show the colorbar",
    "auto_range": "colour range from the data (False = manual range)",
    "range": "(cmin, cmax) for a manual range, or a preset: minmax, p1 "
    "(1-99 %), p5 (5-95 %), sym (± max |v|), sym99 (± 99 % |v|)",
    "range_mode": "minmax, p1, p5, sym, sym99 or manual",
    "range_all": "one colour range for all conditions",
    "range_min": "lower bound of the colour-range slider",
    "range_max": "upper bound of the colour-range slider",
    "color": "colour of lines and arrows",
    "width": "width of lines and arrows",
    "light_ambient": "lighting",
    "light_diffuse": "lighting",
    "light_specular": "lighting",
    "light_roughness": "lighting",
    "light_fresnel": "lighting",
}
GEOMETRY_KEYS = {
    "pos": "slice index along its axis",
    "slice_pos": "alias of pos",
    "axis": "slice axis (plot axis x, y or z)",
    "thickness": "averaging half-thickness",
    "normal": "plane/clip normal (plot coordinates)",
    "origin": "plane/clip origin (plot coordinates)",
    "offset": "shift along the normal",
    "side": "clip side kept: up (+n) or down (-n)",
    "smooth": "planar clip cap",
    "extent": "plane half-size",
    "resolution": "plane sampling",
    "source": "volume sampled by a slice/plane/clip",
    "a": "difference: first volume",
    "b": "difference: second volume (None = reference condition)",
    "ref": "difference: reference condition",
    "field": "region: field shown",
    "support": "region: support volume",
    "threshold": "region: support threshold",
    "mode": "region: shell or core",
    "depth": "region: shell depth",
}
TRANSFORM_KEYS = {
    f"{k}_{a}": "rotation (deg)" if k == "rot" else "translation"
    for k in ("rot", "trans")
    for a in "xyz"
}
EDIT_KEYS = {**STYLE_KEYS, **GEOMETRY_KEYS, **TRANSFORM_KEYS}

CAMERA_VIEWS = {
    "iso": ((1.5, 1.5, 1.5), (0, 0, 1)),
    "x": ((2.2, 0, 0), (0, 0, 1)),
    "-x": ((-2.2, 0, 0), (0, 0, 1)),
    "y": ((0, 2.2, 0), (0, 0, 1)),
    "-y": ((0, -2.2, 0), (0, 0, 1)),
    "z": ((0, 0, 2.2), (0, 1, 0)),
    "-z": ((0, 0, -2.2), (0, 1, 0)),
}


class P:
    """Parameter of an action."""

    def __init__(self, default=REQ, doc="", choices=None, ref=None):
        self.default, self.doc, self.choices, self.ref = (
            default,
            doc,
            choices,
            ref,
        )


class Action:
    """Description of one scripted action (used for running, checking and help)."""

    def __init__(
        self,
        name,
        doc,
        params,
        main=None,
        category="",
        example=None,
        creates=None,
        extra=None,
        export=False,
    ):
        self.name, self.doc, self.params, self.main = name, doc, params, main
        self.category, self.example, self.creates = category, example, creates
        self.extra = (
            extra  # dict of extra accepted keyword names (open params)
        )
        self.export = export

    def signature(self):
        parts = []
        for k, p in self.params.items():
            parts.append(k if p.default is REQ else f"{k}={p.default!r}")
        if self.extra:
            parts.append("**style")
        return f"{self.name}({', '.join(parts)})"


_CREATE = {
    "replace": P(True, "replace a layer that already has this name"),
    "visible": P(True, "show the new layer"),
    "select": P(False, "open the new layer in the layer panel"),
}
_EXPORT = {
    "fps": P(None, "frames per second"),
    "format": P(
        None, "mp4, gif or png (folder of frames)", ("mp4", "gif", "png")
    ),
    "name": P(None, "output name or path, without extension"),
    "smooth": P(
        None, "smoother outlines (render at 2x); default: panel checkbox"
    ),
    "flat": P(None, "one colour per triangle; default: panel checkbox"),
}

ACTIONS = {}


def _register(*actions):
    for a in actions:
        ACTIONS[a.name] = a


_register(
    # --- visibility and editing -------------------------------------------------
    Action(
        "show",
        "Show one layer or a list of layers.",
        {"layer": P(doc="layer name(s)", ref="layers")},
        main="layer",
        category="Layers",
        example={"show": "phase"},
    ),
    Action(
        "hide",
        "Hide one layer or a list of layers.",
        {"layer": P(doc="layer name(s)", ref="layers")},
        main="layer",
        category="Layers",
        example={"hide": ["phase", "amplitude"]},
    ),
    Action(
        "only",
        "Show exactly these layers and hide all others.",
        {"layers": P(doc="layer name(s)", ref="layers")},
        main="layers",
        category="Layers",
        example={"only": ["amplitude", "zmid"]},
    ),
    Action(
        "select",
        "Open a layer in the layer panel.",
        {"layer": P(ref="layer")},
        main="layer",
        category="Layers",
        example={"select": "amplitude"},
    ),
    Action(
        "edit",
        "Change style, geometry or transform parameters of a layer (see help('edit')).",
        {"layer": P(ref="layer")},
        main="layer",
        category="Layers",
        extra=EDIT_KEYS,
        example={
            "edit": "amplitude",
            "thr": 0.3,
            "cmap": "viridis",
            "opacity": 0.8,
        },
    ),
    Action(
        "rename",
        "Rename a layer (references are updated).",
        {"layer": P(ref="layer"), "new": P(doc="new name")},
        main="layer",
        category="Layers",
        example={"rename": "zmid", "new": "phase z-slice"},
    ),
    Action(
        "delete",
        "Delete a layer (and, by default, the layers built from it).",
        {
            "layer": P(ref="layer"),
            "cascade": P(True, "also delete dependent layers"),
        },
        main="layer",
        category="Layers",
        example={"delete": "zmid"},
    ),
    # --- create ------------------------------------------------------------------
    Action(
        "slice",
        "Axis-aligned slice (mean over pos ± thickness).",
        {
            "name": P(doc="layer name"),
            "source": P(
                None, "volume to slice (default: first)", ref="volume"
            ),
            "axis": P("z", "plot axis", ("x", "y", "z")),
            "pos": P(None, "voxel index (default: centre)"),
            "thickness": P(0, "half-thickness in voxels"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        extra=STYLE_KEYS,
        example={
            "slice": "zmid",
            "source": "phase",
            "axis": "z",
            "cmap": "twilight",
        },
    ),
    Action(
        "plane",
        "Arbitrary plane through the volume.",
        {
            "name": P(),
            "source": P(None, "volume sampled (default: first)", ref="volume"),
            "normal": P((0, 0, 1), "normal (plot coordinates)"),
            "origin": P(
                None, "point on the plane (plot coordinates; default: centre)"
            ),
            "offset": P(0.0, "shift along the normal"),
            "thickness": P(0.0, "averaging half-thickness"),
            "extent": P(None, "half-size of the plane"),
            "resolution": P(200, "samples per side"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        extra=STYLE_KEYS,
        example={
            "plane": "p111",
            "source": "het_strain",
            "normal": [1, 1, 1],
            "cmap": "RdBu_r",
        },
    ),
    Action(
        "clip",
        "Isosurface cut by a plane; the cut face is flat and shows the field inside.",
        {
            "name": P(),
            "source": P(None, "volume to cut (default: first)", ref="volume"),
            "normal": P((0, 0, 1), "normal (plot coordinates)"),
            "origin": P(
                None, "point on the cut (plot coordinates; default: centre)"
            ),
            "offset": P(0.0, "shift along the normal"),
            "side": P("up", "side kept", ("up", "down")),
            "smooth": P(True, "planar cap"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        extra=STYLE_KEYS,
        example={
            "clip": "half",
            "source": "amplitude",
            "normal": [1, 0, 0],
            "color_by": "het_strain",
        },
    ),
    Action(
        "difference",
        "a - b, or a(current condition) - a(condition ref) when b is None.",
        {
            "name": P(),
            "a": P(ref="volume"),
            "b": P(None, ref="volume"),
            "ref": P(0, "reference condition"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="volume",
        extra=STYLE_KEYS,
        example={"difference": "dphase", "a": "phase", "ref": 0},
    ),
    Action(
        "region",
        "A field restricted to the surface shell or the core of a support.",
        {
            "name": P(),
            "field": P(ref="volume"),
            "support": P(ref="volume"),
            "threshold": P(None, "support >= threshold (default: 30% of max)"),
            "mode": P("shell", "shell or core", ("shell", "core")),
            "depth": P(None, "shell depth (default: 2 voxels)"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="volume",
        extra=STYLE_KEYS,
        example={
            "region": "strain shell",
            "field": "het_strain",
            "support": "amplitude",
            "depth": 10,
        },
    ),
    Action(
        "lines",
        "Polylines, e.g. dislocation lines.",
        {
            "name": P(),
            "points": P(doc="(N, 3) array, or a list of them (one per line)"),
            "coords": P(
                "index",
                "index (z, y, x voxels), physical or plot",
                ("index", "physical", "plot"),
            ),
            "color": P("black"),
            "width": P(6),
            "labels": P(None, "list of (point, text)"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        example={
            "lines": "dislocation",
            "points": [[10, 10, 10], [30, 30, 30]],
            "color": "red",
        },
    ),
    Action(
        "fitted_line",
        "Straight line from dislocation.fit_line_3d (centroid, direction).",
        {
            "name": P(),
            "centroid": P(),
            "direction": P(),
            "length": P(None, "default: across the volume"),
            "coords": P("index", choices=("index", "physical", "plot")),
            "color": P("black"),
            "width": P(6),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        example={
            "fitted_line": "dislo 1",
            "centroid": [20, 21, 19],
            "direction": [0, 0.7, 0.7],
        },
    ),
    Action(
        "arrows",
        "Arrows (Burgers vectors, q vector, ...).",
        {
            "name": P(),
            "origins": P(),
            "vectors": P(),
            "coords": P("physical", choices=("index", "physical", "plot")),
            "color": P("crimson"),
            "radius": P(None),
            "labels": P(None, "one text per arrow"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        example={
            "arrows": "b",
            "origins": [[100, 100, 100]],
            "vectors": [[0, 40, 40]],
            "labels": ["b"],
        },
    ),
    Action(
        "triad",
        "Crystal axes drawn in the lab frame (uses the orientation matrix).",
        {
            "name": P("crystal axes"),
            "matrix": P(None),
            "labels": P(("[100]", "[010]", "[001]")),
            "origin": P(None),
            "length": P(None),
            "color": P("black"),
            **_CREATE,
        },
        main="name",
        category="Create",
        creates="layer",
        example={"triad": "crystal axes"},
    ),
    Action(
        "scale_bar",
        "Scale bar along a plot axis.",
        {
            "length": P(doc="physical length"),
            "name": P("scale bar"),
            "axis": P("x", choices=("x", "y", "z")),
            "origin": P(None),
            "color": P("black"),
            **_CREATE,
        },
        main="length",
        category="Create",
        creates="layer",
        example={"scale_bar": 50},
    ),
    # --- scene ---------------------------------------------------------------
    Action(
        "condition",
        "Show a condition (index or label).",
        {"index": P(doc="index or label")},
        main="index",
        category="Scene",
        example={"condition": 1},
    ),
    Action(
        "camera",
        "Set the camera: a preset (iso, x, -x, y, -y, z, -z), or eye/up/center.",
        {
            "view": P(None, "preset name or a Plotly camera dict"),
            "eye": P(None),
            "up": P(None),
            "center": P(None),
            "zoom": P(1.0, "> 1 zooms in"),
        },
        main="view",
        category="Scene",
        example={"camera": "iso", "zoom": 1.3},
    ),
    Action(
        "theme",
        "Dark (True) or light (False) theme.",
        {"dark": P(True)},
        main="dark",
        category="Scene",
        example={"theme": True},
    ),
    Action(
        "units",
        "Units shown on the colorbars: {field: unit} or {field: [unit, "
        "scale]} (values shown x scale, e.g. ['%', 100] for a strain).",
        {"units": P(REQ, "{field: 'rad'} or {field: ['%', 100]}")},
        main="units",
        category="Scene",
        example={"units": {"phase": "rad", "het_strain": ["%", 100]}},
    ),
    Action(
        "grid",
        "Show or hide the 3D grid.",
        {"show": P(True)},
        main="show",
        category="Scene",
        example={"grid": False},
    ),
    Action(
        "facets",
        "Facet settings used by color_by='__facet__' and facet tables.",
        {
            "families": P(None, "e.g. '100, 110, 111'"),
            "matrix": P(None, "orientation matrix U"),
            "tolerance": P(None, "degrees"),
        },
        main="families",
        category="Scene",
        example={"facets": "100, 110, 111, 311", "tolerance": 8},
    ),
    Action(
        "patches",
        "Facets of an isosurface as connected patches: colour, (hkl) labels "
        "and a table (optionally saved as CSV).",
        {
            "layer": P(ref="layer"),
            "field": P(None, "statistics of this volume", ref="volume"),
            "min_faces": P(None, "smaller patches are ignored (default 20)"),
            "labels": P(False, "True / 'front' or 'all'"),
            "colour": P(True, "colour the surface by patch"),
            "csv": P(None, "path of a CSV table"),
        },
        main="layer",
        category="Scene",
        example={
            "patches": "amplitude",
            "field": "het_strain",
            "labels": True,
            "csv": "figures/facets.csv",
        },  # fmt: skip
    ),
    Action(
        "probe",
        "Add a probe point and read every volume there.",
        {
            "point": P(),
            "coords": P("plot", choices=("plot", "physical", "index")),
        },
        main="point",
        category="Scene",
        example={"probe": [120, 115, 110]},
    ),
    Action(
        "ortho",
        "Show the linked orthoslices.",
        {
            "show": P(True),
            "field": P(None, ref="volume"),
            "center": P(None, "plot coordinates"),
        },
        main="show",
        category="Scene",
        example={"ortho": True, "field": "phase"},
    ),
    # --- output --------------------------------------------------------------
    Action(
        "image",
        "Save the current view as PNG.",
        {
            "path": P(),
            "width": P(None),
            "height": P(None),
            "scale": P(1),
            "backend": P(None, "software (default), kaleido or vtk"),
            "smooth": P(None, "smoother outlines (render at 2x)"),
            "flat": P(None, "one colour per triangle"),
        },
        main="path",
        category="Output",
        export=True,
        example={"image": "figures/view.png", "scale": 2},
    ),
    Action(
        "html",
        "Save the view as an interactive HTML page (opens in any browser).",
        {
            "path": P(),
            "slider": P(
                False, "True or a list of conditions: slider over them"
            ),
            "plotlyjs": P(
                "embed",
                "embed (offline) or cdn (small file)",
                ("embed", "cdn"),
            ),
            "shared_range": P(True, "same colour range in every condition"),
        },
        main="path",
        category="Output",
        export=True,
        example={"html": "figures/view.html", "slider": True},
    ),
    Action(
        "rotation",
        "Export a camera rotation.",
        {
            "mode": P(
                "orbit_z",
                choices=("orbit_x", "orbit_y", "orbit_z", "axis", "normal"),
            ),
            "frames": P(None),
            "degrees": P(None, "total angle (default 360)"),
            "axis": P(None, "for mode='axis'"),
            "normal_layer": P(None, "for mode='normal'", ref="layer"),
            **_EXPORT,
        },
        main="mode",
        category="Output",
        export=True,
        example={
            "rotation": "orbit_z",
            "frames": 120,
            "format": "mp4",
            "name": "rotation",
        },
    ),
    Action(
        "sweep",
        "Export a sweep of a layer parameter (opacity, thr, pos, offset).",
        {
            "layer": P(ref="layer"),
            "param": P(choices=("opacity", "thr", "pos", "offset")),
            "range": P(doc="(start, stop)"),
            "points": P(None, "number of frames"),
            "skip_empty": P(False, "skip frames where the layer is empty"),
            "min_valid": P(0.01),
            **_EXPORT,
        },
        main="layer",
        category="Output",
        export=True,
        example={
            "sweep": "zmid",
            "param": "pos",
            "range": [0, 63],
            "format": "gif",
        },
    ),
    Action(
        "conditions",
        "Export the visible layers through the conditions, with the "
        "condition label on each frame.",
        {
            "conditions": P(None, "indices or labels (default: all)"),
            "hold": P(None, "frames per condition (default 15)"),
            "degrees": P(None, "camera orbit over the animation (default 0)"),
            "shared_range": P(None, "one colour range (default True)"),
            "fix_axes": P(None, "same axes in every frame (default True)"),
            "label": P(None, "write the condition label (default True)"),
            "label_format": P(None, "'{label}', '{index}/{n}', ..."),
            **_EXPORT,
        },
        main="conditions",
        category="Output",
        export=True,
        example={"conditions": "all", "hold": 20, "format": "gif"},
    ),
    Action(
        "wait",
        "Wait for the running exports before the next step.",
        {},
        category="Output",
        export=True,
        example={"wait": None},
    ),
    Action(
        "mesh",
        "Write a layer as .ply / .vtk / .stl / .obj.",
        {
            "layer": P(ref="layer"),
            "path": P(),
            "frame": P("plot", choices=("plot", "physical")),
        },
        main="layer",
        category="Output",
        example={"mesh": "amplitude", "path": "amplitude.ply"},
    ),
    Action(
        "save_state",
        "Save the view (JSON).",
        {"path": P()},
        main="path",
        category="Output",
        example={"save_state": "view.json"},
    ),
    Action(
        "load_state",
        "Load a view saved with save_state.",
        {"path": P()},
        main="path",
        category="Output",
        example={"load_state": "view.json"},
    ),
    Action(
        "save_actions",
        "Save the current view as an action script (YAML or JSON).",
        {"path": P()},
        main="path",
        category="Output",
        example={"save_actions": "my_view.yaml"},
    ),
    Action(
        "refresh",
        "Redraw everything.",
        {},
        category="Scene",
        example={"refresh": None},
    ),
)

# former action names -> (new name, converter)
_SUFFIX = {
    "create_slice": ":slice",
    "create_plane": ":plane",
    "create_clip": ":clip",
}


def _legacy(name, kw):
    kw = dict(kw)
    kw.pop("wait", None)  # former run_actions_async flag
    params = kw.pop("params", None)
    if isinstance(params, dict):
        kw.update(params)
    if name in _SUFFIX:
        kw["name"] = f"{kw['name']}{_SUFFIX[name]}"
        if kw.pop("center_pos", False):
            kw["pos"] = None
        return {
            "create_slice": "slice",
            "create_plane": "plane",
            "create_clip": "clip",
        }[name], kw
    if name == "write_image":
        if "rendering_backend" in kw:
            kw["backend"] = kw.pop("rendering_backend")
        return "image", kw
    if name == "animate_layer":
        if "auto_trim" in kw:
            kw["skip_empty"] = kw.pop("auto_trim")
        return "sweep", kw
    if name == "camera" and isinstance(kw.get("camera"), dict):
        kw["view"] = kw.pop("camera")
        return "camera", kw
    if name == "add_lines" and "lines" in kw:
        kw["points"] = kw.pop("lines")
    renames = {
        "create_difference": "difference",
        "create_region": "region",
        "add_lines": "lines",
        "add_arrows": "arrows",
        "set_condition": "condition",
        "select_edit": "select",
        "export_mesh": "mesh",
        "animate_rotation": "rotation",
    }
    return renames.get(name, name), kw


LEGACY = {
    "create_slice",
    "create_plane",
    "create_clip",
    "create_difference",
    "create_region",
    "add_lines",
    "add_arrows",
    "set_condition",
    "select_edit",
    "export_mesh",
    "write_image",
    "animate_layer",
    "animate_rotation",
}


class ActionError(Exception):
    """Raised when a script has problems (nothing ran) or a step failed."""

    def __init__(self, message, problems=None, report=None):
        super().__init__(message)
        self.problems = problems or []
        self.report = report


def _suggest(word, options):
    m = difflib.get_close_matches(
        str(word), [str(o) for o in options], n=1, cutoff=0.6
    )
    return f" — did you mean '{m[0]}'?" if m else ""


def _short(v, n=40):
    t = repr(v)
    return t if len(t) <= n else t[: n - 1] + "…"


def _plain(v):
    """Numpy-free, rounded values for scripts."""
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        return float(f"{float(v):.6g}")
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _secs(v):
    return "" if v is None else f"{v:.2f} s"


class ActionReport(list):
    """Per-step results (a list of dicts). Displays as a live table in Jupyter."""

    def __init__(self, steps):
        super().__init__(
            {
                "index": i,
                "action": n,
                "target": _short(kw.get(ACTIONS[n].main))
                if ACTIONS[n].main
                else "",
                "ok": None,
                "status": "queued",
                "error": None,
                "seconds": None,
            }
            for i, (n, kw) in enumerate(steps)
        )
        self.created = []
        self._widget = None

    @property
    def done(self):
        return all(r["ok"] is not None for r in self)

    @property
    def ok(self):
        return all(r["ok"] for r in self)

    def _set(self, i, **kw):
        self[i].update(kw)
        if self._widget is not None:
            self._widget.value = self._repr_html_()

    def _repr_html_(self):
        colors = {
            "done": "#2ca02c",
            "failed": "#d62728",
            "running": "#1f77b4",
            "queued": "#888",
            "skipped": "#888",
            "exporting": "#1f77b4",
        }
        rows = "".join(
            f"<tr><td>{r['index'] + 1}</td><td>{r['action']}</td><td>{r['target']}</td>"
            f"<td style='color:{colors.get(r['status'], '#000')}'>{r['status']}</td>"
            f"<td>{_secs(r['seconds'])}</td>"
            f"<td>{r['error'] or ''}</td></tr>"
            for r in self
        )
        return (
            "<table class='mv-table'><tr><th>#</th><th>action</th><th>target</th><th>status</th>"
            f"<th>time</th><th>message</th></tr>{rows}</table>"
        )

    def _ipython_display_(self):
        if self._widget is None:
            self._widget = widgets.HTML(self._repr_html_())
        display(self._widget)

    def __repr__(self):
        lines = [
            f"{r['index'] + 1:>3} {r['action']:<12} {r['target']:<28} {r['status']:<9} {r['error'] or ''}"
            for r in self
        ]
        return "ActionReport\n" + "\n".join(lines)


class ActionsMixin:
    # =========================================================================
    # Normalising and checking
    # =========================================================================
    def _load_steps(self, actions):
        steps, problems = self._normalize_all(actions)
        if problems:
            raise ActionError(
                "The script was not run:\n  - " + "\n  - ".join(problems),
                problems,
            )
        return steps

    def _normalize_all(self, actions):
        """Any script -> (steps, problems); problems name the step that failed."""
        steps, problems = [], []
        for i, raw in enumerate(self._load_raw(actions)):
            try:
                steps.append(self._normalize_step(raw))
            except ActionError as e:
                problems.append(f"step {i + 1}: {e}")
                steps.append(
                    ("__invalid__", {})
                )  # keeps the numbering of later steps
        return steps, problems

    def _load_raw(self, actions):
        if isinstance(actions, (str, Path)) and Path(
            actions
        ).suffix.lower() in (".yaml", ".yml", ".json"):
            text = Path(actions).read_text()
            if Path(actions).suffix.lower() == ".json":
                actions = json.loads(text)
            else:
                import yaml

                actions = yaml.safe_load(text)
            if isinstance(actions, dict):
                actions = actions.get("actions", [])
        if isinstance(actions, ActionScript):
            return list(actions.steps)
        if isinstance(actions, (dict, str, tuple)):
            return [actions]
        return list(actions)

    @staticmethod
    def _normalize_step(step):
        """Any accepted step format -> (action name, kwargs). Unknown names are kept for the checker."""
        if isinstance(step, str):
            parts = step.split(maxsplit=1)
            name, args, kw = parts[0], parts[1:], {}
        elif isinstance(step, (tuple, list)):
            if not step:
                raise ActionError("Empty step.")
            name, rest = step[0], list(step[1:])
            kw = rest.pop() if rest and isinstance(rest[-1], dict) else {}
            args = rest
        elif isinstance(step, dict):
            kw = dict(step)
            if "action" in kw:
                name, args = kw.pop("action"), []
            else:
                keys = [k for k in kw if k in ACTIONS or k in LEGACY]
                if len(keys) != 1:
                    raise ActionError(
                        f"Cannot tell the action of {_short(step, 80)}: use one action key "
                        f"(e.g. {{'show': 'phase'}}) or 'action': ..."
                        + (
                            _suggest(next(iter(step)), list(ACTIONS))
                            if step
                            else ""
                        )
                    )
                name = keys[0]
                val = kw.pop(name)
                args = [] if val is None else [val]
        else:
            raise ActionError(f"Unsupported step: {_short(step, 80)}")
        name = str(name).strip()
        if name in LEGACY or name == "camera":
            name, kw = _legacy(name, kw)
        elif isinstance(kw.get("params"), dict) and name == "edit":
            kw.update(kw.pop("params"))
        act = ACTIONS.get(name)
        if act is not None and args:
            if act.main is None:
                if args != [None]:
                    raise ActionError(f"'{name}' takes no main argument.")
            elif len(args) > 1:
                raise ActionError(
                    f"'{name}' takes one main argument ({act.main})."
                )
            else:
                kw = {act.main: args[0], **kw}
        return name, kw

    def validate_actions(self, actions):
        """Check a script without running it; returns a list of problems (empty = fine)."""
        steps, problems = self._normalize_all(actions)
        layers = set(self._layers)
        vols = set(self.volume_keys())
        aliases = {
            "amp": "amplitude",
            "disp": "displacement",
            "strain": "het_strain",
        }

        def known(name, pool):
            return name in pool or aliases.get(name) in pool

        specials = {
            "__self__",
            "__x__",
            "__y__",
            "__z__",
            "__constant__",
            "__facet__",
            "__patch__",
        }
        for i, (name, kw) in enumerate(steps):
            where = f"step {i + 1} ({name})"
            if name == "__invalid__":
                continue
            act = ACTIONS.get(name)
            if act is None:
                problems.append(
                    f"{where}: unknown action{_suggest(name, list(ACTIONS) + sorted(LEGACY))}"
                )
                continue
            for k, p in act.params.items():
                if p.default is REQ and k not in kw:
                    problems.append(
                        f"{where}: missing '{k}' ({p.doc or 'required'})"
                    )
            for k, v in kw.items():
                p = act.params.get(k)
                if p is None:
                    if act.extra and k in act.extra:
                        continue
                    pool = list(act.params) + list(act.extra or [])
                    problems.append(
                        f"{where}: unknown parameter '{k}'{_suggest(k, pool)}"
                    )
                    continue
                if p.choices and v is not None and v not in p.choices:
                    problems.append(
                        f"{where}: {k}={v!r} must be one of {', '.join(map(str, p.choices))}"
                        f"{_suggest(v, p.choices)}"
                    )
                if p.ref and v is not None:
                    names = (
                        v
                        if isinstance(v, (list, tuple)) and p.ref == "layers"
                        else [v]
                    )
                    pool = vols if p.ref == "volume" else layers
                    for n in names:
                        if not known(n, pool):
                            kind = "volume" if p.ref == "volume" else "layer"
                            problems.append(
                                f"{where}: unknown {kind} '{n}'{_suggest(n, pool)}"
                            )
            if "cmap" in kw and kw["cmap"] not in self.cmap_options:
                problems.append(
                    f"{where}: unknown colormap '{kw['cmap']}'{_suggest(kw['cmap'], self.cmap_options)}"
                )
            if (
                "color_by" in kw
                and kw["color_by"] not in specials
                and not known(kw["color_by"], vols)
            ):
                problems.append(
                    f"{where}: cannot colour by '{kw['color_by']}'"
                    f"{_suggest(kw['color_by'], sorted(specials | vols))}"
                )
            for rk in ("range_mode", "range"):
                if isinstance(kw.get(rk), str):
                    try:
                        mva.range_mode(kw[rk])
                    except ValueError:
                        problems.append(
                            f"{where}: {rk}={kw[rk]!r} must be one of "
                            f"{', '.join(mva.RANGE_MODES)}"
                            f"{_suggest(kw[rk], [*mva.RANGE_MODES, *mva.RANGE_ALIASES])}"
                        )
            if "nan_policy" in kw and kw["nan_policy"] not in geo.NAN_POLICIES:
                problems.append(
                    f"{where}: nan_policy must be one of {', '.join(geo.NAN_POLICIES)}"
                )
            conds = []
            if name == "condition" and "index" in kw:
                conds = [kw["index"]]
            if name == "conditions":
                if len(self._conditions) < 2:
                    problems.append(f"{where}: there is only one condition")
                c = kw.get("conditions")
                if c not in (None, "all"):
                    conds = c if isinstance(c, (list, tuple)) else [c]
            for c in conds:
                labels = list(self._cond_labels)
                if isinstance(c, str) and not c.lstrip("-").isdigit():
                    if c not in labels:
                        problems.append(
                            f"{where}: unknown condition '{c}'{_suggest(c, labels)}"
                        )
                elif not 0 <= int(c) < max(1, len(labels)):
                    problems.append(
                        f"{where}: condition {c} out of range (0..{len(labels) - 1})"
                    )
            if (
                name == "camera"
                and isinstance(kw.get("view"), str)
                and kw["view"] not in CAMERA_VIEWS
            ):
                problems.append(
                    f"{where}: unknown camera view '{kw['view']}'{_suggest(kw['view'], CAMERA_VIEWS)}"
                )
            # keep track of the layers the script creates, renames and deletes
            if act.creates and "name" in kw or name in ("triad", "scale_bar"):
                new = kw.get(
                    "name",
                    act.params["name"].default
                    if "name" in act.params
                    else None,
                )
                if (
                    new in layers
                    and new in self._layers
                    and self._layers[new]["type"] == "raw"
                ):
                    problems.append(
                        f"{where}: '{new}' is an input volume; choose another name"
                    )
                layers.add(new)
                if act.creates == "volume":
                    vols.add(new)
            elif name == "rename" and "layer" in kw and "new" in kw:
                old = aliases.get(kw["layer"], kw["layer"])
                layers.discard(old)
                layers.add(kw["new"])
                if old in vols:
                    vols.discard(old)
                    vols.add(kw["new"])
            elif name == "delete" and "layer" in kw:
                layers.discard(kw["layer"])
                vols.discard(kw["layer"])
        return problems

    # =========================================================================
    # Running
    # =========================================================================
    def run_actions(
        self, actions, stop_on_error: bool = True, check: bool = True
    ):
        """Run a script: a list of steps, a YAML/JSON file or ``viewer.script()``.

        The script is checked first; if anything is wrong nothing runs and an
        :class:`ActionError` lists every problem. Steps after an export wait
        for it, so several exports and edits can be chained in one script
        (in a notebook the rest continues in the background; use
        ``await viewer.wait_for_script()`` to wait). Returns an
        :class:`ActionReport` (shows as a live table). See ``viewer.help("actions")``.
        """
        if check:
            problems = self.validate_actions(actions)
            if problems:
                raise ActionError(
                    "The script was not run:\n  - " + "\n  - ".join(problems),
                    problems,
                )
        steps = self._load_steps(actions)
        report = ActionReport(steps)
        self._script_task = None
        self._run_from(steps, 0, report, stop_on_error)
        return report

    async def run_actions_async(
        self, actions, stop_on_error: bool = True, check: bool = True
    ):
        """Like :meth:`run_actions`, but returns only when every step (and export) is done."""
        report = self.run_actions(
            actions, stop_on_error=stop_on_error, check=check
        )
        await self.wait_for_script()
        return report

    async def wait_for_script(self):
        """Wait for a script that continues in the background after an export."""
        while getattr(self, "_script_task", None) is not None:
            task = self._script_task
            await task
            if self._script_task is task:
                self._script_task = None
        await self.wait_for_export()

    def _run_from(self, steps, start, report, stop_on_error):
        self._batch = True
        try:
            for i in range(start, len(steps)):
                task = self._exec_step(i, steps[i], report, stop_on_error)
                if isinstance(task, asyncio.Task) and not task.done():
                    report._set(i, status="exporting")
                    if i + 1 < len(steps):
                        loop = asyncio.get_running_loop()
                        self._script_task = loop.create_task(
                            self._continue_after(
                                task, i, steps, report, stop_on_error
                            )
                        )
                    else:
                        task.add_done_callback(
                            lambda t, i=i: self._export_finished(t, i, report)
                        )
                    return
        finally:
            self._batch = False
            self._rebuild()

    def _export_finished(self, task, i, report):
        err = None if task.cancelled() else task.exception()
        if task.cancelled():
            report._set(i, ok=False, status="failed", error="cancelled")
        elif err is not None:
            report._set(
                i,
                ok=False,
                status="failed",
                error=f"{type(err).__name__}: {err}",
            )
        else:
            report._set(i, ok=True, status="done")
        return err is None and not task.cancelled()

    async def _continue_after(self, task, i, steps, report, stop_on_error):
        try:
            await task
        except BaseException:
            pass
        if not self._export_finished(task, i, report) and stop_on_error:
            for j in range(i + 1, len(steps)):
                report._set(j, ok=False, status="skipped")
            return
        self._run_from(steps, i + 1, report, stop_on_error)

    def _exec_step(self, i, step, report, stop_on_error):
        name, kw = step
        act = ACTIONS[name]
        full = {
            k: p.default for k, p in act.params.items() if p.default is not REQ
        }
        full.update(kw)
        report._set(i, status="running")
        t0 = time.perf_counter()
        try:
            if act.export:
                self._batch = False
                self._rebuild()
            out = getattr(self, f"_act_{name}")(**full)
            if act.export:
                self._batch = True
            if isinstance(out, asyncio.Task):
                report._set(i, seconds=time.perf_counter() - t0)
                return out
            if act.creates and isinstance(out, str):
                report.created.append(out)
            report._set(
                i, ok=True, status="done", seconds=time.perf_counter() - t0
            )
        except Exception as e:
            report._set(
                i,
                ok=False,
                status="failed",
                error=f"{type(e).__name__}: {e}",
                seconds=time.perf_counter() - t0,
            )
            if stop_on_error:
                for j in range(i + 1, len(report)):
                    report._set(j, ok=False, status="skipped")
                raise ActionError(
                    f"Step {i + 1} ({name}) failed: {e}", report=report
                ) from e
        return None

    # =========================================================================
    # Handlers (one per action)
    # =========================================================================
    def _layers_arg(self, v):
        names = [v] if isinstance(v, str) else list(v)
        return [self._resolve_layer_name(n) for n in names]

    def _act_show(self, layer):
        for n in self._layers_arg(layer):
            self._visible_cb[n].value = True

    def _act_hide(self, layer):
        for n in self._layers_arg(layer):
            self._visible_cb[n].value = False

    def _act_only(self, layers):
        keep = set(self._layers_arg(layers))
        for n, cb in self._visible_cb.items():
            cb.value = n in keep

    def _act_select(self, layer):
        self._select_edit_layer(self._resolve_layer_name(layer))

    def _act_edit(self, layer, **params):
        if (
            "range" in params
            and not isinstance(params["range"], str)
            and "auto_range" not in params
        ):
            params["auto_range"] = False
        self.set_layer_params(layer, **params)

    def _act_rename(self, layer, new):
        self.rename_layer(self._resolve_layer_name(layer), new)

    def _act_delete(self, layer, cascade):
        self.delete_layer(
            self._resolve_layer_name(layer), cascade=bool(cascade)
        )

    def _create(self, adder, kw):
        name = kw.pop("name")
        replace, select = kw.pop("replace"), kw.pop("select")
        style = {k: kw.pop(k) for k in list(kw) if k in STYLE_KEYS}
        if name in self._layers:
            if self._layers[name]["type"] == "raw":
                raise ValueError(
                    f"'{name}' is an input volume; choose another name."
                )
            if replace:
                self.delete_layer(name, cascade=False)
        new = adder(name=name, **kw)
        if style:
            if (
                "range" in style
                and not isinstance(style["range"], str)
                and "auto_range" not in style
            ):
                style["auto_range"] = False
            self.set_layer_params(new, **style)
        if select:
            self._select_edit_layer(new)
        return new

    def _act_slice(self, **kw):
        return self._create(self.add_slice, kw)

    def _act_plane(self, **kw):
        return self._create(self.add_plane, kw)

    def _act_clip(self, **kw):
        return self._create(self.add_clip, kw)

    def _act_difference(self, **kw):
        return self._create(self.add_difference, kw)

    def _act_region(self, **kw):
        return self._create(self.add_region, kw)

    def _act_lines(self, points, **kw):
        return self._create(self.add_lines, {"lines": points, **kw})

    def _act_fitted_line(self, **kw):
        return self._create(self.add_fitted_line, kw)

    def _act_arrows(self, **kw):
        return self._create(self.add_arrows, kw)

    def _act_triad(self, **kw):
        return self._create(self.add_axes_triad, kw)

    def _act_scale_bar(self, **kw):
        return self._create(self.add_scale_bar, kw)

    def _act_condition(self, index):
        if isinstance(index, str) and not index.lstrip("-").isdigit():
            if index not in self._cond_labels:
                raise ValueError(
                    f"Unknown condition '{index}'{_suggest(index, self._cond_labels)}"
                )
            index = self._cond_labels.index(index)
        self.set_condition(int(index))

    def _act_camera(self, view, eye, up, center, zoom):
        if isinstance(view, dict):
            cam = dict(view)
        else:
            cur = self.fig.layout.scene.camera
            cam = cur.to_plotly_json() if cur is not None else {}
            if view is not None:
                e, u = CAMERA_VIEWS[view]
                cam.update(
                    eye=dict(zip("xyz", e)),
                    up=dict(zip("xyz", u)),
                    center=dict(x=0, y=0, z=0),
                )
            for key, val in (("eye", eye), ("up", up), ("center", center)):
                if val is not None:
                    cam[key] = (
                        dict(zip("xyz", map(float, val)))
                        if not isinstance(val, dict)
                        else val
                    )
        if zoom and float(zoom) != 1.0:
            e = cam.get("eye") or {"x": 1.5, "y": 1.5, "z": 1.5}
            cam["eye"] = {k: float(v) / float(zoom) for k, v in e.items()}
        self.fig.layout.scene.camera = cam

    def _act_theme(self, dark):
        self.theme_toggle.value = bool(dark)

    def _act_units(self, units):
        if not isinstance(units, dict):
            raise ValueError("units must be a dict {field: unit}")
        self.set_units({k: _unit_arg(v) for k, v in units.items()})

    def _act_grid(self, show):
        self.grid_toggle.value = bool(show)

    def _act_facets(self, families, matrix, tolerance):
        self.set_facet_settings(matrix, families, tolerance)

    def _act_patches(self, layer, field, min_faces, labels, colour, csv):
        layer = self._resolve_layer_name(layer)
        rows = self.facet_patches(
            layer, field, min_faces=min_faces, labels=labels, colour=colour
        )
        if csv:
            self.save_facet_csv(csv, layer, field, by="patch")
        return rows

    def _act_probe(self, point, coords):
        return self.probe(point, coords=coords)

    def _act_ortho(self, show, field, center):
        if field is not None:
            self.ortho_field.value = self._resolve_layer_name(field)
        self.ortho_show.value = bool(show)
        if center is not None:
            self.set_ortho_center(center, coords="plot")

    def _act_image(self, path, width, height, scale, backend, smooth, flat):
        return self.write_image(
            path, width, height, scale, backend, smooth=smooth, flat=flat
        )

    def _act_html(self, path, slider, plotlyjs, shared_range):
        return self.export_html(
            path, slider, plotlyjs, shared_range=shared_range
        )

    def _act_rotation(
        self,
        mode,
        frames,
        degrees,
        axis,
        normal_layer,
        fps,
        format,
        name,
        smooth,
        flat,
    ):
        return self.start_rotation_export(
            mode,
            axis,
            frames,
            fps,
            format,
            name,
            normal_layer,
            degrees,
            queue=True,
            smooth=smooth,
            flat=flat,
        )

    def _act_conditions(
        self,
        conditions,
        hold,
        degrees,
        shared_range,
        fix_axes,
        label,
        label_format,
        fps,
        format,
        name,
        smooth,
        flat,
    ):
        return self.start_condition_export(
            conditions,
            hold,
            degrees,
            fps,
            format,
            name,
            shared_range=shared_range,
            fix_axes=fix_axes,
            label=label,
            label_format=label_format,
            queue=True,
            smooth=smooth,
            flat=flat,
        )

    def _act_sweep(
        self,
        layer,
        param,
        range,
        points,
        skip_empty,
        min_valid,
        fps,
        format,
        name,
        smooth,
        flat,
    ):
        return self.start_layer_export(
            layer,
            param,
            range,
            points,
            fps,
            format,
            name,
            skip_empty,
            min_valid,
            queue=True,
            smooth=smooth,
            flat=flat,
        )

    def _act_wait(self):
        t = self._export_task
        return t if t is not None and not t.done() else None

    def _act_mesh(self, layer, path, frame):
        return self.export_mesh(self._resolve_layer_name(layer), path, frame)

    def _act_save_state(self, path):
        return self.save_state(path)

    def _act_load_state(self, path):
        self.load_state(path)

    def _act_save_actions(self, path):
        return self.save_actions(path)

    def _act_refresh(self):
        pass

    # =========================================================================
    # View -> script
    # =========================================================================
    def to_actions(self, camera=True):
        """The current view as a short action script (list of dicts).

        Run it later with ``viewer.run_actions(script)``; it recreates the
        derived layers, styles, transforms, visibility, condition and camera.
        """
        fp = self.frame.phys_to_plot
        steps = []
        if len(self._conditions) > 1:
            steps.append({"condition": self._cond_index})
        units = {
            k: (lab if sc == 1.0 else [lab, sc])
            for k, (lab, sc) in self.field_units.items()
        }
        if units:
            steps.append({"units": units})
        default_fams = "100, 110, 111"
        fams = ", ".join(lab.strip("{}") for lab, _ in self.facet_families)
        if (
            fams != default_fams
            or not np.allclose(self.orientation_matrix, np.eye(3))
            or self.facet_tolerance != 10
        ):
            steps.append(
                {
                    "facets": fams,
                    "matrix": self.orientation_matrix,
                    "tolerance": self.facet_tolerance,
                }
            )
        for name, s in self._layers.items():
            t = s["type"]
            if t == "slice":
                st = {
                    "slice": name,
                    "source": s["source"],
                    "axis": self.frame.phys_axis_to_plot(s["axis_phys"]),
                    "pos": s["pos"],
                    "thickness": s["thickness"],
                }
            elif t in ("plane", "clip"):
                st = {
                    t: name,
                    "source": s["source"],
                    "normal": fp(s["normal"]),
                    "origin": fp(s["origin0"]),
                    "offset": s.get("offset", 0.0),
                }
                st.update(
                    {
                        "thickness": s["thickness"],
                        "extent": s["extent"],
                        "resolution": s.get("resolution", 200),
                    }
                    if t == "plane"
                    else {
                        "side": s.get("side", "up"),
                        "smooth": bool(s.get("smooth", True)),
                    }
                )
            elif t == "diff":
                st = {
                    "difference": name,
                    "a": s["a"],
                    "b": s.get("b"),
                    "ref": s.get("ref", 0),
                }
            elif t == "region":
                st = {
                    "region": name,
                    "field": s["field"],
                    "support": s["support"],
                    "threshold": s["threshold"],
                    "mode": s["mode"],
                    "depth": s["depth"],
                }
            elif t == "lines":
                st = {
                    "lines": name,
                    "points": list(s.get("lines", [])),
                    "coords": "physical",
                    "labels": [[p, txt] for p, txt in s.get("labels") or []]
                    or None,
                }
            elif t == "arrows":
                st = {
                    "arrows": name,
                    "origins": s["origins"],
                    "vectors": s["vectors"],
                    "coords": "physical",
                    "radius": s.get("radius"),
                    "labels": s.get("texts") or None,
                }
            else:
                st = None
            if st is not None:
                st["visible"] = False
                steps.append(st)
        for name in self._layers:
            diff = self._style_changes(name)
            if diff:
                steps.append({"edit": name, **diff})
        steps.append({"only": self._visible_names()})
        if camera and self.fig.layout.scene.camera is not None:
            cam = self.fig.layout.scene.camera.to_plotly_json()
            if cam:
                steps.append({"camera": cam})
        return [_plain(s) for s in steps]

    def _style_changes(self, name):
        w = self._layer_widgets.get(name, {})
        base = w.get("_defaults") or {}
        cur = self._style(name)
        out = {}
        for k, v in cur.items():
            if k in ("range_min", "range_max", "auto_range") or k not in base:
                continue
            if k == "range":
                if not cur["auto_range"]:
                    out["range"] = list(v)
                continue
            if isinstance(v, float) and isinstance(base[k], float):
                if not np.isclose(v, base[k]):
                    out[k] = v
            elif v != base[k]:
                out[k] = v
        tr = self._layer_transform.get(name) or {}
        for i, a in enumerate("xyz"):
            if tr.get("rot_deg") and tr["rot_deg"][i]:
                out[f"rot_{a}"] = tr["rot_deg"][i]
            if tr.get("trans") and tr["trans"][i]:
                out[f"trans_{a}"] = tr["trans"][i]
        return out

    def save_actions(self, path, camera=True):
        """Write :meth:`to_actions` to a ``.yaml``/``.yml`` or ``.json`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = self.to_actions(camera=camera)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(steps, indent=1))
        else:
            import yaml

            path.write_text(
                yaml.safe_dump(
                    steps,
                    sort_keys=False,
                    default_flow_style=None,
                    allow_unicode=True,
                )
            )
        return path

    def script(self, *steps):
        """A chainable script builder::

        (viewer.script()
            .only("amplitude")
            .slice("zmid", source="phase", cmap="twilight")
            .camera("iso", zoom=1.2)
            .rotation(frames=120, name="spin")
            .run())
        """
        return ActionScript(self, list(steps))


class ActionScript:
    """Chainable list of actions (see :meth:`ActionsMixin.script`).

    Every action is a method (``.show``, ``.edit``, ``.slice``, ...); call
    ``.run()`` (or ``await .run_async()``), ``.check()``, ``.save(path)``.
    """

    def __init__(self, viewer=None, steps=None):
        self.viewer = viewer
        self.steps = list(steps or [])

    def add(self, action, *args, **kw):
        act = ACTIONS.get(action)
        step = {action: args[0] if args else None, **kw}
        if act is not None and act.main is None:
            step = {action: None, **kw}
        self.steps.append(step)
        return self

    def check(self):
        problems = self.viewer.validate_actions(self.steps)
        return problems or "No problems found."

    def run(self, stop_on_error=True):
        return self.viewer.run_actions(self.steps, stop_on_error=stop_on_error)

    async def run_async(self, stop_on_error=True):
        return await self.viewer.run_actions_async(
            self.steps, stop_on_error=stop_on_error
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = [_plain(s) for s in self.steps]
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(steps, indent=1))
        else:
            import yaml

            path.write_text(
                yaml.safe_dump(
                    steps,
                    sort_keys=False,
                    default_flow_style=None,
                    allow_unicode=True,
                )
            )
        return path

    def to_list(self):
        return [_plain(s) for s in self.steps]

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return "ActionScript\n" + "\n".join(
            f"{i + 1:>3}. {format_step(s)}" for i, s in enumerate(self.steps)
        )

    def _repr_html_(self):
        items = "".join(
            f"<li><code>{format_step(s)}</code></li>" for s in self.steps
        )
        return f"<b>ActionScript</b> ({len(self.steps)} steps)<ol>{items}</ol>"


def format_step(step):
    """``{'slice': 'zmid', 'axis': 'z'}`` -> ``slice('zmid', axis='z')``."""
    name, kw = ActionsMixin._normalize_step(step)
    act = ACTIONS.get(name)
    main = act.main if act else None
    args = [repr(kw.pop(main))] if main and main in kw else []
    args += [f"{k}={_short(_plain(v), 50)}" for k, v in kw.items()]
    return f"{name}({', '.join(args)})"


def _make_method(name):
    act = ACTIONS[name]

    def method(self, *args, **kw):
        return self.add(name, *args, **kw)

    method.__name__ = name
    lines = [act.doc, "", f"    {act.signature()}", ""]
    for k, p in act.params.items():
        req = "required" if p.default is REQ else f"default {p.default!r}"
        lines.append(
            f"    {k}: {p.doc} ({req})" if p.doc else f"    {k} ({req})"
        )
    if act.extra:
        lines.append(
            "    **style: any style key (cmap, color_by, opacity, thr, ...)"
        )
    method.__doc__ = "\n".join(lines)
    return method


for _name in ACTIONS:
    setattr(ActionScript, _name, _make_method(_name))
