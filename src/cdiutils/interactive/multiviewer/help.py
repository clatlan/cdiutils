"""Built-in help and the guided tutorial of :class:`MultiVolumeViewer` (mixin).

``viewer.help()`` gives an overview, ``viewer.help("actions")`` the list of
script actions, ``viewer.help("slice")`` one action in detail (with examples
using your own layer names), and ``viewer.help("coordinates" | "layers" |
"conditions" | "colours" | "facets" | "probe" | "export" | "state" | "edit")`` a topic.

``viewer.tutorial()`` returns a step-by-step guide that runs each feature on
the loaded data and shows the matching code.
"""

import html
import re

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from .actions import (
    ACTIONS,
    GEOMETRY_KEYS,
    REQ,
    STYLE_KEYS,
    TRANSFORM_KEYS,
    _suggest,
    format_step,
)

_CATEGORIES = ("Layers", "Create", "Scene", "Output")

TOPICS = {
    "overview": "What the viewer is and how to start",
    "actions": "Scripted actions: formats and the list of actions",
    "edit": "Every parameter accepted by edit (style, geometry, transform)",
    "layers": "Layer types",
    "coordinates": "Axes, PLOT_ORDER and units",
    "conditions": "Several datasets and differences",
    "colours": "Colour ranges (symmetric, percentiles, all conditions) and units",
    "facets": "Facet colouring and tables",
    "probe": "Probe, line profiles, orthoslices, shell/core statistics",
    "export": "Images, animations and meshes",
    "state": "Saving a view (state file or action script)",
}


def _in_notebook():
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "Terminal" not in type(ip).__name__
    except Exception:
        return False


def _to_text(h):
    t = re.sub(r"<pre[^>]*>", "\n", h)
    t = re.sub(r"</pre>|<br\s*/?>|</p>|</tr>|</li>|</h\d>", "\n", t)
    t = re.sub(r"<li>", "  - ", t)
    t = re.sub(r"</t[dh]>", "  ", t)
    t = re.sub(r"<[^>]+>", "", t)
    return html.unescape(re.sub(r"\n{3,}", "\n\n", t)).strip()


def _code(text):
    return f"<pre style='margin:4px 0;padding:6px 8px;border-radius:4px;background:rgba(127,127,127,0.12)'>{html.escape(text)}</pre>"


def _table(rows, head):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(
        "<tr>"
        + "".join(f"<td style='text-align:left'>{c}</td>" for c in r)
        + "</tr>"
        for r in rows
    )
    return f"<table class='mv-table'><tr>{th}</tr>{body}</table>"


class HelpMixin:
    # =========================================================================
    # Names used in examples
    # =========================================================================
    def _example_names(self):
        vols = self.raw_keys() if self._layers else []

        def pick(words, fallback):
            for v in vols:
                if any(w in v.lower() for w in words):
                    return v
            return fallback

        amp = pick(
            ("amp", "dens", "support", "modulus"),
            vols[0] if vols else "amplitude",
        )
        others = [v for v in vols if v != amp]
        phase = pick(("phase", "phi"), others[0] if others else amp)
        strain = pick(
            ("strain", "eps"), others[1] if len(others) > 1 else phase
        )
        return {"amplitude": amp, "phase": phase, "het_strain": strain}

    def _localize(self, obj):
        names = self._example_names()
        if isinstance(obj, str):
            return names.get(obj, obj)
        if isinstance(obj, dict):
            return {k: self._localize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._localize(v) for v in obj)
        return obj

    # =========================================================================
    # help()
    # =========================================================================
    def help(self, topic=None):
        """Show built-in help. ``topic``: an action name or one of
        overview, actions, edit, layers, coordinates, conditions, facets,
        probe, export, state."""
        topic = "overview" if topic is None else str(topic).strip()
        if topic in ACTIONS:
            body = self._help_action(topic)
        elif topic in TOPICS:
            body = getattr(self, f"_help_{topic}")()
        else:
            body = (
                f"<b>No help for '{html.escape(topic)}'</b>{html.escape(_suggest(topic, list(TOPICS) + list(ACTIONS)))}"
                "<br>" + self._help_topics_list()
            )
        if _in_notebook():
            display(HTML(f"<div style='max-width:900px'>{body}</div>"))
        else:
            print(_to_text(body))

    def _help_topics_list(self):
        rows = [[f"<code>v.help('{k}')</code>", d] for k, d in TOPICS.items()]
        rows.append(
            [
                "<code>v.help('slice')</code>",
                "any action name: parameters and examples",
            ]
        )
        rows.append(
            ["<code>v.tutorial()</code>", "guided tour on your own data"]
        )
        return _table(rows, ["call", "what you get"])

    def _help_overview(self):
        n = self._example_names()
        a, p = n["amplitude"], n["phase"]
        code = (
            f"v.run_actions([\n"
            f"    {{'only': '{a}'}},\n"
            f"    {{'edit': '{a}', 'thr': 0.3, 'color_by': '{p}', 'cmap': 'twilight'}},\n"
            f"    {{'slice': 'zmid', 'source': '{p}', 'axis': 'z'}},\n"
            f"    {{'camera': 'iso', 'zoom': 1.2}},\n"
            f"])"
        )
        return (
            "<h3>MultiVolumeViewer</h3>"
            "<p>Interactive 3D viewer for several volumes on the same grid. <b>Toolbar</b>: camera "
            "(reset, iso/x/y/z), grid, dark theme, rotation, condition slider, figure size preset, quick PNG "
            "and HTML, and a button to hide the panel. <b>3D view</b> (drag to rotate, scroll to zoom) with a "
            "status bar. <b>Panel tabs</b>: <i>Layers</i> (list: tick to show, click a name to edit), "
            "<i>Add</i> (new slices, planes, clips, differences, shell/core, crystal axes, scale bar), "
            "<i>Analyze</i> (probe, profiles, facets, shell/core statistics, orthoslices), <i>Export</i> "
            "(images, animations, HTML, save/load view), <i>Script</i> (the view as a script; check, run, "
            "save) and <i>Help</i>.</p>"
            "<p>Panel edits are computed in the background: the view keeps responding and shows "
            "<i>updating…</i> until the new geometry is ready. <code>v.cache_info()</code> shows the memory "
            "used by cached isosurfaces and slices.</p>"
            "<p>Everything you can click can also be scripted, which makes a figure reproducible:</p>"
            + _code(code)
            + "<p>Tweak a view by hand, then turn it into a script with <code>v.save_actions('my_view.yaml')</code>; "
            "replay it with <code>v.run_actions('my_view.yaml')</code>.</p>"
            + "<p><b>Help topics</b></p>"
            + self._help_topics_list()
        )

    def _help_actions(self):
        n = self._example_names()
        a, p = n["amplitude"], n["phase"]
        parts = [
            "<h3>Scripted actions</h3>",
            "<p>A script is a list of steps. The short form puts the action first, followed by its "
            "main argument; other parameters follow:</p>",
            _code(
                f"{{'show': '{p}'}}\n{{'edit': '{a}', 'thr': 0.3, 'cmap': 'viridis'}}\n"
                f"{{'clip': 'half', 'source': '{a}', 'normal': [1, 0, 0], 'color_by': '{p}'}}"
            ),
            "<p>Also accepted: <code>('edit', 'amplitude', {'thr': 0.3})</code>, <code>'show phase'</code>, the former "
            "<code>{'action': 'edit', 'layer': ..., 'params': {...}}</code>, a YAML/JSON file path, and the builder:</p>",
            _code(
                f"(v.script()\n   .only('{a}')\n   .slice('zmid', source='{p}', cmap='twilight')\n"
                f"   .rotation(frames=120, name='spin')\n   .run())"
            ),
            "<ul><li>The whole script is <b>checked first</b>; typos are reported with suggestions and nothing runs. "
            "<code>v.validate_actions(script)</code> only checks.</li>"
            "<li>Creating a layer that already exists <b>replaces</b> it, so re-running a cell is safe "
            "(<code>replace=False</code> keeps both).</li>"
            "<li>Create actions accept style keys directly (<code>cmap</code>, <code>color_by</code>, "
            "<code>opacity</code>, <code>thr</code>, ...).</li>"
            "<li>Steps after an export <b>wait for it</b>, so several exports can be chained. In a notebook the script "
            "continues in the background: <code>await v.wait_for_script()</code> or "
            "<code>await v.run_actions_async(...)</code> to wait.</li>"
            "<li>The returned report shows each step's status and time.</li></ul>",
        ]
        for cat in _CATEGORIES:
            rows = [
                [
                    f"<code>{html.escape(act.signature())}</code>",
                    html.escape(act.doc),
                ]
                for act in ACTIONS.values()
                if act.category == cat
            ]
            parts.append(
                f"<p><b>{cat}</b></p>"
                + _table(rows, ["action", "what it does"])
            )
        parts.append(
            "<p><code>v.help('&lt;action&gt;')</code> for details and examples; "
            "<code>v.help('edit')</code> for every edit parameter.</p>"
        )
        return "".join(parts)

    def _help_action(self, name):
        act = ACTIONS[name]
        rows = []
        for k, p in act.params.items():
            req = (
                "<b>required</b>"
                if p.default is REQ
                else html.escape(repr(p.default))
            )
            choices = (
                f" — one of {', '.join(map(str, p.choices))}"
                if p.choices
                else ""
            )
            rows.append(
                [
                    f"<code>{k}</code>" + (" (main)" if k == act.main else ""),
                    req,
                    html.escape(p.doc) + choices,
                ]
            )
        ex = self._localize(act.example or {name: None})
        builder = f"v.script().{format_step(ex)}.run()"
        try:
            import yaml

            yml = yaml.safe_dump(
                [ex], sort_keys=False, default_flow_style=None
            ).strip()
        except Exception:
            yml = ""
        extra = (
            "<p>Also accepts any style key: <code>v.help('edit')</code>.</p>"
            if act.extra is STYLE_KEYS
            else "<p>Accepts any key listed by <code>v.help('edit')</code>.</p>"
            if act.extra
            else ""
        )
        return (
            f"<h3>{name}</h3><p>{html.escape(act.doc)}</p>"
            + (
                _table(rows, ["parameter", "default", "meaning"])
                if rows
                else "<p>No parameters.</p>"
            )
            + extra
            + "<p><b>Example</b></p>"
            + _code(f"v.run_actions([{ex!r}])")
            + _code(builder)
            + (_code(yml) if yml else "")
        )

    def _help_edit(self):
        def block(title, keys):
            return f"<p><b>{title}</b></p>" + _table(
                [
                    [f"<code>{k}</code>", html.escape(d)]
                    for k, d in keys.items()
                ],
                ["key", "meaning"],
            )

        n = self._example_names()
        return (
            "<h3>edit</h3><p>Change any of these on a layer (keys that do not apply to the layer type are ignored "
            "for style, rejected for geometry):</p>"
            + _code(
                f"{{'edit': '{n['amplitude']}', 'thr': 0.3, 'color_by': '{n['het_strain']}', 'cmap': 'RdBu_r', "
                "'range': [-0.002, 0.002], 'rot_z': 30}"
            )
            + block("Style", STYLE_KEYS)
            + block(
                "Geometry (slices, planes, clips, derived volumes)",
                GEOMETRY_KEYS,
            )
            + block(
                "Transform (about the volume centre, plot coordinates)",
                TRANSFORM_KEYS,
            )
        )

    def _help_layers(self):
        rows = [
            [
                "raw",
                "an input volume: isosurface (or mask surface), coloured by any field",
            ],
            ["difference", "a - b, or a(current) - a(reference condition)"],
            [
                "region",
                "a field limited to the surface shell or the core of a support",
            ],
            ["slice", "axis-aligned slice, averaged over ± thickness"],
            ["plane", "arbitrary plane, averaged over ± thickness"],
            [
                "clip",
                "isosurface cut by a plane; the flat cap shows the field inside",
            ],
            ["lines / fitted_line", "polylines, e.g. dislocation lines"],
            ["arrows / triad / scale_bar", "vectors, crystal axes, scale bar"],
        ]
        return (
            "<h3>Layer types</h3>"
            + _table(rows, ["type", "what it is"])
            + "<p>Each layer has its own style, lighting, transform and mesh export in the Layers tab.</p>"
        )

    def _help_coordinates(self):
        po = self.PLOT_ORDER
        mapping = (
            "plot X = array axis 0 (z), plot Z = array axis 2 (x)"
            if po == "zyx"
            else "plot X = array axis 2 (x), plot Z = array axis 0 (z)"
        )
        return (
            "<h3>Coordinates</h3><ul>"
            "<li>Arrays are indexed <code>(z, y, x)</code>; <code>voxel_size</code> is <code>(vz, vy, vx)</code>.</li>"
            f"<li><code>PLOT_ORDER='{po}'</code>: {mapping}.</li>"
            "<li>Normals, origins and axis names (UI and scripts) are in <b>plot</b> coordinates, in the "
            f"physical unit{f' ({self.unit})' if self.unit else ''}; the origin is the centre of voxel (0, 0, 0).</li>"
            "<li>Lines and arrows take <code>coords='index'</code> (voxel indices, e.g. from "
            "<code>np.argwhere</code>), <code>'physical'</code> (x, y, z) or <code>'plot'</code>.</li>"
            "<li>Layer rotations turn about the volume centre.</li></ul>"
        )

    def _help_conditions(self):
        n = self._example_names()
        return (
            "<h3>Conditions</h3><p>Pass a list of dicts with the same keys (e.g. one per pressure step):</p>"
            + _code(
                "v = MultiVolumeViewer([d_0bar, d_1bar], voxel_size=(5, 5, 5), unit='nm',\n"
                "                      condition_labels=['0 bar', '1 bar'])"
            )
            + "<p>A slider appears; every layer and setting is kept when you switch. "
            "Scripts: <code>{'condition': '1 bar'}</code>. Changes relative to a reference condition:</p>"
            + _code(
                f"{{'difference': 'dphase', 'a': '{n['phase']}', 'ref': 0, 'cmap': 'RdBu_r'}}"
            )
        )

    def _help_colours(self):
        n = self._example_names()
        rows = [
            ["minmax", "data range (default)"],
            ["p1 / p5", "1–99 % / 5–95 % percentiles: ignores outliers"],
            [
                "sym",
                "± max |value|: zero in the middle (strain, phase, differences)",
            ],
            ["sym99", "± 99th percentile of |value|"],
            ["manual", "the cmin/cmax slider (or range=[cmin, cmax])"],
        ]
        return (
            "<h3>Colour ranges</h3>"
            + _table(rows, ["range", "meaning"])
            + "<p><i>same for all conditions</i> uses one range for every condition (the union of "
            "their ranges), so colours can be compared between conditions and animations do "
            "not flicker.</p>"
            + _code(
                f"{{'edit': '{n['het_strain']}', 'range': 'sym', 'range_all': True, 'cmap': 'RdBu_r'}}\n"
                f"{{'edit': '{n['phase']}', 'range': [-1.5, 1.5]}}   # manual"
            )
            + "<h3>Units</h3><p>Colorbar titles show the unit of the colour field "
            "(<code>rad</code> by default for names containing <i>phase</i>). A scale "
            "multiplies the displayed values, e.g. a strain in %; probes and statistics "
            "keep raw values. Derived layers use the unit of their source. In the panel: "
            "<i>unit</i> and <i>× scale</i> under the colour range.</p>"
            + _code(
                f"v.set_units({n['het_strain']}=('%', 100), {n['phase']}='rad')\n"
                f"{{'units': {{'{n['het_strain']}': ['%', 100]}}}}        # in a script\n"
                f"MultiVolumeViewer(data, field_units={{'{n['het_strain']}': ('%', 100)}})"
            )
        )

    def _help_facets(self):
        n = self._example_names()
        return (
            "<h3>Facets</h3><p>Set <code>color_by='__facet__'</code> on an isosurface or clip. Each face is "
            "assigned to the closest family direction in the crystal frame "
            "(<code>n_crystal = U @ n_lab</code>; lab = physical x, y, z) within the tolerance, otherwise "
            "'other'.</p>"
            + _code(
                "v.run_actions([{'facets': '100, 110, 111, 311', 'matrix': U, 'tolerance': 8},\n"
                f"                {{'edit': '{n['amplitude']}', 'color_by': '__facet__'}}])\n"
                f"v.facet_stats('{n['amplitude']}', '{n['het_strain']}')            # per family\n"
                f"v.facet_stats('{n['amplitude']}', '{n['het_strain']}', by='plane')  # per (hkl)"
            )
            + "<p><b>Patches</b>: edge-connected faces of the same plane form one facet patch, with its "
            "area, mean normal, deviation from the plane, (hkl) of the mean normal (cdiutils' facet-analysis "
            "rule), centre and the mean / std of a field. <code>color_by='__patch__'</code> colours them; "
            "labels write the plane on the facets facing the camera. Analyze tab → Facets → group by "
            "<i>patches</i>, or:</p>"
            + _code(
                f"rows = v.facet_patches('{n['amplitude']}', '{n['het_strain']}', labels=True, colour=True)\n"
                f"v.save_facet_csv('facets.csv', '{n['amplitude']}', '{n['het_strain']}')\n"
                f"{{'patches': '{n['amplitude']}', 'field': '{n['het_strain']}', 'labels': True, 'csv': 'facets.csv'}}"
            )
        )

    def _help_probe(self):
        n = self._example_names()
        return (
            "<h3>Probe and statistics</h3><ul>"
            "<li><b>Probe</b>: turn on <i>Probe clicks</i> and click a surface: values of every volume, "
            "and |AB| between the last two points. Script: <code>{'probe': [x, y, z]}</code>.</li>"
            "<li><b>Line profile</b> A → B: plot and CSV in the panel, or "
            "<code>s, prof = v.line_profile()</code>.</li>"
            "<li><b>Orthoslices</b>: three linked slices; click to move the centre. "
            "<code>{'ortho': True, 'field': ...}</code>.</li>"
            "<li><b>Shell / core</b>: <code>v.region_stats(field, support, threshold, depth)</code>, "
            "or add both as layers.</li></ul>"
            + _code(
                f"v.region_stats('{n['het_strain']}', '{n['amplitude']}', threshold=0.3, depth=10)"
            )
        )

    def _help_export(self):
        n = self._example_names()
        return (
            "<h3>Export</h3><p>Stills and animations use the <b>built-in renderer</b> by default "
            "(numpy + matplotlib): it works on any server, without a browser, display or GPU, and gives the "
            "same view as the screen. Kaleido or VTK can be chosen in the export panel "
            "(<i>Renderer</i>) or with <code>rendering_backend=</code>. Options: <i>Smooth edges</i> "
            "(renders at 2x for smoother outlines; data values are not changed) and <i>Flat colours</i> "
            "(one colour per triangle, no blending inside triangles). Formats: mp4, gif, png (frame folder); "
            "<i>Save as</i> takes a name or a path.</p>"
            + _code(
                "v.run_actions([\n"
                "    {'image': 'figures/view.png', 'scale': 2, 'flat': True},\n"
                "    {'rotation': 'orbit_z', 'frames': 120, 'name': 'figures/spin'},\n"
                "    {'sweep': 'zmid', 'param': 'pos', 'range': [0, 63], 'format': 'gif', 'smooth': True},\n"
                "    {'conditions': 'all', 'hold': 20, 'format': 'gif', 'name': 'figures/pressure'},\n"
                "    {'html': 'figures/view.html', 'slider': True},   # interactive page\n"
                f"    {{'mesh': '{n['amplitude']}', 'path': 'figures/particle.ply'}},\n"
                "])"
            )
            + "<p><b>Conditions</b> (mode <i>Conditions</i> in the panel): the visible layers through the "
            "conditions, <i>Frames each</i> per condition (rendered once), optional rotation, the same colour "
            "range and axes in every frame, and the condition label (text with <code>{label}</code>, "
            "<code>{index}</code>, <code>{n}</code>). With several conditions the label is also written on "
            "stills and other animations unless <i>Condition label</i> is unticked.</p>"
            + "<p>Exports run one after the other; later steps wait. <code>rendering_mode='fast'</code> renders "
            "frames in parallel; the workers stay ready between exports and stop with <code>viewer.close()</code> "
            "(in a script file, put the code under <code>if __name__ == '__main__':</code>). "
            "GIF and MP4 need no system ffmpeg.</p>"
        )

    def _help_state(self):
        return (
            "<h3>Saving a view</h3><ul>"
            "<li><code>v.save_actions('view.yaml')</code>: the view as a readable action script "
            "(edit it by hand, keep it next to the figure); replay with <code>v.run_actions('view.yaml')</code>.</li>"
            "<li><code>v.to_actions()</code>: the same as a Python list.</li>"
            "<li><code>v.save_state('view.json')</code> / <code>v.load_state(...)</code>: exact restore "
            "of every widget.</li></ul>"
        )

    # =========================================================================
    # tutorial()
    # =========================================================================
    def tutorial(self):
        """Guided tour on the loaded data. Display the returned widget (it is
        displayed automatically when it is the last line of a cell)."""
        return Tutorial(self)


class Tutorial(widgets.VBox):
    """Step-by-step guide: each step explains a feature, shows the code and
    can run it on the viewer."""

    PREFIX = "tuto:"

    def __init__(self, viewer):
        super().__init__()
        self.v = viewer
        self.steps = self._build_steps()
        self.i = 0
        self.title = widgets.HTML()
        self.body = widgets.HTML()
        self.out = widgets.Output()
        self.back = widgets.Button(
            description="◀ Back", layout=widgets.Layout(width="auto")
        )
        self.run = widgets.Button(
            description="Run this step",
            button_style="primary",
            layout=widgets.Layout(width="auto"),
        )
        self.next = widgets.Button(
            description="Next ▶", layout=widgets.Layout(width="auto")
        )
        self.clean = widgets.Button(
            description="Remove tutorial layers",
            layout=widgets.Layout(width="auto"),
        )
        self.back.on_click(lambda _b: self._go(-1))
        self.next.on_click(lambda _b: self._go(+1))
        self.run.on_click(lambda _b: self.run_step())
        self.clean.on_click(lambda _b: self.cleanup())
        self.children = [
            self.title,
            self.body,
            widgets.HBox([self.back, self.run, self.next, self.clean]),
            self.out,
        ]
        self.layout = widgets.Layout(
            border="1px solid rgba(127,127,127,0.4)",
            padding="10px",
            max_width="900px",
        )
        self._show()

    # --- steps ------------------------------------------------------------------
    def _build_steps(self):
        v = self.v
        n = v._example_names()
        A, F, S = n["amplitude"], n["phase"], n["het_strain"]
        T = self.PREFIX
        cmapF = "twilight" if "phase" in F.lower() else "viridis"
        steps = [
            dict(
                title="Welcome",
                text="This tour runs each feature on <b>your</b> data and shows the code. Click <i>Run this step</i> "
                "to apply it to the viewer, <i>Next</i> to continue. Layers created here start with "
                f"<code>{T}</code> and can be removed at the end.<br>Display the viewer in a cell above "
                "(<code>v.show()</code>) so you can watch it change.",
                actions=None,
            ),
            dict(
                title="1. Show and style a volume",
                text=f"Show only <b>{A}</b>, make it semi-transparent and colour its surface by <b>{F}</b>. "
                "The same can be done in the Layers tab (click the layer name: opacity, color by, cmap).",
                actions=[
                    {"only": A},
                    {"edit": A, "opacity": 0.6, "color_by": F, "cmap": cmapF},
                ],
            ),
            dict(
                title="2. A slice through the centre",
                text=f"An axis-aligned slice of <b>{F}</b>. <code>axis</code> is a plot axis; <code>pos</code> "
                "defaults to the centre.",
                actions=[
                    {
                        "slice": f"{T}slice",
                        "source": F,
                        "axis": "z",
                        "cmap": cmapF,
                    }
                ],
            ),
            dict(
                title="3. Cut the particle open",
                text=f"A clip keeps one side of a plane. Its flat cap shows <b>{S}</b> inside the particle.",
                actions=[
                    {"hide": [A, f"{T}slice"]},
                    {
                        "clip": f"{T}clip",
                        "source": A,
                        "normal": [1, 0, 0],
                        "color_by": S,
                        "cmap": "RdBu_r",
                    },
                ],
            ),
            dict(
                title="4. Facets",
                text="Colour the surface by facet family, then get the area and the mean field per family. "
                "Set your orientation matrix first if the volume axes are not the crystal axes "
                "(<code>v.help('facets')</code>).",
                actions=[{"edit": f"{T}clip", "color_by": "__facet__"}],
                extra=self._facet_table,
                extra_code=f"v.facet_stats('{A}', '{S}')",
            ),
            dict(
                title="5. Probe and line profile",
                text="Two probe points (centre and an off-centre point) and the profile of every volume between them. "
                "In the Analyze tab: turn on <i>Probe clicks</i> and click on a surface.",
                actions=None,
                extra=self._probe,
                extra_code="v.probe(point)  # plot coordinates\n"
                "s, profiles = v.line_profile()  # last two points",
            ),
        ]
        if len(v._conditions) > 1:
            steps.append(
                dict(
                    title="6. Conditions",
                    text="Switch to the second condition and show how <b>"
                    + F
                    + "</b> changed relative to the first.",
                    actions=[
                        {"condition": 1},
                        {
                            "difference": f"{T}diff",
                            "a": F,
                            "ref": 0,
                            "cmap": "RdBu_r",
                        },
                        {"only": f"{T}diff"},
                    ],
                )
            )
        else:
            steps.append(
                dict(
                    title="6. Conditions",
                    text="With several datasets (e.g. pressure steps) pass a list of dicts; a slider appears and "
                    "<code>difference</code> layers compare a condition to a reference. Nothing to run here.",
                    actions=None,
                    extra_code="MultiVolumeViewer([d0, d1], condition_labels=['0 bar', '1 bar'])\n"
                    f"{{'difference': 'dphase', 'a': '{F}', 'ref': 0}}",
                )
            )
        steps += [
            dict(
                title="7. Camera presets",
                text="Presets: iso, x, -x, y, -y, z, -z; <code>zoom</code> > 1 zooms in.",
                actions=[
                    {"only": [f"{T}clip"]},
                    {"camera": "iso", "zoom": 1.2},
                ],
            ),
            dict(
                title="8. Your view as a script",
                text="Any view, built by clicking or by script, can be written out as a short script and replayed "
                "later, e.g. to regenerate a figure.",
                actions=None,
                extra=self._show_script,
                extra_code="v.save_actions('my_view.yaml')\nv.run_actions('my_view.yaml')",
            ),
            dict(
                title="9. Export",
                text="Run saves a PNG of the current view (<code>tutorial_view.png</code>). Animations work the same "
                "way and run one after another.",
                actions=[{"image": "tutorial_view.png"}],
                extra_code="{'rotation': 'orbit_z', 'frames': 120, 'name': 'spin'}\n"
                f"{{'sweep': '{T}clip', 'param': 'offset', 'range': [-50, 50], 'format': 'gif'}}",
            ),
            dict(
                title="Done",
                text="Remove the tutorial layers with the button. More: <code>v.help()</code>, "
                "<code>v.help('actions')</code>, <code>v.help('&lt;action&gt;')</code>.",
                actions=None,
                extra=self.cleanup,
                extra_code="v.help('actions')",
            ),
        ]
        return steps

    # --- extras (run with the step) ---------------------------------------------
    def _facet_table(self):
        n = self.v._example_names()
        rows = self.v.facet_stats(
            f"{self.PREFIX}clip"
            if f"{self.PREFIX}clip" in self.v._layers
            else n["amplitude"],
            n["het_strain"],
        )
        display(
            HTML(
                _table(
                    [
                        [
                            r["family"],
                            r["faces"],
                            f"{r['area_frac']:.3f}",
                            f"{r.get('mean', np.nan):.4g}",
                            f"{r.get('std', np.nan):.4g}",
                        ]
                        for r in rows
                    ],
                    ["family", "faces", "area fraction", "mean", "std"],
                )
            )
        )

    def _probe(self):
        v = self.v
        c = v.frame.center_phys()
        lo, hi = v.frame.bounds_phys()
        p1 = c + 0.3 * (hi - lo) * np.array([1.0, 0.5, 0.2])
        v.clear_probe()
        v.probe(c, coords="physical")
        vals = v.probe(p1, coords="physical")
        s, prof = v.line_profile(n=150)
        fig, axes = plt.subplots(
            len(prof),
            1,
            figsize=(5, 1.6 * len(prof)),
            sharex=True,
            squeeze=False,
        )
        for ax, (k, y) in zip(axes[:, 0], prof.items()):
            ax.plot(s, y, lw=1.5)
            ax.set_ylabel(k)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel(
            f"distance A → B{f' ({v.unit})' if v.unit else ''}"
        )
        fig.tight_layout()
        plt.show()
        print("Values at B:", {k: round(x, 5) for k, x in vals.items()})

    def _show_script(self):
        import yaml

        text = yaml.safe_dump(
            self.v.to_actions(), sort_keys=False, default_flow_style=None
        )
        display(
            HTML(_code(text if len(text) < 4000 else text[:4000] + "\n..."))
        )

    def cleanup(self):
        """Delete the layers created by the tutorial and show the first volume again."""
        for name in [k for k in self.v._layers if k.startswith(self.PREFIX)]:
            if name in self.v._layers:
                self.v.delete_layer(name)
        a = self.v._example_names()["amplitude"]
        if a in self.v._layers:
            self.v.run_actions(
                [
                    {"only": a},
                    {"edit": a, "opacity": 1.0, "color_by": "__self__"},
                ]
            )
        print("Tutorial layers removed.")

    # --- navigation -------------------------------------------------------------
    def _code_for(self, step):
        parts = []
        if step.get("actions"):
            inner = ",\n".join(f"    {a!r}" for a in step["actions"])
            parts.append(f"v.run_actions([\n{inner},\n])")
        if step.get("extra_code"):
            parts.append(step["extra_code"])
        return "\n".join(parts)

    def _show(self):
        st = self.steps[self.i]
        self.title.value = f"<h3 style='margin:4px 0'>{st['title']}</h3><i>Step {self.i + 1} of {len(self.steps)}</i>"
        code = self._code_for(st)
        self.body.value = f"<p>{st['text']}</p>" + (
            _code(code) if code else ""
        )
        runnable = bool(st.get("actions") or st.get("extra"))
        self.run.disabled = not runnable
        self.back.disabled = self.i == 0
        self.next.disabled = self.i == len(self.steps) - 1

    def _go(self, d):
        self.i = int(np.clip(self.i + d, 0, len(self.steps) - 1))
        self.out.clear_output()
        self._show()

    def run_step(self, index=None):
        """Run the current step (or step ``index``) on the viewer."""
        if index is not None:
            self.i = int(index)
            self._show()
        st = self.steps[self.i]
        self.out.clear_output()
        with self.out:
            try:
                if st.get("actions"):
                    self.v.run_actions(st["actions"])
                if st.get("extra"):
                    st["extra"]()
                print("✓ done")
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
