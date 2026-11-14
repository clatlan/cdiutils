"""Tests of the MultiVolumeViewer modules: geometry, analysis, layers, tools and exports (headless)."""

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive.multiviewer import analysis as mva  # noqa: E402
from cdiutils.interactive.multiviewer import geometry as geo  # noqa: E402
from cdiutils.interactive.multiviewer_3d import MultiVolumeViewer  # noqa: E402


def make(n=40, shift=0.0):
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    c = n / 2
    cube = np.maximum.reduce([abs(z - c), abs(y - c), abs(x - c)]) / 12
    octa = (abs(z - c) + abs(y - c) + abs(x - c)) / 19
    amp = np.clip(1.2 - np.maximum(cube, octa), 0, 1).astype(np.float32)
    ph = (
        0.03 * (x - c) * np.sin(0.15 * (y - c)) + shift * 0.01 * (z - c)
    ).astype(np.float32)
    ph[amp < 0.05] = np.nan
    st = np.gradient(np.nan_to_num(ph), axis=0).astype(np.float32)
    st[amp < 0.05] = np.nan
    return {"amplitude": amp, "phase": ph, "het_strain": st}


VOX = (5.0, 4.0, 3.0)  # anisotropic on purpose


@pytest.fixture(params=["zyx", "xyz"])
def viewer(request):
    v = MultiVolumeViewer(
        [make(), make(shift=1.0)],
        voxel_size=VOX,
        unit="nm",
        PLOT_ORDER=request.param,
        condition_labels=["p0", "p1"],
    )
    yield v
    v.close()


# =============================================================================
# Geometry
# =============================================================================
@pytest.mark.parametrize("order", ["xyz", "zyx"])
def test_frame_roundtrip(order):
    f = geo.PlotFrame(order, VOX, (10, 20, 30))
    p = np.random.rand(5, 3)
    assert np.allclose(f.plot_to_phys(f.phys_to_plot(p)), p)
    assert np.allclose(f.phys_to_idx(f.idx_to_phys(p)), p)
    for a in "xyz":
        assert f.phys_axis_to_plot(f.plot_axis_to_phys(a)) == a
    assert f.n_along("x") == 30 and f.n_along("z") == 10
    if order == "zyx":
        assert f.plot_axis_to_phys("x") == "z"
        assert np.allclose(f.phys_to_plot([1, 2, 3]), [3, 2, 1])


def test_offset_bounds():
    lo, hi = np.zeros(3), np.array([10.0, 20.0, 30.0])
    a, b = geo.offset_bounds([5, 10, 15], [0, 0, 1], lo, hi)
    assert np.isclose(a, -15) and np.isclose(b, 15)
    a, b = geo.offset_bounds([5, 10, 15], [1, 1, 0], lo, hi)
    assert np.isclose(b, 5 * np.sqrt(2))


def test_clip_cap_is_planar():
    d = make()
    amp = d["amplitude"]
    o = np.array(
        [20 * VOX[2], 20 * VOX[1], 20 * VOX[0]]
    )  # physical xyz centre
    f = geo.clip_implicit(amp, 0.5, [0, 0, 1], o, "up", VOX, smooth=True)
    verts, faces, normals = geo.isosurface(f, 0.0, VOX)
    zphys = verts[:, 0] * VOX[0]
    assert zphys.min() >= o[2] - 1e-6  # nothing below the plane
    cap = np.abs(zphys - o[2]) < 1e-6
    assert cap.sum() > 20  # an exactly planar cap exists


def test_isosurface_normals_outward():
    amp = make()["amplitude"]
    verts, faces, n = geo.isosurface(amp, 0.5, VOX)
    p = verts[:, ::-1] * np.array(VOX[::-1])
    assert np.mean(np.sum((p - p.mean(0)) * n, 1) > 0) > 0.95


def test_axis_slice_shapes():
    f = geo.PlotFrame("xyz", VOX, (10, 20, 30))
    vol = np.random.rand(10, 20, 30)
    for ax, shp in (("z", (20, 30)), ("y", (10, 30)), ("x", (10, 20))):
        d, X, Y, Z = geo.axis_slice(vol, f, ax, 4, 1)
        assert d.shape == shp == X.shape == Y.shape == Z.shape
    d, X, Y, Z = geo.axis_slice(vol, f, "z", 4, 0)
    assert np.allclose(d, vol[4]) and np.allclose(Z, 4 * VOX[0])


def test_grid_to_mesh_drops_nan():
    X, Y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    Z = np.zeros_like(X)
    Z[0, 0] = np.nan
    V, F, _ = geo.grid_to_mesh(X, Y, Z)
    assert len(F) == 2 * 2 * 3 - 2 and F.max() < len(V)


# =============================================================================
# Analysis
# =============================================================================
def test_parse_families_and_classify():
    fams = mva.parse_families("100, 110, 111")
    assert [f[0] for f in fams] == ["{100}", "{110}", "{111}"]
    assert mva.parse_families("1 1 3, -1 0 0")[0][0] == "{311}"
    n = np.array([[0, 0, 1], [1, 1, 0], [1, -1, 1], [0.3, 0.2, 0.9]], float)
    n /= np.linalg.norm(n, axis=1)[:, None]
    fam, plane, ang = mva.classify_normals(n, fams, tol_deg=5)
    assert list(fam[:3]) == [0, 1, 2] and fam[3] == -1
    assert plane[2] == "(1-11)"
    U = mva.parse_matrix("0 1 0; 1 0 0; 0 0 1")
    fam2, _, _ = mva.classify_normals(n, fams, U, tol_deg=5)
    assert list(fam2[:3]) == [0, 1, 2]


def test_shell_core():
    sup = make()["amplitude"] > 0.3
    shell, core, _ = mva.shell_core_masks(sup, VOX, 2 * max(VOX))
    assert (
        (shell & core).sum() == 0
        and (shell | core).sum() == sup.sum()
        and core.sum() > 0
    )


@pytest.mark.parametrize("ext", [".stl", ".ply", ".obj", ".vtk"])
def test_write_mesh(tmp_path, ext):
    V = np.random.rand(4, 3)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    p = mva.write_mesh(tmp_path / f"m{ext}", V, F, np.arange(4.0))
    assert p.stat().st_size > 0
    if ext == ".stl":
        assert p.stat().st_size == 84 + 50 * 2


# =============================================================================
# Viewer
# =============================================================================
def test_initial_render(viewer):
    assert viewer._trace_order == ["amplitude"]
    assert len(viewer.fig.data[0].x) > 100


def test_create_all_layer_types(viewer):
    v = viewer
    s = v.add_slice("s", "phase", axis="x", thickness=1)
    p = v.add_plane("p", "het_strain", normal=(1, 1, 0))
    c = v.add_clip("c", "amplitude", normal=(0, 0, 1), offset=10)
    d = v.add_difference("d", "phase", ref=0)
    r = v.add_region("r", "het_strain", "amplitude", 0.3, "shell", 10)
    L = v.add_lines("dislo", [np.array([[5, 5, 5], [30, 30, 30]])])
    a = v.add_axes_triad()
    sb = v.add_scale_bar(50)
    for n in (s, p, c, d, r, L, a, sb):
        v._visible_cb[n].value = True
    assert set(v._trace_order) >= {s, p, c, d, r, L, a, sb}
    types = {n: v.fig.data[i].type for i, n in enumerate(v._trace_order)}
    assert (
        types[s] == types[p] == "surface"
        and types[c] == "mesh3d"
        and types[L] == "scatter3d"
    )
    assert len(v.fig.layout.scene.annotations) == 4  # triad labels + scale bar
    # slice is perpendicular to the plot x axis
    X = np.asarray(v.fig.data[v._trace_order.index(s)].x, float)
    assert np.nanstd(X) < 1e-9


def test_clip_cap_in_viewer(viewer):
    v = viewer
    c = v.add_clip("c", "amplitude", normal=(0, 0, 1))
    tr = v.fig.data[v._trace_order.index(c)]
    zc = v.frame.phys_to_plot(v.frame.center_phys())[2]
    assert np.min(tr.z) >= zc - 1e-6


def test_condition_and_difference(viewer):
    v = viewer
    d = v.add_difference("d", "phase", ref=0)
    assert np.nanmax(np.abs(v.get_volume(d))) == 0
    v.set_condition(1)
    assert np.nanmax(np.abs(v.get_volume(d))) > 0
    assert v.cond_label.value == "<b>p1</b>"


def test_color_by_variants(viewer):
    v = viewer
    for sel in ("__x__", "__constant__", "phase", "__facet__", "__self__"):
        v.set_layer_params("amplitude", color_by=sel)
        tr = v.fig.data[0]
        assert len(tr.x) > 0
    v.set_layer_params("amplitude", color_by="__facet__")
    assert v.fig.data[0].intensitymode == "cell"
    v.set_layer_params("amplitude", color_by="__self__")
    assert v.fig.data[0].intensitymode == "vertex"


def test_coord_coloring_matches_display(viewer):
    v = viewer
    s = v.add_slice("s", "phase", axis="z")
    v.set_layer_params(s, color_by="__y__")
    tr = v.fig.data[v._trace_order.index(s)]
    assert np.allclose(
        np.asarray(tr.surfacecolor, float),
        np.asarray(tr.y, float),
        equal_nan=True,
    )
    v.set_layer_params("amplitude", color_by="__z__")
    tr = v.fig.data[0]
    assert np.allclose(tr.intensity, tr.z)


def test_transform_and_probe_inverse(viewer):
    v = viewer
    v.set_layer_params("amplitude", rot_z=30, trans_x=10)
    p = np.array([10.0, 20.0, 30.0])
    assert np.allclose(
        v._from_display(v._to_display(p, "amplitude"), "amplitude"), p
    )
    vals = v.probe(v.frame.phys_to_plot(v.frame.center_phys()))
    assert set(vals) == set(v.volume_keys()) and np.isfinite(vals["amplitude"])


def test_line_profile(viewer):
    v = viewer
    lo, hi = v.frame.bounds_phys()
    s, prof = v.line_profile(
        v.frame.phys_to_plot(lo), v.frame.phys_to_plot(hi), ["amplitude"], n=50
    )
    assert len(s) == 50 and np.isclose(s[-1], np.linalg.norm(hi - lo))
    assert np.nanmax(prof["amplitude"]) > 0.9


def test_facet_stats(viewer):
    rows = viewer.facet_stats("amplitude", "het_strain")
    fams = {r["family"] for r in rows}
    assert {"{100}", "{111}"} <= fams
    assert np.isclose(sum(r["area_frac"] for r in rows), 1.0)
    rows = viewer.facet_stats("amplitude", by="plane")
    assert any(r["family"] == "(001)" for r in rows)


def test_region_stats(viewer):
    r = viewer.region_stats("het_strain", "amplitude", 0.3, 10)
    assert r["shell"]["n"] + r["core"]["n"] == r["all"]["n"] > 0


def test_rename_delete_cascade(viewer):
    v = viewer
    s = v.add_slice("s", "phase")
    v.set_layer_params("amplitude", color_by="phase")
    v.rename_layer("phase", "phi")
    assert (
        v._layers[s]["source"] == "phi"
        and "phi" in v.dict_data
        and "phi" in v._conditions[1]
    )
    assert v._style("amplitude")["color_by"] == "phi"
    v.delete_layer("phi")
    assert s not in v._layers and "phi" not in v.dict_data
    assert v._style("amplitude")["color_by"] == "__self__"


def test_state_roundtrip(viewer, tmp_path):
    v = viewer
    v.add_plane("p", "phase", normal=(0, 1, 1), offset=5)
    v.add_lines("L", [np.array([[1, 1, 1], [10, 10, 10]])])
    v.set_layer_params(
        "amplitude", thr=0.42, opacity=0.5, rot_x=15, cmap="viridis"
    )
    path = v.save_state(tmp_path / "s.json")
    v.delete_layer("p")
    v.set_layer_params("amplitude", thr=0.2, rot_x=0)
    v.load_state(path)
    assert "p" in v._layers and "L" in v._layers
    st = v._style("amplitude")
    assert (
        np.isclose(st["thr"], 0.42)
        and st["cmap"] == "viridis"
        and np.isclose(st["op"], 0.5)
    )
    assert v._layer_transform["amplitude"]["rot_deg"][0] == 15
    json.loads(path.read_text())


def test_run_actions_backward_compat(viewer):
    v = viewer
    res = v.run_actions(
        [
            {"action": "show", "layer": "phase"},
            {
                "action": "create_slice",
                "name": "zmid",
                "axis": "z",
                "center_pos": True,
                "thickness": 1,
            },
            {
                "action": "create_plane",
                "name": "pl",
                "normal": [1, 0, 0],
                "origin": [0, 0, 0],
            },
            {
                "action": "create_clip",
                "name": "cl",
                "source": "amplitude",
                "side": "down",
            },
            {
                "action": "edit",
                "layer": "zmid:slice",
                "params": {"slice_pos": 3, "opacity": 0.7, "cmap": "viridis"},
            },
            {
                "action": "edit",
                "layer": "amp",
                "params": {"range": [0.1, 0.9], "auto_range": False},
            },
            {"action": "hide", "layer": "phase"},
        ]
    )
    assert all(r["ok"] for r in res)
    assert v._layers["zmid:slice"]["pos"] == 3
    assert np.isclose(v.fig.data[0].cmin, 0.1) and np.isclose(
        v.fig.data[0].cmax, 0.9
    )


def test_mesh_export(viewer, tmp_path):
    v = viewer
    p = v.export_mesh("amplitude", tmp_path / "a.ply")
    assert p.stat().st_size > 1000
    s = v.add_slice("s", "phase")
    v.export_mesh(s, tmp_path / "s.vtk", frame="physical")


def test_orthoslices(viewer):
    v = viewer
    v.ortho_show.value = True
    assert v._ortho_figs[0].data[0].z is not None
    v.set_ortho_center([0, 0, 0])
    assert np.allclose(v._ortho_center, 0)


def test_linked_cameras():
    a = MultiVolumeViewer(make(24), voxel_size=VOX)
    b = MultiVolumeViewer(make(24), voxel_size=VOX)
    a.link_cameras(b)
    a.fig.layout.scene.camera = dict(eye=dict(x=2, y=0.5, z=0.3))
    assert np.isclose(b.fig.layout.scene.camera.eye.x, 2)


def test_layer_export_png_frames(viewer, tmp_path, monkeypatch):
    """Exercise the export pipeline with a fake renderer (no browser/X needed)."""
    from cdiutils.interactive.multiviewer import anim as _mv_anim

    calls = []

    def fake_render(fig_json, w, h, s, backend, fontsize, options=None):
        calls.append(fig_json)
        import io

        import imageio.v2 as iio

        buf = io.BytesIO()
        iio.imwrite(buf, np.zeros((8, 8, 3), np.uint8), format="png")
        return buf.getvalue()

    monkeypatch.setattr(_mv_anim, "render_png", fake_render)
    viewer._backend = "kaleido"
    s = viewer.add_slice("s", "phase", axis="z")

    async def go():
        t = viewer.start_layer_export(
            s,
            "pos",
            (0, 39),
            points=10,
            fmt="png",
            name=str(tmp_path / "frames"),
        )
        out = await t
        t2 = viewer.start_rotation_export(
            "orbit_z", frames=6, fmt="png", name=str(tmp_path / "rot")
        )
        await t2
        return out

    out = asyncio.run(go())
    assert (
        len(list(out.glob("*.png"))) == 10
        and len(list((tmp_path / "rot").glob("*.png"))) == 6
    )
    zs = [
        np.nanmean(
            np.asarray(c["data"][viewer._trace_order.index(s)]["z"], float)
        )
        for c in calls[:10]
    ]
    assert (
        np.all(np.diff(zs) >= 0) and zs[-1] > zs[0]
    )  # slice moves along plot z
    assert viewer._layers[s]["pos"] == (40 - 1) // 2  # live state untouched


def test_abs_iso_and_signed_iso(viewer):
    v = viewer
    v.set_layer_params("phase", abs_iso=True, thr=0.1)
    v._visible_cb["phase"].value = True
    n_abs = len(v.fig.data[v._trace_order.index("phase")].x)
    v.set_layer_params("phase", abs_iso=False, thr=0.1)
    n_signed = len(v.fig.data[v._trace_order.index("phase")].x)
    assert n_abs > n_signed > 0  # |phase| = 0.1 includes the negative side


def test_create_panel_plane_and_clip(viewer):
    v = viewer
    v.add_layer_kind.value = "clip"
    v.add_layer_source.value = "amplitude"
    v.add_plane_nx.value, v.add_plane_ny.value, v.add_plane_nz.value = 1, 0, 0
    v.add_clip_side.value = "down"
    v._on_create_layer_clicked(None)
    spec = v._layers["amplitude:clip"]
    assert spec["side"] == "down"
    assert np.allclose(v.frame.phys_to_plot(spec["normal"]), [1, 0, 0])


def test_fitted_line_spans_volume(viewer):
    from cdiutils.analysis.dislocation import fit_line_3d

    pts = np.argwhere(np.ones((3, 3, 3)))[:, :] * 0 + np.array([[5, 10, 15]])
    pts = np.vstack([pts, [[25, 10, 15]]])  # a line along z (array axis 0)
    centroid, direction = fit_line_3d(pts.astype(float))
    name = viewer.add_fitted_line("dislo", centroid, direction)
    seg = viewer._layers[name]["lines"][0]
    lo, hi = viewer.frame.bounds_phys()
    assert np.isclose(
        abs(seg[1, 2] - seg[0, 2]), hi[2] - lo[2]
    )  # full z extent
    assert np.allclose(seg[:, :2], seg[0, :2])


def test_plotter_layers_entry_point():
    from cdiutils.interactive import Plotter

    p = Plotter(
        make(20), plot="layers", layers_kwargs={"rendering_mode": "fast"}
    )
    assert p.viewer._trace_order == ["amplitude"]
