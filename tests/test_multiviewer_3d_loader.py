"""Opening cdiutils results, the launcher form and HTML export."""

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")
pytest.importorskip("h5py")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer import loader  # noqa: E402
from cdiutils.interactive.multiviewer.launcher import (  # noqa: E402
    parse_units,
)


def props(n=24, r=6.0, k=1.0, shape=None):
    """Structural properties as the pipeline stores them (0 outside)."""
    shape = shape or (n, n, n)
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(float)
    z, y, x = z - shape[0] / 2, y - shape[1] / 2, x - shape[2] / 2
    amp = 50 * np.clip(1.2 - np.sqrt(x**2 + y**2 + z**2) / r, 0, 1)
    sup = (amp > 0.3 * amp.max()).astype(float)
    inside = sup > 0
    return {
        "amplitude": amp,
        "support": sup,
        "phase": np.where(inside, 0.1 * x, 0.0),
        "displacement": np.where(inside, 0.2 * x, 0.0),
        "displacement_gradient": np.zeros((3, *shape)),
        "het_strain": np.where(inside, k * 0.05 * x / n, 0.0),
        "het_strain_from_dspacing": np.where(inside, 0.01, 0.0),
        "dspacing": np.where(inside, 2.26, 0.0),
        "lattice_parameter": np.where(inside, 3.92, 0.0),
        "hkl": np.array([1, 1, 1]),
        "g_vector": np.array([0.1, 0.2, 0.3]),
        "voxel_size": np.array([5.0, 5.0, 5.0]),
    }


def write_cxi(path, p, scan, iso=0.3, voxel=(5.0, 5.0, 5.0)):
    """Same calls as BcdiPipeline._save_postprocessed_data."""
    from cdiutils.io.cxi import CXIFile

    with CXIFile(str(path), "w") as cxi:
        cxi.stamp()
        cxi.create_cxi_group("process", description="post-processing")
        cxi.create_cxi_group(
            "result",
            description="Orthogonalisation procedure",
            voxel_size=voxel,
            units="nm",
        )
        cxi.create_cxi_group(
            "result",
            description="Surface determination",
            estimated_isosurface=iso,
            used_isosurface=iso,
        )
        cxi.create_cxi_group(
            "parameters",
            scan=scan,
            convention="cxi",
            voxel_size=voxel,
            isosurface=iso,
            pressure=f"{scan - 100} bar",
            support={"support_method": "isosurface"},
        )
        for key, data in p.items():
            if isinstance(data, np.ndarray):
                path_ = cxi.create_cxi_image(
                    data=data, data_space="Direct space", title=key
                )
                cxi.softlink(f"entry_1/{key}", path_)
    return path


@pytest.fixture
def results(tmp_path):
    out = []
    for i, (scan, r, k) in enumerate([(101, 5, 1), (102, 7, 2), (103, 8, 3)]):
        d = tmp_path / f"S{scan}"
        d.mkdir()
        p = props(r=r, k=k, shape=(24, 24, 24) if i < 2 else (26, 24, 22))
        write_cxi(d / f"S{scan}_postprocessed_data.cxi", p, scan)
        np.savez_compressed(d / f"S{scan}_structural_properties.npz", **p)
        out.append(d)
    return tmp_path, out


def test_find_results_prefers_cxi(results):
    root, dirs = results
    files = loader.find_results(root)
    assert [f.name for f in files] == [
        f"S{s}_postprocessed_data.cxi" for s in (101, 102, 103)
    ]
    npz = dirs[0] / "S101_structural_properties.npz"
    assert loader.find_results(npz) == [npz]
    assert "het_strain" in loader.peek_fields(npz)
    assert "displacement_gradient" not in loader.peek_fields(npz)  # 4D
    assert set(loader.peek_fields(files[0])) >= {"amplitude", "het_strain"}


@pytest.mark.parametrize("ext", ["cxi", "npz"])
def test_read_result(results, ext):
    root, dirs = results
    name = (
        "S101_postprocessed_data.cxi"
        if ext == "cxi"
        else "S101_structural_properties.npz"
    )
    r = loader.read_result(dirs[0] / name)
    assert r.scan == "101" and r.voxel_size == (5.0, 5.0, 5.0)
    assert list(r.fields) == list(loader.DEFAULT_FIELDS)
    amp, st = r.fields["amplitude"], r.fields["het_strain"]
    assert np.nanmax(amp) == pytest.approx(1.0)
    assert np.isnan(st[0, 0, 0]) and np.isfinite(st[12, 12, 12])
    assert np.isfinite(amp).all()  # amplitude is not masked
    if ext == "cxi":
        assert r.isosurface == pytest.approx(0.3) and r.convention == "cxi"
        assert r.params["pressure"] == "1 bar"
    raw = loader.read_result(
        dirs[0] / name, fields=["het_strain"], mask_outside=False
    )
    assert list(raw.fields) == ["het_strain"]
    assert raw.fields["het_strain"][0, 0, 0] == 0
    with pytest.raises(KeyError, match="available"):
        loader.read_result(dirs[0] / name, fields=["strian"])


def test_from_cdiutils_single(results, capsys):
    root, dirs = results
    v = MultiVolumeViewer.from_cdiutils(dirs[0])
    out = capsys.readouterr().out
    assert "MultiVolumeViewer configuration" in out
    assert "het_strain" in out and "%" in out and "5, 5, 5 nm" in out
    assert v.unit == "nm" and np.allclose(v.voxel_size, 5)
    assert v._style("amplitude")["thr"] == pytest.approx(0.3)
    assert v._unit_of("het_strain") == ("%", 1.0)
    assert v._unit_of("displacement") == ("Å", 1.0)
    assert v.cdiutils_info[0]["scan"] == "101"
    v.close()


def test_from_cdiutils_conditions(results, capsys):
    root, dirs = results
    with _no_warning():
        v = MultiVolumeViewer.from_cdiutils(
            root,
            fields=["amplitude", "het_strain"],
            label_from="pressure",
            script=[{"only": "amplitude"}],
        )
    out = capsys.readouterr().out
    assert "26x24x22" in out
    assert "(26, 24, 22) -> (26, 24, 24) (centred crop/pad)" in out
    assert v._cond_labels == ["1 bar", "2 bar", "3 bar"]
    assert v.raw_keys() == ["amplitude", "het_strain"]
    assert v.dict_data["amplitude"].shape == (26, 24, 24)
    v.set_condition(2)
    assert v.dict_data["amplitude"].shape == (26, 24, 24)
    v.close()
    w = MultiVolumeViewer.from_cdiutils(
        [dirs[1], dirs[0]],
        labels=["b", "a"],
        voxel_size=4,
        verbose=False,
        field_units={"het_strain": ("ppm", 1e4)},
    )
    assert w._cond_labels == ["b", "a"] and np.allclose(w.voxel_size, 4)
    assert w._unit_of("het_strain") == ("ppm", 1e4)
    w.close()


class _no_warning:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("error")

    def __exit__(self, *a):
        self._w.__exit__(*a)


def test_missing_voxel_size_is_explained(tmp_path):
    p = props()
    p.pop("voxel_size")
    np.savez(tmp_path / "S7_structural_properties.npz", **p)
    with pytest.raises(ValueError, match="voxel_size="):
        MultiVolumeViewer.from_cdiutils(tmp_path, verbose=False)
    with pytest.raises(FileNotFoundError, match="No such file or folder"):
        MultiVolumeViewer.from_cdiutils(tmp_path / "nothing", verbose=False)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No cdiutils result"):
        MultiVolumeViewer.from_cdiutils(tmp_path / "empty", verbose=False)


def test_launcher(results):
    root, dirs = results
    lz = MultiVolumeViewer.launcher(root, figsize=(5, 5))
    assert len(lz.scans.options) == 3 and len(lz.scans.value) == 1
    lz.scans.value = tuple(v for _, v in lz.scans.options)
    assert lz._field_cbs["het_strain"].value
    assert not lz._field_cbs.get("support")
    lz._field_cbs["dspacing"].value = False
    lz.labels.value = "a, b, c"
    lz.units.value = "displacement=nm*0.1"
    lz.voxel.value = "4"
    cfg = lz.config()
    assert cfg["voxel_size"] == [4.0, 4.0, 4.0]
    assert cfg["field_units"] == {"displacement": ("nm", 0.1)}
    lz.show_config()
    text = lz.summary_text
    assert lz.status.value == "", lz.status.value
    assert "MultiVolumeViewer configuration" in text and "a  " in text
    lz.open_viewer()
    v = lz.viewer
    assert v is not None and lz.children[-1] is v
    assert v._cond_labels == ["a", "b", "c"] and "dspacing" not in v._layers
    assert v._unit_of("displacement") == ("nm", 0.1)
    lz.back()
    assert lz.viewer is None and lz.children == (lz.form,)
    lz.scans.value = ()
    lz.open_viewer()
    assert "Select at least one" in lz.status.value


def test_parse_units():
    assert parse_units("a=%, b = nm*0.1 ,") == {
        "a": ("%", 1.0),
        "b": ("nm", 0.1),
    }
    with pytest.raises(ValueError):
        parse_units("a")


def test_html_export(results, tmp_path):
    root, dirs = results
    v = MultiVolumeViewer.from_cdiutils(root, verbose=False)
    one = v.export_html(tmp_path / "view", plotlyjs="cdn")
    text = one.read_text()
    assert one.suffix == ".html" and "cdn.plot.ly" in text
    assert "S101" in text  # condition label
    many = v.export_html(tmp_path / "conds.html", conditions=True)
    text = many.read_text()
    assert "frames" in text and "S103" in text and "play" in text
    assert many.stat().st_size > one.stat().st_size  # plotly.js embedded
    v.close()


def test_html_action_and_edge_colours(results, tmp_path):
    root, dirs = results
    v = MultiVolumeViewer.from_cdiutils(root, verbose=False)
    v.run_actions(
        [
            {"edit": "amplitude", "color_by": "het_strain"},
            {"html": str(tmp_path / "a"), "slider": [0, "S103"]},
        ]
    )
    assert (tmp_path / "a.html").exists()
    # the isosurface lies on the support edge: colours are still defined
    tr = v.fig.data[v._trace_order.index("amplitude")]
    vals = np.asarray(tr.intensity, float)
    assert np.isfinite(vals).all() and np.ptp(vals) > 0
    v.close()


def test_nan_aware_interpolator():
    from cdiutils.interactive.multiviewer.geometry import NanAwareInterpolator

    a = np.full((3, 3, 3), np.nan)
    a[1:, :, :] = 2.0
    a[2, 1, 1] = 4.0
    ax = [np.arange(3.0)] * 3
    f = NanAwareInterpolator(ax, a)
    out = f([[0.5, 1, 1], [1.5, 1, 1], [0.0, 0.0, 0.0], [5, 5, 5]])
    assert out[0] == pytest.approx(2.0) and out[1] == pytest.approx(3.0)
    assert np.isnan(out[2]) and np.isnan(out[3])


@pytest.mark.parametrize("conv", ["cxi", "xu"])
def test_vti_round_trip(tmp_path, conv):
    pytest.importorskip("vtk")
    from cdiutils.io.vtk import save_as_vti

    p = props(shape=(20, 24, 28))
    path = tmp_path / "S9_structural_properties.vti"
    save_as_vti(
        str(path),
        voxel_size=(3, 4, 5),
        cxi_convention=conv == "cxi",
        amplitude=p["amplitude"],
        support=p["support"],
        het_strain=p["het_strain"],
    )
    r = loader.read_result(path, vti_convention=conv)
    assert r.voxel_size == (3.0, 4.0, 5.0) and r.scan == "9"
    assert np.allclose(r.fields["amplitude"], p["amplitude"] / 50)
    assert any("convention" in n for n in r.notes)


def write_preprocessed(path, scan, shape=(20, 22, 24), sample="Cu_BCDI"):
    """Same layout as BcdiPipeline._save_preprocessed_data."""
    from cdiutils.io.cxi import CXIFile

    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(float)
    c = np.array(shape) / 2
    inten = 1e4 * np.exp(
        -((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) / 8
    )
    with CXIFile(str(path), "w") as cxi:
        cxi.stamp()
        cxi.create_cxi_group(
            "result",
            qx_xu=np.linspace(2.0, 2.1, shape[0]),
            qy_xu=np.linspace(-0.05, 0.05, shape[1]),
            qz_xu=np.linspace(-0.06, 0.06, shape[2]),
        )
        cxi.create_cxi_group("sample", "sample_name", sample_name=sample)
        cxi.create_cxi_group("parameters", scan=scan)
        p = cxi.create_cxi_image(inten, data_type="cropped detector data")
        cxi.softlink("entry_1/cropped_detector_data", p)
        p = cxi.create_cxi_image(inten, data_type="orthogonalised data")
        cxi.softlink("entry_1/orthogonalised_detector_data", p)


def test_scans_as_layers(results, capsys):
    root, dirs = results
    v = MultiVolumeViewer.from_cdiutils(
        dirs[:2], scans_as="layers", fields=["amplitude", "het_strain"]
    )
    assert "scans as layers (side)" in capsys.readouterr().out
    assert v.raw_keys() == [
        "S101 | amplitude",
        "S101 | het_strain",
        "S102 | amplitude",
        "S102 | het_strain",
    ]
    assert v._visible_names() == ["S101 | amplitude", "S102 | amplitude"]
    assert v._unit_of("S102 | het_strain") == ("%", 1.0)
    assert v._style("S102 | amplitude")["thr"] == pytest.approx(0.3)
    t0 = v._layer_transform["S101 | amplitude"]["trans"][0]
    t1 = v._layer_transform["S102 | amplitude"]["trans"][0]
    assert t0 == 0 and t1 > 0
    assert v._layer_transform["S102 | het_strain"]["trans"][0] == t1
    v.close()
    w = MultiVolumeViewer.from_cdiutils(
        dirs[:2], scans_as="layers", arrange="overlay", verbose=False
    )
    assert not any(t["trans"][0] for t in w._layer_transform.values())
    w.close()


def test_sample_labels_from_folders(tmp_path):
    d = tmp_path / "Cu_BCDI" / "S1"
    d.mkdir(parents=True)
    np.savez(d / "S1_structural_properties.npz", **props())
    v = MultiVolumeViewer.from_cdiutils(
        tmp_path, label_from="sample", verbose=False
    )
    assert v.cdiutils_info[0]["sample"] == "Cu_BCDI"
    v.close()
    r, labels, _ = loader.load(tmp_path, label_from="sample")
    assert labels == ["Cu_BCDI S1"]


@pytest.mark.parametrize("kind", ["cropped", "orthogonalised"])
def test_diffraction_source(tmp_path, kind, capsys):
    for scan in (5, 6):
        d = tmp_path / "Cu" / f"S{scan}"
        d.mkdir(parents=True)
        write_preprocessed(d / f"S{scan}_preprocessed_data.cxi", scan)
    v = MultiVolumeViewer.from_cdiutils(
        tmp_path, source="diffraction", diffraction=kind, label_from="sample"
    )
    out = capsys.readouterr().out
    # the sample name stored in the file wins over the folder name
    assert v.raw_keys() == ["intensity"]
    assert v._cond_labels == ["Cu_BCDI S5", "Cu_BCDI S6"]
    a = v.dict_data["intensity"]
    assert a.max() == pytest.approx(np.log1p(1e4), rel=1e-3)
    if kind == "cropped":
        assert v.unit == "index" and np.allclose(v.voxel_size, 1)
        assert "index units" in out
    else:
        assert v.unit == "1/Å"
        assert v.voxel_size[0] == pytest.approx(0.1 / 19)
    v.close()


def test_launcher_layers_and_diffraction(results, tmp_path):
    root, dirs = results
    lz = MultiVolumeViewer.launcher(root)
    lz.scans.value = tuple(v for _, v in lz.scans.options)[:2]
    lz.scans_as.value = "side"
    lz.open_viewer()
    assert lz.status.value == "", lz.status.value
    assert "S102 | amplitude" in lz.viewer._layers
    lz.back()
    d = tmp_path / "diff" / "Cu" / "S5"
    d.mkdir(parents=True)
    write_preprocessed(d / "S5_preprocessed_data.cxi", 5)
    lz.folder.value = str(tmp_path / "diff")
    lz.source.value = "cropped"
    assert len(lz.scans.options) == 1
    lz.open_viewer()
    assert lz.status.value == "", lz.status.value
    assert lz.viewer.raw_keys() == ["intensity"]
    lz.back()


def _non_finite_widget_values(viewer):
    """Widget traits that would break the JSON messages to the browser."""
    import math

    import ipywidgets as w

    bad, seen, todo = [], set(), [viewer]
    while todo:
        x = todo.pop()
        if id(x) in seen or not isinstance(x, w.Widget):
            continue
        seen.add(id(x))
        for k in x.keys:
            try:
                val = getattr(x, k)
            except Exception:
                continue
            for q in val if isinstance(val, (list, tuple)) else [val]:
                if isinstance(q, float) and not math.isfinite(q):
                    bad.append((type(x).__name__, x.description, k))
                elif isinstance(q, w.Widget):
                    todo.append(q)
    return bad


def test_no_nan_in_widgets(results):
    root, dirs = results
    for kw in ({}, {"scans_as": "layers"}):
        v = MultiVolumeViewer.from_cdiutils(root, verbose=False, **kw)
        v.run_actions([{"edit": v.raw_keys()[0], "color_by": "__facet__"}])
        assert not _non_finite_widget_values(v)
        v.close()


def test_repeated_scan_numbers_get_unique_labels(tmp_path):
    for sample in ("Cu_BCDI", "Cu_BCDI_0002", "Cu_BCDI_0003"):
        d = tmp_path / sample / "S1"
        d.mkdir(parents=True)
        np.savez(d / "S1_structural_properties.npz", **props())
    d = tmp_path / "Cu_BCDI_0003" / "S34"
    d.mkdir()
    np.savez(d / "S34_structural_properties.npz", **props())
    r, labels, _ = loader.load(tmp_path, fields=["amplitude"])
    assert labels == [
        "Cu_BCDI S1",
        "Cu_BCDI_0002 S1",
        "Cu_BCDI_0003 S1",
        "S34",
    ]
    v = MultiVolumeViewer.from_cdiutils(
        tmp_path, scans_as="layers", fields=["amplitude"], verbose=False
    )
    assert len(v.raw_keys()) == 4  # nothing overwritten
    v.close()
    with pytest.raises(ValueError, match="unique"):
        loader.load(tmp_path, labels=["a", "a", "b", "c"])


def test_resample_different_voxel_sizes(tmp_path, capsys):
    p1 = props(shape=(20, 24, 24), r=5)
    p1["voxel_size"] = np.array([10.0, 5.0, 5.0])
    p2 = props(shape=(40, 24, 24), r=5)
    p2["voxel_size"] = np.array([5.0, 5.0, 5.0])
    for scan, p in ((1, p1), (2, p2)):
        d = tmp_path / f"S{scan}"
        d.mkdir()
        np.savez(d / f"S{scan}_structural_properties.npz", **p)
    v = MultiVolumeViewer.from_cdiutils(tmp_path, scans_as="layers")
    out = capsys.readouterr().out
    assert "resampled (linear)" in out
    assert np.allclose(v.voxel_size, 5)
    assert v.dict_data["S1 | amplitude"].shape == (40, 24, 24)
    s1 = v.get_volume("S1 | het_strain")
    assert np.isnan(s1[0, 0, 0]) and np.isfinite(s1[20, 12, 12])
    # side by side: true proportions
    ar = v.fig.layout.scene.aspectratio
    assert v.fig.layout.scene.aspectmode == "manual" and max(
        ar.x, ar.y, ar.z
    ) == pytest.approx(1)
    v.close()
    with pytest.warns(UserWarning, match="voxel sizes differ"):
        w = MultiVolumeViewer.from_cdiutils(
            tmp_path, resample=False, verbose=False
        )
    assert np.allclose(w.voxel_size, (10, 5, 5))
    w.close()
