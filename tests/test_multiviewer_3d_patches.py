"""Facet patches: segmentation, table, colour, labels, CSV."""

import csv

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer import analysis as mva  # noqa: E402


@pytest.fixture
def v():
    """A cuboctahedron: 8 {111} and 6 {100} facets."""
    n = 56
    z, y, x = np.mgrid[:n, :n, :n].astype(float) - n / 2 + 0.3
    d = np.maximum(np.maximum(abs(x), abs(y)), abs(z)) / 16
    o = (abs(x) + abs(y) + abs(z)) / 24
    amp = np.clip(1.5 - np.maximum(d, o), 0, 1).astype(np.float32)
    strain = (1e-3 * x / n).astype(np.float32)
    viewer = MultiVolumeViewer(
        {"amplitude": amp, "strain": strain},
        voxel_size=(5, 5, 5),
        unit="nm",
        facet_families="100, 111",
    )
    viewer.set_layer_params("amplitude", thr=0.5)
    viewer.run_actions([{"camera": "iso"}])
    yield viewer
    viewer.close()


def test_patches_of_a_cuboctahedron(v):
    rows = v.facet_patches("amplitude", "strain")
    assert len(rows) == 14
    fams = [r["family"] for r in rows]
    assert fams.count("{111}") == 8 and fams.count("{100}") == 6
    assert len({r["plane"] for r in rows}) == 14
    for r in rows:
        assert r["deviation_deg"] < 2 and r["hkl_of_mean"] == r["plane"]
    assert rows[0]["area"] >= rows[-1]["area"]  # sorted by area
    assert sum(r["area_frac"] for r in rows) < 1  # edges are not facets
    # strain = x: (100) and (-100) facets have opposite means
    by = {r["plane"]: r for r in rows}
    assert by["(100)"]["mean"] == pytest.approx(
        -by["(-100)"]["mean"], rel=0.05
    )
    assert by["(100)"]["mean"] > 0


def test_colour_labels_and_csv(v, tmp_path):
    v.facet_patches("amplitude", labels=True, colour=True)
    t = v.fig.data[v._trace_order.index("amplitude")]
    assert t.intensitymode == "cell" and t.showscale is False
    assert len(set(np.round(t.intensity, 3))) == 15  # 14 patches + edges
    front = len(v._layers["amplitude:facets"]["labels"])
    assert 4 <= front < 14  # only facets facing the camera
    v.facet_patches("amplitude", labels="all")
    assert len(v._layers["amplitude:facets"]["labels"]) == 14
    p = v.save_facet_csv(tmp_path / "f.csv", "amplitude", "strain")
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 14 and {"plane", "area", "mean", "std"} <= set(rows[0])


def test_min_faces_and_action(v, tmp_path):
    assert len(v.facet_patches("amplitude", min_faces=10**6)) == 0
    rep = v.run_actions(
        [
            {
                "patches": "amplitude",
                "field": "strain",
                "min_faces": 20,
                "labels": True,
                "csv": str(tmp_path / "a.csv"),
            }
        ]
    )
    assert rep.ok and (tmp_path / "a.csv").exists()
    assert v._style("amplitude")["color_by"] == "__patch__"


def test_panel(v):
    v.facet_layer.value = "amplitude"
    v.facet_field.value = "strain"
    v.facet_by.value = "patch"
    v.facet_btn.click()
    assert "(111)" in v.facet_out.value and "deviation" in v.facet_out.value
    v.patch_show.click()
    assert v._style("amplitude")["color_by"] == "__patch__"


def test_face_patches_unit():
    # two triangles sharing an edge, same plane -> one patch; other -> -1
    faces = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 2]])
    plane = np.array(["(001)", "(001)", "other"], dtype=object)
    fam = np.array([0, 0, -1])
    ids = mva.face_patches(faces, plane, fam, 5, min_faces=1, min_area_frac=0)
    assert ids.tolist() == [0, 0, -1]
