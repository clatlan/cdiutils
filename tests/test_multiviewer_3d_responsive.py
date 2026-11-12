"""Background refresh, colour-only updates and bounded caches."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive import multiviewer_3d as mv3d  # noqa: E402


def make(n=32, r=8.0, k=1.0):
    z, y, x = np.mgrid[:n, :n, :n].astype(float) - n / 2
    amp = np.clip(1.2 - np.sqrt(x**2 + y**2 + z**2) / r, 0, 1)
    return {
        "amplitude": amp.astype(np.float32),
        "strain": (k * 1e-3 * x).astype(np.float32),
    }


@pytest.fixture
def v():
    viewer = MultiVolumeViewer(
        [make(), make(r=11, k=2)], voxel_size=(5, 5, 5), unit="nm"
    )
    viewer.set_layer_params("amplitude", color_by="strain", cmap="viridis")
    yield viewer
    viewer.close()


def trace(v, name="amplitude"):
    return v.fig.data[v._trace_order.index(name)]


def test_panel_edits_run_in_background_and_merge(v, monkeypatch):
    calls = []
    real = v._rebuild
    monkeypatch.setattr(v, "_rebuild", lambda: (calls.append(1), real()))
    n0 = len(trace(v).x)

    async def go():
        w = v._layer_widgets["amplitude"]
        for thr in (0.3, 0.4, 0.5):  # rapid edits
            w["thr"].value = thr
        assert len(trace(v).x) == n0  # not applied yet: kernel stays free
        assert "updating" in v.busy_label.value
        await v.wait_for_refresh()
        v.cond_slider.value = 1  # condition slider too
        await v.wait_for_refresh()

    asyncio.run(go())
    assert v._style("amplitude")["thr"] == pytest.approx(0.5)
    assert len(trace(v).x) != n0 and v._cond_index == 1
    assert v.busy_label.value == ""
    assert len(calls) == 1  # the condition change rebuilt once


def test_scripts_stay_synchronous(v):
    async def go():
        v.run_actions([{"edit": "amplitude", "thr": 0.6}])
        return len(trace(v).x)

    n = asyncio.run(go())
    v.set_layer_params("amplitude", thr=0.6)
    assert n == len(trace(v).x)


def test_colour_only_changes_keep_geometry(v, monkeypatch):
    built = []
    real = v._build_trace
    monkeypatch.setattr(
        v,
        "_build_trace",
        lambda *a, **k: (built.append(a[0]), real(*a, **k))[1],
    )
    x0 = trace(v).x
    w = v._layer_widgets["amplitude"]
    w["cmap"].value = "RdBu_r"
    w["range_mode"].value = "sym"
    t = trace(v)
    assert not built and t.x is x0  # geometry untouched
    assert t.cmin == pytest.approx(-t.cmax)
    assert tuple(w["range_slider"].value) == pytest.approx((t.cmin, t.cmax))
    assert t.colorscale[0][1] != v.get_colorscale("viridis")[0][1]
    w["range_mode"].value = "manual"
    w["range_slider"].value = (-1e-3, 1e-3)
    assert (trace(v).cmin, trace(v).cmax) == pytest.approx((-1e-3, 1e-3))
    assert not built
    w["cmap"].value = "red"  # solid colour: colorbar disappears -> rebuild
    assert built


def test_byte_bounded_cache():
    c = mv3d._LRU(100, max_bytes=3 * 8000)
    for k in range(5):
        c.get_or(k, lambda: np.zeros(1000))  # 8000 bytes each
    assert list(c) == [2, 3, 4] and c.nbytes == 24000
    c.get_or(2, lambda: None)  # hit moves to the end
    c.get_or(5, lambda: np.zeros(1000))
    assert list(c) == [4, 2, 5]
    c.clear()
    assert c.nbytes == 0 and not c


def test_cache_info_and_clear(v):
    v.cond_slider.value = 1
    info = v.cache_info()
    assert info["isosurfaces"] >= 2 and info["isosurfaces_MB"] > 0
    assert info["limit_MB"] == 2048
    v.clear_caches()
    assert v.cache_info()["isosurfaces"] >= 1  # visible layers rebuilt
    w = MultiVolumeViewer(make(), cache_mb=64)
    assert w.cache_info()["limit_MB"] == 64
    w.close()
