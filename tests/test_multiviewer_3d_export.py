"""Exports of MultiVolumeViewer: GIF/MP4 writers, scene deltas, render pool."""

import asyncio
import io

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer import export as ex  # noqa: E402
from cdiutils.interactive.multiviewer.rendering import (  # noqa: E402
    render_png,
)


def blob(n=24):
    z, y, x = np.mgrid[:n, :n, :n].astype(float) - n / 2
    amp = np.clip(1.2 - np.sqrt(x**2 + y**2 + z**2) / 8, 0, 1)
    return {
        "amplitude": amp.astype(np.float32),
        "phase": (0.05 * x).astype(np.float32),
    }


@pytest.fixture
def v():
    viewer = MultiVolumeViewer(blob(), voxel_size=(5, 5, 5), unit="nm")
    viewer.anim_width.value, viewer.anim_height.value = 160, 120
    viewer.anim_scale.value = 1
    yield viewer
    viewer.close()


def _png(color):
    import imageio.v2 as iio

    buf = io.BytesIO()
    img = np.zeros((20, 30, 3), np.uint8)
    img[:] = color
    iio.imwrite(buf, img, format="png")
    return buf.getvalue()


def test_gif_needs_no_system_ffmpeg(tmp_path, monkeypatch):
    import shutil

    from PIL import Image

    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    sink = ex.FrameSink(tmp_path / "a" / "anim.gif", "gif", fps=10)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (250, 250, 250)]
    for c in colors:
        sink.add(_png(c))
    sink.close()
    with Image.open(tmp_path / "a" / "anim.gif") as im:
        assert im.n_frames == 4 and im.info["duration"] == 100
        for i, c in enumerate(colors):
            im.seek(i)
            got = np.asarray(im.convert("RGB"))[10, 15]
            assert np.abs(got.astype(int) - c).max() <= 8, (i, got)


def test_mp4_uses_bundled_ffmpeg(tmp_path):
    import imageio_ffmpeg

    sink = ex.FrameSink(tmp_path / "m.mp4", "mp4", fps=5)
    for c in [(255, 0, 0)] * 3 + [(0, 0, 255)] * 3:
        sink.add(_png(c))
    sink.close()
    gen = imageio_ffmpeg.read_frames(str(tmp_path / "m.mp4"))
    meta = next(gen)
    w, h = meta["size"]
    frames = [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in gen]
    assert len(frames) == 6
    assert frames[0][..., 0].mean() > 200 and frames[-1][..., 2].mean() > 200


def test_merge_delta():
    base = {
        "data": [{"type": "mesh3d", "name": "a"}, {"name": "b"}],
        "layout": {"scene": {"camera": {"eye": 1}, "bgcolor": "w"}},
    }
    fig = ex.merge_delta(
        base, {"camera": {"eye": 2}, "traces": {1: {"name": "c"}}}
    )
    assert fig["layout"]["scene"] == {"camera": {"eye": 2}, "bgcolor": "w"}
    assert [t["name"] for t in fig["data"]] == ["a", "c"]
    assert base["layout"]["scene"]["camera"] == {"eye": 1}  # untouched
    assert base["data"][1]["name"] == "b"


def test_cached_primitives_render_identically(v):
    base = v._export_base_json(160, 120)
    full = render_png(base, 160, 120, 1)
    sc = ex._Scene(base)
    for _ in range(2):  # second call reuses the decoded triangles
        assert render_png(sc.figure({}, True), 160, 120, 1) == full
    assert sc.prims


def _frames_of(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.n_frames


def test_safe_exports_with_deltas(v, tmp_path):
    out = v.start_rotation_export(
        "orbit_z", frames=3, fmt="gif", name=str(tmp_path / "rot")
    )
    assert _frames_of(out) == 3
    v.add_slice("zmid", "phase", axis="z")
    out = v.start_layer_export(
        "zmid", "pos", (2, 20), points=3, fmt="gif", name=str(tmp_path / "sw")
    )
    assert _frames_of(out) == 3


def test_render_pool_is_reused_and_closed(v, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_MP_CONTEXT", "spawn")
    v.rendering_mode, v.render_workers, v.render_in_flight = "fast", 2, 2
    ex.shutdown_pool()

    async def go():
        a = await v.start_rotation_export(
            "orbit_z", frames=4, fmt="png", name=str(tmp_path / "a")
        )
        pool = ex._POOL["executor"]
        b = await v.start_rotation_export(
            "orbit_y", frames=4, fmt="png", name=str(tmp_path / "b")
        )
        return a, b, pool

    a, b, pool = asyncio.run(go())
    assert pool is not None and ex._POOL["executor"] is pool  # reused
    assert len(list(a.glob("*.png"))) == len(list(b.glob("*.png"))) == 4
    import tempfile
    from pathlib import Path

    left = list(Path(tempfile.gettempdir()).glob("mv_scene_*"))
    assert not left  # scene files are removed after each export
    # same frames as sequential rendering
    v.rendering_mode = "safe"
    c = v.start_rotation_export(
        "orbit_z", frames=4, fmt="png", name=str(tmp_path / "c")
    )
    for f in sorted(a.glob("*.png")):
        assert f.read_bytes() == (c / f.name).read_bytes()
    v.close()
    assert not ex.pool_alive()
