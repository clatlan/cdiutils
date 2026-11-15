"""Animation through conditions and the condition label."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer import anim as _mv_anim  # noqa: E402
from cdiutils.interactive.multiviewer import export as ex  # noqa: E402
from cdiutils.interactive.multiviewer.actions import ActionError  # noqa: E402


def make(n=24, r=6.0, k=1.0):
    z, y, x = np.mgrid[:n, :n, :n].astype(float) - n / 2
    amp = np.clip(1.2 - np.sqrt(x**2 + y**2 + z**2) / r, 0, 1)
    return {
        "amplitude": amp.astype(np.float32),
        "het_strain": (k * 1e-3 * (x / n + 0.1)).astype(np.float32),
    }


@pytest.fixture
def v():
    viewer = MultiVolumeViewer(
        [make(r=5), make(r=7, k=2), make(r=9, k=4)],
        voxel_size=(5, 5, 5),
        unit="nm",
        condition_labels=["0 bar", "10 bar", "20 bar"],
    )
    viewer.run_actions(
        [{"edit": "amplitude", "color_by": "het_strain", "range": "sym"}]
    )
    viewer.anim_width.value, viewer.anim_height.value = 160, 120
    viewer.anim_scale.value = 1
    yield viewer
    viewer.close()


def _capture(monkeypatch):
    seen = []

    def fake(fig_json, *a, **k):
        seen.append(fig_json)
        return _fake_png()

    monkeypatch.setattr(_mv_anim, "render_png", fake)
    return seen


def _fake_png():
    import io

    import imageio.v2 as iio

    buf = io.BytesIO()
    iio.imwrite(buf, np.zeros((8, 8, 3), np.uint8), format="png")
    return buf.getvalue()


def test_condition_export_frames(v, tmp_path, monkeypatch):
    seen = _capture(monkeypatch)
    out = v.start_condition_export(
        hold=3, fmt="png", name=str(tmp_path / "c"), label_format="{index}/{n}"
    )
    assert len(list(out.glob("*.png"))) == 9  # 3 conditions x 3
    assert len(seen) == 3  # repeated frames are not rendered again
    labels = [s["layout"]["annotations"][-1]["text"] for s in seen]
    assert labels == ["<b>1/3</b>", "<b>2/3</b>", "<b>3/3</b>"]
    # same colour range and axes in every frame, sizes grow
    rngs = {(s["data"][0]["cmin"], s["data"][0]["cmax"]) for s in seen}
    assert len(rngs) == 1
    axes = {
        tuple(tuple(s["layout"]["scene"][f"{a}axis"]["range"]) for a in "xyz")
        for s in seen
    }
    assert len(axes) == 1
    sizes = [np.ptp(np.asarray(s["data"][0]["x"], float)) for s in seen]
    assert sizes[0] < sizes[1] < sizes[2]
    assert v._cond_index == 0  # the live view is untouched


def test_subset_labels_and_rotation(v, tmp_path, monkeypatch):
    seen = _capture(monkeypatch)
    v.start_condition_export(
        ["20 bar", 0],
        hold=2,
        degrees=90,
        shared_range=False,
        fix_axes=False,
        fmt="png",
        name=str(tmp_path / "s"),
    )
    assert len(seen) == 4  # rotation: every frame rendered
    assert [s["layout"]["annotations"][-1]["text"] for s in seen] == [
        "<b>20 bar</b>"
    ] * 2 + ["<b>0 bar</b>"] * 2
    eyes = [s["layout"]["scene"]["camera"]["eye"]["x"] for s in seen]
    assert len(set(np.round(eyes, 6))) == 4
    assert seen[0]["data"][0]["cmax"] != seen[2]["data"][0]["cmax"]
    assert "range" not in seen[0]["layout"]["scene"]["xaxis"]


def test_label_on_stills_and_rotations(v, tmp_path, monkeypatch):
    seen = _capture(monkeypatch)
    v.set_condition(1)
    v.write_image(tmp_path / "a.png")
    v.write_image(tmp_path / "b.png", label=False)
    v.start_rotation_export(frames=2, fmt="png", name=str(tmp_path / "r"))
    texts = [
        [a["text"] for a in s["layout"].get("annotations") or []] for s in seen
    ]
    assert texts[0] == ["<b>10 bar</b>"] and texts[1] == []
    assert texts[2] == texts[3] == ["<b>10 bar</b>"]
    v.anim_label.value = False
    assert v._export_base_json(100, 100)["layout"].get("annotations") in (
        None,
        [],
        (),
    )


def test_real_gif_and_script(v, tmp_path):
    async def go():
        rep = v.run_actions(
            [
                {
                    "conditions": "all",
                    "hold": 2,
                    "format": "gif",
                    "name": str(tmp_path / "anim"),
                },
                {"edit": "amplitude", "opacity": 0.5},
            ]
        )
        await v.wait_for_script()
        return rep

    rep = asyncio.run(go())
    assert rep.ok
    from PIL import Image

    with Image.open(tmp_path / "anim.gif") as im:
        # Pillow merges identical frames and adds their durations
        assert im.n_frames == 3
        im.seek(0)
        assert im.info["duration"] == pytest.approx(
            2000 / 30, abs=10
        )  # GIF: 10 ms steps
    with pytest.raises(ActionError) as e:
        v.run_actions([{"conditions": ["0 bar", "5 bar"]}])
    assert "unknown condition '5 bar'" in "\n".join(e.value.problems)


def test_repeat_frames_in_exporter(tmp_path):
    frames = []

    class Sink:
        n = 0

        def add(self, png):
            frames.append(png)
            self.n += 1

    def produce(k):
        return {"k": k} if k % 2 == 0 else ex.REPEAT

    n = asyncio.run(
        ex.run_ordered_export(
            5, produce, lambda f: f"png{f['k']}".encode(), (), Sink()
        )
    )
    assert n == 5 and frames == [b"png0", b"png0", b"png2", b"png2", b"png4"]


def test_single_condition_has_no_label_or_mode():
    w = MultiVolumeViewer(make(), voxel_size=(5, 5, 5))
    assert w.anim_label_row.layout.display == "none"
    w.anim_master_mode.value = "conditions"
    assert w.anim_save_btn.disabled
    with pytest.raises(RuntimeError):
        w.start_condition_export()
    w.close()
