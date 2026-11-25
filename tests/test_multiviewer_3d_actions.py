"""Scripted actions, help and tutorial of MultiVolumeViewer (headless)."""

import asyncio
import io

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer import anim as _mv_anim  # noqa: E402
from cdiutils.interactive.multiviewer.actions import (  # noqa: E402
    ACTIONS,
    ActionError,
    ActionsMixin,
)


def make(n=32, shift=0.0):
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    c = n / 2
    r = np.maximum(np.maximum(abs(z - c), abs(y - c)), abs(x - c)) / 10
    amp = np.clip(
        1.2 - np.maximum(r, (abs(z - c) + abs(y - c) + abs(x - c)) / 16), 0, 1
    ).astype(np.float32)
    ph = (0.03 * (x - c) + shift * 0.01 * (z - c)).astype(np.float32)
    ph[amp < 0.05] = np.nan
    st = np.gradient(np.nan_to_num(ph), axis=0).astype(np.float32)
    return {"amplitude": amp, "phase": ph, "het_strain": st}


@pytest.fixture
def v():
    viewer = MultiVolumeViewer(
        [make(), make(shift=1.0)],
        voxel_size=(5, 5, 5),
        unit="nm",
        condition_labels=["0 bar", "1 bar"],
    )
    yield viewer
    viewer.close()


def _fake_png(*_a, **_k):
    import imageio.v2 as iio

    buf = io.BytesIO()
    iio.imwrite(buf, np.zeros((8, 8, 3), np.uint8), format="png")
    return buf.getvalue()


def test_all_step_formats_are_equivalent(v):
    want = ("edit", {"layer": "amplitude", "thr": 0.3})
    forms = [
        {"edit": "amplitude", "thr": 0.3},
        ("edit", "amplitude", {"thr": 0.3}),
        {"action": "edit", "layer": "amplitude", "params": {"thr": 0.3}},
        {"action": "edit", "layer": "amplitude", "thr": 0.3},
    ]
    for f in forms:
        assert ActionsMixin._normalize_step(f) == want
    assert ActionsMixin._normalize_step("show phase") == (
        "show",
        {"layer": "phase"},
    )
    assert ActionsMixin._normalize_step({"refresh": None}) == ("refresh", {})
    assert v.script().edit("amplitude", thr=0.3).steps == [
        {"edit": "amplitude", "thr": 0.3}
    ]


def test_script_is_checked_before_running(v):
    n0 = len(v._layers)
    bad = [
        {"slice": "zmid", "source": "phase"},  # fine, but must not run
        {"edit": "amplitude", "cmpa": "viridis"},
        {"slise": "x"},
        {"show": "phse"},
        {"clip": "c", "side": "left", "color_by": "het_strian"},
        {"condition": "2 bar"},
        {"camera": "top"},
        {"edit": "zmid", "cmap": "viridiss"},
    ]
    with pytest.raises(ActionError) as e:
        v.run_actions(bad)
    msgs = "\n".join(e.value.problems)
    for needle in (
        "'cmap'",
        "'slice'",
        "'phase'",
        "up, down",
        "'het_strain'",
        "'2 bar'",
        "'top'",
        "'viridis'",
        "step 8",
    ):
        assert needle in msgs, needle
    assert len(v._layers) == n0  # nothing ran


def test_create_with_style_and_replace(v):
    for _ in range(2):  # re-running a cell does not duplicate layers
        r = v.run_actions(
            [
                {
                    "slice": "zmid",
                    "source": "phase",
                    "axis": "x",
                    "cmap": "twilight",
                    "opacity": 0.5,
                }
            ]
        )
    assert [k for k in v._layers if k.startswith("zmid")] == ["zmid"]
    st = v._style("zmid")
    assert st["cmap"] == "twilight" and np.isclose(st["op"], 0.5)
    assert r.ok and r.created == ["zmid"]
    v.run_actions([{"slice": "zmid", "source": "phase", "replace": False}])
    assert "zmid#2" in v._layers
    with pytest.raises(ActionError):
        v.run_actions(
            [{"slice": "amplitude", "source": "phase"}]
        )  # raw names are protected


def test_scene_actions(v):
    v.run_actions(
        [
            {"only": ["phase"]},
            {"camera": "x", "zoom": 2},
            {"condition": "1 bar"},
            {"theme": True},
            {"grid": False},
            {"facets": "100, 111", "tolerance": 7},
            {"probe": [80, 80, 80]},
            {"ortho": True, "field": "phase"},
        ]
    )
    assert v._trace_order == ["phase"]
    assert np.isclose(v.fig.layout.scene.camera.eye.x, 1.1)
    assert (
        v._cond_index == 1 and v.theme_toggle.value and not v.grid_toggle.value
    )
    assert [f[0] for f in v.facet_families] == [
        "{100}",
        "{111}",
    ] and v.facet_tolerance == 7
    assert len(v._probe_points) == 1 and v.ortho_show.value


def test_failed_step_stops_and_reports(v):
    with pytest.raises(ActionError) as e:
        v.run_actions(
            [{"show": "phase"}, {"edit": "phase", "pos": 3}, {"hide": "phase"}]
        )
    rep = e.value.report
    assert [r["status"] for r in rep] == ["done", "failed", "skipped"]
    rep2 = v.run_actions(
        [{"edit": "phase", "pos": 3}, {"hide": "phase"}], stop_on_error=False
    )
    assert [r["status"] for r in rep2] == ["failed", "done"]


def test_view_to_script_round_trip(v, tmp_path):
    v.run_actions(
        [
            {"only": ["amplitude"]},
            {
                "clip": "cut",
                "source": "amplitude",
                "normal": [1, 0, 0],
                "offset": 12,
                "color_by": "het_strain",
                "cmap": "RdBu_r",
            },
            {"difference": "dphase", "a": "phase", "ref": 0, "visible": False},
            {
                "lines": "dislo",
                "points": [[5, 5, 5], [20, 20, 20]],
                "color": "#ff0000",
            },
            {
                "arrows": "b",
                "origins": [[50, 50, 50]],
                "vectors": [[0, 30, 0]],
                "labels": ["b"],
            },
            {"edit": "amplitude", "thr": 0.45, "rot_z": 20},
            {"camera": "iso", "zoom": 1.3},
        ]
    )
    path = v.save_actions(tmp_path / "view.yaml")
    w = MultiVolumeViewer(
        [make(), make(shift=1.0)], voxel_size=(5, 5, 5), unit="nm"
    )
    rep = w.run_actions(path)
    assert rep.ok
    assert list(w._layers) == list(v._layers)
    assert w._visible_names() == v._visible_names()
    for name in v._layers:
        a, b = v._style(name), w._style(name)
        for k in ("thr", "cmap", "color_by", "op", "color"):
            assert a[k] == pytest.approx(b[k]), (name, k)
    assert np.allclose(w._layers["cut"]["normal"], v._layers["cut"]["normal"])
    assert w._layers["cut"]["offset"] == pytest.approx(
        v._layers["cut"]["offset"]
    )
    assert w._layer_transform["amplitude"]["rot_deg"][2] == 20
    assert w._layers["b"]["texts"] == ["b"]
    assert w.fig.layout.scene.camera.eye.x == pytest.approx(
        v.fig.layout.scene.camera.eye.x, rel=1e-5
    )


def test_exports_are_chained_in_order(v, tmp_path, monkeypatch):
    monkeypatch.setattr(
        _mv_anim,
        "render_png",
        lambda fig_json, *a: (seen.append(fig_json), _fake_png())[1],
    )
    v._backend = "kaleido"
    seen = []
    script = [
        {"slice": "zmid", "source": "phase", "axis": "z"},
        {
            "sweep": "zmid",
            "param": "pos",
            "range": [0, 31],
            "points": 4,
            "format": "png",
            "name": str(tmp_path / "sweep"),
        },
        {"edit": "zmid", "opacity": 0.3},  # must apply after the sweep
        {
            "rotation": "orbit_z",
            "frames": 3,
            "format": "png",
            "name": str(tmp_path / "rot"),
        },
        {"hide": "zmid"},
    ]

    async def go():
        rep = v.run_actions(script)  # returns at once in a running loop
        assert rep[1]["status"] == "exporting" and rep[2]["status"] == "queued"
        await v.wait_for_script()
        return rep

    rep = asyncio.run(go())
    assert [r["status"] for r in rep] == ["done"] * 5
    assert (
        len(list((tmp_path / "sweep").glob("*.png"))) == 4
        and len(list((tmp_path / "rot").glob("*.png"))) == 3
    )
    idx = 1  # zmid trace index (amplitude, zmid)
    ops = [s["data"][idx].get("opacity", 1.0) for s in seen]
    assert ops[:4] == [1.0] * 4 and ops[4:] == [0.3] * 3
    assert "zmid" not in v._trace_order


def test_builder_save_and_docs(v, tmp_path):
    s = (
        v.script()
        .only("amplitude")
        .slice("zmid", source="phase", cmap="twilight")
        .camera("iso", zoom=1.2)
    )
    assert len(s) == 3 and "slice('zmid'" in repr(s)
    assert s.check() == "No problems found."
    p = s.save(tmp_path / "s.yaml")
    assert v.run_actions(p).ok
    assert "slice(name" in type(s).slice.__doc__
    assert set(ACTIONS) <= set(dir(type(s)))


def test_help_topics_and_actions(v, capsys):
    from cdiutils.interactive.multiviewer.help import TOPICS

    for topic in [None, *TOPICS, *ACTIONS, "slise"]:
        v.help(topic)
    out = capsys.readouterr().out
    assert (
        "did you mean 'slice'" in out
        and "Scripted actions" in out
        and "PLOT_ORDER='zyx'" in out
    )


def test_tutorial_runs_every_step(v, monkeypatch, capsys):
    monkeypatch.setattr(type(v), "write_image", lambda self, *a, **k: None)
    t = v.tutorial()
    for i in range(len(t.steps)):
        t.run_step(i)
        assert not any(
            "Error" in str(o.get("text", "")) for o in t.out.outputs
        ), (i, t.out.outputs)
    assert not [
        k for k in v._layers if k.startswith("tuto:")
    ]  # last step cleans up


@pytest.mark.parametrize("fmt", ["mp4", "gif", "png"])
def test_export_name_can_be_a_path(v, tmp_path, monkeypatch, fmt):
    from cdiutils.interactive.multiviewer.export import output_path

    assert output_path("run_1.5bar", "mp4", "x").name == "run_1.5bar.mp4"
    assert output_path("out/", "gif", "rotation") == output_path(
        "out/rotation", "gif", "x"
    )
    monkeypatch.setattr(_mv_anim, "render_png", _fake_png)
    v._backend = "kaleido"
    target = tmp_path / "new" / "nested" / "spin"
    out = v.start_rotation_export(
        "orbit_z", frames=2, fmt=fmt, name=str(target)
    )
    out = asyncio.run(out) if asyncio.iscoroutine(out) else out
    assert (
        target.with_name(f"spin.{fmt}") if fmt != "png" else target
    ).exists()
