"""The tabbed layout: toolbar, layer list, Add tiles, Script tab, presets."""

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer.ui import TAB_TITLES  # noqa: E402


def make(n=24, k=1.0):
    z, y, x = np.mgrid[:n, :n, :n].astype(float) - n / 2
    amp = np.clip(1.2 - np.sqrt(x**2 + y**2 + z**2) / 7, 0, 1)
    return {
        "amplitude": amp.astype(np.float32),
        "phase": (0.05 * x).astype(np.float32),
    }


@pytest.fixture
def v():
    viewer = MultiVolumeViewer(
        [make(), make(k=2)],
        voxel_size=(5, 5, 5),
        unit="nm",
        condition_labels=["a", "b"],
    )
    yield viewer
    viewer.close()


def test_tabs_toolbar_and_status(v):
    assert tuple(v.tabs.get_title(i) for i in range(6)) == TAB_TITLES
    assert v.children[1] is v.toolbar and v.body.children[1] is v.right_panel
    assert v.cond_row in v.toolbar.children
    s = v.status_html.value
    assert "2 layers" in s and "1 visible" in s and "condition <b>a</b>" in s
    v.set_condition(1)
    assert "condition <b>b</b>" in v.status_html.value
    v.panel_toggle.value = False
    assert v.right_panel.layout.display == "none"
    v.panel_toggle.value = True
    v.view_btns["x"].click()
    assert v.fig.layout.scene.camera.eye.x == pytest.approx(2.2)


def test_panel_on_the_left_and_width():
    w = MultiVolumeViewer(make(), panel_side="left", panel_width="400px")
    assert w.body.children[0] is w.right_panel
    assert w.right_panel.layout.width == "400px"
    w.close()
    with pytest.raises(ValueError):
        MultiVolumeViewer(make(), panel_side="top")


def test_layer_list_selects_and_shows_colours(v):
    rows = v.visible_keys_box.children
    assert [r.children[2].description for r in rows] == ["amplitude", "phase"]
    rows[1].children[2].click()  # the name button selects the layer
    assert v.edit_key.value == "phase"
    assert "mv-row-sel" in rows[1]._dom_classes
    assert "mv-row-sel" not in rows[0]._dom_classes
    rows[1].children[0].value = True  # the checkbox shows it
    assert "phase" in v._trace_order
    v._layer_widgets["phase"]["cmap"].value = "viridis"
    sw = v._rows["phase"][1].value
    assert "253,231,36" in sw  # viridis' yellow end
    assert sw.count("%") <= 8
    v.add_slice("zmid", "phase", axis="z")
    names = [r.children[2].description for r in v.visible_keys_box.children]
    assert names[-1] == "zmid" and v.edit_key.value == "zmid"


def test_add_tiles(v):
    tiles = v._build_add_tiles().children[0].children
    by = {t.description.split(" ", 1)[1]: t for t in tiles}
    by["Clip"].click()
    assert v.add_layer_kind.value == "clip"
    by["Crystal axes"].click()
    assert any(s["type"] == "arrows" for s in v._layers.values())
    by["Lines"].click()
    assert "add_lines" in v.add_tile_msg.value


def test_figure_presets(v):
    assert v.set_figure_preset("paper") == (1200, 900, 2)
    assert (v.anim_width.value, v.anim_height.value) == (1200, 900)
    assert v.fig_preset.value == "paper"
    v.fig_preset.value = "slide"
    assert v.anim_width.value == 1600
    assert v.set_figure_preset(width=1000) == (1000, 900, 1)
    with pytest.raises(ValueError):
        v.set_figure_preset("billboard")
    w = MultiVolumeViewer(
        make(),
        figure_preset="journal",
        figure_presets={"journal": (850, 600, 3)},
    )
    assert (w.anim_width.value, w.anim_height.value, w.anim_scale.value) == (
        850,
        600,
        3,
    )
    w.close()


def test_script_tab(v, tmp_path):
    v.set_layer_params("amplitude", thr=0.4)
    v.script_from_view.click()
    assert "thr: 0.4" in v.script_text.value
    v.script_check.click()
    assert "No problems" in v.script_msg.value
    v.script_text.value = "- {edit: amplitude, cmpa: viridis}"
    v.script_check.click()
    assert "cmap" in v.script_msg.value and "✗" in v.script_msg.value
    v.script_text.value = "- {edit: amplitude, thr: 0.25}\n- {hide: amplitude}"
    v.script_run.click()
    assert v._style("amplitude")["thr"] == pytest.approx(0.25)
    assert "amplitude" not in v._trace_order
    v.script_path.value = str(tmp_path / "s.yaml")
    v.script_save.click()
    v.script_text.value = ""
    v.script_open.click()
    assert "thr: 0.25" in v.script_text.value


def test_png_button(v, tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(
        type(v), "write_image", lambda self, out, *a, **k: saved.append(out)
    )
    v.anim_name.value = str(tmp_path / "fig")
    v.png_btn.click()
    assert saved and saved[0].name == "fig.png"
