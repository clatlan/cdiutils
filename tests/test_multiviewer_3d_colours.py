"""Colour ranges (presets, symmetric, all conditions) and units."""

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer import analysis as mva  # noqa: E402
from cdiutils.interactive.multiviewer import geometry as geo  # noqa: E402
from cdiutils.interactive.multiviewer.actions import ActionError  # noqa: E402


def make(n=28, k=1.0):
    z, y, x = np.mgrid[:n, :n, :n].astype(float) - n / 2
    amp = np.clip(1.2 - np.sqrt(x**2 + y**2 + z**2) / 9, 0, 1)
    ph = 0.05 * x
    ph[amp < 0.05] = np.nan
    strain = k * 1e-3 * (0.2 + x / n)  # asymmetric around zero
    return {
        "amplitude": amp.astype(np.float32),
        "phase": ph.astype(np.float32),
        "het_strain": strain.astype(np.float32),
    }


@pytest.fixture
def v():
    viewer = MultiVolumeViewer(
        [make(), make(k=3.0)],
        voxel_size=(5, 5, 5),
        unit="nm",
        condition_labels=["a", "b"],
    )
    viewer.add_slice("zs", "het_strain", axis="z")
    viewer.run_actions([{"only": ["amplitude", "zs"]}])
    yield viewer
    viewer.close()


def trace(v, name):
    return v.fig.data[v._trace_order.index(name)]


def test_color_limits_modes():
    vals = np.r_[np.linspace(-1, 2, 1000), 50.0, np.nan]
    assert mva.color_limits(vals, "minmax") == (-1.0, 50.0)
    lo, hi = mva.color_limits(vals, "p1")
    assert -1 < lo < -0.9 and 1.9 < hi < 2.0
    assert mva.color_limits(vals, "sym") == (-50.0, 50.0)
    lo, hi = mva.color_limits(vals, "sym99")
    assert lo == -hi and 1.9 < hi < 2.1
    assert mva.color_limits([np.nan], "sym") == (0.0, 1.0)
    assert mva.range_mode("symmetric") == "sym"
    with pytest.raises(ValueError):
        mva.range_mode("symetric")


def test_presets_from_scripts_and_panel(v):
    data = trace(v, "zs").surfacecolor
    v.run_actions([{"edit": "zs", "range": "sym"}])
    t = trace(v, "zs")
    assert t.cmin == pytest.approx(-t.cmax)
    assert t.cmax == pytest.approx(np.nanmax(np.abs(data)), rel=1e-6)
    w = v._layer_widgets["zs"]
    assert w["range_mode"].value == "sym" and w["range_slider"].disabled
    assert tuple(w["range_slider"].value) == pytest.approx((t.cmin, t.cmax))
    w["range_mode"].value = "p5"  # panel
    t = trace(v, "zs")
    lo, hi = np.nanpercentile(data, [5, 95])
    assert (t.cmin, t.cmax) == pytest.approx((lo, hi), rel=1e-5)
    # manual: explicit range or the former auto_range flag
    v.run_actions([{"edit": "zs", "range": [-1e-3, 2e-3]}])
    assert w["range_mode"].value == "manual" and not w["range_slider"].disabled
    assert (trace(v, "zs").cmin, trace(v, "zs").cmax) == pytest.approx(
        (-1e-3, 2e-3)
    )
    v.set_layer_params("zs", auto_range=True)
    assert w["range_mode"].value == "minmax"
    v.set_layer_params("zs", auto_range=False)
    assert w["range_mode"].value == "manual"


def test_bad_preset_is_reported(v):
    with pytest.raises(ActionError) as e:
        v.run_actions([{"edit": "zs", "range": "symetric"}])
    assert "did you mean 'symmetric'" in "\n".join(e.value.problems)


def test_same_range_for_all_conditions(v):
    v.run_actions([{"edit": "zs", "range": "sym"}])
    own = [trace(v, "zs").cmax]
    v.set_condition(1)
    own.append(trace(v, "zs").cmax)
    assert own[1] == pytest.approx(3 * own[0], rel=0.05)
    v.run_actions([{"edit": "zs", "range_all": True}])
    both = [trace(v, "zs").cmax]
    v.set_condition(0)
    both.append(trace(v, "zs").cmax)
    assert both[0] == pytest.approx(both[1]) == pytest.approx(max(own))
    assert trace(v, "zs").cmin == pytest.approx(-max(own))
    # an isosurface coloured by strain also shares its range
    v.run_actions(
        [
            {
                "edit": "amplitude",
                "color_by": "het_strain",
                "range": "minmax",
                "range_all": True,
            }
        ]
    )
    r0 = (trace(v, "amplitude").cmin, trace(v, "amplitude").cmax)
    v.set_condition(1)
    assert (
        trace(v, "amplitude").cmin,
        trace(v, "amplitude").cmax,
    ) == pytest.approx(r0)


def test_geometry_of_other_conditions_is_kept(v, monkeypatch):
    v.set_condition(1)
    v.set_condition(0)
    calls = []
    real = geo.isosurface
    monkeypatch.setattr(
        geo, "isosurface", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    v.set_condition(1)
    v.set_condition(0)
    assert not calls


def test_units(v):
    v.add_slice("ps", "phase", axis="z")
    v.add_difference("dphase", "phase", ref=0)
    v.run_actions([{"only": ["amplitude", "zs", "ps", "dphase"]}])
    v.set_layer_params("dphase", color_by="__self__")
    titles = {
        n: trace(v, n).colorbar.title.text
        for n in ("ps", "zs", "dphase")
        if trace(v, n).colorbar is not None
    }
    assert titles["ps"].endswith("(rad)") and "(" not in titles["zs"]
    raw = np.array(trace(v, "zs").surfacecolor, dtype=float)
    v.set_units(het_strain=("%", 100))
    t = trace(v, "zs")
    assert t.colorbar.title.text.endswith("(%)")
    assert np.allclose(
        np.array(t.surfacecolor, float), raw * 100, equal_nan=True
    )
    assert t.cmax == pytest.approx(np.nanmax(raw) * 100, rel=1e-5)
    # the panel: unit widgets follow the colour field and edit it
    w = v._layer_widgets["zs"]
    assert w["u_unit"].value == "%" and w["u_scale"].value == 100
    w["u_unit"].value = "per cent"
    assert v.field_units["het_strain"] == ("per cent", 100.0)
    v.set_layer_params("zs", color_by="__x__")
    assert w["u_unit"].value == "nm" and w["u_unit"].disabled
    v.set_units(het_strain=None)
    assert "het_strain" not in v.field_units


def test_units_and_ranges_round_trip(v, tmp_path):
    v.run_actions(
        [
            {"units": {"het_strain": ["%", 100]}},
            {"edit": "zs", "range": "sym99", "range_all": True},
        ]
    )
    steps = v.to_actions()
    assert {"units": {"het_strain": ["%", 100.0]}} in steps
    edit = next(s for s in steps if s.get("edit") == "zs")
    assert edit["range_mode"] == "sym99" and edit["range_all"] is True
    assert "auto_range" not in edit and "range" not in edit
    w = MultiVolumeViewer(
        [make(), make(k=3.0)], voxel_size=(5, 5, 5), unit="nm"
    )
    assert w.run_actions(v.save_actions(tmp_path / "c.yaml")).ok
    assert w.field_units == v.field_units
    assert (trace(w, "zs").cmin, trace(w, "zs").cmax) == pytest.approx(
        (trace(v, "zs").cmin, trace(v, "zs").cmax)
    )
    u = MultiVolumeViewer(
        [make(), make(k=3.0)], voxel_size=(5, 5, 5), unit="nm"
    )
    u.load_state(v.save_state(tmp_path / "s.json"))
    assert u.field_units == v.field_units
    assert u._layer_widgets["zs"]["range_mode"].value == "sym99"
    assert u._layer_widgets["zs"]["range_all"].value
    w.close()
    u.close()


def test_constructor_units_and_help(capsys):
    v = MultiVolumeViewer(
        make(), voxel_size=(5, 5, 5), field_units={"het_strain": ("%", 100)}
    )
    assert v._unit_of("het_strain") == ("%", 100.0)
    assert v._unit_of("phase") == ("rad", 1.0)
    # single condition: no "same for all conditions" box
    assert v._layer_widgets["phase"]["range_all"].layout.display == "none"
    v.help("colours")
    assert "± max" in capsys.readouterr().out
    v.close()
