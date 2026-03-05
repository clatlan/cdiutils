# tests/test_multiviewer_3d.py
from __future__ import annotations

import numpy as np
import pytest


def _import_multiviewer():
    """
    multiviewer_3d.py raises RuntimeError at import-time if imageio_ffmpeg
    isn't installed. Skip cleanly in that case.
    """
    pytest.importorskip("ipywidgets")
    pytest.importorskip("plotly")
    pytest.importorskip("skimage")  # marching cubes dependency
    pytest.importorskip("imageio_ffmpeg")  # required by module import

    # Replace this import path with the real one in your package
    # e.g. from cdiutils.interactive.multiviewer_3d import MultiVolumeViewer
    from cdiutils.interactive.multiviewer_3d import (
        MultiVolumeViewer,  # noqa: E402
    )

    return MultiVolumeViewer


@pytest.fixture
def dict_data_small():
    shape = (16, 16, 16)
    amp = np.ones(shape, dtype=float)
    phase = np.linspace(-np.pi, np.pi, num=np.prod(shape), dtype=float).reshape(shape)
    mask = np.zeros(shape, dtype=float)
    mask[4:12, 4:12, 4:12] = 1.0

    # add some NaNs
    phase = phase.copy()
    phase[0, 0, 0] = np.nan
    phase[1, 1, 1] = np.inf

    return {"amp": amp, "phase": phase, "mask": mask}


def test_validate_dict_data_rejects_bad_inputs(dict_data_small):
    MultiVolumeViewer = _import_multiviewer()
    v = MultiVolumeViewer(dict_data=None)

    with pytest.raises(TypeError):
        v._validate_dict_data(None)

    with pytest.raises(ValueError):
        v._validate_dict_data({})

    with pytest.raises(TypeError):
        v._validate_dict_data({123: np.zeros((4, 4, 4))})

    with pytest.raises(TypeError):
        v._validate_dict_data({"a": [1, 2, 3]})

    with pytest.raises(ValueError):
        v._validate_dict_data({"a": np.zeros((4, 4))})

    with pytest.raises(TypeError):
        v._validate_dict_data({"a": (np.zeros((4, 4, 4)) + 1j)})

    # good case should pass
    v._validate_dict_data(dict_data_small)


def test_set_data_registers_raw_layers(dict_data_small):
    MultiVolumeViewer = _import_multiviewer()
    v = MultiVolumeViewer(dict_data=None, voxel_size=(2.0, 3.0, 4.0), unit="nm")

    v.set_data(dict_data_small)

    # core state
    assert v._shape0 == (16, 16, 16)
    assert set(v.dict_data.keys()) == {"amp", "phase", "mask"}

    # layer registry should contain raw layers
    for k in ("amp", "phase", "mask"):
        assert k in v._layers
        assert v._layers[k]["type"] == "raw"


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("none", "preserve"),
        ("mean", "finite"),
        ("zero", "finite"),
        ("min", "finite"),
        ("max", "finite"),
        ("unknown_policy", "preserve"),
    ],
)
def test_apply_nan_policy(dict_data_small, policy, expected):
    MultiVolumeViewer = _import_multiviewer()
    v = MultiVolumeViewer(dict_data=None)

    arr = dict_data_small["phase"]
    out = v._apply_nan_policy(arr, policy=policy)

    if expected == "preserve":
        assert np.isnan(out).any() or np.isinf(out).any()
    else:
        assert np.isfinite(out).all()


def test_make_mesh_trace_for_raw_layer_smoke(dict_data_small):
    MultiVolumeViewer = _import_multiviewer()
    v = MultiVolumeViewer(dict_data=dict_data_small)

    # ensure raw layer widgets exist (created during set_data)
    assert "amp" in v._layer_widgets

    trace, used_cbar = v._make_mesh_trace_for_key("amp", cbar_index=0)

    # plotly object, should be a Mesh3d in most cases
    import plotly.graph_objects as go

    assert hasattr(trace, "to_plotly_json")
    assert isinstance(trace, go.Mesh3d)
    assert isinstance(used_cbar, bool)


def test_create_slice_layer_adds_new_layer(dict_data_small):
    MultiVolumeViewer = _import_multiviewer()
    v = MultiVolumeViewer(dict_data=dict_data_small, voxel_size=(1, 1, 1))

    # simulate the UI "Create layer" action: create a slice from default source
    v.add_layer_kind.value = "slice"
    v.add_layer_name.value = "t0"

    # choose axis/pos/thickness deterministically
    v.add_slice_axis.value = "z"
    v.add_slice_pos.value = 8
    v.add_slice_thickness.value = 0

    before = set(v._layers.keys())
    v._on_create_layer_clicked(None)
    after = set(v._layers.keys())

    new = list(after - before)
    assert len(new) == 1
    new_name = new[0]

    assert v._layers[new_name]["type"] == "slice"
    assert "data2d" in v._layers[new_name]
    assert v._layers[new_name]["data2d"].ndim == 2
