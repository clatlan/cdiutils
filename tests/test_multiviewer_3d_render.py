"""Built-in software renderer of MultiVolumeViewer (numpy + matplotlib)."""

import asyncio
import io

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")
pytest.importorskip("skimage")

import imageio.v2 as iio  # noqa: E402

from cdiutils.interactive import MultiVolumeViewer  # noqa: E402
from cdiutils.interactive.multiviewer.raster import render_scene  # noqa: E402
from cdiutils.interactive.multiviewer.rendering import (  # noqa: E402
    render_png,
    resolve_backend,
)

NO_AXES = {
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "zaxis": {"visible": False},
}


def quad(x0, x1, y0, y1, z, color, opacity=1.0):
    """Mesh3d square in the plane z (two triangles), constant colour, no shading."""
    return {
        "type": "mesh3d",
        "x": [x0, x1, x1, x0],
        "y": [y0, y0, y1, y1],
        "z": [z] * 4,
        "i": [0, 0],
        "j": [1, 2],
        "k": [2, 3],
        "color": color,
        "opacity": opacity,
        "lighting": {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0},
    }


def scene(*traces, eye=(0, 0, 2.0), up=(0, 1, 0)):
    return {
        "data": list(traces),
        "layout": {
            "scene": {
                **NO_AXES,
                "camera": {
                    "eye": dict(zip("xyz", eye)),
                    "up": dict(zip("xyz", up)),
                },
            }
        },
    }


def png_array(png):
    return np.asarray(iio.imread(io.BytesIO(png)))[..., :3].astype(int)


def test_default_backend_is_builtin():
    assert resolve_backend("auto")[0] == "software"
    assert resolve_backend(None)[0] == "software"
    v = MultiVolumeViewer(dict_data=None)
    assert v.rendering_backend == "software"


def test_opaque_depth_order_does_not_depend_on_trace_order():
    near, far = (
        quad(-1, 1, -1, 1, 0.5, "red"),
        quad(-1, 1, -1, 1, -0.5, "blue"),
    )
    for traces in ((near, far), (far, near)):
        rgba, *_ = render_scene(scene(*traces), 200, 200)
        c = rgba[100, 100]
        assert c[3] == 1 and c[0] > 0.9 and c[2] < 0.1


def test_transparent_layer_blends_over_opaque():
    rgba, *_ = render_scene(
        scene(
            quad(-1, 1, -1, 1, 0.5, "blue", 0.5),
            quad(-1, 1, -1, 1, -0.5, "red"),
        ),
        200,
        200,
    )
    r, g, b, a = rgba[100, 100]
    assert (
        a == pytest.approx(1)
        and r == pytest.approx(0.5, abs=0.02)
        and b == pytest.approx(0.5, abs=0.02)
    )


def test_no_cracks_between_triangles():
    n = 40
    X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    tr = {
        "type": "surface",
        "x": X,
        "y": Y,
        "z": 0 * X,
        "surfacecolor": X,
        "showscale": False,
    }
    rgba, *_ = render_scene(scene(tr), 300, 300)
    inner = rgba[60:240, 60:240, 3]
    assert inner.min() == 1.0  # every pixel inside the square is covered


def test_camera_orientation_matches_plotly_axes():
    # front view from -y, z up: +x must be on the right, +z at the top
    right = quad(0.6, 1.0, -0.2, 0.2, 0.0, "red")
    left = quad(-1.0, -0.6, -0.2, 0.2, 0.0, "blue")
    for t in (right, left):  # make them face the camera (plane y = const)
        t["y"], t["z"] = [0, 0, 0, 0], [-0.2, -0.2, 0.2, 0.2]
    rgba, *_ = render_scene(
        scene(right, left, eye=(0, -2.2, 0), up=(0, 0, 1)), 300, 200
    )
    red = np.argwhere((rgba[..., 0] > 0.8) & (rgba[..., 2] < 0.2))
    blue = np.argwhere((rgba[..., 2] > 0.8) & (rgba[..., 0] < 0.2))
    assert red[:, 1].mean() > blue[:, 1].mean()


def test_flat_and_smooth_options_change_the_image():
    tr = {
        "type": "mesh3d",
        "x": [-1, 1, 0],
        "y": [-1, -1, 1],
        "z": [0, 0, 0],
        "i": [0],
        "j": [1],
        "k": [2],
        "intensity": [0, 0.5, 1],
        "colorscale": [[0, "black"], [1, "white"]],
        "cmin": 0,
        "cmax": 1,
        "lighting": {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0},
    }
    base, *_ = render_scene(scene(tr), 200, 200)
    flat, *_ = render_scene(scene(tr), 200, 200, flat=True)
    smooth, *_ = render_scene(scene(tr), 200, 200, smooth=True)
    inside = base[..., 3] == 1
    assert base[..., 0][inside].std() > 0.1  # interpolated colours
    assert (
        np.ptp(flat[..., 0][flat[..., 3] == 1]) < 1e-6
    )  # one colour per triangle
    edge = (smooth[..., 3] > 0) & (smooth[..., 3] < 1)
    assert (
        edge.any() and not ((base[..., 3] > 0) & (base[..., 3] < 1)).any()
    )  # soft edges only when smooth


def test_viewer_still_has_axes_and_colorbar(tmp_path):
    n = 32
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    amp = np.clip(
        1.2 - np.sqrt((x - 16) ** 2 + (y - 16) ** 2 + (z - 16) ** 2) / 10, 0, 1
    ).astype(np.float32)
    v = MultiVolumeViewer(
        {"amplitude": amp, "phase": (x - 16).astype(np.float32) / 16},
        voxel_size=(5, 5, 5),
        unit="nm",
    )
    v.run_actions(
        [{"edit": "amplitude", "color_by": "phase", "cmap": "viridis"}]
    )
    p = v.write_image(tmp_path / "view.png", width=600, height=400)
    img = png_array(p.read_bytes())
    assert img.shape == (400, 600, 3)
    right = img[:, 480:]  # colorbar strip
    assert (np.abs(np.diff(right.astype(float), axis=0)).sum() > 0) and (
        right != 255
    ).any()
    dark = img.sum(-1) < 200  # axis lines and labels drawn in black
    assert dark[:, :480].sum() > 200


def test_animation_without_kaleido(tmp_path):
    n = 24
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    amp = np.clip(
        1.2
        - np.maximum(abs(x - 12), np.maximum(abs(y - 12), abs(z - 12))) / 8,
        0,
        1,
    ).astype(np.float32)
    v = MultiVolumeViewer(
        {"amplitude": amp},
        voxel_size=(5, 5, 5),
        export_width=200,
        export_height=150,
    )
    v.anim_width.value, v.anim_height.value, v.anim_scale.value = 200, 150, 1
    out = v.start_rotation_export(
        "orbit_z", frames=3, fmt="png", name=str(tmp_path / "frames")
    )
    out = asyncio.run(out) if asyncio.iscoroutine(out) else out
    files = sorted((tmp_path / "frames").glob("*.png"))
    assert len(files) == 3
    a, b = (png_array(f.read_bytes()) for f in files[:2])
    assert np.abs(a - b).sum() > 0  # the camera moved


def _kaleido_ok():
    try:
        from cdiutils.interactive.multiviewer.rendering import (
            _test_kaleido_backend,
        )

        return _test_kaleido_backend(timeout=20)[0]
    except Exception:
        return False


@pytest.mark.skipif(not _kaleido_ok(), reason="Kaleido/Chrome not available")
def test_silhouette_matches_kaleido():
    n = 48
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    c = n / 2
    amp = np.clip(
        1.2
        - np.maximum(
            abs(z - c) / 18, np.maximum(abs(y - c) / 11, abs(x - c) / 6)
        ),
        0,
        1,
    )
    v = MultiVolumeViewer(
        {"amplitude": amp.astype(np.float32)}, voxel_size=(5, 4, 3)
    )
    v.fig.update_layout(scene=NO_AXES)
    for eye in [(1.25, 1.25, 1.25), (0.2, -1.8, 0.9)]:
        v.fig.layout.scene.camera = dict(
            eye=dict(zip("xyz", eye)), up=dict(x=0, y=0, z=1)
        )
        js = v._export_base_json(400, 300)
        mk = (
            np.abs(
                png_array(render_png(js, 400, 300, 1, "kaleido", 14)) - 255
            ).sum(-1)
            > 30
        )
        ms = (
            np.abs(
                png_array(render_png(js, 400, 300, 1, "software", 14)) - 255
            ).sum(-1)
            > 30
        )
        assert (mk & ms).sum() / (mk | ms).sum() > 0.97
