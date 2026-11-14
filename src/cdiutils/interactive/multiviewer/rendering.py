"""
Static PNG rendering backends for :class:`MultiVolumeViewer`.

``software`` Built-in renderer (default): numpy for the 3D part, matplotlib
             for axes, colorbars and text (:mod:`.raster`, :mod:`.overlay`).
             No browser, display or GPU needed.
``kaleido``  Plotly/Kaleido (identical to the interactive look; needs Chrome).
``vtk``      Converts the traces to VTK and renders offscreen (Linux: needs an
             X display, Xvfb is started on demand).

Kaleido and VTK are tested on first use and cached for the kernel lifetime;
when they are unavailable the software renderer is used with a warning.
"""

import base64
import copy
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

_XVFB_PROCESS = None
_XVFB_DISPLAY = None
_BACKEND_CACHE: dict = {}
_BACKEND_LOCK = threading.Lock()
AXIS_SCREEN_SIZE = None  # VTK axis text size; None -> 0.9 * fontsize


# =============================================================================
# X display (VTK on Linux)
# =============================================================================
def _x_display_is_usable(display_name) -> bool:
    if not display_name:
        return False
    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is None:
        return True  # cannot probe; trust the existing DISPLAY
    r = subprocess.run(
        [xdpyinfo, "-display", str(display_name)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return r.returncode == 0


def _find_free_x_display(start=90, end=200) -> str:
    for n in range(start, end):
        if (
            not Path(f"/tmp/.X11-unix/X{n}").exists()
            and not Path(f"/tmp/.X{n}-lock").exists()
        ):
            return f":{n}"
    raise RuntimeError("No free X display number was found.")


def ensure_x_display(
    display_name=None, screen="1920x1080x24", timeout=10.0
) -> str:
    """Reuse a working DISPLAY or start Xvfb (kept alive for the kernel lifetime)."""
    global _XVFB_PROCESS, _XVFB_DISPLAY
    current = os.environ.get("DISPLAY")
    if _x_display_is_usable(current):
        return str(current)
    if platform.system() != "Linux":
        raise RuntimeError(
            "VTK Xvfb rendering is supported only on Linux; use Kaleido "
            "or a VTK build with EGL/OSMesa on this system."
        )
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        raise RuntimeError(
            "No usable X display was found and Xvfb is not installed."
        )
    if (
        _XVFB_PROCESS is not None
        and _XVFB_PROCESS.poll() is None
        and _x_display_is_usable(_XVFB_DISPLAY)
    ):
        os.environ["DISPLAY"] = str(_XVFB_DISPLAY)
        return str(_XVFB_DISPLAY)
    chosen = str(display_name or _find_free_x_display())
    proc = subprocess.Popen(
        [xvfb, chosen, "-screen", "0", str(screen), "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    os.environ["DISPLAY"] = chosen
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if _x_display_is_usable(chosen):
            _XVFB_PROCESS, _XVFB_DISPLAY = proc, chosen
            return chosen
        time.sleep(0.1)
    try:
        proc.terminate()
    except Exception:
        pass
    raise RuntimeError(f"Xvfb could not start on display {chosen}.")


def stop_xvfb():
    """Terminate the Xvfb process started by this module, if any."""
    global _XVFB_PROCESS, _XVFB_DISPLAY
    if _XVFB_PROCESS is not None and _XVFB_PROCESS.poll() is None:
        _XVFB_PROCESS.terminate()
    _XVFB_PROCESS = _XVFB_DISPLAY = None


# =============================================================================
# Plotly JSON helpers
# =============================================================================
def decode_plotly_array(value, dtype=float):
    """Decode Plotly typed-array JSON (``bdata``) or ordinary array-likes."""
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, dict) and "bdata" in value:
        arr = np.frombuffer(
            base64.b64decode(value["bdata"]),
            dtype=np.dtype(value.get("dtype", "f8")),
        )
        shape = value.get("shape")
        if shape is not None:
            dims = (
                tuple(
                    int(p)
                    for p in str(shape).replace("x", ",").split(",")
                    if p.strip()
                )
                if isinstance(shape, str)
                else tuple(int(p) for p in shape)
            )
            if dims:
                arr = arr.reshape(dims)
        return arr.astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def parse_color(value, default=(0.5, 0.5, 0.5)):
    from matplotlib.colors import to_rgb

    if value is None:
        return default
    text = str(value).strip()
    for prefix in ("rgba(", "rgb("):
        if text.startswith(prefix):
            vals = [float(v) for v in text[len(prefix) : -1].split(",")]
            return tuple(v / 255.0 for v in vals[:3])
    try:
        return tuple(float(v) for v in to_rgb(text))
    except Exception:
        return default


_UNIT_POWER = {
    "pm": -12,
    "nm": -9,
    "um": -6,
    "µm": -6,
    "μm": -6,
    "mm": -3,
    "m": 0,
    "km": 3,
}
_POWER_UNIT = {-12: "pm", -9: "nm", -6: "um", -3: "mm", 0: "m", 3: "km"}


def _axis_title_text(axis):
    t = (axis or {}).get("title") or {}
    return t.get("text", "") if isinstance(t, dict) else str(t)


def _vtk_safe_text(text):
    return (
        str(text or "")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("×", "x")
        .replace("−", "-")
        .replace("<sup>", "^")
        .replace("</sup>", "")
    )


def _strip_html(text):
    out = str(text or "")
    for a, b in (
        ("<b>", ""),
        ("</b>", ""),
        ("<br>", "\n"),
        ("<br/>", "\n"),
        ("<br />", "\n"),
    ):
        out = out.replace(a, b)
    return out


def prepare_fig_json_for_vtk(fig_json):
    """Rescale coordinates to a readable engineering unit (e.g. 1200 nm -> 1.2 um).

    Only the copy sent to VTK is changed; the interactive figure is untouched.
    """
    out = copy.deepcopy(fig_json)
    layout = dict(out.get("layout") or {})
    scene = dict(layout.get("scene") or {})
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for tr in out.get("data", []):
        if float(tr.get("opacity", 1.0)) <= 1e-6 or str(
            tr.get("type", "")
        ).lower() not in {"mesh3d", "surface", "scatter3d"}:
            continue
        for i, key in enumerate("xyz"):
            a = decode_plotly_array(tr.get(key), float)
            a = a[np.isfinite(a)]
            if a.size:
                lo[i], hi[i] = min(lo[i], a.min()), max(hi[i], a.max())
    spans = (hi - lo)[np.isfinite(hi - lo)]
    unit = _axis_title_text(scene.get("xaxis")).rsplit("(", 1)
    unit = unit[1].split(")", 1)[0].strip() if len(unit) == 2 else None
    power = _UNIT_POWER.get(unit) if unit else None
    if not spans.size or spans.max() <= 0 or power is None:
        return out
    shift = int(np.floor(np.log10(spans.max()) / 3.0) * 3)
    shift = shift if shift >= 3 else 0
    new_unit = _POWER_UNIT.get(power + shift)
    if new_unit is None:
        shift, new_unit = 0, unit
    s = 10.0 ** (-shift)
    if s != 1.0:
        for tr in out.get("data", []):
            if str(tr.get("type", "")).lower() in {
                "mesh3d",
                "surface",
                "scatter3d",
            }:
                for key in "xyz":
                    if tr.get(key) is not None:
                        tr[key] = (
                            decode_plotly_array(tr[key], float) * s
                        ).tolist()
        anns = list(scene.get("annotations") or [])
        for a in anns:
            for key in "xyz":
                if a.get(key) is not None:
                    a[key] = float(a[key]) * s
        scene["annotations"] = anns
    for name in ("xaxis", "yaxis", "zaxis"):
        ax = dict(scene.get(name) or {})
        r = ax.get("range")
        if s != 1.0 and r is not None and len(r) == 2:
            ax["range"] = [float(r[0]) * s, float(r[1]) * s]
        letter = _axis_title_text(ax).split("(")[0].strip() or name[0].upper()
        ax["title"] = {"text": f"{letter} ({new_unit})"}
        scene[name] = ax
    layout["scene"] = scene
    out["layout"] = layout
    return out


# =============================================================================
# VTK conversion
# =============================================================================
def _vtk_lookup_table(colorscale, n=256):
    import vtk

    stops = []
    for item in colorscale or [[0.0, "blue"], [1.0, "red"]]:
        try:
            stops.append((float(item[0]), parse_color(item[1])))
        except Exception:
            continue
    stops = sorted(stops) or [(0.0, (0, 0, 1)), (1.0, (1, 0, 0))]
    pos = np.array([s[0] for s in stops])
    rgb = np.array([s[1] for s in stops])
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(int(n))
    lut.Build()
    t = np.linspace(0, 1, int(n))
    cols = np.stack([np.interp(t, pos, rgb[:, k]) for k in range(3)], axis=1)
    for i, c in enumerate(cols):
        lut.SetTableValue(i, float(c[0]), float(c[1]), float(c[2]), 1.0)
    return lut


def _polydata(points, tris=None, lines=None):
    import vtk
    from vtkmodules.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
    )

    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(
        numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True)
    )
    poly.SetPoints(pts)
    if tris is not None and len(tris):
        t = np.asarray(tris, dtype=np.int64)
        cells = vtk.vtkCellArray()
        cells.SetCells(
            len(t),
            numpy_to_vtkIdTypeArray(
                np.column_stack([np.full(len(t), 3), t]).ravel(), deep=True
            ),
        )
        poly.SetPolys(cells)
    if lines:
        cells = vtk.vtkCellArray()
        for run in lines:
            cells.InsertNextCell(len(run))
            for pid in run:
                cells.InsertCellPoint(int(pid))
        poly.SetLines(cells)
    return poly


def _set_scalars(poly, values, cell=False, name="scalars"):
    from vtkmodules.util.numpy_support import numpy_to_vtk

    arr = numpy_to_vtk(
        np.nan_to_num(np.asarray(values, dtype=np.float64)), deep=True
    )
    arr.SetName(name)
    (poly.GetCellData() if cell else poly.GetPointData()).SetScalars(arr)


def _trace_to_polydata(tr):
    """Return (polydata, kind) with kind in {'surface', 'line'}; None if empty."""
    t = str(tr.get("type", "")).lower()
    if t == "mesh3d":
        P = np.column_stack(
            [decode_plotly_array(tr.get(k), float) for k in "xyz"]
        )
        F = np.column_stack(
            [decode_plotly_array(tr.get(k), np.int64) for k in "ijk"]
        )
        if not len(P) or not len(F):
            return None, None
        poly = _polydata(P, tris=F)
        inten = tr.get("intensity")
        if inten is not None:
            v = decode_plotly_array(inten, float).reshape(-1)
            cell = str(tr.get("intensitymode", "vertex")) == "cell"
            if v.size == (len(F) if cell else len(P)):
                _set_scalars(poly, v, cell=cell)
        return poly, "surface"
    if t == "surface":
        from .geometry import grid_to_mesh

        Z = decode_plotly_array(tr.get("z"), float)
        X = decode_plotly_array(tr.get("x"), float)
        Y = decode_plotly_array(tr.get("y"), float)
        if Z.ndim != 2:
            return None, None
        if X.ndim == 1:
            X = np.broadcast_to(X[None, :], Z.shape)
        if Y.ndim == 1:
            Y = np.broadcast_to(Y[:, None], Z.shape)
        C = (
            decode_plotly_array(tr.get("surfacecolor"), float)
            if tr.get("surfacecolor") is not None
            else Z
        )
        pts, tris, sc = grid_to_mesh(X, Y, Z, C if C.shape == Z.shape else Z)
        if not len(tris):
            return None, None
        poly = _polydata(pts, tris=tris)
        _set_scalars(poly, sc)
        return poly, "surface"
    if t == "scatter3d" and "lines" in str(tr.get("mode", "lines")):
        P = np.column_stack(
            [decode_plotly_array(tr.get(k), float) for k in "xyz"]
        )
        ok = np.all(np.isfinite(P), axis=1)
        runs, cur = [], []
        for i, good in enumerate(ok):
            if good:
                cur.append(i)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        runs = [r for r in runs if len(r) > 1]
        if not runs:
            return None, None
        P = np.nan_to_num(P)
        return _polydata(P, lines=runs), "line"
    return None, None


def render_fig_json_to_png_vtk(fig_json, w, h, s, fontsize=18):
    """Render Mesh3d / Surface / Scatter3d(lines) traces through VTK."""
    ensure_x_display()
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError(
            "VTK rendering requires the 'vtk' Python package."
        ) from exc

    layout = dict(fig_json.get("layout") or {})
    scene = dict(layout.get("scene") or {})
    width, height = (
        max(1, int(w) * max(1, int(s))),
        max(1, int(h) * max(1, int(s))),
    )
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    if hasattr(renderer, "SetBackgroundAlpha"):
        renderer.SetBackgroundAlpha(0.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetSize(width, height)
    window.SetMultiSamples(8)
    window.AddRenderer(renderer)

    actors, bars = [], []
    for tr in fig_json.get("data", []):
        opacity = float(tr.get("opacity", 1.0))
        if opacity <= 1e-6 or tr.get("visible", True) is False:
            continue
        poly, kind = _trace_to_polydata(tr)
        if poly is None or poly.GetNumberOfPoints() == 0:
            continue
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        prop = actor.GetProperty()
        prop.SetOpacity(opacity)
        if kind == "line":
            line = tr.get("line") or {}
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            prop.SetColor(*parse_color(line.get("color", "black")))
            prop.SetLineWidth(float(line.get("width", 2)) * max(1, int(s)))
            if hasattr(prop, "SetRenderLinesAsTubes"):
                prop.SetRenderLinesAsTubes(True)
        else:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly)
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOff()
            mapper.SetInputConnection(normals.GetOutputPort())
            has_point = poly.GetPointData().GetScalars() is not None
            has_cell = poly.GetCellData().GetScalars() is not None
            if has_point or has_cell:
                lut = _vtk_lookup_table(tr.get("colorscale"))
                mapper.SetLookupTable(lut)
                cmin, cmax = (
                    float(tr.get("cmin", 0.0) or 0.0),
                    float(tr.get("cmax", 1.0) or 1.0),
                )
                if not np.isfinite(cmin):
                    cmin = 0.0
                if not np.isfinite(cmax) or cmax <= cmin:
                    cmax = cmin + 1.0
                mapper.SetScalarRange(cmin, cmax)
                mapper.ScalarVisibilityOn()
                if has_cell and not has_point:
                    mapper.SetScalarModeToUseCellData()
            else:
                mapper.ScalarVisibilityOff()
                prop.SetColor(*parse_color(tr.get("color", "lightgray")))
            prop.SetInterpolationToPhong()
            lt = tr.get("lighting") or {}
            prop.SetAmbient(float(lt.get("ambient", 0.35)))
            prop.SetDiffuse(float(lt.get("diffuse", 0.65)))
            prop.SetSpecular(float(lt.get("specular", 0.25)))
            prop.SetSpecularPower(
                max(1.0, 100.0 * (1.0 - float(lt.get("roughness", 0.5))))
            )
            if (
                bool(tr.get("showscale", False))
                and mapper.GetScalarVisibility()
            ):
                bar = vtk.vtkScalarBarActor()
                bar.SetLookupTable(mapper.GetLookupTable())
                title = (tr.get("colorbar") or {}).get("title") or {}
                title = (
                    title.get("text", "") if isinstance(title, dict) else title
                )
                bar.SetTitle(_vtk_safe_text(str(title or tr.get("name", ""))))
                bar.SetNumberOfLabels(5)
                size = max(10, int(round(0.028 * height)))
                for p in (
                    bar.GetTitleTextProperty(),
                    bar.GetLabelTextProperty(),
                ):
                    p.SetFontFamilyToArial()
                    p.SetColor(0.0, 0.0, 0.0)
                    p.SetFontSize(size)
                    p.BoldOn()
                if hasattr(bar, "SetUnconstrainedFontSize"):
                    bar.SetUnconstrainedFontSize(True)
                if hasattr(bar, "SetTitleRatio"):
                    bar.SetTitleRatio(0.16)
                bar.SetMaximumWidthInPixels(max(40, width // 10))
                bar.SetMaximumHeightInPixels(max(120, int(height * 0.7)))
                bar.SetWidth(0.08)
                bar.SetHeight(0.7)
                renderer.AddViewProp(bar)
                bars.append(bar)
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        actors.append(actor)

    if not actors:
        raise RuntimeError(
            "VTK renderer found no visible Mesh3d, Surface or line traces."
        )
    for i, bar in enumerate(bars):  # right-aligned, side by side
        bar.SetPosition(0.97 - 0.12 * (len(bars) - i), 0.12)

    bounds = _scene_bounds(scene) or renderer.ComputeVisiblePropBounds()
    c = np.array(
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
    )
    span = max(
        float(
            max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            )
        ),
        1e-12,
    )
    cam = dict(scene.get("camera") or {})
    eye = dict(cam.get("eye") or {"x": 1.5, "y": 1.5, "z": 1.5})
    up = dict(cam.get("up") or {"x": 0, "y": 0, "z": 1})
    ctr = dict(cam.get("center") or {})
    focal = c + span * np.array([float(ctr.get(k, 0.0)) for k in "xyz"])
    ev = np.array([float(eye.get(k, 1.5)) for k in "xyz"])
    if not np.all(np.isfinite(ev)) or np.linalg.norm(ev) < 1e-12:
        ev = np.array([1.5, 1.5, 1.5])
    uv = np.array([float(up.get(k, 0.0)) for k in "xyz"])
    if not np.all(np.isfinite(uv)) or np.linalg.norm(uv) < 1e-12:
        uv = np.array([0.0, 0.0, 1.0])
    camera = renderer.GetActiveCamera()
    if (
        str((cam.get("projection") or {}).get("type", "perspective")).lower()
        == "orthographic"
    ):
        camera.ParallelProjectionOn()
    else:
        camera.ParallelProjectionOff()
        camera.SetViewAngle(30.0)
    renderer.ResetCamera(bounds)
    dist = float(
        np.linalg.norm(
            np.asarray(camera.GetPosition())
            - np.asarray(camera.GetFocalPoint())
        )
    )
    if not np.isfinite(dist) or dist <= 0:
        dist = 3.0 * span
    camera.SetFocalPoint(*focal)
    camera.SetPosition(*(focal + ev / np.linalg.norm(ev) * dist))
    camera.SetViewUp(*uv)
    camera.Zoom(0.82)
    renderer.ResetCameraClippingRange(bounds)
    _add_annotations(renderer, scene, s)
    _add_cube_axes(renderer, camera, scene, bounds, fontsize, max(1, int(s)))
    renderer.ResetCameraClippingRange(bounds)
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.SetInputBufferTypeToRGBA()
    grab.ReadFrontBufferOff()
    grab.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(True)
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()
    out = bytes(memoryview(writer.GetResult()))
    window.Finalize()
    return out


def _scene_bounds(scene):
    out = []
    for name in ("xaxis", "yaxis", "zaxis"):
        r = dict(scene.get(name) or {}).get("range")
        if (
            r is None
            or len(r) != 2
            or not np.all(np.isfinite(np.asarray(r, float)))
        ):
            return None
        out += sorted(float(v) for v in r)
    return tuple(out)


def _add_annotations(renderer, scene, scale):
    import vtk

    for a in scene.get("annotations") or []:
        if a.get("visible", True) is False:
            continue
        try:
            pos = [float(a.get(k, 0.0)) for k in "xyz"]
        except (TypeError, ValueError):
            continue
        text = _strip_html(a.get("text", ""))
        if not text:
            continue
        font = dict(a.get("font") or {})
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(_vtk_safe_text(text))
        actor.SetPosition(*pos)
        p = actor.GetTextProperty()
        p.SetColor(*parse_color(font.get("color", "black")))
        p.SetFontFamilyToArial()
        p.SetFontSize(
            max(
                1, int(round(float(font.get("size", 18)) * max(1, int(scale))))
            )
        )
        if "<b>" in str(a.get("text", "")):
            p.BoldOn()
        p.SetJustificationToCentered()
        p.SetVerticalJustificationToCentered()
        renderer.AddActor(actor)


def _add_cube_axes(renderer, camera, scene, bounds, fontsize, scale=1):
    import vtk

    lay = {k: dict(scene.get(k) or {}) for k in ("xaxis", "yaxis", "zaxis")}
    vis = [
        bool(lay[k].get("visible", True)) for k in ("xaxis", "yaxis", "zaxis")
    ]
    if not any(vis):
        return
    axes = vtk.vtkCubeAxesActor()
    axes.SetBounds(bounds)
    axes.SetCamera(camera)
    if hasattr(axes, "SetXAxisVisibility"):
        axes.SetXAxisVisibility(vis[0])
        axes.SetYAxisVisibility(vis[1])
        axes.SetZAxisVisibility(vis[2])
    for i, (k, setter) in enumerate(
        (
            ("xaxis", axes.SetXTitle),
            ("yaxis", axes.SetYTitle),
            ("zaxis", axes.SetZTitle),
        )
    ):
        setter(_vtk_safe_text(_axis_title_text(lay[k]) or k[0].upper()))
    if hasattr(axes, "SetLabelOffset"):
        try:
            axes.SetLabelOffset(8.0)
        except TypeError:
            axes.SetLabelOffset([0.0, 8.0])
    if hasattr(axes, "SetTitleOffset"):
        try:
            axes.SetTitleOffset(18.0)
        except TypeError:  # VTK >= 9.2 takes an (x, y) offset
            axes.SetTitleOffset([0.0, 18.0])
    m = max(
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
    )
    fmt = f"%.{3 if m < 0.1 else 2 if m < 1 else 1 if m < 10 else 0}f"
    axes.SetXLabelFormat(fmt)
    axes.SetYLabelFormat(fmt)
    axes.SetZLabelFormat(fmt)
    axes.SetFlyModeToOuterEdges()
    axes.SetGridLineLocation(axes.VTK_GRID_LINES_FURTHEST)
    grid = any(
        bool(lay[k].get("showgrid", True)) and v for k, v in zip(lay, vis)
    )
    axes.SetDrawXGridlines(grid and vis[0])
    axes.SetDrawYGridlines(grid and vis[1])
    axes.SetDrawZGridlines(grid and vis[2])
    fs = max(8, int(fontsize))
    if hasattr(axes, "SetScreenSize"):
        axes.SetScreenSize(
            float(AXIS_SCREEN_SIZE if AXIS_SCREEN_SIZE else 0.5 * fs) * scale
        )
    if hasattr(axes, "SetUseTextActor3D"):
        axes.SetUseTextActor3D(False)
    for i in range(3):
        for p in (axes.GetTitleTextProperty(i), axes.GetLabelTextProperty(i)):
            p.SetFontFamilyToArial()
            p.SetColor(0.0, 0.0, 0.0)
            p.SetFontSize(fs * scale)
            p.BoldOn()
        axes.GetLabelTextProperty(i).SetOrientation(45.0)
    for p in (
        axes.GetXAxesLinesProperty(),
        axes.GetYAxesLinesProperty(),
        axes.GetZAxesLinesProperty(),
    ):
        p.SetColor(0, 0, 0)
        p.SetLineWidth(2.0)
    for p in (
        axes.GetXAxesGridlinesProperty(),
        axes.GetYAxesGridlinesProperty(),
        axes.GetZAxesGridlinesProperty(),
    ):
        p.SetColor(0, 0, 0)
        p.SetOpacity(0.35)
    renderer.AddActor(axes)


# =============================================================================
# Backend dispatch (module-level so it can run in worker processes)
# =============================================================================
def render_png(
    fig_json, w, h, s, backend="software", fontsize=18, options=None
) -> bytes:
    """Render a figure JSON to PNG bytes.

    ``options`` (software backend): ``smooth`` (render at 2x, smoother
    outlines) and ``flat`` (one colour per triangle).
    """
    backend = str(backend or "software").lower().strip()
    if backend in ("software", "auto"):
        from .overlay import render_software_png

        opts = dict(options or {})
        return render_software_png(
            fig_json,
            w,
            h,
            s,
            fontsize=fontsize,
            smooth=bool(opts.get("smooth", False)),
            flat=bool(opts.get("flat", False)),
        )
    if "_mv_prims" in fig_json:
        fig_json = {k: v for k, v in fig_json.items() if k != "_mv_prims"}
    if backend == "vtk":
        return render_fig_json_to_png_vtk(
            prepare_fig_json_for_vtk(fig_json), w, h, s, fontsize=fontsize
        )
    if backend not in ("kaleido", "plotly"):
        raise ValueError(f"Unknown rendering backend: {backend!r}")
    import plotly.graph_objects as go
    import plotly.io as pio

    return pio.to_image(
        go.Figure(fig_json),
        format="png",
        width=int(w),
        height=int(h),
        scale=int(s),
    )


def _test_vtk_backend():
    try:
        fig = {
            "data": [
                {
                    "type": "mesh3d",
                    "x": [0, 1, 0],
                    "y": [0, 0, 1],
                    "z": [0, 0, 0],
                    "i": [0],
                    "j": [1],
                    "k": [2],
                    "intensity": [0, 0.5, 1],
                }
            ],
            "layout": {"scene": {}},
        }
        png = render_fig_json_to_png_vtk(fig, 160, 120, 1, fontsize=14)
        return (
            (True, "VTK is available.")
            if png.startswith(b"\x89PNG")
            else (False, "VTK returned no PNG.")
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _test_kaleido_backend(timeout=25):
    code = (
        "import plotly.graph_objects as go, plotly.io as pio\n"
        "d = pio.to_image(go.Figure(go.Scatter(x=[0, 1], y=[0, 1])), format='png', width=160, height=120)\n"
        "assert d.startswith(b'\\x89PNG')\n"
    )
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout)),
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"Kaleido test timed out after {timeout} s (Chrome may be unavailable).",
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if p.returncode == 0:
        return True, "Kaleido is available."
    return False, (p.stderr or p.stdout or "unknown Kaleido error").strip()[
        -1200:
    ]


def resolve_backend(requested="auto"):
    """Resolve the export backend (cached).

    ``software`` (default, also ``auto``) needs only numpy and matplotlib.
    ``kaleido`` and ``vtk`` are tested on first use; if unavailable, the
    software renderer is used with a warning.
    """
    requested = str(requested or "software").lower().strip()
    if requested == "plotly":
        requested = "kaleido"
    if requested in (
        "software",
        "auto",
    ):  # numpy + matplotlib: always available
        return "software", {
            "software": "Built-in renderer (numpy + matplotlib)."
        }
    if requested not in ("kaleido", "vtk"):
        raise ValueError(
            "rendering_backend must be 'software' (default), 'kaleido' or 'vtk'; "
            f"got {requested!r}"
        )
    with _BACKEND_LOCK:
        if requested in _BACKEND_CACHE:
            return _BACKEND_CACHE[requested]
        ok, msg = (
            _test_kaleido_backend()
            if requested == "kaleido"
            else _test_vtk_backend()
        )
        diag = {requested: msg}
        if ok:
            out = (requested, diag)
        else:
            warnings.warn(
                f"Rendering backend {requested!r} is unavailable ({msg}); using the built-in "
                "software renderer instead.",
                RuntimeWarning,
                stacklevel=3,
            )
            diag["software"] = "Built-in renderer (numpy + matplotlib)."
            out = ("software", diag)
        _BACKEND_CACHE[requested] = out
        return out
