"""
Software renderer for MultiVolumeViewer exports: numpy only.

It draws the traces the viewer builds (Mesh3d, Surface, Scatter3d lines and
markers) with a depth buffer, per-vertex lighting and correct blending of
transparent layers, using the same camera as Plotly's 3D scene:

- each axis is scaled by ``1 / geometric_mean(ranges)`` (``aspectmode`` data /
  auto), centred on the middle of the data box;
- the camera looks from ``center + eye`` to ``center`` (Plotly scene units)
  with a vertical field of view of 45 degrees.

Options:

- ``smooth``: render at 2x and downsample (smoother outlines; it only blends
  pixels on edges, data values are not changed);
- ``flat``: one colour per triangle, from the mean of its vertex values,
  instead of interpolating colours inside triangles.

The output is an RGBA float image; the background is transparent so that
:mod:`.overlay` can draw the axes behind and the labels in front.
"""

import numpy as np

from .rendering import decode_plotly_array, parse_color

FOVY = np.pi / 4
SURFACE_LIGHT = dict(
    ambient=0.8, diffuse=0.8, specular=0.05, roughness=0.5, fresnel=0.2
)
MESH_LIGHT = dict(
    ambient=0.8, diffuse=0.8, specular=0.05, roughness=0.5, fresnel=0.2
)
# Plotly's default light positions, in camera coordinates (x right, y up, z towards the viewer)
MESH_LIGHTPOS = (1e5, 1e5, 0.0)
SURFACE_LIGHTPOS = (10.0, 1e4, 0.0)


# =============================================================================
# Colours
# =============================================================================
def colorscale_lut(colorscale, n=256):
    stops = []
    for item in colorscale or [[0, "#440154"], [1, "#fde725"]]:
        try:
            stops.append((float(item[0]), parse_color(item[1])))
        except Exception:
            continue
    stops.sort(key=lambda s: s[0])
    pos = np.array([s[0] for s in stops])
    rgb = np.array([s[1] for s in stops], dtype=float)
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, pos, rgb[:, k]) for k in range(3)], axis=1)


def map_colors(values, colorscale, cmin, cmax):
    lut = colorscale_lut(colorscale)
    v = np.asarray(values, dtype=float)
    lo, hi = float(cmin), float(cmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(v), np.nanmax(v) + 1e-12
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    t = np.nan_to_num(t, nan=0.0)
    return lut[np.round(t * (len(lut) - 1)).astype(int)]


# =============================================================================
# Scene extraction from Plotly JSON
# =============================================================================
class Primitive:
    """Triangles (``kind='tri'``), polylines (``'line'``) or points (``'point'``)."""

    def __init__(self, kind, points, opacity=1.0, **kw):
        self.kind = kind
        self.points = np.asarray(points, dtype=float)
        self.opacity = float(opacity)
        self.__dict__.update(kw)


def _grid_mesh(X, Y, Z, C):
    from .geometry import grid_to_mesh

    return grid_to_mesh(X, Y, Z, C)


def extract_primitives(fig_json):
    """Plotly figure JSON -> list of :class:`Primitive` (plot coordinates)."""
    return [p for tr in fig_json.get("data", []) for p in extract_trace(tr)]


def extract_trace(tr):
    """One Plotly trace -> list of :class:`Primitive` (decoded once, reusable)."""
    prims = []
    if tr.get("visible", True) is False:
        return prims
    t = str(tr.get("type", "")).lower()
    op = float(
        tr.get("opacity", 1.0) if tr.get("opacity") is not None else 1.0
    )
    if op <= 1e-6:
        return prims
    if t == "mesh3d":
        P = np.column_stack(
            [decode_plotly_array(tr.get(k), float) for k in "xyz"]
        )
        F = np.column_stack(
            [decode_plotly_array(tr.get(k), np.int64) for k in "ijk"]
        )
        if not len(P) or not len(F):
            return prims
        cell = str(tr.get("intensitymode", "vertex")) == "cell"
        inten = tr.get("intensity")
        vcol = fcol = None
        if inten is not None and len(decode_plotly_array(inten, float)):
            vals = decode_plotly_array(inten, float).reshape(-1)
            cols = map_colors(
                vals,
                tr.get("colorscale"),
                tr.get("cmin", np.nan),
                tr.get("cmax", np.nan),
            )
            if cell and len(vals) == len(F):
                fcol = cols
            elif len(vals) == len(P):
                vcol = cols
        if vcol is None and fcol is None:
            vcol = np.tile(
                parse_color(tr.get("color", "lightgray")), (len(P), 1)
            )
        light = dict(MESH_LIGHT, **(tr.get("lighting") or {}))
        lp = tr.get("lightposition") or {}
        lpos = tuple(float(lp.get(k, d)) for k, d in zip("xyz", MESH_LIGHTPOS))
        prims.append(
            Primitive(
                "tri",
                P,
                op,
                faces=F,
                vcol=vcol,
                fcol=fcol,
                light=light,
                lightpos=lpos,
                flatshading=bool(tr.get("flatshading", False)),
            )
        )
    elif t == "surface":
        Z = decode_plotly_array(tr.get("z"), float)
        if Z.ndim != 2:
            return prims
        X = decode_plotly_array(tr.get("x"), float)
        Y = decode_plotly_array(tr.get("y"), float)
        if X.ndim == 1:
            X = np.broadcast_to(X[None, :], Z.shape)
        if Y.ndim == 1:
            Y = np.broadcast_to(Y[:, None], Z.shape)
        sc = tr.get("surfacecolor")
        C = decode_plotly_array(sc, float) if sc is not None else Z
        pts, tris, vals = _grid_mesh(X, Y, Z, C if C.shape == Z.shape else Z)
        if not len(tris):
            return prims
        vcol = map_colors(
            vals,
            tr.get("colorscale"),
            tr.get("cmin", np.nan),
            tr.get("cmax", np.nan),
        )
        light = dict(SURFACE_LIGHT, **(tr.get("lighting") or {}))
        lp = tr.get("lightposition") or {}
        lpos = tuple(
            float(lp.get(k, d)) for k, d in zip("xyz", SURFACE_LIGHTPOS)
        )
        prims.append(
            Primitive(
                "tri",
                pts,
                op,
                faces=tris,
                vcol=vcol,
                fcol=None,
                light=light,
                lightpos=lpos,
                flatshading=False,
            )
        )
    elif t == "scatter3d":
        P = np.column_stack(
            [decode_plotly_array(tr.get(k), float) for k in "xyz"]
        )
        if not len(P):
            return prims
        mode = str(tr.get("mode", "lines+markers"))
        if "lines" in mode:
            line = tr.get("line") or {}
            prims.append(
                Primitive(
                    "line",
                    P,
                    op,
                    color=parse_color(line.get("color", "black")),
                    width=float(line.get("width", 2)),
                )
            )
        if "markers" in mode:
            mk = tr.get("marker") or {}
            col = mk.get("color", "black")
            prims.append(
                Primitive(
                    "point",
                    P[np.all(np.isfinite(P), axis=1)],
                    op,
                    color=parse_color(
                        col if isinstance(col, str) else "black"
                    ),
                    width=float(mk.get("size", 6)),
                )
            )
    return prims


# =============================================================================
# Camera (Plotly-compatible)
# =============================================================================
class Camera:
    """Plotly gl3d camera: maps plot coordinates to pixels and view depth."""

    def __init__(self, scene, bounds, width, height):
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        rng = np.where(hi - lo > 0, hi - lo, 1.0)
        mode = str(scene.get("aspectmode", "auto"))
        if mode == "cube":
            self.scale = 1.0 / rng
        elif mode == "manual" and scene.get("aspectratio"):
            ar = scene["aspectratio"]
            self.scale = np.array([ar.get(k, 1.0) for k in "xyz"], float) / rng
        else:
            g = float(np.exp(np.mean(np.log(rng))))
            asp = rng / g
            if mode == "auto" and asp.max() / asp.min() > 4:
                self.scale = 1.0 / rng
            else:
                self.scale = 1.0 / np.full(3, g)
        self.mid = 0.5 * (lo + hi)
        cam = scene.get("camera") or {}
        eye = cam.get("eye") or {"x": 1.25, "y": 1.25, "z": 1.25}
        up = cam.get("up") or {"x": 0, "y": 0, "z": 1}
        ctr = cam.get("center") or {"x": 0, "y": 0, "z": 0}
        self.eye = np.array([float(eye.get(k, 1.25)) for k in "xyz"])
        self.center = np.array([float(ctr.get(k, 0.0)) for k in "xyz"])
        upv = np.array([float(up.get(k, 0.0)) for k in "xyz"])
        self.ortho = (
            str((cam.get("projection") or {}).get("type", "perspective"))
            == "orthographic"
        )
        f = self.center - self.eye
        f /= np.linalg.norm(f) or 1.0
        if np.linalg.norm(np.cross(f, upv)) < 1e-9:
            upv = (
                np.array([0.0, 1.0, 0.0])
                if abs(f[1]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
        s = np.cross(f, upv)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        self.R = np.stack([s, u, -f])
        self.W, self.H = int(width), int(height)
        self.focal = 0.5 * self.H / np.tan(FOVY / 2)
        self.dist = float(np.linalg.norm(self.eye - self.center))

    def to_view(self, P):
        """Plot coordinates -> camera coordinates (x right, y up, depth > 0)."""
        n = (np.asarray(P, float) - self.mid) * self.scale
        c = (n - self.eye) @ self.R.T
        return c[..., 0], c[..., 1], -c[..., 2]

    def project(self, P):
        """Plot coordinates -> (px, py, depth); py grows downwards."""
        x, y, d = self.to_view(P)
        if self.ortho:
            k = self.focal / max(self.dist, 1e-9)
            return self.W / 2 + k * x, self.H / 2 - k * y, d
        dd = np.where(d > 1e-9, d, np.nan)
        return (
            self.W / 2 + self.focal * x / dd,
            self.H / 2 - self.focal * y / dd,
            d,
        )

    def cam_pos(self, P):
        """Camera-space positions (x right, y up, z towards the viewer)."""
        n = (np.asarray(P, float) - self.mid) * self.scale
        return (n - self.eye) @ self.R.T

    def cam_vec(self, V):
        """Directions (plot-normalised) -> camera space (unit)."""
        c = np.asarray(V, float) @ self.R.T
        return c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)

    def view_dir(self, P):
        """Unit vectors from points to the eye, in camera space."""
        c = -self.cam_pos(P)
        return c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)

    def light_dir(self, P, lightpos):
        """Unit vectors from points to a light given in camera space."""
        c = np.asarray(lightpos, float) - self.cam_pos(P)
        return c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)


def scene_bounds(prims):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for p in prims:
        P = (
            p.points[np.all(np.isfinite(p.points), axis=1)]
            if p.points.ndim == 2
            else p.points
        )
        if len(P):
            lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    if not np.all(np.isfinite(lo)):
        return np.zeros(3), np.ones(3)
    same = hi <= lo
    hi[same] = lo[same] + 1.0
    return lo, hi


def axis_bounds(scene, prims):
    """Axis ranges as Plotly sets them: explicit ranges where given, otherwise
    the data bounds padded by 1/32 of the range on each side."""
    lo, hi = scene_bounds(prims)
    pad = (hi - lo) / 32.0
    lo, hi = lo - pad, hi + pad
    for i, ax in enumerate(("xaxis", "yaxis", "zaxis")):
        r = (scene.get(ax) or {}).get("range")
        if (
            r is not None
            and len(r) == 2
            and np.all(np.isfinite(np.asarray(r, float)))
        ):
            lo[i], hi[i] = sorted(map(float, r))
    return lo, hi


# =============================================================================
# Shading
# =============================================================================
def vertex_normals(P, F, cam):
    """Area-weighted vertex normals in plot-normalised space."""
    Q = (P - cam.mid) * cam.scale
    a, b, c = Q[F[:, 0]], Q[F[:, 1]], Q[F[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = np.zeros_like(Q)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    fl = np.linalg.norm(fn, axis=1, keepdims=True)
    return vn / np.maximum(
        np.linalg.norm(vn, axis=1, keepdims=True), 1e-12
    ), fn / np.maximum(fl, 1e-30)


def _beckmann(ndh, rough):
    c2 = np.maximum(ndh, 1e-4) ** 2
    t2 = (c2 - 1.0) / c2
    r2 = rough * rough
    return np.exp(t2 / r2) / (np.pi * r2 * c2 * c2)


def shade(base, normals, view, light, light_dir):
    """Plotly's lighting model (gl-mesh3d / gl-surface3d shaders).

    ``normals``, ``view`` (point -> eye) and ``light_dir`` are unit vectors
    in camera space. Colour = min(ambient + diffuse * max(n.l, 0), 1) * rgb
    + specular * Cook-Torrance(roughness, fresnel). Normals are turned
    towards the viewer (both sides of a surface are lit).
    """
    n = np.asarray(normals, float)
    v = np.asarray(view, float)
    ndv = np.sum(n * v, axis=-1)
    n = np.where(ndv[..., None] < 0, -n, n)
    L = np.broadcast_to(light_dir, n.shape)
    amb = float(light.get("ambient", 0.8))
    dif = float(light.get("diffuse", 0.8))
    spe = float(light.get("specular", 0.05))
    rough = max(float(light.get("roughness", 0.5)), 0.02)
    fres = float(light.get("fresnel", 0.2))
    vdn = np.maximum(np.sum(v * n, axis=-1), 0.0)
    ldn = np.maximum(np.sum(L * n, axis=-1), 0.0)
    h = L + v
    h /= np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
    ndh = np.maximum(np.sum(n * h, axis=-1), 0.0)
    vdh = np.maximum(np.sum(v * h, axis=-1), 1e-6)
    ldh = np.maximum(np.sum(L * h, axis=-1), 1e-6)
    g = np.minimum(1.0, np.minimum(2 * ndh * vdn / vdh, 2 * ndh * ldn / ldh))
    f = np.power(np.maximum(1.0 - vdn, 0.0), fres)
    cook = g * f * _beckmann(ndh, rough) / np.maximum(np.pi * vdn, 1e-6)
    spec = np.clip(cook, 0.0, 1.0)
    k = np.minimum(amb + dif * ldn, 1.0)[..., None]
    return np.clip(base * k + spe * spec[..., None], 0, 1)


# =============================================================================
# Rasterisation
# =============================================================================
def _raster_triangles(px, py, pd, cols, F, W, H, face_cols=None):
    """Fragments of triangles: (pixel index, depth, rgb)."""
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    x0, x1, x2, y0, y1, y2 = px[a], px[b], px[c], py[a], py[b], py[c]
    ok = (
        np.isfinite(x0)
        & np.isfinite(x1)
        & np.isfinite(x2)
        & (pd[a] > 0)
        & (pd[b] > 0)
        & (pd[c] > 0)
    )
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok &= np.abs(area) > 1e-12
    xmin = (
        np.floor(np.nanmin([x0, x1, x2], axis=0)).astype(np.int64, copy=False)
        if len(F)
        else np.zeros(0, int)
    )
    ymin = (
        np.floor(np.nanmin([y0, y1, y2], axis=0)).astype(np.int64, copy=False)
        if len(F)
        else np.zeros(0, int)
    )
    xmax = np.ceil(np.nanmax([x0, x1, x2], axis=0)) if len(F) else np.zeros(0)
    ymax = np.ceil(np.nanmax([y0, y1, y2], axis=0)) if len(F) else np.zeros(0)
    ok &= (xmax >= 0) & (ymax >= 0) & (xmin < W) & (ymin < H)
    span = (np.maximum(xmax - xmin, ymax - ymin) + 1).astype(np.int64)
    ok &= span <= max(W, H) + 2
    out_p, out_d, out_c = [], [], []
    eps = -1e-7
    idx_all = np.where(ok)[0]
    if not len(idx_all):
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3))
    sizes = np.unique(np.minimum(span[idx_all], 4096))
    for k in sizes:
        idx = idx_all[np.minimum(span[idx_all], 4096) == k]
        if k > 64:  # large triangles: one at a time over their box
            chunks = [idx[i : i + 1] for i in range(len(idx))]
        else:
            per = max(1, 3_000_000 // (k * k))
            chunks = [idx[i : i + per] for i in range(0, len(idx), per)]
        gx, gy = np.meshgrid(np.arange(k), np.arange(k))
        gx, gy = gx.ravel(), gy.ravel()
        for ch in chunks:
            qx = xmin[ch, None] + gx + 0.5
            qy = ymin[ch, None] + gy + 0.5
            A = area[ch, None]
            w0 = (
                (x1[ch, None] - qx) * (y2[ch, None] - qy)
                - (x2[ch, None] - qx) * (y1[ch, None] - qy)
            ) / A
            w1 = (
                (x2[ch, None] - qx) * (y0[ch, None] - qy)
                - (x0[ch, None] - qx) * (y2[ch, None] - qy)
            ) / A
            w2 = 1.0 - w0 - w1
            inside = (
                (w0 >= eps)
                & (w1 >= eps)
                & (w2 >= eps)
                & (qx >= 0)
                & (qx < W)
                & (qy >= 0)
                & (qy < H)
            )
            ti, gi = np.nonzero(inside)
            if not len(ti):
                continue
            t = ch[ti]
            W0, W1, W2 = w0[ti, gi], w1[ti, gi], w2[ti, gi]
            out_p.append(
                qy[ti, gi].astype(np.int64) * W + qx[ti, gi].astype(np.int64)
            )
            # perspective-correct depth is not needed for visibility at this scale
            out_d.append(W0 * pd[a[t]] + W1 * pd[b[t]] + W2 * pd[c[t]])
            if face_cols is not None:
                out_c.append(face_cols[t])
            else:
                out_c.append(
                    W0[:, None] * cols[a[t]]
                    + W1[:, None] * cols[b[t]]
                    + W2[:, None] * cols[c[t]]
                )
    if not out_p:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3))
    return np.concatenate(out_p), np.concatenate(out_d), np.concatenate(out_c)


def _disc_offsets(radius):
    r = max(0.5, float(radius))
    R = int(np.ceil(r))
    dx, dy = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))
    m = dx**2 + dy**2 <= r * r + 0.25
    return dx[m], dy[m]


def _raster_lines(px, py, pd, color, width, W, H):
    """Thick polylines: discs along each segment (screen-space width)."""
    ok = np.isfinite(px) & np.isfinite(py) & (pd > 0)
    segs = np.where(ok[:-1] & ok[1:])[0]
    if not len(segs):
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3))
    L = np.hypot(px[segs + 1] - px[segs], py[segs + 1] - py[segs])
    n = np.maximum(np.ceil(L * 2).astype(int), 1)
    rep = np.repeat(segs, n + 1)
    t = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
    sx = px[rep] + t * (px[rep + 1] - px[rep])
    sy = py[rep] + t * (py[rep + 1] - py[rep])
    sd = pd[rep] + t * (pd[rep + 1] - pd[rep])
    dx, dy = _disc_offsets(width / 2)
    qx = (np.floor(sx)[:, None] + dx).astype(np.int64).ravel()
    qy = (np.floor(sy)[:, None] + dy).astype(np.int64).ravel()
    qd = np.repeat(sd, len(dx)) * (
        1 - 1e-4
    )  # small bias: lines drawn on surfaces stay visible
    m = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
    return (
        qy[m] * W + qx[m],
        qd[m],
        np.tile(np.asarray(color, float), (int(m.sum()), 1)),
    )


def _composite(frags, W, H):
    """Opaque depth test, then back-to-front blending of transparent fragments."""
    rgb = np.zeros((H * W, 3))
    alpha = np.zeros(H * W)
    depth = np.full(H * W, np.inf)
    opq = [f for f in frags if f[3] >= 1.0 - 1e-6]
    trn = [f for f in frags if f[3] < 1.0 - 1e-6]
    if opq:
        p = np.concatenate([f[0] for f in opq])
        d = np.concatenate([f[1] for f in opq])
        c = np.concatenate([f[2] for f in opq])
        order = np.lexsort((-d, p))  # per pixel: far to near
        p, d, c = p[order], d[order], c[order]
        last = np.r_[
            np.nonzero(np.diff(p))[0], len(p) - 1
        ]  # nearest fragment of each pixel
        rgb[p[last]] = c[last]
        depth[p[last]] = d[last]
        alpha[p[last]] = 1.0
    if trn:
        p = np.concatenate([f[0] for f in trn])
        d = np.concatenate([f[1] for f in trn])
        c = np.concatenate([f[2] for f in trn])
        a = np.concatenate([np.full(len(f[0]), f[3]) for f in trn])
        vis = d < depth[p]
        p, d, c, a = p[vis], d[vis], c[vis], a[vis]
        if len(p):
            order = np.lexsort((-d, p))  # per pixel: far to near
            p, c, a = p[order], c[order], a[order]
            starts = np.r_[0, np.nonzero(np.diff(p))[0] + 1]
            lt = np.log(np.maximum(1.0 - a, 1e-12))
            cs = np.cumsum(lt)
            grp_total = np.add.reduceat(lt, starts)
            grp_id = np.repeat(
                np.arange(len(starts)), np.diff(np.r_[starts, len(p)])
            )
            before = (
                cs - np.r_[0.0, cs][starts][grp_id]
            )  # inclusive prefix within group
            trans_after = np.exp(
                grp_total[grp_id] - before
            )  # product of (1 - a) of nearer fragments
            contrib = c * (a * trans_after)[:, None]
            cpix = np.add.reduceat(contrib, starts, axis=0)
            apix = np.add.reduceat(a * trans_after, starts)
            pix = p[starts]
            keep = np.exp(grp_total)
            rgb[pix] = rgb[pix] * keep[:, None] + cpix
            alpha[pix] = alpha[pix] * keep + apix
    # un-premultiply for display over any background
    out = np.zeros((H * W, 4))
    nz = alpha > 1e-9
    out[nz, :3] = rgb[nz] / alpha[nz, None]
    out[:, 3] = np.clip(alpha, 0, 1)
    return out.reshape(H, W, 4), depth.reshape(H, W)


def render_scene(fig_json, width, height, smooth=False, flat=False):
    """Render the 3D content of a Plotly figure JSON.

    Returns ``(rgba, camera, prims, bounds)``; ``rgba`` is (H, W, 4) floats in
    [0, 1] with a transparent background.
    """
    layout = dict(fig_json.get("layout") or {})
    scene = dict(layout.get("scene") or {})
    pre = fig_json.get("_mv_prims")  # decoded once by the export workers
    prims = (
        [q for ps in pre for q in ps]
        if pre is not None
        else extract_primitives(fig_json)
    )
    bounds = axis_bounds(scene, prims)
    ss = 2 if smooth else 1
    W, H = int(width) * ss, int(height) * ss
    cam = Camera(scene, bounds, W, H)
    frags = []
    for p in prims:
        if p.kind == "tri":
            P = p.points
            px, py, pd = cam.project(P)
            vn, fn = vertex_normals(P, p.faces, cam)
            if p.fcol is not None or flat or p.flatshading:
                base = (
                    p.fcol
                    if p.fcol is not None
                    else p.vcol[p.faces].mean(axis=1)
                )
                C = P[p.faces].mean(axis=1)
                nrm = (
                    fn
                    if (flat or p.flatshading or p.fcol is not None)
                    else vn[p.faces].mean(axis=1)
                )
                fc = shade(
                    base,
                    cam.cam_vec(nrm),
                    cam.view_dir(C),
                    p.light,
                    cam.light_dir(C, p.lightpos),
                )
                f = _raster_triangles(
                    px, py, pd, None, p.faces, W, H, face_cols=fc
                )
            else:
                vc = shade(
                    p.vcol,
                    cam.cam_vec(vn),
                    cam.view_dir(P),
                    p.light,
                    cam.light_dir(P, p.lightpos),
                )
                f = _raster_triangles(px, py, pd, vc, p.faces, W, H)
        elif p.kind == "line":
            px, py, pd = cam.project(p.points)
            f = _raster_lines(px, py, pd, p.color, p.width * ss, W, H)
        else:
            px, py, pd = cam.project(p.points)
            pts = np.column_stack([px, py, pd])
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            dx, dy = _disc_offsets(p.width * ss / 2)
            qx = (np.floor(pts[:, 0])[:, None] + dx).astype(np.int64).ravel()
            qy = (np.floor(pts[:, 1])[:, None] + dy).astype(np.int64).ravel()
            qd = np.repeat(pts[:, 2], len(dx)) * (1 - 2e-4)
            m = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
            f = (
                qy[m] * W + qx[m],
                qd[m],
                np.tile(np.asarray(p.color, float), (int(m.sum()), 1)),
            )
        if len(f[0]):
            frags.append((f[0], f[1], f[2], p.opacity))
    rgba, _ = _composite(frags, W, H)
    if ss > 1:
        rgba = _downsample(rgba, ss)
        cam = Camera(scene, bounds, int(width), int(height))
    return rgba, cam, prims, bounds


def _downsample(rgba, k):
    H, W = rgba.shape[0] // k, rgba.shape[1] // k
    r = rgba[: H * k, : W * k].reshape(H, k, W, k, 4)
    a = r[..., 3]
    asum = a.sum(axis=(1, 3))
    rgb = (r[..., :3] * a[..., None]).sum(axis=(1, 3)) / np.maximum(
        asum, 1e-12
    )[..., None]
    return np.concatenate([rgb, (asum / (k * k))[..., None]], axis=-1)
