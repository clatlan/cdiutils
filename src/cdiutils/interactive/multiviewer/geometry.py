"""
Pure-numpy geometry helpers for :class:`MultiVolumeViewer`.

Conventions
-----------
- Volumes are indexed ``(z, y, x)``; ``voxel_size`` is given as ``(vz, vy, vx)``.
- *Physical* points are ``(x, y, z)`` in the unit of ``voxel_size``, with the
  origin at the centre of voxel ``(0, 0, 0)``.
- *Plot* points are what Plotly receives. ``PLOT_ORDER="xyz"`` is the identity;
  ``PLOT_ORDER="zyx"`` swaps x and z (plot x = physical z).

Nothing in this module depends on widgets or Plotly, so it can be unit tested.
"""

import warnings

import numpy as np
from skimage.measure import marching_cubes

DIM = {"x": 2, "y": 1, "z": 0}  # physical axis -> array dimension
NAN_POLICIES = ("none", "mean", "zero", "min", "max")


class PlotFrame:
    """Coordinate conversions between voxel indices, physical and plot space."""

    def __init__(self, plot_order: str, voxel_size, shape):
        if plot_order not in ("xyz", "zyx"):
            raise ValueError("PLOT_ORDER must be 'xyz' or 'zyx'")
        self.plot_order = plot_order
        self.voxel_zyx = np.asarray(voxel_size, dtype=float).reshape(3)
        self.voxel_xyz = self.voxel_zyx[::-1].copy()
        self.shape = tuple(int(s) for s in shape)
        # plot[i] = phys[perm[i]]
        self.perm = (0, 1, 2) if plot_order == "xyz" else (2, 1, 0)
        self.inv = tuple(int(i) for i in np.argsort(self.perm))

    # --- points / vectors -------------------------------------------------
    def phys_to_plot(self, p):
        return np.asarray(p, dtype=float)[..., list(self.perm)]

    def plot_to_phys(self, p):
        return np.asarray(p, dtype=float)[..., list(self.inv)]

    def idx_to_phys(self, v_zyx):
        return np.asarray(v_zyx, dtype=float)[..., ::-1] * self.voxel_xyz

    def phys_to_idx(self, p_xyz):
        return (np.asarray(p_xyz, dtype=float) / self.voxel_xyz)[..., ::-1]

    # --- axis names ---------------------------------------------------------
    def plot_axis_to_phys(self, ax: str) -> str:
        return "xyz"[self.perm["xyz".index(ax)]]

    def phys_axis_to_plot(self, ax: str) -> str:
        return "xyz"[self.inv["xyz".index(ax)]]

    # --- extents ----------------------------------------------------------------
    def n_along(self, ax_phys: str) -> int:
        return self.shape[DIM[ax_phys]]

    def coords(self, ax_phys: str) -> np.ndarray:
        return (
            np.arange(self.n_along(ax_phys))
            * self.voxel_xyz["xyz".index(ax_phys)]
        )

    def bounds_phys(self):
        hi = (np.asarray(self.shape[::-1], dtype=float) - 1.0) * self.voxel_xyz
        return np.zeros(3), hi

    def center_phys(self):
        lo, hi = self.bounds_phys()
        return 0.5 * (lo + hi)

    def diagonal(self) -> float:
        lo, hi = self.bounds_phys()
        return float(np.linalg.norm(hi - lo))


# =============================================================================
# Scalar helpers
# =============================================================================
def apply_nan_policy(arr, policy: str = "none"):
    """Replace non-finite values. ``'none'`` keeps NaNs (rendered as holes)."""
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if policy == "none" or not bad.any():
        return arr
    good = arr[~bad]
    if policy == "zero" or good.size == 0:
        v = 0.0
    elif policy == "mean":
        v = float(np.mean(good))
    elif policy == "min":
        v = float(np.min(good))
    elif policy == "max":
        v = float(np.max(good))
    else:
        return arr
    out = arr.copy()
    out[bad] = v
    return out


def finite_range(arr, fallback=(0.0, 1.0)):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return fallback
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def fill_outside(vol, below: float):
    """Copy of ``vol`` with non-finite values set to ``below`` (treated as outside)."""
    out = np.array(vol, dtype=float, copy=True)
    out[~np.isfinite(out)] = below
    return out


def low_value(vol, level: float) -> float:
    """A value safely below both ``level`` and the data (used for 'outside')."""
    lo, hi = finite_range(vol)
    span = max(hi - lo, abs(level), 1.0)
    return min(lo, level) - 10.0 * span


# =============================================================================
# Isosurfaces
# =============================================================================
def isosurface(vol, level: float, voxel_zyx, step: int = 1):
    """Marching cubes in physical units.

    Returns ``(verts_idx_zyx, faces, normals_xyz)`` or ``None`` when ``level``
    is outside the data range. Normals are unit vectors in physical (x, y, z),
    pointing from high to low values (outwards for an object above ``level``).
    """
    vol = np.ascontiguousarray(
        vol, dtype=np.float32
    )  # what marching_cubes uses internally
    finite = vol[np.isfinite(vol)]
    lv = np.float32(level)
    if finite.size == 0 or not (finite.min() < lv < finite.max()):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verts, faces, normals, _ = marching_cubes(
            vol,
            level=float(level),
            spacing=tuple(float(v) for v in voxel_zyx),
            step_size=int(max(1, step)),
            allow_degenerate=False,
        )
    if faces.size == 0:
        return None
    verts_idx = verts / np.asarray(voxel_zyx, dtype=float)
    return verts_idx, faces.astype(np.int64), normals[:, ::-1].astype(float)


def face_normals(verts_xyz, faces):
    """Unit normals and areas of triangles (physical coordinates)."""
    a, b, c = (verts_xyz[faces[:, i]] for i in range(3))
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(area2, 1e-30)[:, None]
    return n, 0.5 * area2


def orient_faces_like(face_n, vertex_normals, faces):
    """Flip face normals so they agree with the (outward) vertex normals."""
    vn = vertex_normals[faces].mean(axis=1)
    sign = np.sign(np.sum(face_n * vn, axis=1))
    sign[sign == 0] = 1.0
    return face_n * sign[:, None]


# =============================================================================
# Planes, half-spaces and offsets
# =============================================================================
def unit(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return (
        v / n
        if n > 0 and np.isfinite(n)
        else np.asarray(fallback, dtype=float)
    )


def plane_basis(normal):
    n = unit(normal)
    a = (
        np.array([1.0, 0.0, 0.0])
        if abs(n[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v / np.linalg.norm(v)


def effective_origin(origin0, normal, offset):
    return np.asarray(origin0, dtype=float).reshape(3) + unit(normal) * float(
        offset
    )


def offset_bounds(origin0, normal, lo, hi, eps=1e-12):
    """Interval of offsets keeping ``origin0 + n_hat * offset`` inside ``[lo, hi]``."""
    o0 = np.asarray(origin0, dtype=float).reshape(3)
    n = unit(normal)
    off_min, off_max = -np.inf, np.inf
    for i in range(3):
        if abs(n[i]) < eps:
            if o0[i] < lo[i] - 1e-9 or o0[i] > hi[i] + 1e-9:
                return 0.0, 0.0
            continue
        a, b = sorted(((lo[i] - o0[i]) / n[i], (hi[i] - o0[i]) / n[i]))
        off_min, off_max = max(off_min, a), min(off_max, b)
    if (
        not (np.isfinite(off_min) and np.isfinite(off_max))
        or off_max < off_min
    ):
        return 0.0, 0.0
    return float(off_min), float(off_max)


def clamp_offset(origin0, normal, offset, lo, hi):
    a, b = offset_bounds(origin0, normal, lo, hi)
    return float(np.clip(float(offset), a, b)), a, b


def signed_distance(shape, voxel_zyx, normal, origin):
    """Signed distance (physical units) of every voxel to a plane, shape ``(nz, ny, nx)``."""
    n = unit(normal)
    o = np.asarray(origin, dtype=float).reshape(3)
    nz, ny, nx = shape
    vz, vy, vx = voxel_zyx
    dx = (np.arange(nx) * vx - o[0]) * n[0]
    dy = (np.arange(ny) * vy - o[1]) * n[1]
    dz = (np.arange(nz) * vz - o[2]) * n[2]
    return dz[:, None, None] + dy[None, :, None] + dx[None, None, :]


def clip_implicit(vol, level, normal, origin, side, voxel_zyx, smooth=True):
    """Implicit field whose 0-isosurface is ``{vol >= level}`` cut by a half-space.

    ``smooth=True`` intersects the two implicit functions with ``min``, which
    yields an exactly planar cap. ``smooth=False`` reproduces the former
    voxel-masked (staircase) cut.
    """
    v = np.asarray(vol, dtype=float) - float(level)
    finite = np.isfinite(v)
    if not finite.any():
        return None
    d = signed_distance(v.shape, voxel_zyx, normal, origin)
    if str(side).lower() == "down":
        d = -d
    inside = v[finite & (v > 0)]
    scale = float(np.median(inside)) if inside.size else 1.0
    floor = -10.0 * max(float(np.nanmax(np.abs(v[finite]))), 1.0)
    v = np.where(finite, v, floor)
    if not smooth:
        v[d < 0] = floor
        return v
    alpha = 0.25 * scale / float(np.min(voxel_zyx))
    return np.minimum(v, alpha * d)


# =============================================================================
# Slices and planes
# =============================================================================
def axis_slice(vol, frame: PlotFrame, ax_phys: str, pos: int, thick: int):
    """Mean over ``pos ± thick`` along a physical axis.

    Returns ``data2d, X, Y, Z`` (physical grids of the same shape).
    """
    vol = np.asarray(vol, dtype=float)
    dim = DIM[ax_phys]
    n = vol.shape[dim]
    pos = int(np.clip(pos, 0, n - 1))
    lo, hi = max(0, pos - int(thick)), min(n - 1, pos + int(thick))
    sl = [slice(None)] * 3
    sl[dim] = slice(lo, hi + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.nanmean(vol[tuple(sl)], axis=dim)
    x, y, z = frame.coords("x"), frame.coords("y"), frame.coords("z")
    c = pos * frame.voxel_xyz["xyz".index(ax_phys)]
    if ax_phys == "z":  # (ny, nx)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, c)
    elif ax_phys == "y":  # (nz, nx)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, c)
    else:  # (nz, ny)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, c)
    return data, X, Y, Z


def sample_plane(
    interp_idx,
    frame: PlotFrame,
    normal,
    origin,
    thickness,
    extent,
    resolution: int = 200,
    n_thick: int = 5,
):
    """Sample a field on an arbitrary plane.

    ``interp_idx`` maps ``(..., 3)`` zyx index points to values (NaN outside).
    Returns ``data2d, X, Y, Z`` in physical coordinates.
    """
    n, u, v = plane_basis(normal)
    r0 = np.asarray(origin, dtype=float).reshape(3)
    s = np.linspace(-float(extent), float(extent), int(resolution))
    UU, VV = np.meshgrid(s, s)
    P = r0 + UU[..., None] * u + VV[..., None] * v
    tt = (
        [0.0]
        if thickness <= 0
        else np.linspace(-thickness, thickness, max(1, int(n_thick)))
    )
    vals = [interp_idx(frame.phys_to_idx(P + n * t)) for t in tt]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.nanmean(np.stack(vals, axis=0), axis=0)
    return data, P[..., 0], P[..., 1], P[..., 2]


def grid_to_mesh(X, Y, Z, C=None):
    """Triangulate a structured surface; cells with any non-finite corner are dropped."""
    X, Y, Z = (np.asarray(a, dtype=float) for a in (X, Y, Z))
    nr, nc = X.shape
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    valid = np.all(np.isfinite(pts), axis=1)
    ids = np.arange(nr * nc).reshape(nr, nc)
    a, b = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()
    c, d = ids[1:, 1:].ravel(), ids[1:, :-1].ravel()
    ok = valid[a] & valid[b] & valid[c] & valid[d]
    tri = np.concatenate(
        [np.stack([a, b, c], 1)[ok], np.stack([a, c, d], 1)[ok]]
    )
    used = np.unique(tri)
    remap = -np.ones(nr * nc, dtype=np.int64)
    remap[used] = np.arange(used.size)
    sc = None if C is None else np.asarray(C, dtype=float).reshape(-1)[used]
    return pts[used], remap[tri], sc


# =============================================================================
# Rotations
# =============================================================================
def rotation_matrix_xyz(rx, ry, rz):
    """``Rz @ Ry @ Rx`` with angles in degrees."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = (
        np.cos(rx),
        np.sin(rx),
        np.cos(ry),
        np.sin(ry),
        np.cos(rz),
        np.sin(rz),
    )
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rodrigues(axis, angle_rad):
    x, y, z = unit(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


# =============================================================================
# Arrow meshes (rendered as Mesh3d so both export backends draw them)
# =============================================================================
def arrow_mesh(
    origins, vectors, radius, head_frac=0.25, head_scale=2.2, n_seg=16
):
    """Triangle mesh of arrows (cylinder shaft + cone head)."""
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    verts, faces = [], []
    for o, vec in zip(origins, vectors):
        L = float(np.linalg.norm(vec))
        if L <= 0:
            continue
        n, u, v = plane_basis(vec)
        r, rh = float(radius), float(radius) * head_scale
        lh = min(head_frac * L, 0.9 * L)
        ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
        base = sum(len(x) for x in verts)
        b0 = o + r * ring
        b1 = o + (L - lh) * n + r * ring
        h0 = o + (L - lh) * n + rh * ring
        tip = (o + L * n)[None]
        c0 = o[None]
        c1 = (o + (L - lh) * n)[None]
        block = np.concatenate([b0, b1, h0, tip, c0, c1])
        k = n_seg
        i_b0, i_b1, i_h0, i_tip, i_c0, i_c1 = (
            0,
            k,
            2 * k,
            3 * k,
            3 * k + 1,
            3 * k + 2,
        )
        f = []
        for j in range(k):
            jn = (j + 1) % k
            f += [
                (i_b0 + j, i_b0 + jn, i_b1 + jn),
                (i_b0 + j, i_b1 + jn, i_b1 + j),
            ]
            f += [(i_h0 + j, i_h0 + jn, i_tip)]
            f += [(i_c0, i_b0 + jn, i_b0 + j), (i_c1, i_h0 + j, i_h0 + jn)]
        verts.append(block)
        faces.append(np.asarray(f, dtype=np.int64) + base)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(verts), np.concatenate(faces)


class NanAwareInterpolator:
    """Trilinear interpolation that ignores undefined (NaN) corners.

    The value is the weighted mean of the finite corners of the cell
    (normalised convolution): ``interp(v * m) / interp(m)`` with ``m`` the
    finite mask. Points whose cell has no finite corner are NaN. Used to
    colour isosurfaces lying on the edge of a support, where plain
    interpolation returns NaN as soon as one corner is outside.
    """

    def __init__(self, axes, vol):
        from scipy.interpolate import RegularGridInterpolator

        vol = np.asarray(vol, dtype=np.float32)
        m = np.isfinite(vol)
        kw = dict(bounds_error=False, fill_value=0.0)
        self._v = RegularGridInterpolator(axes, np.where(m, vol, 0), **kw)
        self._w = RegularGridInterpolator(axes, m.astype(np.float32), **kw)

    def __call__(self, pts):
        w = self._w(pts)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w > 1e-6, self._v(pts) / w, np.nan)
