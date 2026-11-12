"""
Analysis helpers for :class:`MultiVolumeViewer`: facet classification, line
profiles, surface-shell / core statistics and mesh file writers.

All functions are pure numpy/scipy and independent of the widgets.
"""

import itertools
import re
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# =============================================================================
# Facets
# =============================================================================
FACET_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]
OTHER_COLOR = "#c8c8c8"


def parse_hkl(item):
    """'111', '1 1 1', '1,-1,0', (1, 1, 1) or '-113' -> tuple of 3 ints."""
    if isinstance(item, (tuple, list, np.ndarray)):
        h = tuple(int(v) for v in item)
    else:
        s = str(item).strip().strip("{}()[]<>")
        parts = (
            re.findall(r"-?\d+", s)
            if re.search(r"[\s,;]", s)
            else re.findall(r"-?\d", s)
        )
        h = tuple(int(p) for p in parts)
    if len(h) != 3 or not any(h):
        raise ValueError(f"Invalid hkl: {item!r}")
    return h


def parse_families(spec):
    """Families as text ('100, 110, 111') or a list -> list of (label, hkl)."""
    if isinstance(spec, str):
        items = [s for s in re.split(r"[;,]|\s{2,}", spec) if s.strip()]
        if len(items) == 1 and " " in items[0].strip():
            toks = items[0].split()
            if all(len(re.findall(r"-?\d", t)) == 3 for t in toks):
                items = toks
    else:
        items = list(spec)
    out = []
    for it in items:
        h = tuple(sorted((abs(v) for v in parse_hkl(it)), reverse=True))
        lab = "{" + "".join(str(v) for v in h) + "}"
        if lab not in [o[0] for o in out]:
            out.append((lab, h))
    return out


def family_vectors(hkl):
    """All signed permutations of a family, as unit vectors (unique)."""
    vecs = set()
    for p in itertools.permutations(hkl):
        for s in itertools.product((1, -1), repeat=3):
            vecs.add(tuple(int(a * b) for a, b in zip(p, s)))
    arr = np.array(sorted(vecs), dtype=float)
    return arr / np.linalg.norm(arr, axis=1)[:, None], sorted(vecs)


def parse_matrix(text):
    """'1 0 0; 0 1 0; 0 0 1' (rows) -> 3x3 float array."""
    if text is None:
        return np.eye(3)
    if not isinstance(text, str):
        m = np.asarray(text, dtype=float)
    else:
        rows = [r for r in re.split(r"[;\n]", text) if r.strip()]
        m = np.array(
            [[float(v) for v in re.split(r"[\s,]+", r.strip())] for r in rows]
        )
    if m.shape != (3, 3):
        raise ValueError("Orientation matrix must be 3x3.")
    return m


def classify_normals(normals_lab, families, orientation=None, tol_deg=10.0):
    """Assign each normal to the closest facet family.

    Parameters
    ----------
    normals_lab : (N, 3) unit normals in the physical (x, y, z) frame.
    families : output of :func:`parse_families`.
    orientation : 3x3 matrix ``U`` with ``n_crystal = U @ n_lab`` (identity
        when the volume axes are the crystal axes).
    tol_deg : maximum angle to the family direction; beyond it -> -1 ("other").

    Returns ``(family_id, plane_label, angle_deg)``.
    """
    U = (
        np.eye(3)
        if orientation is None
        else np.asarray(orientation, dtype=float)
    )
    n = np.asarray(normals_lab, dtype=float) @ U.T
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    best_cos = np.full(len(n), -2.0)
    fam = np.full(len(n), -1, dtype=int)
    plane = np.empty(len(n), dtype=object)
    for f, (_lab, hkl) in enumerate(families):
        vecs, ints = family_vectors(hkl)
        dots = n @ vecs.T
        j = np.argmax(dots, axis=1)
        c = dots[np.arange(len(n)), j]
        better = c > best_cos
        best_cos[better] = c[better]
        fam[better] = f
        lbl = np.array(
            [
                "(" + "".join(str(v) for v in ints[k]) + ")"
                for k in range(len(ints))
            ],
            dtype=object,
        )
        plane[better] = lbl[j[better]]
    ang = np.degrees(np.arccos(np.clip(best_cos, -1, 1)))
    fam[ang > float(tol_deg)] = -1
    plane[fam < 0] = "other"
    return fam, plane, ang


def facet_colorscale(n_families):
    """Discrete colorscale for ids 0..n-1 plus 'other' (=n)."""
    k = n_families + 1
    cols = [FACET_COLORS[i % len(FACET_COLORS)] for i in range(n_families)] + [
        OTHER_COLOR
    ]
    cs = []
    for i, c in enumerate(cols):
        cs += [[i / k, c], [(i + 1) / k, c]]
    return cs, k


# =============================================================================
# Facet patches
# =============================================================================
PATCH_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#86bcb6", "#d37295", "#8cd17d",
]  # fmt: skip


def face_patches(faces, plane, fam, n_verts, areas=None, min_faces=20,
                 min_area_frac=0.002):  # fmt: skip
    """Connected patches of mesh faces with the same plane label.

    Two faces belong to the same patch when they share an edge and were
    assigned the same crystal plane (e.g. ``(1-11)``) by
    :func:`classify_normals`. Faces of no family and patches smaller than
    ``min_faces`` faces or ``min_area_frac`` of the area get ``-1``.
    Returns ``patch_id`` per face, numbered by decreasing area.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces, dtype=np.int64)
    nf = len(faces)
    if not nf:
        return np.zeros(0, dtype=int)
    areas = np.ones(nf) if areas is None else np.asarray(areas, float)
    codes = {p: i for i, p in enumerate(sorted(set(plane)))}
    lab = np.array([codes[p] for p in plane])
    lab[np.asarray(fam) < 0] = -1
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(nf), 3)
    key = e[:, 0] * np.int64(n_verts) + e[:, 1]
    o = np.argsort(key, kind="stable")
    k, f = key[o], fid[o]
    same = k[1:] == k[:-1]
    a, b = f[:-1][same], f[1:][same]
    keep = (lab[a] == lab[b]) & (lab[a] >= 0)
    g = coo_matrix(
        (np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(nf, nf)
    )
    _, comp = connected_components(g, directed=False)
    comp[lab < 0] = -1
    total = float(areas.sum()) or 1.0
    ids = [c for c in np.unique(comp) if c >= 0]
    size = {
        c: (int(np.sum(comp == c)), float(areas[comp == c].sum())) for c in ids
    }
    big = [
        c for c in ids
        if size[c][0] >= min_faces and size[c][1] / total >= min_area_frac
    ]  # fmt: skip
    big.sort(key=lambda c: -size[c][1])
    out = np.full(nf, -1, dtype=int)
    for i, c in enumerate(big):
        out[comp == c] = i
    return out


def miller_label(normal_crystal):
    """Approximate (hkl) of a normal, with cdiutils' facet analysis rule."""
    try:
        from cdiutils.facetanalysis.facet_utils import get_miller_indices

        h = get_miller_indices(np.asarray(normal_crystal, dtype=float))
    except Exception:
        n = np.asarray(normal_crystal, dtype=float)
        h = np.round(n / np.max(np.abs(n))).astype(int).tolist()
    return "(" + "".join(str(int(v)) for v in h) + ")"


def patch_table(patch, plane, fam, areas, normals_cry, centroids, family_labels,
                values=None):  # fmt: skip
    """One row per patch: plane, family, faces, area, area fraction, mean
    normal (crystal frame) and its deviation from the plane, (hkl) of the
    mean normal, centroid and area-weighted mean / std of ``values``."""
    rows, total = [], float(np.sum(areas)) or 1.0
    for pid in range(int(patch.max()) + 1 if len(patch) else 0):
        m = patch == pid
        if not m.any():
            continue
        a = areas[m]
        w = a / a.sum()
        n = (normals_cry[m] * w[:, None]).sum(axis=0)
        n /= np.linalg.norm(n) or 1.0
        f = int(np.bincount(fam[m][fam[m] >= 0]).argmax())
        c = (centroids[m] * w[:, None]).sum(axis=0)
        p = plane[m][0]
        ideal = np.array(
            [int(t) for t in re.findall(r"-?\d", str(p))], dtype=float
        )
        dev = (
            float(
                np.degrees(
                    np.arccos(
                        np.clip(abs(n @ ideal) / np.linalg.norm(ideal), -1, 1)
                    )
                )
            )
            if len(ideal) == 3 and np.linalg.norm(ideal) > 0
            else float("nan")
        )
        row = {
            "patch": pid,
            "plane": str(p),
            "family": family_labels[f] if 0 <= f < len(family_labels) else "?",
            "faces": int(m.sum()),
            "area": float(a.sum()),
            "area_frac": float(a.sum() / total),
            "nx": float(n[0]),
            "ny": float(n[1]),
            "nz": float(n[2]),
            "deviation_deg": dev,
            "hkl_of_mean": miller_label(n),
            "cx": float(c[0]),
            "cy": float(c[1]),
            "cz": float(c[2]),
        }
        if values is not None:
            v = values[m]
            ok = np.isfinite(v)
            if ok.any():
                ww = a[ok] / a[ok].sum()
                mu = float(np.sum(ww * v[ok]))
                row.update(
                    mean=mu,
                    std=float(np.sqrt(np.sum(ww * (v[ok] - mu) ** 2))),
                )
        rows.append(row)
    return rows


def patch_colorscale(n):
    """Discrete colorscale for patch ids 0..n-1, with -1 (none) grey."""
    k = n + 1
    cols = [OTHER_COLOR] + [
        PATCH_COLORS[i % len(PATCH_COLORS)] for i in range(n)
    ]
    cs = []
    for i, c in enumerate(cols):
        cs += [[i / k, c], [(i + 1) / k, c]]
    return cs, k


def facet_table(fam, areas, labels, values=None):
    """Area-weighted statistics per family (``values`` sampled per face)."""
    rows, total = [], float(np.sum(areas)) or 1.0
    for f, lab in list(enumerate(labels)) + [(-1, "other")]:
        m = fam == f
        if not m.any():
            continue
        a = areas[m]
        row = {
            "family": lab,
            "faces": int(m.sum()),
            "area_frac": float(a.sum() / total),
        }
        if values is not None:
            v = values[m]
            ok = np.isfinite(v)
            if ok.any():
                w = a[ok] / a[ok].sum()
                mu = float(np.sum(w * v[ok]))
                row.update(
                    mean=mu, std=float(np.sqrt(np.sum(w * (v[ok] - mu) ** 2)))
                )
        rows.append(row)
    return rows


# =============================================================================
# Line profiles
# =============================================================================
def line_points(p0, p1, n=200):
    p0, p1 = np.asarray(p0, float).reshape(3), np.asarray(p1, float).reshape(3)
    t = np.linspace(0.0, 1.0, int(n))
    pts = p0 + t[:, None] * (p1 - p0)
    return pts, t * float(np.linalg.norm(p1 - p0))


# =============================================================================
# Surface shell / core regions
# =============================================================================
def shell_core_masks(support, voxel_zyx, depth):
    """Split a boolean support into a surface shell (distance <= depth) and a core."""
    support = np.asarray(support, dtype=bool)
    dist = distance_transform_edt(
        support, sampling=tuple(float(v) for v in voxel_zyx)
    )
    shell = support & (dist <= float(depth))
    return shell, support & ~shell, dist


def masked_stats(field, mask):
    v = np.asarray(field, dtype=float)[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(p50),
        "p5": float(p5),
        "p95": float(p95),
    }


# =============================================================================
# Mesh writers (STL / PLY / OBJ / legacy VTK); no extra dependency
# =============================================================================
def write_mesh(path, verts, faces, scalars=None, scalar_name="scalar"):
    """Write a triangle mesh. Format from the suffix: .stl, .ply, .obj, .vtk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    s = (
        None
        if scalars is None
        else np.nan_to_num(np.asarray(scalars, dtype=np.float32))
    )
    ext = path.suffix.lower()
    if ext == ".stl":
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
        rec = np.zeros(
            len(f),
            dtype=[
                ("n", "<f4", 3),
                ("a", "<f4", 3),
                ("b", "<f4", 3),
                ("c", "<f4", 3),
                ("attr", "<u2"),
            ],
        )
        rec["n"], rec["a"], rec["b"], rec["c"] = n, a, b, c
        with open(path, "wb") as fh:
            fh.write(b"MultiVolumeViewer".ljust(80, b" "))
            fh.write(struct.pack("<I", len(f)))
            fh.write(rec.tobytes())
    elif ext == ".ply":
        per_vertex = s is not None and s.size == len(v)
        hdr = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(v)}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if per_vertex:
            hdr.append(f"property float {scalar_name}")
        hdr += [
            f"element face {len(f)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        with open(path, "wb") as fh:
            fh.write(("\n".join(hdr) + "\n").encode())
            vv = np.column_stack([v, s]) if per_vertex else v
            fh.write(vv.astype("<f4").tobytes())
            ff = np.zeros(len(f), dtype=[("k", "u1"), ("i", "<i4", 3)])
            ff["k"], ff["i"] = 3, f
            fh.write(ff.tobytes())
    elif ext == ".obj":
        with open(path, "w") as fh:
            fh.write("# MultiVolumeViewer\n")
            np.savetxt(fh, v, fmt="v %.6g %.6g %.6g")
            np.savetxt(fh, f + 1, fmt="f %d %d %d")
    elif ext == ".vtk":
        with open(path, "w") as fh:
            fh.write(
                "# vtk DataFile Version 3.0\nMultiVolumeViewer\nASCII\nDATASET POLYDATA\n"
            )
            fh.write(f"POINTS {len(v)} float\n")
            np.savetxt(fh, v, fmt="%.6g")
            fh.write(f"POLYGONS {len(f)} {4 * len(f)}\n")
            np.savetxt(fh, np.column_stack([np.full(len(f), 3), f]), fmt="%d")
            if s is not None and s.size in (len(v), len(f)):
                kind = "POINT_DATA" if s.size == len(v) else "CELL_DATA"
                fh.write(
                    f"{kind} {s.size}\nSCALARS {scalar_name} float 1\nLOOKUP_TABLE default\n"
                )
                np.savetxt(fh, s, fmt="%.6g")
    else:
        raise ValueError(
            f"Unsupported mesh format {ext!r}; use .stl, .ply, .obj or .vtk"
        )
    return path


# --------------------------------------------------------------------------
# Colour ranges and units
# --------------------------------------------------------------------------

#: Colour-range modes: key -> label shown in the panel.
RANGE_MODES = {
    "minmax": "min – max",
    "p1": "1 – 99 %",
    "p5": "5 – 95 %",
    "sym": "± max |v|",
    "sym99": "± 99 % |v|",
    "manual": "manual",
}
RANGE_ALIASES = {
    "auto": "minmax",
    "data": "minmax",
    "min-max": "minmax",
    "symmetric": "sym",
    "percentile": "p1",
    "1-99": "p1",
    "5-95": "p5",
}


def range_mode(value) -> str:
    """Canonical colour-range mode (raises on unknown names)."""
    key = str(value).strip().lower()
    key = RANGE_ALIASES.get(key, key)
    if key not in RANGE_MODES:
        raise ValueError(
            f"Unknown colour range {value!r}; use one of {list(RANGE_MODES)}"
        )
    return key


def color_limits(values, mode="minmax"):
    """``(cmin, cmax)`` of the finite ``values`` for a range ``mode``.

    ``minmax``: data range; ``p1`` / ``p5``: 1-99 % / 5-95 % percentiles
    (robust to outliers); ``sym`` / ``sym99``: symmetric around zero from
    the largest / 99th-percentile absolute value (strain, phase, differences).
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    if mode in ("p1", "p5"):
        q = 1 if mode == "p1" else 5
        lo, hi = (float(x) for x in np.percentile(v, [q, 100 - q]))
    elif mode in ("sym", "sym99"):
        a = np.abs(v)
        m = float(a.max() if mode == "sym" else np.percentile(a, 99))
        lo, hi = -m, m
    else:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def parse_unit(value):
    """``(label, scale)`` from ``"rad"``, ``("%", 100)``,
    ``{"unit": "%", "scale": 100}`` or ``None``."""
    if value is None:
        return "", 1.0
    if isinstance(value, str):
        return value, 1.0
    if isinstance(value, dict):
        return str(value.get("unit", "")), float(value.get("scale", 1.0))
    label, scale = value
    return str(label or ""), float(scale)


def guess_unit(name: str):
    """Default unit of a field from its name (only unambiguous cases)."""
    return ("rad", 1.0) if "phase" in str(name).lower() else ("", 1.0)
