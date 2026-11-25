"""
Read cdiutils post-processing results for :class:`MultiVolumeViewer`.

The BCDI pipeline writes, per scan, ``S{scan}_postprocessed_data.cxi``,
``S{scan}_structural_properties.npz`` and (with vtk)
``S{scan}_structural_properties.vti``. :func:`read_result` reads any of
them; :func:`find_results` finds them in a folder.

cdiutils units are kept: voxel size in nm, ``het_strain*`` already in %,
displacement, d-spacing and lattice parameter in Å, phase in rad.
"""

import os
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Units of the cdiutils fields (label, display scale).
CDIUTILS_UNITS = {
    "phase": ("rad", 1.0),
    "het_strain": ("%", 1.0),
    "het_strain_with_ramp": ("%", 1.0),
    "het_strain_from_dspacing": ("%", 1.0),
    "numpy_het_strain": ("%", 1.0),
    "displacement": ("Å", 1.0),
    "dspacing": ("Å", 1.0),
    "lattice_parameter": ("Å", 1.0),
}
#: Fields loaded when ``fields`` is not given (those present).
DEFAULT_FIELDS = (
    "amplitude",
    "phase",
    "displacement",
    "het_strain",
    "het_strain_from_dspacing",
    "dspacing",
    "lattice_parameter",
)
#: Fields not masked outside the support.
UNMASKED = ("amplitude", "support", "surface")
PATTERNS = (
    "*_postprocessed_data.cxi",
    "*_structural_properties.npz",
    "*_structural_properties.vti",
)
DIFFRACTION_PATTERNS = ("*_preprocessed_data.cxi",)
INTENSITY_TRANSFORMS = ("log1p", "sqrt", "linear")


@dataclass
class Result:
    """One cdiutils result file."""

    path: Path
    fields: dict
    voxel_size: tuple | None = None
    isosurface: float | None = None
    scan: str | None = None
    convention: str | None = None
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    sample: str | None = None
    unit: str | None = None  # voxel-size unit when not nm (diffraction)
    file_shape: tuple | None = None  # before any common-grid crop/pad

    @property
    def shape(self):
        return next(iter(self.fields.values())).shape if self.fields else None


# --------------------------------------------------------------------------
# Finding files
# --------------------------------------------------------------------------


def _scan_of(path) -> str | None:
    m = re.search(r"S(\d+)_", Path(path).name)
    return m.group(1) if m else None


def _natural(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path))
    ]


def find_results(path, recursive=True, source="reconstruction") -> list:
    """Result files of a file, a scan folder or a folder of scans.

    One file per scan is returned, preferring ``.cxi`` > ``.npz`` > ``.vti``,
    sorted naturally (S2 before S10). ``source="diffraction"`` looks for
    ``S*_preprocessed_data.cxi`` instead.
    """
    patterns = DIFFRACTION_PATTERNS if source == "diffraction" else PATTERNS
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"No such file or folder: {p}")
    best = {}
    for rank, pat in enumerate(patterns):
        hits = p.rglob(pat) if recursive else p.glob(pat)
        for f in hits:
            if ".ipynb_checkpoints" in f.parts:
                continue
            key = (f.parent, _scan_of(f) or f.stem)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, f)
    return sorted((f for _, f in best.values()), key=_natural)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _scalar(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _scalar(v[()])
    if isinstance(v, np.generic):
        return v.item()
    return v


def _h5_params(group, depth=0):
    out = {}
    for k, v in group.items():
        try:
            if hasattr(v, "keys"):
                if depth < 3:
                    out[k] = _h5_params(v, depth + 1)
            elif v.size <= 16:
                val = v[()]
                out[k] = (
                    _scalar(val)
                    if np.ndim(val) == 0
                    else [_scalar(x) for x in np.ravel(val)]
                )
        except Exception:  # unreadable entries are skipped
            continue
    return out


def _peek_cxi(path):
    import h5py

    names = []
    with h5py.File(path, "r") as f:
        e = f.get("entry_1")
        if e is None:
            return names
        for k in e:
            if k.startswith(("image", "data", "result", "process")):
                continue
            g = e.get(k)
            if hasattr(g, "keys") and "data" in g and g["data"].ndim == 3:
                names.append(k)
    return names


def _read_cxi(path, fields):
    import h5py

    res = Result(Path(path), {})
    with h5py.File(path, "r") as f:
        e = f.get("entry_1")
        if e is None:
            raise ValueError(f"{path}: no entry_1 group (not a CXI result?)")
        for k in e:
            g = e.get(k)
            if g is None or not hasattr(g, "keys"):
                continue
            if k.startswith("result"):
                if "voxel_size" in g and res.voxel_size is None:
                    res.voxel_size = tuple(
                        float(x) for x in np.ravel(g["voxel_size"][()])
                    )
                if "used_isosurface" in g:
                    iso = _scalar(g["used_isosurface"][()])
                    if isinstance(iso, (int, float)) and np.isfinite(iso):
                        res.isosurface = float(iso)
            elif k.startswith("parameters"):
                res.params = _h5_params(g)
            elif k.startswith("sample") and "sample_name" in g:
                res.sample = str(_scalar(g["sample_name"][()]))
            elif k.startswith(("image", "data", "process")):
                continue
            elif "data" in g and g["data"].ndim == 3:
                if fields is None or k in fields:
                    res.fields[k] = np.asarray(g["data"][()])
    p = res.params
    res.convention = p.get("convention")
    if p.get("scan") is not None:
        res.scan = str(p["scan"])
    if res.voxel_size is None and p.get("voxel_size") is not None:
        v = np.ravel(p["voxel_size"]).astype(float)
        res.voxel_size = tuple(np.repeat(v, 3) if v.size == 1 else v)
    if res.isosurface is None and isinstance(
        p.get("isosurface"), (int, float)
    ):
        res.isosurface = float(p["isosurface"])
    return res


def _peek_npz(path):
    names = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with z.open(info) as fh:
                fmt = np.lib.format
                major, _ = fmt.read_magic(fh)
                read = (
                    fmt.read_array_header_1_0
                    if major == 1
                    else fmt.read_array_header_2_0
                )
                shape, _, _ = read(fh)
            if len(shape) == 3:
                names.append(info.filename[:-4])
    return names


def _read_npz(path, fields):
    res = Result(Path(path), {})
    with np.load(path, allow_pickle=False) as z:
        keys = set(_peek_npz(path))
        for k in z.files:
            if k == "voxel_size":
                res.voxel_size = tuple(float(x) for x in np.ravel(z[k]))
            elif k in keys and (fields is None or k in fields):
                res.fields[k] = np.asarray(z[k])
    return res


def _read_vti(path, fields, convention="cxi"):
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as e:
        raise ImportError(
            f"Reading {Path(path).name} needs the vtk package; use the .npz "
            "or .cxi file of the same scan instead."
        ) from e
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    img = reader.GetOutput()
    nx, ny, nz = img.GetDimensions()
    spacing = tuple(float(s) for s in img.GetSpacing())
    pd = img.GetPointData()
    res = Result(Path(path), {}, convention=convention)
    cxi = str(convention).lower() == "cxi"
    res.voxel_size = spacing[::-1] if cxi else spacing
    for i in range(pd.GetNumberOfArrays()):
        a = pd.GetArray(i)
        if a.GetNumberOfComponents() != 1:
            continue
        name = a.GetName()
        if fields is not None and name not in fields:
            continue
        arr = vtk_to_numpy(a).reshape(nz, ny, nx)
        res.fields[name] = arr.copy() if cxi else arr.transpose(2, 1, 0)
    res.notes.append(
        f"vti file: axis convention assumed {convention!r} "
        "(pass vti_convention= to change)"
    )
    return res


def read_diffraction(path, kind="cropped", intensity="log1p") -> Result:
    """Measured intensity from ``S*_preprocessed_data.cxi`` (reciprocal space).

    ``kind``: ``"cropped"`` (detector frame: pixel / rocking-step indices) or
    ``"orthogonalised"`` (regular q grid; its spacing in 1/Å is used when the
    file stores it). ``intensity``: ``log1p``, ``sqrt`` or ``linear``.
    """
    import h5py

    if intensity not in INTENSITY_TRANSFORMS:
        raise ValueError(
            f"intensity must be one of {INTENSITY_TRANSFORMS}, got {intensity!r}"
        )
    keys = {
        "cropped": "entry_1/cropped_detector_data",
        "orthogonalised": "entry_1/orthogonalised_detector_data",
    }
    if kind not in keys:
        raise ValueError("kind must be 'cropped' or 'orthogonalised'")
    res = Result(Path(path), {})
    with h5py.File(path, "r") as f:
        g = f.get(keys[kind])
        if g is None:
            raise KeyError(f"{Path(path).name}: no {keys[kind]}")
        data = np.squeeze(np.asarray(g["data"][()] if "data" in g else g[()]))
        if data.ndim != 3:
            raise ValueError(f"{keys[kind]} is not 3D: {data.shape}")
        e = f["entry_1"]
        for k in e:
            if k.startswith("parameters"):
                res.params = _h5_params(e[k])
            elif k.startswith("sample") and "sample_name" in e[k]:
                res.sample = str(_scalar(e[k]["sample_name"][()]))
            elif kind == "orthogonalised" and k.startswith("result"):
                q = [e[k].get(n) for n in ("qx_xu", "qy_xu", "qz_xu")]
                if all(x is not None for x in q):
                    q = [np.ravel(x[()]).astype(float) for x in q]
                    if [len(x) for x in q] == list(data.shape):
                        res.voxel_size = tuple(
                            float(np.mean(np.diff(x))) for x in q
                        )
                        res.unit = "1/Å"
    a = np.nan_to_num(data.astype(np.float32), nan=0, posinf=0, neginf=0)
    a = np.maximum(a, 0)
    if intensity == "log1p":
        a = np.log1p(a)
    elif intensity == "sqrt":
        a = np.sqrt(a)
    res.fields = {"intensity": a}
    scan = res.params.get("scan")
    res.scan = str(scan) if scan is not None else _scan_of(path)
    if res.voxel_size is None:
        res.unit = "index"
        res.voxel_size = (1.0, 1.0, 1.0)
        res.notes.append(
            f"{Path(path).name}: {kind} diffraction in index units "
            "(detector pixels / rocking steps), not calibrated q"
        )
    res.notes.append(f"intensity shown as {intensity}(I)")
    res.file_shape = a.shape
    return res


def peek_fields(path) -> list:
    """Names of the 3D fields of a result file, without loading them."""
    p = Path(path)
    if p.suffix == ".cxi":
        return _peek_cxi(p)
    if p.suffix == ".npz":
        return _peek_npz(p)
    if p.suffix == ".vti":
        return list(_read_vti(p, None).fields)
    raise ValueError(f"Unsupported file type: {p.name}")


def read_result(
    path,
    fields=None,
    mask_outside=True,
    normalise_amplitude=True,
    vti_convention="cxi",
) -> Result:
    """Read one cdiutils result (``.cxi``, ``.npz`` or ``.vti``).

    ``fields``: names to load (default: :data:`DEFAULT_FIELDS` present in
    the file, or ``"all"``). ``mask_outside``: values outside the support
    set to NaN (cdiutils stores 0 there), except amplitude/support.
    ``normalise_amplitude``: amplitude divided by its maximum, so the
    pipeline's isosurface applies.
    """
    p = Path(path).expanduser()
    wanted = None if fields in (None, "all") else set(fields)
    need = None if wanted is None else wanted | {"amplitude", "support"}
    if p.suffix == ".cxi":
        res = _read_cxi(p, need)
    elif p.suffix == ".npz":
        res = _read_npz(p, need)
    elif p.suffix == ".vti":
        res = _read_vti(p, need, vti_convention)
    else:
        raise ValueError(
            f"Unsupported file type {p.suffix!r}; expected a cdiutils .cxi, "
            ".npz or .vti result"
        )
    if not res.fields:
        raise ValueError(f"{p}: no 3D field found")
    res.scan = res.scan or _scan_of(p)
    if res.sample is None and res.scan and p.parent.name == f"S{res.scan}":
        res.sample = p.parent.parent.name  # pipeline: <sample>/S<scan>/
    support = res.fields.get("support")
    if fields is None:
        keep = [k for k in DEFAULT_FIELDS if k in res.fields]
        if not keep:  # unknown layout: everything 3D except helpers
            keep = [k for k in res.fields if k not in ("support", "surface")]
    elif fields == "all":
        keep = list(res.fields)
    else:
        missing = [k for k in fields if k not in res.fields]
        if missing:
            raise KeyError(
                f"{p.name}: no field {missing}; available: {list(res.fields)}"
            )
        keep = list(fields)
    out = {}
    for k in keep:
        a = np.asarray(res.fields[k])
        if a.dtype.kind not in "fiub":
            continue  # complex or object arrays are not shown
        a = a.astype(np.float32, copy=False)
        if k == "amplitude" and normalise_amplitude:
            m = np.nanmax(a)
            if np.isfinite(m) and m > 0:
                a = a / m
        if mask_outside and support is not None and k not in UNMASKED:
            a = np.where(np.asarray(support) > 0, a, np.nan).astype(np.float32)
        out[k] = a
    res.fields = out
    res.file_shape = res.shape
    return res


# --------------------------------------------------------------------------
# Several scans
# --------------------------------------------------------------------------


def _zoom_nan(a, factors):
    """Linear resampling that keeps NaN regions (normalised interpolation)."""
    from scipy.ndimage import zoom

    a = np.asarray(a, dtype=np.float32)
    m = np.isfinite(a)
    if m.all():
        return zoom(a, factors, order=1).astype(np.float32)
    v = zoom(np.where(m, a, 0), factors, order=1)
    w = zoom(m.astype(np.float32), factors, order=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0.5, v / w, np.nan).astype(np.float32)


def resample_to_common(results):
    """Resample results to the finest voxel size along each axis."""
    target = np.min([r.voxel_size for r in results], axis=0)
    notes = []
    for r in results:
        f = np.asarray(r.voxel_size, float) / target
        if np.allclose(f, 1, rtol=1e-3):
            continue
        r.fields = {k: _zoom_nan(a, f) for k, a in r.fields.items()}
        notes.append(
            f"{r.path.name}: resampled (linear) from voxel "
            + ", ".join(f"{v:.4g}" for v in r.voxel_size)
            + " to "
            + ", ".join(f"{v:.4g}" for v in target)
            + f" -> {r.shape}"
        )
        r.voxel_size = tuple(float(v) for v in target)
    return notes


def _fit_shape(a, shape, fill):
    """Centre-crop / pad ``a`` to ``shape``."""
    out = np.full(shape, fill, dtype=a.dtype)
    src, dst = [], []
    for n_in, n_out in zip(a.shape, shape):
        if n_in >= n_out:
            s = (n_in - n_out) // 2
            src.append(slice(s, s + n_out))
            dst.append(slice(0, n_out))
        else:
            s = (n_out - n_in) // 2
            src.append(slice(0, n_in))
            dst.append(slice(s, s + n_in))
    out[tuple(dst)] = a[tuple(src)]
    return out


def common_grid(results, shape=None):
    """Put every result on one grid (by default the largest size along each
    axis; smaller volumes are padded around their centre)."""
    shapes = [r.shape for r in results]
    target = tuple(
        int(n) for n in (shape if shape else np.max(shapes, axis=0))
    )
    notes = []
    for r in results:
        if r.shape != target:
            notes.append(
                f"{r.path.name}: {r.shape} -> {target} (centred crop/pad)"
            )
            r.fields = {
                k: _fit_shape(a, target, 0.0 if k == "amplitude" else np.nan)
                for k, a in r.fields.items()
            }
    keys = [set(r.fields) for r in results]
    shared = [k for k in results[0].fields if all(k in s for s in keys)]
    dropped = sorted(set().union(*keys) - set(shared))
    if dropped:
        notes.append(f"fields not in every scan, dropped: {dropped}")
    for r in results:
        r.fields = {k: r.fields[k] for k in shared}
    return notes


def labels_for(results, labels=None, label_from="scan"):
    """Condition labels: explicit, from the scan number, a parameter or a
    callable ``f(result) -> str``.

    Labels must be unique (they name conditions or layers): default labels
    that repeat (e.g. ``S1`` in several samples) get the sample name, then
    a number; repeated explicit labels are an error.
    """
    if labels is not None:
        labels = [str(x) for x in labels]
        if len(labels) != len(results):
            raise ValueError(
                f"{len(labels)} labels for {len(results)} results"
            )
        dup = sorted({x for x in labels if labels.count(x) > 1})
        if dup:
            raise ValueError(f"labels must be unique; repeated: {dup}")
        return labels
    out = []
    for i, r in enumerate(results):
        if callable(label_from):
            out.append(str(label_from(r)))
        elif label_from == "scan":
            out.append(f"S{r.scan}" if r.scan else r.path.stem)
        elif label_from == "sample":
            scan = f"S{r.scan}" if r.scan else r.path.stem
            out.append(f"{r.sample} {scan}" if r.sample else scan)
        elif label_from in ("file", "name"):
            out.append(r.path.stem)
        else:
            v = r.params.get(label_from)
            out.append(str(v) if v is not None else f"#{i}")
    return _unique(out, results)


def _unique(labels, results):
    out = list(labels)
    for _ in range(2):
        dup = {x for x in out if out.count(x) > 1}
        if not dup:
            return out
        for i, r in enumerate(results):
            if (
                out[i] in dup
                and r.sample
                and not out[i].startswith(f"{r.sample} ")
            ):
                out[i] = f"{r.sample} {out[i]}"
    seen = {}
    for i, x in enumerate(out):  # still repeated: number them
        if out.count(x) > 1:
            seen[x] = seen.get(x, 0) + 1
            out[i] = f"{x} #{seen[x]}"
    return out


def load(
    paths,
    fields=None,
    labels=None,
    label_from="scan",
    mask_outside=True,
    normalise_amplitude=True,
    vti_convention="cxi",
    shape=None,
    source="reconstruction",
    diffraction="cropped",
    intensity="log1p",
    resample=True,
):
    """Read one or several results; returns ``(results, labels, notes)``.

    ``paths``: a file, a folder (all scans found in it) or a list of them.
    ``source="diffraction"`` reads the measured intensity of the
    preprocessed files instead (see :func:`read_diffraction`).
    ``resample``: scans with different voxel sizes are resampled (linear,
    NaN-aware) to the finest voxel size along each axis, so they are shown
    with their true size; ``False`` keeps the arrays and uses the first
    voxel size (with a warning).
    """
    if source not in ("reconstruction", "diffraction"):
        raise ValueError("source must be 'reconstruction' or 'diffraction'")
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files = []
    for p in paths:
        found = find_results(p, source=source)
        if not found:
            pats = (
                DIFFRACTION_PATTERNS if source == "diffraction" else PATTERNS
            )
            raise FileNotFoundError(
                f"No cdiutils result in {p} (looked for "
                + ", ".join(pats)
                + ")"
            )
        files += found
    if source == "diffraction":
        results = [read_diffraction(f, diffraction, intensity) for f in files]
    else:
        results = [
            read_result(
                f, fields, mask_outside, normalise_amplitude, vti_convention
            )
            for f in files
        ]
    notes = [n for r in results for n in r.notes]
    if len(results) > 1:
        vs = [r.voxel_size for r in results if r.voxel_size is not None]
        if vs and not all(np.allclose(v, vs[0], rtol=1e-3) for v in vs):
            if resample and len(vs) == len(results):
                notes += resample_to_common(results)
            else:
                msg = f"voxel sizes differ between scans {vs}; using {vs[0]}"
                warnings.warn(msg, stacklevel=2)
                notes.append(msg)
        notes += common_grid(results, shape)
    elif shape is not None:
        notes += common_grid(results, shape)
    return results, labels_for(results, labels, label_from), notes


def summary(results, labels, voxel_size, unit, field_units, thr, notes):
    """Plain-text configuration printed before the viewer opens."""
    lines = ["MultiVolumeViewer configuration", "=" * 31]
    rows = [("condition", "scan", "file", "shape", "voxel size", "iso")]
    for lab, r in zip(labels, results):
        rows.append(
            (
                lab,
                r.scan or "-",
                r.path.name,
                "x".join(map(str, r.file_shape or r.shape)),
                "-"
                if r.voxel_size is None
                else ", ".join(f"{v:.4g}" for v in r.voxel_size),
                "-" if r.isosurface is None else f"{r.isosurface:.3g}",
            )
        )
    widths = [max(len(str(row[i])) for row in rows) for i in range(6)]
    for j, row in enumerate(rows):
        lines.append(
            "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip()
        )
        if j == 0:
            lines.append("  ".join("-" * w for w in widths))
    lines.append("")
    lines.append(
        "voxel size used : "
        + ", ".join(f"{v:.4g}" for v in voxel_size)
        + f" {unit} (array axes)"
    )
    lines.append(f"amplitude iso   : {thr:.3g}" if thr is not None else "")
    lines.append("fields          :")
    for k in results[0].fields:
        lab, sc = field_units.get(k, ("", 1.0))
        a = results[0].fields[k]
        fin = a[np.isfinite(a)]
        rng = f"[{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "(empty)"
        u = f"{lab}" + (f" (x{sc:g})" if sc != 1 else "")
        lines.append(f"  {k:<26} {u:<10} {rng}")
    if notes:
        lines.append("notes           :")
        lines += [f"  - {n}" for n in notes]
    return "\n".join(x for x in lines if x is not None)
