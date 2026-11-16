"""Planar defects (twin boundaries, stacking faults) in a BCDI amplitude."""

import numpy as np
from scipy import ndimage, signal


def _closed_interior(support: np.ndarray, closing: int, depth: int):
    """Support closed over low-amplitude sheets, filled, then eroded."""
    pad = closing + 1
    closed = ndimage.binary_closing(
        np.pad(support, pad), iterations=closing
    )[pad:-pad, pad:-pad, pad:-pad] | support
    filled = ndimage.binary_fill_holes(closed)
    return filled, ndimage.binary_erosion(filled, iterations=depth)


def find_planar_defects(
    amplitude: np.ndarray,
    support: np.ndarray,
    level: float = 0.35,
    depth: int = 2,
    closing: int = 3,
    min_voxels: int = 30,
    reference: tuple = (0.0, 1.0, 0.0),
    radius: float = 10.0,
    prominence: float = 0.08,
) -> list:
    """
    Find planar low-amplitude sheets inside a crystal and the boundaries
    they contain.

    A twin boundary or stacking fault seen in a reflection whose
    displacement jumps across it shows as a thin sheet of low amplitude in
    the reconstruction. The support is first closed (`closing` iterations)
    so that sheets cutting the particle are inside it, filled and eroded by
    `depth` voxels; voxels below `level` (amplitude normalised to its 99th
    percentile in the support) are grouped into connected sheets. Each
    sheet is fitted by principal components: its normal is the axis of
    least variance. Several close parallel boundaries (a thin lamella) can
    merge into one sheet, so the boundaries are then located as minima of
    the mean amplitude along the normal, in a cylinder of `radius` voxels
    around the sheet centre (which keeps partial boundaries from being
    diluted by the rest of the particle).

    Args:
        amplitude (np.ndarray): 3D amplitude.
        support (np.ndarray): 3D support.
        level (float, optional): sheet threshold on the normalised
            amplitude. Defaults to 0.35.
        depth (int, optional): voxels removed from the surface. Defaults
            to 2.
        closing (int, optional): closing iterations of the support.
            Defaults to 3.
        min_voxels (int, optional): smallest sheet kept. Defaults to 30.
        reference (tuple, optional): direction, in array axes, against
            which the angle of each normal is given (e.g. the measured Q;
            (0, 1, 0) is the vertical axis of the CXI frame). Defaults to
            (0, 1, 0).
        radius (float, optional): radius of the cylinder used for the
            profile, in voxels. Defaults to 10.
        prominence (float, optional): smallest depth of a boundary
            minimum in the profile. Defaults to 0.08.

    Returns:
        list: one dict per sheet, largest first: voxels, centre, normal,
            angle_to_reference (deg, 0-90), planarity (least / middle
            variance), thickness (2 sigma along the normal, voxels),
            boundaries (positions of the minima along the normal relative
            to the centre, voxels), boundary_amplitude (mean amplitude at
            each minimum), spacing (between consecutive boundaries).
    """
    support = np.asarray(support) > 0
    norm = np.nanpercentile(amplitude[support], 99)
    amp = np.asarray(amplitude, dtype=float) / norm
    _, inner = _closed_interior(support, closing, depth)
    labels, n_sheets = ndimage.label(
        inner & (amp < level), structure=np.ones((3, 3, 3))
    )
    ref = np.asarray(reference, float)
    ref = ref / np.linalg.norm(ref)
    grid = np.argwhere(inner).astype(float)
    values = amp[inner]
    sheets = []
    for k in range(1, n_sheets + 1):
        points = np.argwhere(labels == k).astype(float)
        if len(points) < min_voxels:
            continue
        centre = points.mean(0)
        variances, axes = np.linalg.eigh(np.cov((points - centre).T))
        normal = axes[:, 0]
        # amplitude profile along the normal, in a cylinder around the
        # sheet centre, 1-voxel bins
        rel = grid - centre
        along = rel @ normal
        across = np.linalg.norm(rel - along[:, None] * normal, axis=1)
        near = across < radius
        edges = np.arange(np.floor(along[near].min()),
                          np.ceil(along[near].max()) + 1)
        index = np.digitize(along[near], edges)
        counts = np.bincount(index, minlength=len(edges) + 1)[1:-1]
        sums = np.bincount(index, weights=values[near],
                           minlength=len(edges) + 1)[1:-1]
        profile = np.where(counts > 5, sums / np.maximum(counts, 1), np.nan)
        positions = edges[:-1] + 0.5
        good = ~np.isnan(profile)
        minima, _ = signal.find_peaks(-profile[good], prominence=prominence)
        boundaries = positions[good][minima]
        sheets.append({
            "voxels": int(len(points)),
            "centre": centre.tolist(),
            "normal": normal.tolist(),
            "angle_to_reference": float(np.degrees(
                np.arccos(min(abs(normal @ ref), 1.0)))),
            "planarity": float(variances[0] / variances[1]),
            "thickness": float(2 * np.sqrt(max(variances[0], 0.0))),
            "boundaries": boundaries.tolist(),
            "boundary_amplitude": profile[good][minima].tolist(),
            "spacing": np.diff(boundaries).tolist(),
        })
    return sorted(sheets, key=lambda s: -s["voxels"])
