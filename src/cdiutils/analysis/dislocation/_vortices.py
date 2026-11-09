import numpy as np
from scipy import ndimage

from cdiutils.analysis.dislocation._geometry import fit_line_3d


def _wrap(phase_difference):
    """Wrap phase differences into [-pi, pi)."""
    return (phase_difference + np.pi) % (2 * np.pi) - np.pi


def find_phase_vortices(
    phase: np.ndarray,
    support: np.ndarray,
    amplitude: np.ndarray = None,
    voxel_size: float = 1.0,
    erode: int = 2,
    min_voxels: int = 6,
) -> tuple:
    """
    Find the lines where the phase winds, the signature of dislocations.

    Around a dislocation line the displacement changes by the Burgers
    vector, so the phase changes by a multiple of 2 pi along any closed
    loop around it (Sun & Singer, Chem. Phys. Rev. 5, 031310 (2024),
    p. 5). Every elementary square (plaquette) of voxels is tested on
    the three planes; plaquettes with a non-zero winding inside the
    eroded support are grouped into connected lines, and each line is
    fitted with fit_line_3d. Lines with no displacement along Q, within
    `erode` voxels of the surface, or below the resolution are not
    found.

    Args:
        phase (np.ndarray): the 3D phase (rad), NaN or anything outside
            the support.
        support (np.ndarray): the 3D support.
        amplitude (np.ndarray, optional): the 3D amplitude, to report
            the amplitude on and around each line (a dislocation core
            usually has low amplitude). Defaults to None.
        voxel_size (float, optional): voxel size, for the line length.
            Defaults to 1.
        erode (int, optional): voxels removed from the support surface.
            Defaults to 2.
        min_voxels (int, optional): smallest line kept. Defaults to 6.

    Returns:
        tuple: (list of dicts, one per line: voxels, length, centre,
            direction, winding sign, distance to the surface and, with
            an amplitude, amp_core and amp_ring normalised to the 99th
            percentile; boolean mask of all winding plaquettes).
    """
    support = np.asarray(support) > 0
    core = ndimage.binary_erosion(support, iterations=erode)
    phase = np.nan_to_num(phase)
    hits = np.zeros(phase.shape, bool)
    winding = np.zeros(phase.shape, np.int8)
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        shifted_a = np.roll(phase, -1, axis_a)
        shifted_b = np.roll(phase, -1, axis_b)
        shifted_ab = np.roll(shifted_a, -1, axis_b)
        loop = (_wrap(shifted_a - phase) + _wrap(shifted_ab - shifted_a)
                + _wrap(shifted_b - shifted_ab) + _wrap(phase - shifted_b))
        turns = np.rint(loop / (2 * np.pi)).astype(np.int8)
        # all four corners of the plaquette inside the eroded support
        inside = (core & np.roll(core, -1, axis_a)
                  & np.roll(core, -1, axis_b)
                  & np.roll(np.roll(core, -1, axis_a), -1, axis_b))
        found = (turns != 0) & inside
        hits |= found
        winding[found] = turns[found]

    labels, n_lines = ndimage.label(hits, structure=np.ones((3, 3, 3)))
    surface_distance = ndimage.distance_transform_edt(support)
    normalised = None
    if amplitude is not None:
        normalised = amplitude / np.nanpercentile(amplitude[support], 99)
    lines = []
    for k in range(1, n_lines + 1):
        line = labels == k
        if line.sum() < min_voxels:
            continue
        points = np.argwhere(line)
        centre, direction = fit_line_3d(points)
        along = (points - centre) @ direction
        result = {
            "voxels": int(line.sum()),
            "length": float(np.ptp(along) * voxel_size),
            "centre": centre.tolist(),
            "direction": direction.tolist(),
            "winding": int(np.sign(winding[line].sum()) or 1),
            "surface_distance": float(surface_distance[line].mean()
                                      * voxel_size),
        }
        if normalised is not None:
            ring = (ndimage.binary_dilation(line, iterations=3)
                    & ~ndimage.binary_dilation(line) & support)
            result["amp_core"] = float(np.nanmean(normalised[line]))
            result["amp_ring"] = (float(np.nanmean(normalised[ring]))
                                  if ring.any() else np.nan)
        lines.append(result)
    return lines, hits
