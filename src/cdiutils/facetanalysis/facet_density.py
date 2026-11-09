"""Facet analysis from the density of surface normals.

Facets are the peaks of the density of surface-normal orientations on the
unit sphere, as in FacetAnalyser (Grothausmann et al., Ultramicroscopy 122,
65 (2012), the "Gaussian splat"), so their number comes from the data and is
not chosen in advance (unlike the spectral clustering of
FacetAnalysisProcessor).

The crystal orientation has its out-of-plane direction fixed by the measured
specular reflection ([111] for a (111) peak, [100] for a (200) peak) and one
rotation about it, fitted on the facets (and on a planar defect, required to
be a {111} plane), then refined on all indexed facets. Facets are indexed
lowest index first with tolerances that keep the chance of a coincidental
match near 3-5 %. Facets equivalent by the symmetry that keeps the top
direction in place form an orbit (for [111]: the permutations of h, k, l).
The two faces of a planar gap between separate parts of the support (twin
lamella, faulted region) look into the particle: they are internal surfaces,
kept apart from the facets.

With a single reflection the in-plane crystal axes are only defined up to
the symmetry around the top direction, so a facet label stands for any member
of its orbit; labels are consistent within one analysis (one rotation).

Arrays follow the CXI convention used by cdiutils post-processing
(Z beam, Y vertical up, X outboard); vectors are written in array order.
"""
from functools import lru_cache
from itertools import permutations, product
from math import gcd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, PowerNorm
from scipy import ndimage


# (highest Miller index, accepted misfit in degrees). With 2-4 degree
# precision on the normals, a random direction falls within 4 deg of a
# {100}, {110} or {111} direction 3 % of the time but within 4 deg of some
# direction up to index 3 34 % of the time, hence the tighter tolerances.
INDEX_LEVELS = ((1, 4.0), (2, 2.5), (3, 1.5))

FAMILY_COLOURS = {
    "{111}": "tab:blue", "{100}": "tab:orange", "{110}": "tab:purple",
    "{210}": "tab:olive", "{211}": "tab:cyan", "{221}": "tab:pink",
    "{311}": "tab:brown",
}
CONDITION_COLOURS = ("tab:blue", "tab:green", "tab:red", "tab:purple",
                     "tab:orange", "tab:brown")


@lru_cache(maxsize=None)
def _direction_table(max_index):
    """Cached table of the cubic directions up to max_index."""
    names, hkls = [], []
    for name, members in FacetDensityAnalyser.cubic_families(
            max_index).items():
        for hkl in members:
            names.append(name)
            hkls.append(hkl)
    unit = np.array(hkls, float)
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    return tuple(names), tuple(hkls), unit


class FacetDensityAnalyser:
    """Find, index and measure the facets of one reconstruction.

    Typical use on a reference scan::

        analyser = FacetDensityAnalyser(support, top=(1, 1, 1))
        reference = analyser.analyse()
        facets, boundary = analyser.statistics(
            {"strain": het_strain_from_dspacing, "dspacing": dspacing})

    and on the other scans of a series, with the reference orientation::

        other = FacetDensityAnalyser.from_reference(support, reference)
        facets, boundary = other.statistics({...})
        peaks = other.scan_peaks()

    Args:
        support (np.ndarray): 3D support of the reconstruction.
        top (tuple): out-of-plane crystal direction (the reflection).
        top_lab (tuple): lab direction of the reflection, array order
            (vertical for a specular reflection).
        voxel_size (float): voxel size in nm, for heights and positions.
        sigma (float): smoothing of the support for the normals (voxels).
        tol_deg (float): voxel-to-facet angular tolerance.
        min_count (int): voxels within tol_deg needed for a density peak.
        min_patch (int): smallest connected patch counted in a facet.
        kernel_width_deg (float): width of the density kernel.
    """

    def __init__(self, support, top=(1, 1, 1), top_lab=(0, 1, 0),
                 voxel_size=10.0, sigma=1.5, tol_deg=5.0, min_count=60,
                 min_patch=30, kernel_width_deg=3.0):
        self.support = np.asarray(support) > 0
        self.top = tuple(top)
        self.top_lab = np.asarray(top_lab, float)
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.tol_deg = tol_deg
        self.min_count = min_count
        self.min_patch = min_patch
        self.kernel_width_deg = kernel_width_deg
        self.rotation = None
        self.facets = []
        self.orientation_cost_deg = None

        self.surface, self.normals = self.surface_normals(self.support,
                                                          sigma)
        self.idx = np.argwhere(self.surface)
        self.gap = self.planar_gap(self.support)
        self.gap_face = ndimage.binary_dilation(self.gap)[self.surface]
        self.boundary_normal = None
        if self.gap_face.sum() > 50:
            faces = self.normals[self.gap_face]
            # both faces with a common sign give the plane normal
            signs = np.sign(faces @ faces[0])
            normal = (faces * signs[:, None]).mean(0)
            self.boundary_normal = normal / np.linalg.norm(normal)

    # ---------------------------------------------------------- crystal

    @staticmethod
    def cubic_families(max_index=3):
        """Families {hkl} of the cubic system up to a maximum index.

        Args:
            max_index (int): largest |h|, |k|, |l|.

        Returns:
            dict: family name (e.g. "{113}") -> list of its directions as
            integer tuples.
        """
        families = {}
        for hkl in product(range(max_index + 1), repeat=3):
            if hkl == (0, 0, 0) or gcd(gcd(*hkl[:2]), hkl[2]) != 1:
                continue
            base = tuple(sorted(hkl, reverse=True))
            name = "{" + "".join(str(i) for i in base) + "}"
            if name in families:
                continue
            members = set()
            for perm in permutations(base):
                for signs in product((1, -1), repeat=3):
                    members.add(tuple(int(p * s)
                                      for p, s in zip(perm, signs)))
            families[name] = sorted(members)
        return families

    @staticmethod
    def symmetry_operations(top):
        """Cubic point operations that keep the top direction in place.

        Args:
            top (tuple): out-of-plane crystal direction.

        Returns:
            list: 3x3 integer matrices (for [111]: the six permutations,
            the 3-fold axis and three mirrors, C3v).
        """
        top = np.array(top, float)
        ops = []
        for perm in permutations(range(3)):
            for signs in product((1, -1), repeat=3):
                matrix = np.zeros((3, 3), int)
                for row, (col, sign) in enumerate(zip(perm, signs)):
                    matrix[row, col] = sign
                if np.allclose(matrix @ top, top):
                    ops.append(matrix)
        return ops

    @classmethod
    def orbit_label(cls, hkl, top=(1, 1, 1)):
        """Facets equivalent to hkl by the symmetry around the top.

        Args:
            hkl (tuple): facet indices.
            top (tuple): out-of-plane direction.

        Returns:
            str: the sorted members, e.g. "(-111)(1-11)(11-1)".
        """
        members = {tuple(int(v) for v in op @ np.array(hkl))
                   for op in cls.symmetry_operations(top)}
        return "".join("(" + "".join(str(v) for v in m) + ")"
                       for m in sorted(members))

    @staticmethod
    def fibonacci_sphere(n=20000):
        """Nearly uniform unit vectors on the sphere, shape (n, 3)."""
        i = np.arange(n) + 0.5
        polar = np.arccos(1 - 2 * i / n)
        azimuth = np.pi * (1 + 5 ** 0.5) * i
        return np.stack([np.cos(polar),
                         np.sin(polar) * np.sin(azimuth),
                         np.sin(polar) * np.cos(azimuth)], axis=1)

    @staticmethod
    def angle_deg(u, v):
        """Angles in degrees between the rows of u and of v (n, m)."""
        return np.degrees(np.arccos(np.clip(np.atleast_2d(u)
                                            @ np.atleast_2d(v).T, -1, 1)))

    @classmethod
    def chance_level(cls, max_index, misfit_deg, n_grid=20000):
        """Probability that a random direction indexes within misfit_deg.

        Args:
            max_index (int): families up to this index are allowed.
            misfit_deg (float): accepted misfit.
            n_grid (int): number of uniform random directions.

        Returns:
            float: fraction of the sphere within misfit_deg.
        """
        unit = _direction_table(max_index)[2]
        misfit = cls.angle_deg(cls.fibonacci_sphere(n_grid), unit).min(1)
        return float((misfit < misfit_deg).mean())

    # ---------------------------------------------------------- surface

    @staticmethod
    def surface_normals(support, sigma=1.5):
        """Outer voxel layer of a support and its outward normals.

        Args:
            support (np.ndarray): 3D support.
            sigma (float): smoothing before the gradient (voxels).

        Returns:
            tuple: surface mask and (n, 3) unit normals in the order of
            np.argwhere(surface).
        """
        support = np.asarray(support) > 0
        surface = support & ~ndimage.binary_erosion(support)
        smooth = ndimage.gaussian_filter(support.astype(float), sigma)
        normals = -np.array(np.gradient(smooth))[:, surface].T
        normals /= np.linalg.norm(normals, axis=1)[:, None] + 1e-12
        return surface, normals

    @staticmethod
    def planar_gap(support, min_share=0.1, closing=4):
        """Voxels of a gap between separate parts of the support.

        Only parts larger than min_share of the largest count, so that a
        concave surface is not mistaken for a gap.

        Args:
            support (np.ndarray): 3D support.
            min_share (float): minimum size of a part, relative.
            closing (int): closing iterations used to fill the gap.

        Returns:
            np.ndarray: boolean mask of the gap (empty for one part).
        """
        support = np.asarray(support) > 0
        labels, n_parts = ndimage.label(support)
        if n_parts < 2:
            return np.zeros_like(support)
        sizes = np.bincount(labels.ravel())[1:]
        big = np.nonzero(sizes >= min_share * sizes.max())[0] + 1
        if len(big) < 2:
            return np.zeros_like(support)
        parts = np.isin(labels, big)
        return ndimage.binary_closing(parts, iterations=closing) & ~support

    # ------------------------------------------------------ peaks, index

    def normal_density(self, normals, grid, chunk=2000):
        """Kernel density of normal orientations on grid directions.

        Args:
            normals (np.ndarray): (n, 3) unit normals.
            grid (np.ndarray): (m, 3) unit directions.
            chunk (int): grid points per block (memory).

        Returns:
            np.ndarray: (m,) density (von Mises-Fisher kernel).
        """
        kappa = 1 / np.radians(self.kernel_width_deg) ** 2
        density = np.zeros(len(grid))
        for start in range(0, len(grid), chunk):
            block = grid[start:start + chunk]
            density[start:start + chunk] = np.exp(
                kappa * (block @ normals.T - 1)).sum(1)
        return density

    def find_peaks(self, min_sep_deg=8.0):
        """Facet directions: local maxima of the external normal density.

        Args:
            min_sep_deg (float): minimum angle between two peaks.

        Returns:
            list: dicts with the mean direction of the voxels within
            tol_deg of each peak ('direction') and their number ('count'),
            largest first.
        """
        normals = self.normals[~self.gap_face]
        grid = self.fibonacci_sphere()
        density = self.normal_density(normals, grid)
        order = np.argsort(density)[::-1]
        cos_sep = np.cos(np.radians(min_sep_deg))
        cos_tol = np.cos(np.radians(self.tol_deg))
        peaks = []
        for i in order:
            if density[i] < 0.02 * density[order[0]]:
                break
            seed = grid[i]
            if any(p["seed"] @ seed > cos_sep for p in peaks):
                continue
            # a peak must be the maximum of its neighbourhood
            if density[grid @ seed > cos_sep].max() > density[i]:
                continue
            members = normals @ seed > cos_tol
            if members.sum() < self.min_count:
                continue
            mean = normals[members].mean(0)
            peaks.append(dict(seed=seed,
                              direction=mean / np.linalg.norm(mean),
                              count=int(members.sum())))
        return peaks

    @staticmethod
    def _rotation_about(axis, angle):
        """Rotation matrix of an angle (rad) about an axis."""
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        x, y, z = axis
        c, s = np.cos(angle), np.sin(angle)
        t = 1 - c
        return np.array([
            [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
            [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
            [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
        ])

    @classmethod
    def _align(cls, a, b):
        """Rotation taking direction a onto direction b."""
        a = np.asarray(a, float) / np.linalg.norm(a)
        b = np.asarray(b, float) / np.linalg.norm(b)
        cross = np.cross(a, b)
        sin, cos = np.linalg.norm(cross), a @ b
        if sin < 1e-9:
            if cos > 0:
                return np.eye(3)
            other = [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]
            return cls._rotation_about(np.cross(a, other), np.pi)
        skew = np.array([[0, -cross[2], cross[1]],
                         [cross[2], 0, -cross[0]],
                         [-cross[1], cross[0], 0]])
        return np.eye(3) + skew + skew @ skew * (1 - cos) / sin ** 2

    def fit_orientation(self, peaks, max_tilt_deg=2.0):
        """Crystal-to-lab rotation from the facet peaks.

        The top direction is put along top_lab (tilt up to max_tilt_deg)
        and the rotation about it scanned in 0.5 deg steps; the cost is
        the size-weighted mean angle of the peaks to the nearest low-index
        direction, plus the angle of a planar gap to the nearest {111}.

        Args:
            peaks (list): output of find_peaks.
            max_tilt_deg (float): allowed tilt of the top direction.

        Returns:
            tuple: (rotation, cost in degrees).
        """
        _, hkls, unit = _direction_table(1)
        is_111 = np.array([sorted(map(abs, h)) == [1, 1, 1] for h in hkls])
        normals = np.array([p["direction"] for p in peaks])
        weights = np.array([p["count"] for p in peaks], float)
        weights /= weights.sum()
        top_lab = self.top_lab / np.linalg.norm(self.top_lab)
        perp = np.cross(top_lab, [1, 0, 0])
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(top_lab, perp)
        tilts = [(0.0, 0.0)] + [
            (np.radians(t), np.radians(p))
            for t in np.arange(1, max_tilt_deg + 0.1, 1)
            for p in range(0, 360, 30)]
        best = (np.inf, None)
        for tilt, tilt_azimuth in tilts:
            axis = np.cos(tilt_azimuth) * perp + np.sin(tilt_azimuth) * perp2
            tilted = self._rotation_about(axis, tilt) @ top_lab
            base = self._align(self.top, tilted)
            for phi in np.radians(np.arange(0, 360, 0.5)):
                rotation = self._rotation_about(tilted, phi) @ base
                lab = unit @ rotation.T
                cost = (weights * self.angle_deg(normals, lab).min(1)).sum()
                if self.boundary_normal is not None:
                    cos = np.abs(lab[is_111] @ self.boundary_normal)
                    cost += np.degrees(np.arccos(np.clip(cos, -1, 1))).min()
                if cost < best[0]:
                    best = (cost, rotation)
        return best[1], best[0]

    def refine_orientation(self, rotation, directions, hkls, weights,
                           max_tilt_deg=2.0):
        """Weighted best rotation (Kabsch) of indexed facets.

        Rejected (the input rotation is returned) if it tilts the top
        direction more than max_tilt_deg from top_lab.

        Args:
            rotation (np.ndarray): starting rotation.
            directions (np.ndarray): (n, 3) measured lab normals.
            hkls (list): (n,) indices of the facets.
            weights (np.ndarray): (n,) facet sizes.
            max_tilt_deg (float): allowed tilt of the top direction.

        Returns:
            np.ndarray: refined rotation.
        """
        crystal = np.array(hkls, float)
        crystal /= np.linalg.norm(crystal, axis=1)[:, None]
        lab = np.asarray(directions, float)
        cov = (lab * np.asarray(weights, float)[:, None]).T @ crystal
        u, _, vt = np.linalg.svd(cov)
        sign = np.sign(np.linalg.det(u @ vt))
        refined = u @ np.diag([1, 1, sign]) @ vt
        top = refined @ (np.array(self.top, float)
                         / np.linalg.norm(self.top))
        tilt = self.angle_deg(top, self.top_lab / np.linalg.norm(
            self.top_lab))[0, 0]
        return refined if tilt <= max_tilt_deg else rotation

    def index(self, direction, rotation=None, levels=INDEX_LEVELS):
        """Index a lab direction, lowest index first.

        Args:
            direction (np.ndarray): (3,) lab unit vector.
            rotation (np.ndarray): crystal-to-lab rotation (default: the
                fitted one).
            levels (tuple): (max_index, tolerance_deg), increasing.

        Returns:
            dict: family, hkl, misfit_deg, angle_to_top, orbit, kind
            ('facet' or 'vicinal'), level and chance (probability of a
            random match at that level).
        """
        rotation = self.rotation if rotation is None else rotation
        top_unit = np.array(self.top, float) / np.linalg.norm(self.top)
        for max_index, tolerance in levels:
            names, hkls, unit = _direction_table(max_index)
            misfit = self.angle_deg(direction, unit @ rotation.T)[0]
            i = int(misfit.argmin())
            if misfit[i] < tolerance:
                kind = "facet"
                chance = self.chance_level(max_index, tolerance)
                break
        else:
            kind, chance = "vicinal", float("nan")
        return dict(family=names[i], hkl=list(hkls[i]),
                    misfit_deg=float(misfit[i]),
                    angle_to_top=float(self.angle_deg(unit[i],
                                                      top_unit)[0, 0]),
                    orbit=self.orbit_label(hkls[i], self.top), kind=kind,
                    level=max_index, chance=chance)

    def analyse(self, n_refine=3):
        """Peaks, orientation and indexing of this (reference) scan.

        Args:
            n_refine (int): refinement iterations of the orientation.

        Returns:
            dict: reference description (top, tolerances, orientation
            cost, rotation, facets, boundary), reusable with
            from_reference.
        """
        peaks = self.find_peaks()
        rotation, _ = self.fit_orientation(peaks)
        for _ in range(n_refine):
            indexed = [(p, self.index(p["direction"], rotation))
                       for p in peaks]
            indexed = [(p, i) for p, i in indexed if i["kind"] == "facet"]
            rotation = self.refine_orientation(
                rotation, [p["direction"] for p, _ in indexed],
                [i["hkl"] for _, i in indexed],
                [p["count"] for p, _ in indexed])
        self.rotation = rotation
        self.facets = []
        for peak in peaks:
            info = self.index(peak["direction"])
            info["count"] = peak["count"]
            info["lab_direction"] = peak["direction"].tolist()
            info["crystal_direction"] = (rotation.T
                                         @ peak["direction"]).tolist()
            self.facets.append(info)
        weights = [f["count"] for f in self.facets]
        self.orientation_cost_deg = float(np.average(
            [f["misfit_deg"] for f in self.facets], weights=weights))
        return self.reference()

    def reference(self):
        """Description of the fitted orientation and facets (dict)."""
        boundary = None
        if self.boundary_normal is not None:
            boundary = self.index(self.boundary_normal, levels=((1, 10.0),))
            boundary["lab_normal"] = self.boundary_normal.tolist()
        return dict(top=list(self.top), tol_deg=self.tol_deg,
                    min_count=self.min_count,
                    orientation_cost_deg=self.orientation_cost_deg,
                    rotation=self.rotation.tolist(), facets=self.facets,
                    boundary=boundary)

    @classmethod
    def from_reference(cls, support, reference, **kwargs):
        """Analyser of another scan with a reference orientation.

        Args:
            support (np.ndarray): 3D support of the scan.
            reference (dict): output of analyse (or reference()).
            **kwargs: other FacetDensityAnalyser arguments.

        Returns:
            FacetDensityAnalyser: with the reference rotation and facets.
        """
        analyser = cls(support, top=tuple(reference["top"]),
                       tol_deg=reference["tol_deg"],
                       min_count=reference["min_count"], **kwargs)
        analyser.rotation = np.array(reference["rotation"])
        analyser.facets = reference["facets"]
        analyser.orientation_cost_deg = reference["orientation_cost_deg"]
        return analyser

    # ------------------------------------------------- assignment, values

    def facet_directions(self):
        """Lab directions of the facets, from their crystal directions."""
        return np.array([self.rotation @ np.array(f["crystal_direction"])
                         for f in self.facets])

    def assign(self):
        """Label every outer surface voxel.

        A voxel joins the nearest facet when its normal is within tol_deg
        and it lies in a connected patch (26-neighbours, same facet) of at
        least min_patch voxels. Gap faces are internal surfaces: -2 (normal
        along the plane normal) and -3 (opposite side), whatever their
        orientation. Others: -1.

        Returns:
            np.ndarray: (n,) labels in the order of self.idx.
        """
        label = np.full(len(self.idx), -1)
        if self.facets:
            angles = self.angle_deg(self.normals, self.facet_directions())
            label = np.where(angles.min(1) < self.tol_deg,
                             angles.argmin(1), -1)
        if self.boundary_normal is not None:
            side = np.where(self.normals @ self.boundary_normal > 0, -2, -3)
            label[self.gap_face] = side[self.gap_face]
        grid = np.full(self.support.shape, -9, int)
        grid[tuple(self.idx.T)] = label
        for k in np.unique(label[label >= 0]):
            patches, _ = ndimage.label(grid == k,
                                       structure=np.ones((3, 3, 3)))
            sizes = np.bincount(patches.ravel())
            small = (patches > 0) & (sizes[patches] < self.min_patch)
            grid[small] = -1
        return grid[tuple(self.idx.T)]

    def _below(self, values, depths=(1.5, 2.5)):
        """Mean of a 3D quantity a few voxels below each surface voxel."""
        out = np.full(len(self.idx), np.nan)
        for i, (point, normal) in enumerate(zip(self.idx, self.normals)):
            picked = []
            for depth in depths:
                q = tuple(np.rint(point - depth * normal).astype(int))
                if self.support[q]:
                    picked.append(values[q])
            if picked:
                out[i] = np.nanmean(picked)
        return out

    def statistics(self, quantities, near_gap_voxels=3):
        """Mean quantities over each facet and each gap face.

        Args:
            quantities (dict): name -> 3D array (e.g. strain, dspacing).
                Each is averaged over the outer layer ('<name>_surface')
                and 1.5-2.5 voxels below ('<name>_below').
            near_gap_voxels (float): distance defining 'share_at_gap'.

        Returns:
            tuple: (facets, boundary) DataFrames, one row per facet and
            per gap face.
        """
        label = self.assign()
        bottom = np.nonzero(self.support.any((0, 2)))[0].min()
        height = (self.idx[:, 1] - bottom) * self.voxel_size
        values = {}
        for name, array in quantities.items():
            values[f"{name}_surface"] = array[tuple(self.idx.T)]
            values[f"{name}_below"] = self._below(array)
        gap_distance = np.full(len(self.idx), np.inf)
        if self.gap.any():
            gap_distance = ndimage.distance_transform_edt(
                ~self.gap)[tuple(self.idx.T)]

        rows = []
        for k, facet in enumerate(self.facets):
            member = label == k
            if not member.any():
                continue
            row = dict(facet_id=k, kind=facet["kind"],
                       family=facet["family"],
                       hkl=str(tuple(facet["hkl"])), orbit=facet["orbit"],
                       angle_to_top=round(facet["angle_to_top"], 1),
                       voxels=int(member.sum()),
                       height_nm=float(height[member].mean()),
                       share_at_gap=float((gap_distance[member]
                                           <= near_gap_voxels).mean()))
            for name, value in values.items():
                row[name] = float(np.nanmean(value[member]))
            rows.append(row)

        bounds = []
        if self.boundary_normal is not None:
            info = self.index(self.boundary_normal, levels=((1, 10.0),))
            along = np.argwhere(self.gap) @ self.boundary_normal
            centre = np.array(ndimage.center_of_mass(self.support))
            parts, _ = ndimage.label(self.support)
            sizes = sorted(np.bincount(parts.ravel())[1:].tolist())
            for side, name in ((-2, "+n side"), (-3, "-n side")):
                member = label == side
                row = dict(side=name, plane=str(tuple(info["hkl"])),
                           misfit_deg=info["misfit_deg"],
                           offset_nm=float(along.mean()
                                           - centre @ self.boundary_normal)
                           * self.voxel_size,
                           roughness_nm=float(along.std()) * self.voxel_size,
                           voxels=int(member.sum()),
                           height_nm=float(height[member].mean()),
                           part_volumes=str(sizes[-2:]))
                for value_name, value in values.items():
                    row[value_name] = float(np.nanmean(value[member]))
                bounds.append(row)
        return pd.DataFrame(rows), pd.DataFrame(bounds)

    def scan_peaks(self):
        """Peaks of this scan, matched to the facets of the reference.

        Returns:
            pd.DataFrame: count, family, hkl, kind and misfit of each peak,
            and the matched facet index (reference_id, -1 if none within
            tol_deg).
        """
        directions = self.facet_directions()
        rows = []
        for peak in self.find_peaks():
            info = self.index(peak["direction"])
            match = self.angle_deg(peak["direction"], directions)[0]
            rows.append(dict(
                count=peak["count"], family=info["family"],
                hkl=str(tuple(info["hkl"])), kind=info["kind"],
                misfit_deg=info["misfit_deg"],
                reference_id=(int(match.argmin())
                              if match.min() < self.tol_deg else -1)))
        return pd.DataFrame(rows)

    # ------------------------------------------------------ series tables

    @staticmethod
    def summarise(facets, min_share=0.8, min_voxels=60):
        """Facets present in most scans of a series, with mean values.

        Args:
            facets (pd.DataFrame): concatenated statistics of the scans
                (one row per facet and scan, with 'scan_key' and
                'strain_surface' columns).
            min_share (float): minimum share of scans with the facet.
            min_voxels (int): minimum median size in outer voxels.

        Returns:
            pd.DataFrame: one row per kept facet.
        """
        n_scans = facets.scan_key.nunique()
        summary = facets.groupby(
            ["facet_id", "kind", "family", "hkl", "orbit", "angle_to_top"]
        ).agg(scans=("scan_key", "nunique"), voxels=("voxels", "median"),
              height_nm=("height_nm", "mean"),
              strain=("strain_surface", "mean"),
              strain_sd=("strain_surface", "std")).reset_index()
        keep = ((summary.scans >= min_share * n_scans)
                & (summary.voxels >= min_voxels))
        return summary[keep].sort_values(["angle_to_top", "voxels"],
                                         ascending=[True, False])

    @staticmethod
    def persistent_new_facets(peaks, state="state"):
        """Unmatched peaks found in every scan of a state with repeats.

        Args:
            peaks (pd.DataFrame): concatenated scan_peaks tables with
                'state' (e.g. the scan folder) and 'scan_key' columns.
            state (str): column defining the states.

        Returns:
            pd.DataFrame: state, family, hkl, kind, scans, mean misfit and
            size of the new facets.
        """
        n_scans = peaks.groupby(state).scan_key.nunique()
        new = peaks[peaks.reference_id < 0]
        grouped = new.groupby([state, "family", "hkl", "kind"]).agg(
            scans=("scan_key", "nunique"),
            misfit_deg=("misfit_deg", "mean"),
            count=("count", "mean")).reset_index()
        complete = grouped.scans == grouped[state].map(n_scans)
        repeated = grouped[state].map(n_scans) >= 2
        return grouped[complete & repeated]

    # ------------------------------------------------------------ plots

    @staticmethod
    def family_colour(family):
        """Plot colour of a family (grey if it has none)."""
        return FAMILY_COLOURS.get(family, "tab:gray")

    @staticmethod
    def compact(hkl):
        """'(1, 0, -2)' -> '(10-2)'."""
        return ("(" + "".join(hkl.strip("()").replace(" ", "").split(","))
                + ")")

    def plot_map(self, axes, facet_ids=None):
        """Outer surface coloured by facet family, three views.

        Args:
            axes (list): three axes (two side views and the top view).
            facet_ids (list): facets to colour (default: all).
        """
        label = self.assign()
        facet_ids = (range(len(self.facets)) if facet_ids is None
                     else facet_ids)
        colours = np.full(len(self.idx), "lightgray", dtype=object)
        for k in facet_ids:
            colours[label == k] = self.family_colour(
                self.facets[k]["family"])
        colours[label == -2] = "black"
        colours[label == -3] = "dimgray"
        views = ((2, 1, 0, "side view (x, y)"),
                 (0, 1, 2, "side view (z, y)"),
                 (2, 0, 1, "top view (x, z)"))
        for ax, (horizontal, vertical, depth, title) in zip(axes, views):
            order = np.argsort(self.idx[:, depth])
            ax.scatter(self.idx[order, horizontal],
                       self.idx[order, vertical], c=colours[order], s=3)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    @staticmethod
    def stereographic(vectors_cxi):
        """Stereographic coordinates as in cdiutils.analysis.pole_figure.

        Upper hemisphere along the vertical (xrayutilities z), in degrees
        from the centre; downward vectors are drawn at their opposite,
        since facet streaks are centrosymmetric.

        Args:
            vectors_cxi (np.ndarray): (n, 3) vectors, CXI array order.

        Returns:
            tuple: (x, y) arrays.
        """
        # CXI (z, y, x) -> xrayutilities (x, y, z): swap the last two
        # components, as Geometry.swap_convention does for one vector
        u = np.atleast_2d(vectors_cxi)[:, [0, 2, 1]].astype(float)
        u = u / np.linalg.norm(u, axis=1)[:, None]
        u = np.where(u[:, 2:3] < 0, -u, u)
        return -u[:, 0] / (1 + u[:, 2]) * 90, -u[:, 1] / (1 + u[:, 2]) * 90

    @staticmethod
    def _stereo_frame(ax):
        """Square axes with the equator circle."""
        ax.set_aspect("equal")
        ax.set_xlim(-92, 92)
        ax.set_ylim(-92, 92)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(plt.Circle((0, 0), 90, fill=False, lw=0.6, color="k"))

    def mark_facets(self, ax, facet_ids=None, annotate=False):
        """Circles at the facet normals on a stereographic plot."""
        facet_ids = (range(len(self.facets)) if facet_ids is None
                     else facet_ids)
        directions = self.facet_directions()
        for k in facet_ids:
            x, y = self.stereographic(directions[k])
            colour = self.family_colour(self.facets[k]["family"])
            ax.plot(x, y, "o", mfc="none", mec=colour, ms=9, mew=1.6)
            if annotate:
                name = "".join(str(i) for i in self.facets[k]["hkl"])
                ax.annotate(name, (x[0], y[0]), fontsize=6, xytext=(4, 3),
                            textcoords="offset points")

    def plot_normal_density(self, ax, facet_ids=None):
        """Density of the surface normals with the facets marked."""
        x, y = self.stereographic(self.normals)
        hist, edges_x, edges_y = np.histogram2d(
            x, y, bins=120, range=[[-90, 90], [-90, 90]])
        hist = ndimage.gaussian_filter(hist, 1.5)
        ax.pcolormesh(edges_x, edges_y, hist.T, cmap="Greys",
                      norm=PowerNorm(0.35, 0, hist.max()), shading="auto")
        self._stereo_frame(ax)
        self.mark_facets(ax, facet_ids, annotate=True)

    def plot_pole_figures(self, axes, intensity, q_grid, facet_ids=None,
                          fractions=(0.35, 0.6)):
        """Pole figures of the Bragg peak with the facet normals.

        Args:
            axes (list): one axis per shell.
            intensity (np.ndarray): orthogonalised intensity.
            q_grid (list): qx, qy, qz 1D grids (xrayutilities).
            facet_ids (list): facets to mark.
            fractions (tuple): shell radii as fractions of the half
                q range.
        """
        from cdiutils.analysis import pole_figure

        half_range = min(np.ptp(g) for g in q_grid) / 2
        for ax, fraction in zip(axes, fractions):
            (gx, gy, projected), (fig, _) = pole_figure(
                intensity, q_grid, axis="2", radius=fraction * half_range,
                dr=0.06 * half_range, resolution=200)
            plt.close(fig)
            positive = projected[projected > 0]
            norm = LogNorm(max(np.nanpercentile(positive, 5), 1e-3),
                           np.nanmax(projected))
            ax.pcolormesh(gx, gy, projected, norm=norm, cmap="turbo",
                          shading="auto")
            ax.set_title(f"|Δq| = {fraction * half_range:.3f} 1/Å",
                         fontsize=9)
            self._stereo_frame(ax)
            self.mark_facets(ax, facet_ids)

    @staticmethod
    def plot_orbits(ax, facets, summary, condition="condition"):
        """Facet strain grouped by symmetry orbit.

        Args:
            ax: matplotlib axis.
            facets (pd.DataFrame): concatenated statistics of the scans,
                with a condition column.
            summary (pd.DataFrame): output of summarise.
            condition (str): column used for the colours.
        """
        conditions = list(dict.fromkeys(facets[condition]))
        colours = dict(zip(conditions, CONDITION_COLOURS))
        used = facets[facets.facet_id.isin(summary.facet_id)]
        orbits = summary.groupby("orbit").angle_to_top.first().sort_values()
        for position, orbit in enumerate(orbits.index):
            members = summary[summary.orbit == orbit]
            for offset, member in enumerate(members.itertuples()):
                values = used[used.facet_id == member.facet_id]
                x = position + 0.12 * offset
                for name, group in values.groupby(condition, sort=False):
                    ax.errorbar(x, group.strain_surface.mean(),
                                yerr=group.strain_surface.std(), fmt="o",
                                ms=5, color=colours[name], capsize=2)
                ax.annotate(FacetDensityAnalyser.compact(member.hkl),
                            (x, member.strain), fontsize=6, xytext=(3, 3),
                            textcoords="offset points")
        labels = [f"{summary[summary.orbit == o].family.iloc[0]} "
                  f"{orbits[o]:.0f}°" for o in orbits.index]
        ax.set_xticks(range(len(orbits)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        for name in conditions:
            ax.plot([], [], "o", color=colours[name], label=name)
        ax.legend(fontsize=7)
        ax.set_ylabel("facet strain, outer layer (%)")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)

    @staticmethod
    def plot_series(ax, table, items, positions, labels, key="facet_id",
                    value="strain_surface", legend="hkl"):
        """A quantity of facets (or gap faces) through a series.

        Args:
            ax: matplotlib axis.
            table (pd.DataFrame): statistics with a 'position' column
                (order of the scan in the series).
            items (list): values of the key column to draw.
            positions (list): all positions of the series.
            labels (list): tick labels of the positions.
            key (str): column identifying an item ('facet_id' or 'side').
            value (str): column to draw.
            legend (str): column used for the legend.
        """
        for item in items:
            rows = table[table[key] == item].sort_values("position")
            if rows.empty:
                continue
            ax.plot(rows.position, rows[value], "o-", ms=3, lw=1,
                    label=str(rows[legend].iloc[0]).replace(" ", ""))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
