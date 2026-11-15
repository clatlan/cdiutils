"""
Supported (epitaxial) crystals for BCDI experiment planning.

The :class:`SupportedCrystal` class describes a crystal sitting on a
single-crystal substrate (e.g. a dewetted metal particle on sapphire)
and answers the questions that come up when preparing a beamtime:

- where the Bragg peaks are (surface-frame and beamline angles) as a
  function of the energy,
- how close the substrate Bragg nodes are to each crystal node,
- what the reciprocal space of the stack looks like,
- what the equilibrium (Winterbottom) shape is, using wulffpack,
- how many particles a dewetted film gives per beam footprint,
- what a Bragg peak of a (strained) particle looks like.

All vectors are expressed in the *sample frame*: z along the surface
normal, x along the in-plane projection of a reference reciprocal
direction of the crystal. Reciprocal vectors include the 2 pi factor
and are in 1/Angstrom, as in xrayutilities.
"""

import textwrap

import numpy as np
import pandas as pd
import xrayutilities as xu
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist

from cdiutils.geometry import Geometry

from .objects import (
    add_displacement_field,
    get_winterbottom,
    make_from_wulffpack,
    simulate_diffraction,
)


def three_index(hkl: tuple) -> tuple[int, int, int]:
    """
    Convert Miller-Bravais (h, k, i, l) indices to (h, k, l).

    Args:
        hkl: 3-index or 4-index Miller indices.

    Returns:
        3-index Miller indices.

    Raises:
        ValueError: If hkl does not have 3 or 4 indices.
    """
    if len(hkl) == 4:
        return (hkl[0], hkl[1], hkl[3])
    if len(hkl) == 3:
        return tuple(hkl)
    raise ValueError("hkl must have 3 or 4 indices")


def miller_label(hkl: tuple, hexagonal: bool = False) -> str:
    """
    Compact label of a reflection, e.g. '(10-13)' or '(111)'.

    Args:
        hkl: 3-index or 4-index Miller indices.
        hexagonal: If True, print 4-index Miller-Bravais indices.

    Returns:
        The label.
    """
    h, k, l_index = three_index(hkl)
    indices = (h, k, -(h + k), l_index) if hexagonal else (h, k, l_index)
    return "(" + "".join(str(int(index)) for index in indices) + ")"


def orientation_matrix(
    material: xu.materials.Crystal,
    normal_hkl: tuple,
    inplane_hkl: tuple,
) -> np.ndarray:
    """
    Rotation from a crystal's Cartesian frame to the sample frame.

    The sample frame has z along the reciprocal direction
    `normal_hkl` (the surface normal) and x along the in-plane
    projection of the reciprocal direction `inplane_hkl`.

    Args:
        material: xrayutilities crystal.
        normal_hkl: Reciprocal-lattice direction along the surface
            normal, e.g. (0, 0, 1) for a c-plane hcp crystal.
        inplane_hkl: Reciprocal-lattice direction that defines the
            sample x axis.

    Returns:
        (3, 3) matrix U with q_sample = U @ q_crystal.
    """
    z_axis = material.Q(*three_index(normal_hkl))
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = material.Q(*three_index(inplane_hkl))
    x_axis = x_axis - (x_axis @ z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis])


class SupportedCrystal:
    """
    A crystal on a single-crystal substrate, for experiment planning.

    The epitaxy is given by two reciprocal directions per material:
    the one along the surface normal and the one along the sample x
    axis. For Ru(0001) on c-plane sapphire with
    Ru<1-210> // Al2O3<10-10>, the Ru (10-10)* and sapphire (11-20)*
    directions are parallel, so ``inplane_hkl=(1, 0, 0)`` and
    ``substrate_inplane_hkl=(1, 1, 0)``.

    Args:
        crystal: xrayutilities crystal of the particle.
        normal_hkl: Crystal reciprocal direction along the normal.
        inplane_hkl: Crystal reciprocal direction along sample x.
        substrate: xrayutilities crystal of the substrate, or None.
        substrate_normal_hkl: Substrate direction along the normal.
        substrate_inplane_hkl: Substrate direction along sample x.
        surface_energies: Facet energies for wulffpack (Miller
            indices -> energy), needed for the shape methods.
        primitive_structure: ASE Atoms of the crystal (point group
            for wulffpack), needed for the shape methods.
        interface_hkl: 3-index Miller indices of the interface facet
            (pointing into the substrate), e.g. (0, 0, -1).
        beamline: Beamline name for :class:`cdiutils.Geometry`, used
            to compute diffractometer angles. Default 'id01'.

    Example:
        >>> import xrayutilities as xu
        >>> ru = xu.materials.Crystal(
        ...     "Ru", xu.materials.SGLattice(
        ...         194, 2.705, 4.281, atoms=["Ru"], pos=["2c"]))
        >>> sample = SupportedCrystal(
        ...     ru, inplane_hkl=(1, 0, 0),
        ...     substrate=xu.materials.Al2O3,
        ...     substrate_inplane_hkl=(1, 1, 0))
        >>> sample.reflection_angles((1, 0, -1, 4), energy=9000)
    """

    def __init__(
        self,
        crystal: xu.materials.Crystal,
        normal_hkl: tuple = (0, 0, 1),
        inplane_hkl: tuple = (1, 0, 0),
        substrate: xu.materials.Crystal | None = None,
        substrate_normal_hkl: tuple = (0, 0, 1),
        substrate_inplane_hkl: tuple = (1, 0, 0),
        surface_energies: dict | None = None,
        primitive_structure=None,
        interface_hkl: tuple = (0, 0, -1),
        beamline: str = "id01",
    ):
        self.crystal = crystal
        self.substrate = substrate
        self.orientation = orientation_matrix(crystal, normal_hkl, inplane_hkl)
        self.substrate_orientation = None
        if substrate is not None:
            self.substrate_orientation = orientation_matrix(
                substrate, substrate_normal_hkl, substrate_inplane_hkl
            )
        self.surface_energies = surface_energies
        self.primitive_structure = primitive_structure
        self.interface_hkl = tuple(interface_hkl)
        self.beamline = beamline
        self.hexagonal = np.isclose(crystal.lattice.gamma, 120.0)
        self._qconversion = None
        self._substrate_nodes = {}

    # ------------------------------------------------------------------
    # reciprocal space
    # ------------------------------------------------------------------
    def q_vector(self, hkl: tuple) -> np.ndarray:
        """
        Q vector of a crystal reflection in the sample frame.

        Args:
            hkl: 3-index or 4-index Miller indices.

        Returns:
            (3,) Q vector in 1/Angstrom (x, y, z sample frame).
        """
        return self.orientation @ self.crystal.Q(*three_index(hkl))

    def label(self, hkl: tuple) -> str:
        """
        Label of a crystal reflection (4-index for hexagonal).

        Args:
            hkl: Miller indices.

        Returns:
            The label, e.g. '(10-14)'.
        """
        return miller_label(hkl, self.hexagonal)

    def substrate_nodes(
        self,
        q_max: float,
        energy: float,
        min_structure_factor: float = 1.0,
    ) -> tuple[np.ndarray, list]:
        """
        Allowed substrate reciprocal nodes with Q_z >= 0, |Q| <= q_max.

        Structure factors are computed in the substrate's own frame
        (before the epitaxial rotation), so extinctions are exact.

        Args:
            q_max: Largest |Q| (1/Angstrom).
            energy: X-ray energy (eV) for the structure factors.
            min_structure_factor: Nodes with a smaller |F| are
                treated as forbidden.

        Returns:
            Tuple (q_vectors (n, 3) in the sample frame, list of hkl).

        Raises:
            ValueError: If no substrate was given.
        """
        if self.substrate is None:
            raise ValueError("No substrate defined.")
        key = (round(q_max, 3), round(energy), min_structure_factor)
        if key in self._substrate_nodes:
            return self._substrate_nodes[key]

        basis_lengths = [
            np.linalg.norm(self.substrate.Q(*unit))
            for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ]
        ranges = [
            np.arange(-n, n + 1)
            for n in (int(np.ceil(q_max / b)) + 1 for b in basis_lengths)
        ]
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T
        q_crystal = np.array([self.substrate.Q(*hkl) for hkl in grid])
        q_sample = q_crystal @ self.substrate_orientation.T
        keep = (np.linalg.norm(q_sample, axis=1) <= q_max) & (
            q_sample[:, 2] >= -1e-9
        )
        structure_factors = np.abs(
            self.substrate.StructureFactorForQ(q_crystal[keep], energy)
        )
        allowed = structure_factors >= min_structure_factor
        nodes = (
            q_sample[keep][allowed],
            [tuple(int(v) for v in hkl) for hkl in grid[keep][allowed]],
        )
        self._substrate_nodes[key] = nodes
        return nodes

    def nearest_substrate_node(
        self, hkl: tuple, energy: float
    ) -> tuple[tuple, float]:
        """
        Nearest allowed substrate node to a crystal reflection.

        Args:
            hkl: Crystal Miller indices.
            energy: X-ray energy (eV).

        Returns:
            Tuple (substrate hkl, |Delta Q| in 1/Angstrom).
        """
        q = self.q_vector(hkl)
        nodes, labels = self.substrate_nodes(np.linalg.norm(q) + 1.0, energy)
        distances = np.linalg.norm(nodes - q, axis=1)
        index = int(np.argmin(distances))
        return labels[index], float(distances[index])

    # ------------------------------------------------------------------
    # angles
    # ------------------------------------------------------------------
    @property
    def qconversion(self) -> xu.experiment.QConversion:
        """xrayutilities QConversion of the beamline geometry."""
        if self._qconversion is None:
            geometry = Geometry.from_setup(self.beamline)
            geometry.cxi_to_xu()
            self._qconversion = xu.experiment.QConversion(
                sampleAxis=geometry.sample_circles,
                detectorAxis=geometry.detector_circles,
                r_i=geometry.beam_direction,
            )
        return self._qconversion

    def _diffractometer_angles(
        self, q: np.ndarray, energy: float, eta: float
    ) -> tuple[float, float, float]:
        """
        Solve (phi, nu, delta) for a Q vector at a given eta.

        Closed-form solution for the ID01 circles in xrayutilities
        convention (beam along x): sample eta about -y (outer) and
        phi about -z (the surface normal), detector nu about -z
        (outer) and delta about -y. Then
        k_f = k (cos d cos n, -cos d sin n, sin d) and
        Q_sample = Rz(phi) Ry(eta) Q_lab. Of the two mirror solutions
        the one with nu >= 0 is returned. The tests check the result
        against :attr:`qconversion`.

        Args:
            q: Target Q in the sample frame (1/Angstrom).
            energy: X-ray energy (eV).
            eta: Incidence angle (deg).

        Returns:
            Tuple (phi, nu, delta) in degrees (NaNs if unreachable).

        Raises:
            NotImplementedError: If the beamline circles differ from
                the ID01 ones.
        """
        if list(self.qconversion.sampleAxis) != ["y-", "z-"] or list(
            self.qconversion.detectorAxis
        ) != ["z-", "y-"]:
            raise NotImplementedError(
                "Angles are only implemented for ID01-like circles."
            )
        wavevector = 2 * np.pi / xu.en2lam(energy)
        eta_rad = np.radians(eta)
        q_norm_sq = q @ q
        q_inplane = np.hypot(q[0], q[1])
        # elastic condition: Q_lab,x = -|Q|^2 / (2k); with
        # u = Rz(-phi) Q_sample, Q_lab,x = cos(eta) u_x - sin(eta) q_z
        u_x = (-q_norm_sq / (2 * wavevector) + np.sin(eta_rad) * q[2]) / (
            np.cos(eta_rad)
        )
        if q_inplane < 1e-9:
            if abs(u_x) > 1e-6:
                return (np.nan, np.nan, np.nan)
            phi_candidates = [0.0]
        else:
            if abs(u_x) > q_inplane * (1 + 1e-9):
                return (np.nan, np.nan, np.nan)
            q_azimuth = np.arctan2(q[1], q[0])
            # clip: the coplanar case sits exactly at |u_x| = q_inplane
            half_angle = np.arccos(np.clip(u_x / q_inplane, -1.0, 1.0))
            phi_candidates = [q_azimuth + half_angle, q_azimuth - half_angle]

        solutions = []
        for phi in phi_candidates:
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            u = np.array(
                [
                    q[0] * cos_phi + q[1] * sin_phi,
                    -q[0] * sin_phi + q[1] * cos_phi,
                    q[2],
                ]
            )
            # Q_lab = Ry(-eta) u
            q_lab = np.array(
                [
                    np.cos(eta_rad) * u[0] - np.sin(eta_rad) * u[2],
                    u[1],
                    np.sin(eta_rad) * u[0] + np.cos(eta_rad) * u[2],
                ]
            )
            k_out = q_lab + np.array([wavevector, 0.0, 0.0])
            delta = np.degrees(
                np.arcsin(np.clip(k_out[2] / wavevector, -1, 1))
            )
            nu = np.degrees(np.arctan2(-k_out[1], k_out[0]))
            if abs(nu) > 90.0:
                # same k_f with |nu| <= 90 and delta past the vertical
                nu -= np.sign(nu) * 180.0
                delta = 180.0 - delta
            phi_deg = (np.degrees(phi) + 180.0) % 360.0 - 180.0
            solutions.append((phi_deg, nu, delta))
        valid = [s for s in solutions if s[1] > -1e-6] or solutions
        return min(valid, key=lambda s: abs(s[1]))

    def reflection_angles(
        self,
        hkl: tuple,
        energy: float,
        incidence: str | float = "symmetric",
    ) -> dict:
        """
        Angular position of a crystal reflection at one energy.

        Args:
            hkl: Crystal Miller indices.
            energy: X-ray energy (eV).
            incidence: 'symmetric' (incidence = exit angle, uses the
                in-plane detector arm), 'coplanar' (nu = 0, only if
                the reflection is inclined less than theta) or an
                incidence angle in degrees.

        Returns:
            Dictionary with d (A), |Q| (1/A), 2theta, psi (angle
            between Q and the normal), azimuth (in-plane angle of Q
            from sample x), incidence and exit angles, footprint
            factor (1/sin incidence) and the diffractometer angles
            eta, phi, nu, delta (deg). phi is relative to the sample
            x axis: align it on one measured reflection.
        """
        q = self.q_vector(hkl)
        q_norm = np.linalg.norm(q)
        q_inplane = np.hypot(q[0], q[1])
        wavevector = 2 * np.pi / xu.en2lam(energy)
        result = {
            "reflection": self.label(hkl),
            "energy (eV)": energy,
            "d (A)": 2 * np.pi / q_norm,
            "|Q| (1/A)": q_norm,
            "psi (deg)": np.degrees(np.arctan2(q_inplane, q[2])),
            "azimuth (deg)": (
                np.degrees(np.arctan2(q[1], q[0]))
                if q_inplane > 1e-9 * q_norm
                else np.nan
            ),
        }
        nan_keys = (
            "2theta (deg)",
            "incidence (deg)",
            "exit (deg)",
            "footprint",
            "eta",
            "phi",
            "nu",
            "delta",
        )
        if q_norm > 2 * wavevector:
            result.update({key: np.nan for key in nan_keys})
            return result

        theta = np.degrees(np.arcsin(q_norm / (2 * wavevector)))
        if incidence == "symmetric":
            alpha_in = np.degrees(np.arcsin(q[2] / (2 * wavevector)))
        elif incidence == "coplanar":
            alpha_in = theta - result["psi (deg)"]
        else:
            alpha_in = float(incidence)
        sin_exit = q[2] / wavevector - np.sin(np.radians(alpha_in))
        if alpha_in <= 0 or not -1 <= sin_exit <= 1:
            result.update({key: np.nan for key in nan_keys})
            result["2theta (deg)"] = 2 * theta
            return result

        phi, nu, delta = self._diffractometer_angles(q, energy, alpha_in)
        result.update(
            {
                "2theta (deg)": 2 * theta,
                "incidence (deg)": alpha_in,
                "exit (deg)": np.degrees(np.arcsin(sin_exit)),
                "footprint": 1 / np.sin(np.radians(alpha_in)),
                "eta": alpha_in,
                "phi": phi,
                "nu": nu,
                "delta": delta,
            }
        )
        return result

    def angles_vs_energy(
        self,
        hkls: list,
        energies: list,
        incidence: str | float | dict = "symmetric",
    ) -> pd.DataFrame:
        """
        Angular positions of several reflections at several energies.

        Args:
            hkls: List of crystal Miller indices.
            energies: List of energies (eV).
            incidence: Geometry for all reflections (see
                :meth:`reflection_angles`), or a dict hkl -> geometry
                (reflections missing from the dict use 'symmetric').

        Returns:
            DataFrame with one row per (reflection, energy).
        """
        rows = []
        for hkl in hkls:
            if isinstance(incidence, dict):
                geometry = incidence.get(tuple(hkl), "symmetric")
            else:
                geometry = incidence
            for energy in energies:
                row = self.reflection_angles(hkl, energy, geometry)
                row["geometry"] = (
                    geometry
                    if isinstance(geometry, str)
                    else f"incidence {geometry:g}°"
                )
                rows.append(row)
        columns = [
            "reflection",
            "geometry",
            "energy (eV)",
            "2theta (deg)",
            "eta",
            "phi",
            "nu",
            "delta",
            "exit (deg)",
            "footprint",
        ]
        return pd.DataFrame(rows)[columns].round(3)

    def reflection_table(
        self,
        hkls: list,
        energy: float,
        incidence: str | float = "symmetric",
        particle_sizes_nm: tuple = (100.0, 200.0),
        fringes: int = 10,
    ) -> pd.DataFrame:
        """
        Planning table: angles and distance to the substrate nodes.

        The 'margin' columns compare |Delta Q| to the nearest
        substrate node with a +-`fringes` window of a particle of each
        size (window = fringes x 2 pi / size): below 1 the bulk
        substrate Bragg node is inside the measured volume.

        Args:
            hkls: List of crystal Miller indices.
            energy: X-ray energy (eV).
            incidence: See :meth:`reflection_angles`.
            particle_sizes_nm: Particle sizes for the margin columns.
            fringes: Half-width of the BCDI window, in fringes.

        Returns:
            DataFrame with one row per reflection.
        """
        rows = []
        for hkl in hkls:
            row = self.reflection_angles(hkl, energy, incidence)
            coplanar = self.reflection_angles(hkl, energy, "coplanar")
            row["coplanar incidence (deg)"] = coplanar["incidence (deg)"]
            if self.substrate is not None:
                node, distance = self.nearest_substrate_node(hkl, energy)
                row["substrate node"] = miller_label(
                    node, np.isclose(self.substrate.lattice.gamma, 120.0)
                )
                row["|dQ| (1/A)"] = distance
                for size in particle_sizes_nm:
                    window = fringes * 2 * np.pi / (10 * size)
                    row[f"margin {size:.0f} nm"] = distance / window
            rows.append(row)
        table = pd.DataFrame(rows).drop(columns=["energy (eV)"])
        return table.round(3)

    def peak_shifts(
        self, strains: dict, hkls: list, energy: float
    ) -> pd.DataFrame:
        """
        Shift of 2theta (deg) produced by homogeneous strains.

        Uses d(2theta) = -2 strain tan(theta).

        Args:
            strains: Name -> strain along Q.
            hkls: Crystal Miller indices.
            energy: X-ray energy (eV).

        Returns:
            DataFrame (strain names x reflections) of 2theta shifts.
        """
        wavevector = 2 * np.pi / xu.en2lam(energy)
        table = {}
        for hkl in hkls:
            q_norm = np.linalg.norm(self.q_vector(hkl))
            theta = np.arcsin(q_norm / (2 * wavevector))
            table[self.label(hkl)] = {
                name: np.degrees(-2 * strain * np.tan(theta))
                for name, strain in strains.items()
            }
        return pd.DataFrame(table).round(4)

    # ------------------------------------------------------------------
    # plots of the stack
    # ------------------------------------------------------------------
    def _crystal_nodes(self, q_max: float, energy: float) -> tuple:
        """Allowed crystal nodes (sample frame) with Q_z >= 0."""
        basis_lengths = [
            np.linalg.norm(self.crystal.Q(*unit))
            for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ]
        ranges = [
            np.arange(-n, n + 1)
            for n in (int(np.ceil(q_max / b)) + 1 for b in basis_lengths)
        ]
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T
        q_crystal = np.array([self.crystal.Q(*hkl) for hkl in grid])
        q_sample = q_crystal @ self.orientation.T
        keep = (np.linalg.norm(q_sample, axis=1) <= q_max) & (
            q_sample[:, 2] >= -1e-9
        )
        factors = np.abs(
            self.crystal.StructureFactorForQ(q_crystal[keep], energy)
        )
        allowed = factors >= 1.0
        return (
            q_sample[keep][allowed],
            [tuple(int(v) for v in hkl) for hkl in grid[keep][allowed]],
            factors[allowed],
        )

    def plot_reciprocal_space_map(
        self,
        energy: float,
        azimuths_deg: tuple | None = None,
        highlight: list | None = None,
    ) -> plt.Figure:
        """
        Q_par-Q_z maps of crystal and substrate nodes in vertical planes.

        Args:
            energy: X-ray energy (eV); nodes beyond |Q| = 2k are left
                out.
            azimuths_deg: Azimuths (deg from sample x) of the planes.
                Default (0, 30) for hexagonal and (0, 45) otherwise.
            highlight: Crystal reflections to circle in red.

        Returns:
            The figure.
        """
        if azimuths_deg is None:
            azimuths_deg = (0.0, 30.0) if self.hexagonal else (0.0, 45.0)
        q_max = 2 * 2 * np.pi / xu.en2lam(energy)
        crystal = self._crystal_nodes(q_max, energy)
        substrate = None
        if self.substrate is not None:
            nodes, labels = self.substrate_nodes(q_max, energy)
            substrate = (nodes, labels)
        highlight = [three_index(h) for h in (highlight or [])]

        fig, axes = plt.subplots(
            1,
            len(azimuths_deg),
            figsize=(7 * len(azimuths_deg), 6.2),
            sharey=True,
            squeeze=False,
        )
        for ax, azimuth in zip(axes[0], azimuths_deg):
            direction = np.array(
                [np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))]
            )

            def in_plane(q_vectors):
                """Signed Q_par and a mask of nodes in this plane."""
                q_par = q_vectors[:, :2] @ direction
                q_perp = q_vectors[:, :2] @ [-direction[1], direction[0]]
                return q_par, np.abs(q_perp) < 1e-3

            if substrate is not None:
                q_par, mask = in_plane(substrate[0])
                ax.scatter(
                    q_par[mask],
                    substrate[0][mask, 2],
                    s=30,
                    marker="s",
                    c="0.6",
                    zorder=1,
                )
            q_par, mask = in_plane(crystal[0])
            for index in np.where(mask)[0]:
                hkl = crystal[1][index]
                ax.scatter(
                    q_par[index],
                    crystal[0][index, 2],
                    s=2 * crystal[2][index],
                    c="tab:blue",
                    zorder=2,
                )
                ax.annotate(
                    self.label(hkl),
                    (q_par[index], crystal[0][index, 2]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color="tab:blue",
                )
                if hkl in highlight:
                    ax.scatter(
                        q_par[index],
                        crystal[0][index, 2],
                        s=300,
                        facecolors="none",
                        edgecolors="tab:red",
                        linewidths=2,
                        zorder=3,
                    )
            angle = np.linspace(0, np.pi, 200)
            ax.plot(q_max * np.cos(angle), q_max * np.sin(angle), "k:", lw=1)
            ax.axvline(0, color="0.85", lw=4, zorder=0)
            ax.set_title(f"in-plane azimuth {azimuth:g}° from sample x")
            ax.set_xlabel("Q$_∥$ (1/Å)")
            ax.set_xlim(-q_max, q_max)
            ax.set_ylim(0, q_max)
        axes[0][0].set_ylabel("Q$_z$ (1/Å)")
        handles = [
            Patch(color="tab:blue", label="crystal (area ∝ |F|)"),
            Patch(color="0.85", label="specular rod"),
        ]
        if substrate is not None:
            handles.insert(1, Patch(color="0.6", label="substrate"))
        axes[0][-1].legend(handles=handles, loc="upper right", fontsize=9)
        fig.suptitle(f"{energy / 1000:g} keV, dotted: |Q| = 2k")
        return fig

    @staticmethod
    def table_figure(
        table: pd.DataFrame,
        highlight: dict | None = None,
        decimals: int = 2,
        font_size: int = 11,
    ) -> plt.Figure:
        """
        Render a DataFrame as a figure (for slides).

        Args:
            table: The table; the index is not shown.
            highlight: Value of the first column -> row colour.
            decimals: Rounding of float columns.
            font_size: Font size of the cells.

        Returns:
            The figure.
        """
        shown = table.copy()
        for column in shown.select_dtypes("number"):
            values = shown[column]
            # integer-valued columns (e.g. energies) without decimals
            places = 0 if np.allclose(values.dropna() % 1, 0) else decimals
            shown[column] = values.map(
                lambda value, places=places: (
                    "–" if pd.isna(value) else f"{value + 0.0:.{places}f}"
                ).replace("-0.00", "0.00")
            )
        fig, ax = plt.subplots(
            figsize=(1.35 * len(shown.columns), 0.42 * len(shown) + 0.9)
        )
        ax.axis("off")
        labels = [
            textwrap.fill(str(column).replace(" (", "\n("), 12)
            for column in shown.columns
        ]
        cells = ax.table(
            cellText=shown.astype(str).values,
            colLabels=labels,
            loc="center",
            cellLoc="center",
        )
        cells.auto_set_font_size(False)
        cells.set_fontsize(font_size)
        cells.scale(1, 1.7)
        highlight = highlight or {}
        first_column = list(shown.iloc[:, 0])
        for (row, _), cell in cells.get_celld().items():
            if row == 0:
                cell.set_facecolor("0.9")
                cell.set_height(cell.get_height() * 1.6)
            elif first_column[row - 1] in highlight:
                cell.set_facecolor(highlight[first_column[row - 1]])
        return fig

    @staticmethod
    def interface_misfit(
        crystal_spacing: float, substrate_spacing: float
    ) -> float:
        """
        Interface misfit (positive: crystal in tension if coherent).

        Args:
            crystal_spacing: In-plane nearest-neighbour distance of
                the crystal along the interface direction.
            substrate_spacing: Matching distance of the substrate.

        Returns:
            (substrate - crystal) / crystal.
        """
        return (substrate_spacing - crystal_spacing) / crystal_spacing

    @staticmethod
    def plot_interface_moire(
        crystal_spacing: float,
        substrate_spacing: float,
        half_width: float = 220.0,
    ) -> plt.Figure:
        """
        Coincidence (moiré) map of two aligned hexagonal nets.

        Each crystal atom is coloured by its lateral offset from the
        nearest substrate site; the pattern period equals the
        misfit-dislocation spacing b/f.

        Args:
            crystal_spacing: Crystal nearest-neighbour distance (A).
            substrate_spacing: Substrate site spacing (A).
            half_width: Half size of the square window (A).

        Returns:
            The figure.
        """

        def hexagonal_net(spacing):
            """Hexagonal net with one row along x inside the window."""
            n_max = int(2 * half_width / spacing) + 2
            i, j = np.meshgrid(
                np.arange(-n_max, n_max), np.arange(-n_max, n_max)
            )
            x = spacing * (i + 0.5 * j)
            y = spacing * np.sqrt(3) / 2 * j
            keep = (np.abs(x) < half_width) & (np.abs(y) < half_width)
            return np.column_stack([x[keep], y[keep]])

        crystal_net = hexagonal_net(crystal_spacing)
        offset, _ = cKDTree(hexagonal_net(substrate_spacing)).query(
            crystal_net
        )
        misfit = SupportedCrystal.interface_misfit(
            crystal_spacing, substrate_spacing
        )
        period = (
            crystal_spacing
            * substrate_spacing
            / abs(substrate_spacing - crystal_spacing)
        )
        fig, ax = plt.subplots(figsize=(7, 6.5))
        points = ax.scatter(*crystal_net.T, s=3, c=offset, cmap="viridis")
        fig.colorbar(points, ax=ax, label="lateral offset to substrate (Å)")
        ax.set_title(
            f"misfit {100 * misfit:.2f} % → coincidence period "
            f"{period / 10:.1f} nm"
        )
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_aspect("equal")
        return fig

    @staticmethod
    def plot_strain_budget(
        strains: dict, sensitivity: float = 1e-4
    ) -> plt.Figure:
        """
        Log-scale chart of strain magnitudes (values or ranges).

        Args:
            strains: Name -> strain, or (low, high) for a range. A
                range crossing zero is drawn from 1e-6.
            sensitivity: Reference line (e.g. BCDI strain
                sensitivity).

        Returns:
            The figure.
        """
        fig, ax = plt.subplots(figsize=(11, 0.9 * len(strains) + 1))
        for index, (name, value) in enumerate(strains.items()):
            low, high = np.broadcast_to(value, 2)
            magnitudes = sorted(np.abs([low, high]))
            if low * high < 0:
                magnitudes[0] = 1e-6
            if np.isclose(*magnitudes):
                ax.plot(magnitudes[1], index, "o", ms=11, color="tab:blue")
            else:
                ax.plot(
                    magnitudes,
                    [index, index],
                    lw=10,
                    solid_capstyle="butt",
                    color="tab:blue",
                )
            ax.annotate(
                f"{magnitudes[1]:.1e}",
                (magnitudes[1], index),
                xytext=(10, 6),
                textcoords="offset points",
            )
        ax.axvline(sensitivity, color="0.5", ls="--")
        ax.set_xscale("log")
        ax.set_yticks(range(len(strains)), list(strains))
        ax.set_xlabel("|strain|")
        ax.set_xlim(1e-6, 3e-2)
        ax.set_ylim(-0.6, len(strains) - 0.4)
        return fig

    # ------------------------------------------------------------------
    # shapes (wulffpack)
    # ------------------------------------------------------------------
    def _interface_facet_energy(self) -> float:
        """Surface energy of the facet parallel to the interface."""
        target = tuple(-index for index in self.interface_hkl)
        for key, energy in self.surface_energies.items():
            if three_index(key) == target:
                return energy
        raise ValueError(
            f"No surface energy for the facet {target} parallel to the "
            "interface."
        )

    def winterbottom(self, adhesion_ratio: float):
        """
        Winterbottom particle for a work of adhesion W_adh.

        Args:
            adhesion_ratio: W_adh / gamma_parallel, with gamma_parallel
                the energy of the facet parallel to the interface
                (0: free Wulff shape, 1: half crystal).

        Returns:
            wulffpack Winterbottom particle (interface along -z).

        Raises:
            ValueError: If surface energies or the primitive
                structure are missing.
        """
        if self.surface_energies is None or self.primitive_structure is None:
            raise ValueError(
                "surface_energies and primitive_structure are needed."
            )
        gamma_parallel = self._interface_facet_energy()
        # Dupre: gamma_interface - gamma_substrate = gamma_par - W_adh
        interface_energy = gamma_parallel * (1 - adhesion_ratio)
        if abs(interface_energy) < 1e-3 * gamma_parallel:
            # wulffpack needs the Wulff point strictly inside
            interface_energy = 1e-3 * gamma_parallel
        return get_winterbottom(
            self.surface_energies,
            self.interface_hkl,
            interface_energy,
            primitive_structure=self.primitive_structure,
        )

    @staticmethod
    def shape_metrics(particle, diameter: float | None = None) -> dict:
        """
        Height, diameter, volume and projected area of a particle.

        Args:
            particle: wulffpack particle (interface normal along -z).
            diameter: Wanted largest in-plane extent; None keeps the
                wulffpack units (Angstrom).

        Returns:
            Dictionary with height, diameter, aspect_ratio, volume and
            projected_area in the units of `diameter`.
        """
        vertices = np.vstack(
            [
                facet.vertices
                for form in particle.forms
                for facet in form.facets
            ]
        )
        width = pdist(vertices[:, :2]).max()
        scale = 1.0 if diameter is None else diameter / width
        height = np.ptp(vertices[:, 2])
        return {
            "height": height * scale,
            "diameter": width * scale,
            "aspect_ratio": height / width,
            "volume": ConvexHull(vertices).volume * scale**3,
            "projected_area": ConvexHull(vertices[:, :2]).volume * scale**2,
        }

    def _family_colours(self, n_main: int = 5) -> dict:
        """Colours of the main facet families of the free Wulff shape."""
        from wulffpack import SingleCrystal

        shares = SingleCrystal(
            self.surface_energies, primitive_structure=self.primitive_structure
        ).facet_fractions
        main = sorted(shares, key=shares.get, reverse=True)[:n_main]
        palette = [
            "tab:green",
            "tab:blue",
            "tab:orange",
            "tab:purple",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]
        colours = {family: "0.85" for family in self.surface_energies}
        colours.update(dict(zip(main, palette)))
        colours["interface"] = "0.35"
        return colours

    def plot_winterbottom(self, adhesion_ratios: tuple) -> plt.Figure:
        """
        3D Winterbottom shapes for several adhesion ratios.

        The interface (bottom) is not drawn: matplotlib's painter
        algorithm would draw it over the particle.

        Args:
            adhesion_ratios: W_adh / gamma_parallel values.

        Returns:
            The figure.
        """
        colours = self._family_colours()
        fig = plt.figure(figsize=(5 * len(adhesion_ratios), 5.2))
        for index, ratio in enumerate(adhesion_ratios):
            particle = self.winterbottom(ratio)
            ax = fig.add_subplot(
                1, len(adhesion_ratios), index + 1, projection="3d"
            )
            particle.make_plot(ax, alpha=1.0, colors=colours)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            interface_rgb = to_rgb(colours["interface"])
            for collection in list(ax.collections):
                face = collection.get_facecolor()
                if len(face) and np.allclose(face[0][:3], interface_rgb):
                    collection.remove()
            ax.view_init(elev=25, azim=-60)
            ax.set_axis_off()
            aspect = self.shape_metrics(particle)["aspect_ratio"]
            ax.set_title(f"W/γ$_∥$ = {ratio}: h/D = {aspect:.2f}")
        handles = [
            Patch(color=colour, label=str(family))
            for family, colour in colours.items()
            if colour not in ("0.85", "0.35")
        ]
        handles.append(Patch(color="0.85", label="minor families"))
        fig.legends.clear()
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles), fontsize=10
        )
        return fig

    def adhesion_scan(self, adhesion_ratios) -> pd.DataFrame:
        """
        Aspect ratio and facet shares versus the adhesion ratio.

        Facet shares are fractions of the free (non-interface)
        surface.

        Args:
            adhesion_ratios: W_adh / gamma_parallel values.

        Returns:
            DataFrame indexed by the adhesion ratio.
        """
        rows = []
        for ratio in adhesion_ratios:
            particle = self.winterbottom(ratio)
            shares = particle.facet_fractions
            free = 1 - shares.get("interface", 0.0)
            row = {
                "adhesion ratio": ratio,
                "h/D": self.shape_metrics(particle)["aspect_ratio"],
            }
            row.update(
                {
                    str(family): shares.get(family, 0.0) / free
                    for family in self.surface_energies
                }
            )
            rows.append(row)
        return pd.DataFrame(rows).set_index("adhesion ratio")

    def plot_adhesion_scan(self, adhesion_ratios) -> plt.Figure:
        """
        Plot h/D and the main facet shares versus the adhesion ratio.

        Args:
            adhesion_ratios: W_adh / gamma_parallel values.

        Returns:
            The figure.
        """
        scan = self.adhesion_scan(adhesion_ratios)
        colours = self._family_colours()
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.6))
        fig.subplots_adjust(wspace=0.3)
        left.plot(scan.index, scan["h/D"], "k-")
        left.set_xlabel("W$_{adh}$ / γ$_∥$")
        left.set_ylabel("h / D")
        for family, colour in colours.items():
            if colour in ("0.85", "0.35"):
                continue
            right.plot(
                scan.index,
                100 * scan[str(family)],
                color=colour,
                label=str(family),
            )
        right.set_xlabel("W$_{adh}$ / γ$_∥$")
        right.set_ylabel("share of free surface (%)")
        right.legend(fontsize=9)
        return fig

    def particles_from_films(
        self,
        thicknesses_nm: tuple,
        adhesion_ratios: tuple,
        size_factor: float = 10.0,
        beam_sizes_nm: tuple = (200.0, 500.0),
        incidence_deg: float = 90.0,
    ) -> pd.DataFrame:
        """
        Particles from dewetted films, by volume conservation.

        Each film of thickness t gives Winterbottom particles of
        diameter D = size_factor x t; one particle per V / t of
        substrate area.

        Args:
            thicknesses_nm: Film thicknesses.
            adhesion_ratios: W_adh / gamma_parallel values.
            size_factor: D / t.
            beam_sizes_nm: Square beam sizes for the particle count.
            incidence_deg: Incidence angle; the footprint is
                beam^2 / sin(incidence).

        Returns:
            DataFrame with height, volume, density, spacing, coverage
            and particles per footprint.
        """
        rows = []
        for thickness in thicknesses_nm:
            diameter = size_factor * thickness
            for ratio in adhesion_ratios:
                metrics = self.shape_metrics(
                    self.winterbottom(ratio), diameter
                )
                area_per_particle = metrics["volume"] / thickness  # nm^2
                density = 1e6 / area_per_particle  # per um^2
                row = {
                    "t (nm)": thickness,
                    "D (nm)": diameter,
                    "W/gamma": ratio,
                    "h (nm)": metrics["height"],
                    "V (1e6 nm3)": metrics["volume"] / 1e6,
                    "density (um-2)": density,
                    "spacing (nm)": np.sqrt(area_per_particle),
                    "coverage (%)": 100
                    * metrics["projected_area"]
                    / area_per_particle,
                }
                for beam in beam_sizes_nm:
                    footprint = beam**2 / np.sin(np.radians(incidence_deg))
                    row[f"in {beam:.0f} nm beam"] = density * footprint / 1e6
                rows.append(row)
        return pd.DataFrame(rows).round(2)

    # ------------------------------------------------------------------
    # Bragg peak simulation
    # ------------------------------------------------------------------
    def simulate_bragg_peak(
        self,
        hkl: tuple,
        diameter_nm: float = 100.0,
        adhesion_ratio: float = 1.0,
        voxel_nm: float = 2.0,
        shape: tuple = (128, 128, 128),
        base_strain: float = 0.0,
        decay_nm: float = 25.0,
        poisson_factor: float = 0.86,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Diffraction around a Bragg node of a Winterbottom particle.

        Optional pinned-base strain toy model: in-plane strain
        `base_strain` at the interface gives
        eps_zz(z) = -poisson_factor * base_strain * exp(-z / decay_nm)
        (z: height above the interface); only u_z is applied, so the
        phase is Q_z u_z.

        Args:
            hkl: Crystal Miller indices (sets Q_z for the phase).
            diameter_nm: Largest in-plane extent of the particle.
            adhesion_ratio: W_adh / gamma_parallel.
            voxel_nm: Voxel size.
            shape: Array shape (z, y, x); z is the interface normal.
            base_strain: In-plane strain at the interface.
            decay_nm: Decay length of the strain with height.
            poisson_factor: -eps_zz / eps_xx (2 nu / (1 - nu) for
                isotropic biaxial strain).

        Returns:
            Tuple (complex object, intensity), array axes (z, y, x) in
            the wulffpack frame (z along the interface normal).
        """
        particle = self.winterbottom(adhesion_ratio)
        obj = make_from_wulffpack(shape, particle, diameter_nm / voxel_nm)
        if base_strain:
            occupied = np.where(obj.sum(axis=(1, 2)) > 0)[0]
            height = np.clip(
                (np.arange(shape[0]) - occupied[0]) * voxel_nm, 0, None
            )
            eps_zz = -poisson_factor * base_strain
            u_z = eps_zz * decay_nm * (1 - np.exp(-height / decay_nm))
            displacement = np.zeros((3,) + tuple(shape))
            displacement[0] = u_z[:, None, None]
            # Q in 1/nm without the 2 pi factor
            q_z = self.q_vector(hkl)[2] * 10 / (2 * np.pi)
            obj = add_displacement_field(obj, displacement, (q_z, 0.0, 0.0))
        return obj, simulate_diffraction(obj)

    @staticmethod
    def plot_bragg_peaks(
        simulations: dict, voxel_nm: float = 2.0, half_window: int = 32
    ) -> plt.Figure:
        """
        Central intensity slices and phase of simulated particles.

        Args:
            simulations: Label -> (object, intensity) from
                :meth:`simulate_bragg_peak`.
            voxel_nm: Voxel size, for the phase axes.
            half_window: Half size (pixels) of the intensity slices.

        Returns:
            The figure (one row per simulation).
        """
        fig, axes = plt.subplots(
            len(simulations),
            3,
            figsize=(15, 4.4 * len(simulations)),
            squeeze=False,
            layout="constrained",
        )
        for row, (label, (obj, intensity)) in zip(axes, simulations.items()):
            centre = np.array(intensity.shape) // 2
            window = slice(centre[2] - half_window, centre[2] + half_window)
            vertical = intensity[
                centre[0] - half_window : centre[0] + half_window,
                centre[1],
                window,
            ]
            horizontal = intensity[
                centre[0],
                centre[1] - half_window : centre[1] + half_window,
                window,
            ]
            for ax, image, title in (
                (row[0], vertical, "Q$_z$–Q$_x$"),
                (row[1], horizontal, "Q$_y$–Q$_x$ (in plane)"),
            ):
                ax.imshow(
                    np.log10(image + 1e-6 * image.max()),
                    cmap="turbo",
                    origin="lower",
                )
                ax.set_title(f"{label}: {title}")
                ax.set_xlabel("pixels")
            section = obj[:, centre[1], :]
            extent = np.array([-1, 1, -1, 1]) * obj.shape[0] * voxel_nm / 2
            image = row[2].imshow(
                np.ma.masked_where(np.abs(section) == 0, np.angle(section)),
                cmap="RdBu_r",
                origin="lower",
                extent=extent,
            )
            occupied = np.argwhere(np.abs(obj) > 0)
            half_size = (np.ptp(occupied, axis=0).max() + 8) * voxel_nm / 2
            row[2].set_xlim(-half_size, half_size)
            row[2].set_ylim(-half_size / 1.5, half_size / 1.5)
            row[2].set_title(f"{label}: phase, side view")
            row[2].set_xlabel("x (nm)")
            row[2].set_ylabel("z (nm)")
            fig.colorbar(image, ax=row[2], label="phase (rad)", shrink=0.7)
        return fig
