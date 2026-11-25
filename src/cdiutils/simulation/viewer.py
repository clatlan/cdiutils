"""
Interactive view of a diffractometer in reciprocal space.

:class:`DiffractometerViewer` shows, in the laboratory frame, the
reciprocal lattice of a :class:`~cdiutils.simulation.SupportedCrystal`
(and of its substrate) rotating with the sample circles, the Ewald
sphere, and where the detector circles put the detector on that
sphere. A node diffracts when it lies on the sphere; it is measured
when the detector points at it.

Conventions (xrayutilities, as used by the cdiutils ID01 geometry):
the beam is along +x, z is up, the sample circles are eta (incidence)
and phi (about the surface normal), the detector circles are nu
(horizontal) and delta (vertical). Q vectors are in 1/Angstrom with
the 2 pi factor.
"""

import numpy as np
import xrayutilities as xu
from matplotlib import pyplot as plt

from .epitaxy import three_index

VIEWS = {
    "3D": (22.0, -60.0),
    "side (beam left to right)": (0.0, -90.0),
    "top": (90.0, -90.0),
    "from the detector side (along the beam)": (0.0, 180.0),
}


def sample_rotation_matrix(eta: float, phi: float) -> np.ndarray:
    """
    Rotation from the sample frame to the laboratory frame.

    Args:
        eta: Incidence angle (deg), rotation about -y.
        phi: Rotation about the surface normal (deg).

    Returns:
        (3, 3) matrix M with Q_lab = M @ Q_sample.
    """
    eta_rad, phi_rad = np.radians(eta), np.radians(phi)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    cos_eta, sin_eta = np.cos(eta_rad), np.sin(eta_rad)
    about_normal = np.array(
        [[cos_phi, sin_phi, 0.0], [-sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
    )
    incidence = np.array(
        [[cos_eta, 0.0, -sin_eta], [0.0, 1.0, 0.0], [sin_eta, 0.0, cos_eta]]
    )
    return incidence @ about_normal


def detector_direction(nu: float, delta: float) -> np.ndarray:
    """
    Unit vector from the sample to the detector, laboratory frame.

    Args:
        nu: Horizontal detector angle (deg).
        delta: Vertical detector angle (deg).

    Returns:
        (3,) unit vector along the outgoing wavevector.
    """
    nu_rad, delta_rad = np.radians(nu), np.radians(delta)
    return np.array(
        [
            np.cos(delta_rad) * np.cos(nu_rad),
            -np.cos(delta_rad) * np.sin(nu_rad),
            np.sin(delta_rad),
        ]
    )


class DiffractometerViewer:
    """
    Reciprocal lattice, Ewald sphere and detector versus the angles.

    Args:
        sample: A :class:`~cdiutils.simulation.SupportedCrystal`.
        energy: Initial X-ray energy (eV).
        energy_range: (min, max) of the energy slider (eV). Nodes are
            computed once, up to ``|Q|`` = 2k at the highest energy.
        reflections: Crystal reflections offered by the "go to" menu
            and always labelled. Default: the 25 allowed nodes of
            lowest ``|Q|`` with non-negative (h, k, l).
        phi_offset: Motor phi at which the sample x axis (the
            in-plane reference direction of the crystal) lies along
            the beam; motor phi = computed phi + phi_offset.
        tolerance: Distance to the Ewald sphere (1/Angstrom) below
            which a node is drawn as diffracting.
        detector_distance: Sample-detector distance (m).
        detector_size: Detector side length (m); default a 516 pixel
            Maxipix (55 um pixels).
        show_substrate: Draw the substrate nodes.

    Example:
        >>> viewer = DiffractometerViewer(sample, energy=9000)
        >>> viewer.widget()          # in a notebook
        >>> fig = viewer.figure(eta=20.48, phi=180, nu=0, delta=90.08)
    """

    def __init__(
        self,
        sample,
        energy: float = 9000.0,
        energy_range: tuple = (6000.0, 13000.0),
        reflections: list | None = None,
        phi_offset: float = 0.0,
        tolerance: float = 0.03,
        detector_distance: float = 1.0,
        detector_size: float = 516 * 55e-6,
        show_substrate: bool = True,
    ):
        self.sample = sample
        self.energy = float(energy)
        self.energy_range = tuple(energy_range)
        self.phi_offset = phi_offset
        self.tolerance = tolerance
        self.detector_half_angle = np.degrees(
            np.arctan(detector_size / 2 / detector_distance)
        )

        q_max = 2 * self._wavevector(max(self.energy_range[1], energy))
        nodes, hkls, factors = sample._crystal_nodes(q_max, energy)
        nonzero = np.linalg.norm(nodes, axis=1) > 1e-9
        self.crystal_nodes = nodes[nonzero]
        self.crystal_hkls = [h for h, keep in zip(hkls, nonzero) if keep]
        self.crystal_factors = factors[nonzero]
        self.substrate_nodes = np.empty((0, 3))
        if show_substrate and sample.substrate is not None:
            nodes, _ = sample.substrate_nodes(q_max, energy)
            self.substrate_nodes = nodes[np.linalg.norm(nodes, axis=1) > 1e-9]

        if reflections is None:
            order = np.argsort(np.linalg.norm(self.crystal_nodes, axis=1))
            reflections = [
                self.crystal_hkls[index]
                for index in order
                if min(self.crystal_hkls[index]) >= 0
            ][:25]
        self.reflections = [three_index(hkl) for hkl in reflections]
        # crystal B matrix: columns are the reciprocal basis vectors
        self._b_matrix = np.column_stack(
            [sample.crystal.Q(*unit) for unit in np.eye(3, dtype=int)]
        )

    @staticmethod
    def _wavevector(energy: float) -> float:
        """Wavevector 2 pi / lambda (1/Angstrom) at an energy (eV)."""
        return 2 * np.pi / xu.en2lam(energy)

    # ------------------------------------------------------------------
    # computation
    # ------------------------------------------------------------------
    def state(
        self,
        eta: float,
        phi: float,
        nu: float,
        delta: float,
        energy: float | None = None,
    ) -> dict:
        """
        Everything the view shows, for one set of motor angles.

        Args:
            eta: Incidence motor angle (deg).
            phi: Motor phi (deg); the sample is rotated by
                phi - phi_offset.
            nu: Horizontal detector angle (deg).
            delta: Vertical detector angle (deg).
            energy: X-ray energy (eV); default the viewer's energy.

        Returns:
            Dictionary with the wavevector, the rotation matrix, node
            positions in the lab frame, masks of diffracting nodes,
            the Q vector at the detector centre (lab and sample
            frames), its fractional hkl, the nearest crystal
            reflection and its distance, 2theta, incidence and exit
            angles (deg).
        """
        energy = self.energy if energy is None else energy
        wavevector = self._wavevector(energy)
        rotation = sample_rotation_matrix(eta, phi - self.phi_offset)
        k_in = np.array([wavevector, 0.0, 0.0])
        k_out = wavevector * detector_direction(nu, delta)
        q_lab = k_out - k_in
        q_sample = rotation.T @ q_lab
        normal = rotation @ np.array([0.0, 0.0, 1.0])

        def on_sphere(nodes_lab):
            """Mask of nodes diffracting above the sample horizon."""
            if not len(nodes_lab):
                return np.zeros(0, dtype=bool)
            scattered = nodes_lab + k_in
            distance = np.abs(np.linalg.norm(scattered, axis=1) - wavevector)
            leaves_surface = scattered @ normal > 0
            return (distance < self.tolerance) & leaves_surface & (eta > 0)

        crystal_lab = self.crystal_nodes @ rotation.T
        substrate_lab = self.substrate_nodes @ rotation.T
        distances = np.linalg.norm(crystal_lab - q_lab, axis=1)
        nearest = int(np.argmin(distances))
        hkl = np.linalg.solve(
            self._b_matrix, self.sample.orientation.T @ q_sample
        )
        return {
            "energy": energy,
            "wavevector": wavevector,
            "rotation": rotation,
            "normal": normal,
            "k_out": k_out,
            "q_lab": q_lab,
            "q_sample": q_sample,
            "hkl": hkl,
            "crystal_lab": crystal_lab,
            "substrate_lab": substrate_lab,
            "crystal_visible": np.linalg.norm(self.crystal_nodes, axis=1)
            <= 2 * wavevector,
            "substrate_visible": np.linalg.norm(self.substrate_nodes, axis=1)
            <= 2 * wavevector
            if len(self.substrate_nodes)
            else np.zeros(0, dtype=bool),
            "crystal_diffracting": on_sphere(crystal_lab),
            "substrate_diffracting": on_sphere(substrate_lab),
            "nearest_hkl": self.crystal_hkls[nearest],
            "nearest_distance": float(distances[nearest]),
            "two_theta": float(
                np.degrees(np.arccos(np.clip(k_out[0] / wavevector, -1, 1)))
            ),
            "incidence": float(eta),
            "exit": float(
                np.degrees(
                    np.arcsin(np.clip(k_out @ normal / wavevector, -1, 1))
                )
            ),
        }

    def summary(self, state: dict) -> str:
        """
        One-paragraph text description of a state.

        Args:
            state: Output of :meth:`state`.

        Returns:
            Text with hkl at the detector, the nearest reflection and
            the beam angles.
        """
        h, k, l_index = state["hkl"]
        nearest = self.sample.label(state["nearest_hkl"])
        on_peak = state["nearest_distance"] < self.tolerance
        lines = [
            f"detector centre: H K L = {h:.3f} {k:.3f} {l_index:.3f}, "
            f"|Q| = {np.linalg.norm(state['q_lab']):.3f} 1/Å, "
            f"2θ = {state['two_theta']:.2f}°",
            f"nearest Ru reflection: {nearest}, "
            f"{state['nearest_distance']:.3f} 1/Å away"
            + ("  → ON THE PEAK" if on_peak else ""),
            f"incidence = {state['incidence']:.2f}°, "
            f"exit = {state['exit']:.2f}°"
            + ("  (exit below the surface!)" if state["exit"] <= 0 else ""),
        ]
        diffracting = [
            self.sample.label(hkl)
            for hkl, lit in zip(
                self.crystal_hkls, state["crystal_diffracting"]
            )
            if lit
        ]
        lines.append(
            "on the Ewald sphere: "
            + (", ".join(diffracting) if diffracting else "none")
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def figure(
        self,
        eta: float,
        phi: float,
        nu: float,
        delta: float,
        energy: float | None = None,
        view: str | tuple = "3D",
    ) -> plt.Figure:
        """
        Draw the lab-frame view for one set of angles.

        Args:
            eta, phi, nu, delta: Motor angles (deg).
            energy: X-ray energy (eV); default the viewer's energy.
            view: A key of :data:`VIEWS` or (elevation, azimuth).

        Returns:
            The figure.
        """
        state = self.state(eta, phi, nu, delta, energy)
        wavevector = state["wavevector"]
        centre = np.array([-wavevector, 0.0, 0.0])  # Ewald sphere centre
        fig = plt.figure(figsize=(9.5, 9.0))
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], projection="3d")
        # cubic view box around the Ewald sphere and the sample
        low = np.array([-2.1, -1.55, -0.9]) * wavevector
        high = np.array([1.0, 1.55, 2.2]) * wavevector

        def in_box(points):
            """Mask of the points inside the view box."""
            return np.all((points >= low) & (points <= high), axis=1)

        # Ewald sphere
        polar, azimuth = np.mgrid[0 : np.pi : 19j, 0 : 2 * np.pi : 37j]
        ax.plot_wireframe(
            centre[0] + wavevector * np.sin(polar) * np.cos(azimuth),
            wavevector * np.sin(polar) * np.sin(azimuth),
            wavevector * np.cos(polar),
            color="0.75",
            linewidth=0.3,
        )

        # sample surface (disc) and its normal
        radius = 0.45 * wavevector
        angle = np.linspace(0, 2 * np.pi, 60)
        ring = np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle), 0 * angle]
        )
        ring = ring @ state["rotation"].T
        ax.plot_trisurf(*ring.T, color="tab:brown", alpha=0.25, linewidth=0)
        ax.plot(*ring.T, color="tab:brown", lw=1)
        reference = state["rotation"] @ np.array([radius, 0.0, 0.0])
        ax.plot(*np.column_stack([[0, 0, 0], reference]), color="tab:brown")
        ax.text(*reference, " x$_{sample}$", color="tab:brown", fontsize=8)
        self._arrow(
            ax, [0, 0, 0], 1.3 * radius * state["normal"], "tab:brown", "n"
        )

        # nodes
        shown = state["substrate_visible"]
        lit = state["substrate_diffracting"]
        if len(lit):
            shown = shown & in_box(state["substrate_lab"])
            lit = lit & shown
            ax.scatter(
                *state["substrate_lab"][shown & ~lit].T,
                s=3,
                marker="s",
                c="0.6",
                alpha=0.35,
                depthshade=False,
                label="substrate nodes",
            )
            ax.scatter(
                *state["substrate_lab"][lit].T,
                s=45,
                marker="s",
                c="k",
                depthshade=False,
                label="substrate node on the sphere",
            )
        visible = state["crystal_visible"] & in_box(state["crystal_lab"])
        lit = state["crystal_diffracting"] & visible
        ax.scatter(
            *state["crystal_lab"][visible & ~lit].T,
            s=0.35 * self.crystal_factors[visible & ~lit],
            c="tab:blue",
            alpha=0.7,
            depthshade=False,
            label="crystal nodes",
        )
        ax.scatter(
            *state["crystal_lab"][lit].T,
            s=1.2 * self.crystal_factors[lit],
            c="tab:red",
            depthshade=False,
            label="crystal node on the sphere",
        )
        for index, hkl in enumerate(self.crystal_hkls):
            if lit[index] or (hkl in self.reflections and visible[index]):
                ax.text(
                    *state["crystal_lab"][index],
                    " " + self.sample.label(hkl),
                    fontsize=9,
                    fontweight="bold" if lit[index] else "normal",
                    color="tab:red" if lit[index] else "tab:blue",
                )

        # beams and detector
        self._arrow(ax, centre, -centre, "tab:orange", "k$_i$")
        self._arrow(ax, centre, state["k_out"], "tab:green", "k$_f$")
        ax.plot(*np.column_stack([[0, 0, 0], state["q_lab"]]), "k--", lw=1)
        half = self.detector_half_angle
        corners = [
            centre + wavevector * detector_direction(nu + dn, delta + dd)
            for dn, dd in (
                (-half, -half),
                (half, -half),
                (half, half),
                (-half, half),
                (-half, -half),
            )
        ]
        ax.plot(*np.array(corners).T, color="tab:green", lw=1.5)
        ax.scatter(
            *state["q_lab"],
            s=160,
            marker="D",
            facecolors="none",
            edgecolors="tab:green",
            linewidths=2,
            depthshade=False,
            label="detector centre",
        )

        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])
        ax.set_box_aspect((1, 1, 1), zoom=0.95)
        ax.set_xlabel("Q$_x$ lab, along the beam (1/Å)")
        ax.set_ylabel("Q$_y$ lab (1/Å)")
        ax.set_zlabel("Q$_z$ lab, up (1/Å)")
        elevation, azimuth_view = VIEWS.get(view, view)
        ax.view_init(elev=elevation, azim=azimuth_view)
        if elevation in (0.0, 90.0):
            # flat projection: hide the ticks of the axis seen end-on
            ax.set_proj_type("ortho")
            if elevation == 90.0:
                ax.set_zticks([])
                ax.set_zlabel("")
            elif azimuth_view % 180 == 90:
                ax.set_yticks([])
                ax.set_ylabel("")
            else:
                ax.set_xticks([])
                ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=8)
        fig.text(
            0.03,
            0.985,
            f"eta = {eta:.2f}°, phi = {phi:.2f}°, nu = {nu:.2f}°, "
            f"delta = {delta:.2f}°, {state['energy'] / 1000:.2f} keV\n"
            + self.summary(state),
            fontsize=10,
            va="top",
            family="monospace",
        )
        return fig

    @staticmethod
    def _arrow(ax, start, vector, colour, label):
        """Draw a labelled arrow in a 3D axes."""
        start, vector = np.asarray(start), np.asarray(vector)
        ax.quiver(
            *start, *vector, color=colour, arrow_length_ratio=0.08, linewidth=2
        )
        ax.text(*(start + 0.5 * vector), f" {label}", color=colour)

    # ------------------------------------------------------------------
    # widget
    # ------------------------------------------------------------------
    def widget(
        self,
        eta: float = 20.0,
        phi: float = 0.0,
        nu: float = 0.0,
        delta: float = 40.0,
    ):
        """
        Interactive viewer for a Jupyter notebook.

        Sliders drive the motor angles and the energy; the "go to"
        row sets them on a reflection in the chosen geometry
        (see :meth:`SupportedCrystal.reflection_angles`).

        Args:
            eta, phi, nu, delta: Initial motor angles (deg).

        Returns:
            An ipywidgets container; display it in a notebook cell.
        """
        import ipywidgets as widgets
        from IPython.display import display

        def slider(value, low, high, name, step=0.05):
            """Slider that only fires when released."""
            return widgets.FloatSlider(
                value=value,
                min=low,
                max=high,
                step=step,
                description=name,
                continuous_update=False,
                readout_format=".2f",
                layout=widgets.Layout(width="420px"),
            )

        controls = {
            "eta": slider(eta, 0.0, 90.0, "eta"),
            "phi": slider(phi, -180.0, 180.0, "phi"),
            "nu": slider(nu, -20.0, 130.0, "nu"),
            "delta": slider(delta, -5.0, 130.0, "delta"),
            "energy": slider(
                self.energy, *self.energy_range, "energy (eV)", step=10.0
            ),
        }
        view = widgets.Dropdown(options=list(VIEWS), description="view")
        labels = {self.sample.label(hkl): hkl for hkl in self.reflections}
        target = widgets.Dropdown(options=list(labels), description="go to")
        geometry = widgets.Dropdown(
            options=["coplanar", "symmetric", "fixed eta"],
            description="geometry",
        )
        go_button = widgets.Button(description="go", button_style="success")
        message = widgets.Label()
        output = widgets.Output()
        busy = {"value": False}

        def redraw(_change=None):
            """Redraw the figure for the current slider values."""
            if busy["value"]:
                return
            values = {
                name: control.value for name, control in controls.items()
            }
            with output:
                output.clear_output(wait=True)
                fig = self.figure(view=view.value, **values)
                display(fig)
                plt.close(fig)

        def go_to(_button):
            """Move the sliders onto the selected reflection."""
            hkl = labels[target.value]
            incidence = (
                controls["eta"].value
                if geometry.value == "fixed eta"
                else geometry.value
            )
            angles = self.sample.reflection_angles(
                hkl, controls["energy"].value, incidence
            )
            if np.isnan(angles["eta"]):
                message.value = (
                    f"{target.value} is not reachable in the "
                    f"{geometry.value} geometry at this energy."
                )
                return
            message.value = ""
            motor_phi = (angles["phi"] + self.phi_offset + 180) % 360 - 180
            busy["value"] = True
            controls["eta"].value = angles["eta"]
            controls["phi"].value = motor_phi
            controls["nu"].value = angles["nu"]
            controls["delta"].value = angles["delta"]
            busy["value"] = False
            redraw()

        for control in (*controls.values(), view):
            control.observe(redraw, names="value")
        go_button.on_click(go_to)
        redraw()
        return widgets.VBox(
            [
                widgets.HBox([target, geometry, go_button, message]),
                widgets.HBox(
                    [
                        widgets.VBox(list(controls.values()) + [view]),
                        output,
                    ]
                ),
            ]
        )
