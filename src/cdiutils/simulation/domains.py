"""
Domain models of a reconstructed crystal, fitted to measured intensity.

A Bragg coherent diffraction pattern does not always identify a single
real-space object: a strongly deformed crystal and a set of domains
separated by phase steps can both reproduce a split or streaked Bragg
peak, and phase retrieval from random starts may land on either. This
module tests such interpretations directly against the data. A
crystal is described by a few uniform-density domains, each with its
own complex amplitude and a linear (and optionally quadratic) phase,
and the model intensity is fitted to the measured intensity with a
Poisson deviance, in the detector frame of the phasing.

Functions:
    segment_domains: split a reconstruction into domains separated by
        low-modulus planes and share its filled outline between them.
    fill_domain_gaps: fill only the planes between domains.
    measure_domain_phases: linear ramp, constant phase and coherence of
        each domain of a complex object.
    fit_quadratic_phase: linear + quadratic phase of a region (bent or
        graded crystal), robust to phase wrapping.

Classes:
    DomainModelFit: fit models with phase steps ("A"), tilts ("B"),
        neither ("C"), both ("full") or a quadratic phase ("bend").

Notes:
    The intensity follows the PyNX convention,
    ``|fftshift(fftn(ifftshift(obj)))|**2``, so that a model can be
    compared voxel by voxel with the ``pynx_input_data`` array of a
    phasing and a reconstruction mode taken from the same phasing.
    Coordinates are centred on ``n // 2`` along each axis, as in
    :func:`~cdiutils.simulation.add_linear_phase`.

See Also:
    :doc:`/user_guide/domain_model` for the physical background and the
    ``domain_model_fit.ipynb`` example notebook.
"""

import numpy as np
from scipy.fft import fftn, fftshift, ifftshift

from .objects import _get_centred_coordinates


def fill_domain_gaps(seeds: np.ndarray, width: float = 3.0) -> np.ndarray:
    """
    Fill only the gaps between domains, leaving their outer surface
    unchanged: a voxel outside every seed is added when two different
    seeds are both within ``width`` voxels of it, and given to the
    nearest one.

    Args:
        seeds: Integer domain labels (0 = outside), e.g. the ``seeds``
            output of :func:`segment_domains`.
        width: Largest distance (voxels) to each of the two seeds.

    Returns:
        Integer labels: the seeds plus their filled gaps.
    """
    from scipy import ndimage

    count = int(seeds.max())
    distances = np.stack(
        [
            ndimage.distance_transform_edt(seeds != label)
            for label in range(1, count + 1)
        ]
    )
    ordered = np.sort(distances, axis=0)
    gap = (seeds == 0) & (ordered[0] <= width)
    if count > 1:
        gap &= ordered[1] <= width
    # inside the closing of the union only: fills the planes between
    # domains, not the rim just outside their surface
    structure = ndimage.generate_binary_structure(seeds.ndim, seeds.ndim)
    gap &= ndimage.binary_closing(
        seeds > 0, structure=structure, iterations=int(np.ceil(width))
    )
    filled = seeds.copy()
    filled[gap] = np.argmin(distances, axis=0)[gap] + 1
    return filled.astype(np.int32)


def segment_domains(
    modulus: np.ndarray,
    seed_threshold: float = 0.25,
    outline_threshold: float = 0.2,
    closing_iterations: int = 3,
    min_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a reconstructed object into domains separated by
    low-modulus planes, and share its filled outline between them.

    A phase step between two parts of a crystal makes a band-limited
    reconstruction drop in modulus at their boundary (the modulus goes
    as ``|cos(step / 2)|``), so the domains appear as blobs separated by
    "empty planes". The blobs above ``seed_threshold`` are the domain
    seeds. The outline is the modulus above ``outline_threshold``,
    closed and with its holes filled, so that it bridges the low-modulus
    planes. Every outline voxel is given to the nearest seed.

    Args:
        modulus: Modulus of the reconstructed object (any scale).
        seed_threshold: Relative modulus (fraction of the maximum)
            above which connected voxels form a domain seed.
        outline_threshold: Relative modulus of the particle outline.
        closing_iterations: Binary closing iterations of the outline
            (bridges the low-modulus planes between domains).
        min_size: Seeds with fewer voxels are ignored.

    Returns:
        (domains, seeds): integer arrays of the modulus shape. 0 is
        outside, 1..n the domains, numbered by decreasing seed size.
        ``domains`` covers the filled outline, ``seeds`` only the
        seed blobs (the "gaps as voids" geometry).

    Note:
        The closing also changes the outer surface of the outline. To
        compare "gaps filled" with "gaps as voids" at a fixed outer
        surface, fill ``seeds`` with :func:`fill_domain_gaps` instead.

    Example:
        >>> domains, seeds = segment_domains(np.abs(mode),
        ...                                  seed_threshold=0.25)
        >>> int(domains.max())  # number of domains
        3
    """
    from scipy import ndimage

    relative = modulus / modulus.max()
    labels, count = ndimage.label(relative > seed_threshold)
    sizes = np.bincount(labels.ravel())[1:]
    order = [k + 1 for k in np.argsort(sizes)[::-1] if sizes[k] >= min_size]
    seeds = np.zeros(modulus.shape, dtype=np.int32)
    for number, label in enumerate(order, start=1):
        seeds[labels == label] = number
    outline = ndimage.binary_closing(
        relative > outline_threshold, iterations=closing_iterations
    )
    outline = ndimage.binary_fill_holes(outline | (seeds > 0))
    # index of the nearest seed voxel for every voxel
    _, nearest = ndimage.distance_transform_edt(
        seeds == 0, return_indices=True
    )
    domains = seeds[tuple(nearest)] * outline
    return domains.astype(np.int32), seeds


def measure_domain_phases(obj: np.ndarray, domains: np.ndarray) -> list[dict]:
    """
    Linear phase ramp and constant phase of every domain of a complex
    object.

    The ramp of a domain is the mean phase difference between
    neighbouring voxels along each axis (modulus-weighted, insensitive
    to phase wrapping). Its offset is the phase left after removing the
    ramp, with coordinates centred on the array (:func:`add_linear_phase`
    convention), so offsets of domains that share a ramp are directly
    their phase steps. The coherence ``|sum(z)| / sum(|z|)`` is 1 when the
    domain is exactly a ramp plus a constant.

    Args:
        obj: Complex object.
        domains: Integer domain labels (0 = outside).

    Returns:
        One dict per domain: label, size, ramp (rad / voxel per axis),
        offset (rad), coherence, mean_modulus (relative to the maximum).
    """
    coordinates = _get_centred_coordinates(obj.shape)
    modulus = np.abs(obj) / np.abs(obj).max()
    results = []
    for label in range(1, int(domains.max()) + 1):
        inside = domains == label
        ramp = []
        for axis in range(obj.ndim):
            product = obj * np.conj(np.roll(obj, 1, axis=axis))
            pairs = inside & np.roll(inside, 1, axis=axis)
            ramp.append(float(np.angle(product[pairs].sum())))
        phase = sum(ramp[i] * coordinates[i] for i in range(obj.ndim))
        flattened = (obj * np.exp(-1j * phase))[inside]
        results.append(
            {
                "label": label,
                "size": int(inside.sum()),
                "ramp": ramp,
                "offset": float(np.angle(flattened.sum())),
                "coherence": float(
                    np.abs(flattened.sum()) / np.abs(flattened).sum()
                ),
                "mean_modulus": float(modulus[inside].mean()),
            }
        )
    return results


# order of the quadratic phase coefficients (rad / voxel^2):
# phase = sum_i g_i x_i + sum_(i<=j) h_ij x_i x_j
QUADRATIC_TERMS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _phase_field(coordinates, ramp, quadratic=None):
    """
    Linear (+ quadratic) phase on centred coordinates.

    Args:
        coordinates: Output of :func:`_centred_coordinates`.
        ramp: Three linear coefficients (rad / voxel).
        quadratic: Six quadratic coefficients in the order of
            ``QUADRATIC_TERMS`` (rad / voxel^2), or None.

    Returns:
        Phase array (broadcast to the full shape).
    """
    phase = sum(ramp[i] * coordinates[i] for i in range(3))
    if quadratic is not None:
        for value, (i, j) in zip(quadratic, QUADRATIC_TERMS):
            phase = phase + value * coordinates[i] * coordinates[j]
    return phase


def fit_quadratic_phase(obj: np.ndarray, region: np.ndarray) -> tuple:
    """
    Linear + quadratic phase that best describes a complex object inside
    a region (a bent or graded crystal), by least squares on the local
    phase differences between neighbouring voxels (insensitive to phase
    wrapping): d(phase)/dx_a = g_a + sum_b dh/dx_a terms.

    Args:
        obj: Complex object.
        region: Boolean region of the fit.

    Returns:
        (ramp, quadratic): three linear and six quadratic coefficients
        (``QUADRATIC_TERMS`` order), with coordinates centred on the
        array (:func:`_centred_coordinates`).
    """
    coordinates = _get_centred_coordinates(obj.shape)
    rows, values = [], []
    for axis in range(3):
        product = obj * np.conj(np.roll(obj, 1, axis=axis))
        pairs = region & np.roll(region, 1, axis=axis)
        weights = np.abs(product)[pairs]
        # midpoint between the voxel and its neighbour along this axis
        midpoint = [
            np.broadcast_to(c, obj.shape)[pairs].astype(float)
            for c in coordinates
        ]
        midpoint[axis] = midpoint[axis] - 0.5
        design = np.zeros((weights.size, 9))
        design[:, axis] = 1.0
        for column, (i, j) in enumerate(QUADRATIC_TERMS):
            # derivative of h_ij x_i x_j along this axis
            if i == j == axis:
                design[:, 3 + column] = 2 * midpoint[axis]
            elif i == axis:
                design[:, 3 + column] = midpoint[j]
            elif j == axis:
                design[:, 3 + column] = midpoint[i]
        rows.append(design * np.sqrt(weights)[:, None])
        values.append(np.angle(product[pairs]) * np.sqrt(weights))
    solution, *_ = np.linalg.lstsq(
        np.vstack(rows), np.concatenate(values), rcond=None
    )
    return solution[:3], solution[3:]


class DomainModelFit:
    """
    Fit the measured Bragg intensity with a crystal made of domains of
    uniform density, each with its own complex amplitude and linear
    phase ramp, in the detector frame of the phasing (PyNX convention:
    intensity = ``|fftshift(fftn(ifftshift(object)))|**2``).

    Model intensity: ``|sum_k c_k FFT(rho_k exp(i g_k . r))|**2 + b``, with
    rho_k the domain shapes, c_k = a_k exp(i phi_k) and b a constant
    background. Models:

    * "C": one common ramp, no phase steps (phi_k = 0), free a_k.
    * "A": one common ramp, free phase steps phi_k (phi_1 = 0).
    * "B": no phase steps, free ramp per domain (tilted domains).
    * "full": free phase steps and free ramps.
    * "bend": no phase steps, free ramp and free quadratic phase per
      domain (smooth bending or d-spacing gradient; with one domain, a
      continuous crystal).

    ``equal_amplitudes=True`` forces one a for all domains (density
    equal in every domain). The fit minimises the Poisson deviance on
    the unmasked voxels of ``region``.

    Compare models by their Poisson deviance on the same data and
    mask; a model with more parameters must lower it by much more than
    its extra parameter count to be preferred. Check the selection on a
    known answer first (simulate each model with the real mask and photon
    count, and refit).

    Args:
        data: Measured intensity (photons), detector frame.
        mask: Non-zero for masked voxels.
        domains: Integer domain labels, same shape as data.
        region: Tuple of slices where the deviance is computed
            (default: the whole array).

    Example:
        >>> from cdiutils.simulation import (
        ...     DomainModelFit, measure_domain_phases, segment_domains)
        >>> domains, seeds = segment_domains(np.abs(mode))
        >>> phases = measure_domain_phases(mode, seeds)
        >>> ramps = np.array([p["ramp"] for p in phases])
        >>> fitter = DomainModelFit(iobs, mask, domains)
        >>> result = fitter.fit("A", ramps=ramps)
        >>> model = fitter.model_intensity(result)
    """

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        domains: np.ndarray,
        region: tuple = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.domains = domains
        self.count = int(domains.max())
        self.region = region or tuple(slice(None) for _ in data.shape)
        self.valid = (np.asarray(mask) == 0)[self.region]
        self.observed = self.data[self.region][self.valid]
        self.coordinates = _get_centred_coordinates(data.shape)

    def domain_fields(
        self, ramps: np.ndarray, quadratics: np.ndarray = None
    ) -> np.ndarray:
        """
        Diffracted field of every domain (unit density), on the valid
        voxels of the region.

        Args:
            ramps: (n_domains, 3) phase ramps in rad / voxel.
            quadratics: (n_domains, 6) quadratic phase coefficients
                (``QUADRATIC_TERMS`` order, rad / voxel^2), or None.

        Returns:
            (n_domains, n_valid) complex array.
        """
        fields = []
        for label in range(1, self.count + 1):
            phase = _phase_field(
                self.coordinates,
                ramps[label - 1],
                None if quadratics is None else quadratics[label - 1],
            )
            density = (self.domains == label) * np.exp(1j * phase)
            field = fftshift(fftn(ifftshift(density), workers=-1))
            fields.append(field[self.region][self.valid])
        return np.array(fields)

    def deviance(self, model: np.ndarray) -> float:
        """
        Poisson deviance between the model and the observed intensity.

        Args:
            model: Model intensity on the valid voxels of the region.

        Returns:
            2 * sum(m - d + d ln(d / m)).
        """
        model = np.maximum(model, 1e-12)
        observed = self.observed
        positive = observed > 0
        log_term = np.zeros_like(observed)
        log_term[positive] = observed[positive] * np.log(
            observed[positive] / model[positive]
        )
        return float(2 * np.sum(model - observed + log_term))

    def intensity(
        self,
        fields: np.ndarray,
        amplitudes: np.ndarray,
        phases: np.ndarray,
        background: float,
    ) -> np.ndarray:
        """
        Model intensity on the valid voxels of the region.

        Args:
            fields: Output of :meth:`domain_fields`.
            amplitudes: a_k, one per domain.
            phases: phi_k, one per domain (rad).
            background: Constant background b (photons / voxel).

        Returns:
            Model intensity, same length as the observed vector.
        """
        coefficients = amplitudes * np.exp(1j * phases)
        return np.abs(coefficients @ fields) ** 2 + background

    def fit(
        self,
        model: str = "A",
        ramps: np.ndarray = None,
        equal_amplitudes: bool = False,
        n_starts: int = 8,
        seed: int = 0,
        maxiter: int = 400,
        quadratics: np.ndarray = None,
    ) -> dict:
        """
        Fit one model.

        Args:
            model: "A", "B", "C", "full" or "bend" (see the class
                docstring).
            ramps: (n_domains, 3) starting ramps in rad / voxel. Models
                A and C use their mean (weighted by domain size) for
                every domain; B and full start from them. Default 0.
            equal_amplitudes: One amplitude for all domains.
            n_starts: Random starts of the phase steps (A, full); B and
                C use one start.
            seed: Seed of the random starts.
            maxiter: Iterations of each L-BFGS-B / Powell run.
            quadratics: (n_domains, 6) starting quadratic coefficients
                for "bend" (e.g. from :func:`fit_quadratic_phase`).
                Default 0.

        Returns:
            dict: model, deviance, n_parameters, amplitudes, phases
            (rad, phases[0] = 0), ramps, quadratics (``None`` unless
            "bend"), background, start deviances.

        Raises:
            ValueError: If ``model`` is not one of the five models.

        Example:
            >>> fitter = DomainModelFit(data, mask, domains)
            >>> steps = fitter.fit("A", ramps=ramps, n_starts=8)
            >>> tilts = fitter.fit("B", ramps=ramps)
            >>> steps["deviance"] < tilts["deviance"]
            True
        """
        from scipy.optimize import minimize

        if model not in ("A", "B", "C", "full", "bend"):
            raise ValueError(
                f"model must be 'A', 'B', 'C', 'full' or 'bend', got {model!r}"
            )
        count = self.count
        sizes = np.bincount(self.domains.ravel())[1 : count + 1]
        ramps = np.zeros((count, 3)) if ramps is None else np.array(ramps)
        if model in ("A", "C"):
            common = (sizes[:, None] * ramps).sum(0) / sizes.sum()
            ramps = np.repeat(common[None], count, axis=0)
        free_phases = model in ("A", "full")
        free_ramps = model in ("B", "full", "bend")
        free_quadratics = model == "bend"
        quadratics = (
            np.zeros((count, 6))
            if quadratics is None
            else np.array(quadratics, dtype=float)
        )
        # quadratic coefficients are optimised in units of 1e-3 rad/voxel^2
        # so that all parameters have comparable steps
        quadratic_unit = 1e-3
        n_amplitudes = 1 if equal_amplitudes else count
        fields = self.domain_fields(ramps)
        # starting amplitude: total measured / total unit-density model
        scale = np.sqrt(
            max(self.observed.sum(), 1.0)
            / max((np.abs(fields.sum(0)) ** 2).sum(), 1e-12)
        )

        def unpack(parameters):
            """Split the parameter vector into the model quantities."""
            position = 0
            # clipped: exp() stays finite on wild steps (Powell has no
            # bounds here, they stall it)
            log_amplitudes = np.clip(
                parameters[position : position + n_amplitudes],
                np.log(scale) - 10,
                np.log(scale) + 10,
            )
            position += n_amplitudes
            amplitudes = np.exp(log_amplitudes) * np.ones(count)
            phases = np.zeros(count)
            if free_phases:
                phases[1:] = parameters[position : position + count - 1]
                position += count - 1
            background = np.logaddexp(
                0, np.clip(parameters[position], -30, 30)
            )
            position += 1
            fitted_ramps = ramps
            if free_ramps:
                fitted_ramps = parameters[position : position + 3 * count]
                fitted_ramps = fitted_ramps.reshape(count, 3)
                position += 3 * count
            fitted_quadratics = None
            if free_quadratics:
                fitted_quadratics = quadratic_unit * parameters[
                    position : position + 6 * count
                ].reshape(count, 6)
            return (
                amplitudes,
                phases,
                background,
                fitted_ramps,
                fitted_quadratics,
            )

        cache = {}

        def objective(parameters):
            """Poisson deviance of one parameter vector."""
            (
                amplitudes,
                phases,
                background,
                fitted_ramps,
                fitted_quadratics,
            ) = unpack(parameters)
            current = fields
            if free_ramps:
                key = fitted_ramps.tobytes() + (
                    b""
                    if fitted_quadratics is None
                    else fitted_quadratics.tobytes()
                )
                if key not in cache:
                    cache.clear()
                    cache[key] = self.domain_fields(
                        fitted_ramps, fitted_quadratics
                    )
                current = cache[key]
            return self.deviance(
                self.intensity(current, amplitudes, phases, background)
            )

        rng = np.random.default_rng(seed)
        starts = n_starts if free_phases else 1
        best, start_deviances = None, []
        for _ in range(starts):
            initial = [np.log(scale)] * n_amplitudes
            if free_phases:
                initial += list(rng.uniform(-np.pi, np.pi, count - 1))
            initial += [np.log(np.expm1(max(np.median(self.observed), 1e-3)))]
            if free_ramps:
                initial += list(ramps.ravel())
            if free_quadratics:
                initial += list((quadratics / quadratic_unit).ravel())
            # bounds keep exp(log-amplitude) and the softplus background
            # finite on wild steps (overflow seen on real data)
            bounds = [(np.log(scale) - 10, np.log(scale) + 10)]
            bounds *= n_amplitudes
            if free_phases:
                bounds += [(None, None)] * (count - 1)
            bounds += [(-30, 30)]
            if free_ramps:
                bounds += [(-np.pi, np.pi)] * (3 * count)
            if free_quadratics:
                bounds += [(None, None)] * (6 * count)
            method = "Powell" if free_ramps else "L-BFGS-B"
            result = minimize(
                objective,
                np.array(initial, dtype=float),
                method=method,
                bounds=None if free_ramps else bounds,
                options={"maxiter": maxiter},
            )
            start_deviances.append(float(result.fun))
            if best is None or result.fun < best.fun:
                best = result
        (amplitudes, phases, background, fitted_ramps, fitted_quadratics) = (
            unpack(best.x)
        )
        return {
            "model": model,
            "equal_amplitudes": equal_amplitudes,
            "deviance": float(best.fun),
            "n_parameters": int(best.x.size),
            "amplitudes": amplitudes.tolist(),
            "phases": list(np.angle(np.exp(1j * phases))),
            "ramps": np.asarray(fitted_ramps).tolist(),
            "quadratics": (
                None
                if fitted_quadratics is None
                else fitted_quadratics.tolist()
            ),
            "background": float(background),
            "start_deviances": start_deviances,
        }

    def model_intensity(self, result: dict) -> np.ndarray:
        """
        Full-array model intensity of a fit result (for plots).

        Args:
            result: Output of :meth:`fit`.

        Returns:
            Model intensity with the shape of the data.
        """
        field = np.zeros(self.data.shape, dtype=complex)
        for label in range(1, self.count + 1):
            quadratics = result.get("quadratics")
            phase = _phase_field(
                self.coordinates,
                result["ramps"][label - 1],
                None if quadratics is None else quadratics[label - 1],
            )
            density = (self.domains == label) * np.exp(1j * phase)
            coefficient = result["amplitudes"][label - 1] * np.exp(
                1j * result["phases"][label - 1]
            )
            field += coefficient * fftshift(
                fftn(ifftshift(density), workers=-1)
            )
        return np.abs(field) ** 2 + result["background"]
