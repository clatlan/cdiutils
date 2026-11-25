"""Tests of the domain model of cdiutils.simulation (phase steps vs
tilts between domains, fitted against simulated intensity)."""

import numpy as np

from cdiutils.simulation import (
    DomainModelFit,
    fill_domain_gaps,
    fit_quadratic_phase,
    measure_domain_phases,
    segment_domains,
)

SHAPE = (48, 48, 48)


def two_slabs(step=np.pi * 0.8, gap=1):
    """
    Two slabs of a box separated along axis 2, the second with a
    constant phase step.

    Args:
        step: Phase step of the second slab (rad).
        gap: Width of the empty plane between the slabs (voxels).

    Returns:
        (object, domains): complex object and its true labels.
    """
    domains = np.zeros(SHAPE, dtype=np.int32)
    domains[16:32, 16:32, 12:24] = 1
    domains[16:32, 16:32, 24 + gap : 36] = 2
    obj = (domains == 1) + (domains == 2) * np.exp(1j * step)
    return obj.astype(complex), domains


def simulated_intensity(obj, photons=1e7, seed=1):
    """
    Poisson intensity in the PyNX convention.

    Args:
        obj: Complex object.
        photons: Total expected photons.
        seed: Random seed.

    Returns:
        Intensity array (photons).
    """
    field = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    intensity = np.abs(field) ** 2
    intensity *= photons / intensity.sum()
    return np.random.default_rng(seed).poisson(intensity).astype(float)


def test_segment_domains_bridges_the_gap():
    """The filled outline covers the empty plane between the slabs."""
    obj, truth = two_slabs(gap=1)
    modulus = np.abs(obj)
    domains, seeds = segment_domains(
        modulus,
        seed_threshold=0.5,
        outline_threshold=0.5,
        closing_iterations=2,
        min_size=10,
    )
    assert seeds.max() == 2
    assert domains.max() == 2
    # the gap plane is now inside a domain (except its one-voxel rim on
    # the box surface, a notch that closing does not reach)
    assert np.all(domains[17:31, 17:31, 24] > 0)


def test_measure_domain_phases_recovers_step():
    """Offsets of two domains without ramp differ by the step."""
    step = 2.0
    obj, truth = two_slabs(step=step)
    phases = measure_domain_phases(obj, truth)
    measured = phases[1]["offset"] - phases[0]["offset"]
    assert abs(np.angle(np.exp(1j * (measured - step)))) < 1e-6
    assert phases[0]["coherence"] > 0.999


def test_fit_prefers_phase_steps_and_recovers_them():
    """On data made with a phase step, model A beats C and finds the
    step (up to the twin ambiguity, step -> -step)."""
    step = 2.4
    obj, truth = two_slabs(step=step, gap=0)
    data = simulated_intensity(obj)
    mask = np.zeros(SHAPE)
    fitter = DomainModelFit(data, mask, truth)
    result_a = fitter.fit("A", n_starts=4)
    result_c = fitter.fit("C")
    assert result_a["deviance"] < result_c["deviance"]
    found = abs(result_a["phases"][1])
    assert abs(found - step) < 0.1


def test_fit_prefers_tilts_on_tilted_data():
    """On data made with tilted domains and no step, model B beats A."""
    _, truth = two_slabs(step=0.0, gap=0)
    coordinates = np.ogrid[:48, :48, :48]
    tilt = (truth == 2) * 0.25 * (coordinates[0] - 24)
    obj = (truth > 0) * np.exp(1j * tilt)
    data = simulated_intensity(obj)
    fitter = DomainModelFit(data, np.zeros(SHAPE), truth)
    result_b = fitter.fit(
        "B", ramps=np.array([[0, 0, 0], [0.2, 0, 0]]), maxiter=60
    )
    result_a = fitter.fit("A", n_starts=4)
    assert result_b["deviance"] < result_a["deviance"]
    assert abs(result_b["ramps"][1][0] - 0.25) < 0.03


def test_fill_domain_gaps_keeps_outer_surface():
    """Only the plane between the slabs is filled; nothing is added
    outside the box."""
    _, truth = two_slabs(gap=1)
    filled = fill_domain_gaps(truth, width=1.5)
    assert np.all(filled[16:32, 16:32, 24] > 0)
    outside = np.ones(SHAPE, dtype=bool)
    outside[16:32, 16:32, 12:36] = False
    assert not np.any(filled[outside])


def bent_box(curvature=0.004):
    """
    A box whose phase is quadratic along axis 2 (a bent crystal).

    Args:
        curvature: h_22 in rad / voxel^2 (centred coordinates).

    Returns:
        (object, labels): complex object and a one-domain label array.
    """
    labels = np.zeros(SHAPE, dtype=np.int32)
    labels[16:32, 16:32, 10:38] = 1
    x = np.arange(SHAPE[2]) - SHAPE[2] // 2
    obj = (labels > 0) * np.exp(1j * curvature * x[None, None, :] ** 2)
    return obj.astype(complex), labels


def test_fit_quadratic_phase_recovers_curvature():
    """The least-squares quadratic phase recovers h_22 of a bent box."""
    obj, labels = bent_box(0.004)
    ramp, quadratic = fit_quadratic_phase(obj, labels > 0)
    assert abs(quadratic[2] - 0.004) < 2e-4
    assert np.all(np.abs(quadratic[[0, 1, 3, 4, 5]]) < 2e-4)
    assert np.all(np.abs(ramp) < 1e-2)


def test_bend_model_beats_flat_model_on_bent_data():
    """On data from a bent box, the bend model beats C and recovers
    the curvature."""
    obj, labels = bent_box(0.004)
    data = simulated_intensity(obj)
    fitter = DomainModelFit(data, np.zeros(SHAPE), labels)
    start = np.zeros((1, 6))
    start[0, 2] = 0.003
    result_bend = fitter.fit("bend", quadratics=start, maxiter=40)
    result_c = fitter.fit("C")
    assert result_bend["deviance"] < result_c["deviance"]
    assert abs(result_bend["quadratics"][0][2] - 0.004) < 4e-4


def test_fit_rejects_unknown_model():
    """An unknown model name raises instead of silently fitting C."""
    import pytest

    obj, truth = two_slabs()
    fitter = DomainModelFit(simulated_intensity(obj), np.zeros(SHAPE), truth)
    with pytest.raises(ValueError):
        fitter.fit("D")
