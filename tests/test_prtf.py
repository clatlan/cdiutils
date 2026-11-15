"""Tests of the phase retrieval transfer function
(cdiutils.process.PhasingResultAnalyser.phase_retrieval_transfer_function)
on synthetic known answers."""

import numpy as np
import pytest
from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.ndimage import gaussian_filter

pytest.importorskip("skimage")
from cdiutils.process.phaser import PhasingResultAnalyser  # noqa: E402

SHAPE = (48, 48, 48)


def crystal(seed=0):
    """
    A smooth-edged, mildly strained box at the array centre.

    Args:
        seed: Random seed of the phase field.

    Returns:
        np.ndarray: complex object, origin at the array centre.
    """
    z, y, x = np.indices(SHAPE) - np.array(SHAPE)[:, None, None, None] // 2
    box = (np.abs(z) < 9) & (np.abs(y) < 7) & (np.abs(x) < 11)
    amplitude = gaussian_filter(box.astype(float), 0.7)
    rng = np.random.default_rng(seed)
    phase = gaussian_filter(rng.normal(size=SHAPE), 4) * 20
    return amplitude * np.exp(1j * phase)


def spectrum(obj):
    """
    Centred Fourier transform (the convention of the method).

    Args:
        obj: complex object, origin at the array centre.

    Returns:
        np.ndarray: complex spectrum, origin at the array centre.
    """
    return fftshift(fftn(ifftshift(obj)))


def q_norm():
    """
    ``|q|`` per voxel in 2 pi / N units (no q_lab matrix).

    Returns:
        np.ndarray: ``|q|`` of every voxel.
    """
    offsets = np.indices(SHAPE) - np.array(SHAPE)[:, None, None, None] // 2
    return np.linalg.norm(2 * np.pi * offsets / SHAPE[0], axis=0)


def test_identical_reconstructions_give_unit_prtf():
    """Reconstructions equal to the truth: PRTF 1, no cut-off."""
    obj = crystal()
    intensity = np.abs(spectrum(obj)) ** 2
    result = PhasingResultAnalyser.phase_retrieval_transfer_function(
        [obj, obj, obj], intensity, nb_shells=12
    )
    assert np.allclose(result["prtf"], 1, atol=1e-6)
    assert result["resolution"] is None


def test_random_phases_beyond_cutoff_set_the_resolution():
    """Phases random beyond q_c in each run: the resolution is
    pi / q_c."""
    obj = crystal()
    truth = spectrum(obj)
    intensity = np.abs(truth) ** 2
    norm = q_norm()
    q_cut = 1.2
    rng = np.random.default_rng(1)
    runs = []
    for _ in range(12):
        random_phase = np.exp(2j * np.pi * rng.random(SHAPE))
        run = np.where(norm > q_cut, truth * random_phase, truth)
        runs.append(fftshift(ifftn(ifftshift(run))))
    result = PhasingResultAnalyser.phase_retrieval_transfer_function(
        runs, intensity, nb_shells=24, align=False
    )
    low = result["q"] < q_cut * 0.9
    high = result["q"] > q_cut * 1.1
    assert np.all(result["prtf"][low] > 0.8)
    assert np.all(result["prtf"][high] < 0.5)
    assert result["resolution"] == pytest.approx(np.pi / q_cut, rel=0.1)


def test_alignment_removes_twin_shift_and_phase_offset():
    """A twin, sub-pixel shifted, phase-offset copy is aligned back:
    PRTF close to 1 with align, much lower without."""
    obj = crystal()
    intensity = np.abs(spectrum(obj)) ** 2
    shifted = fftshift(
        ifftn(
            ifftshift(
                spectrum(obj)
                * np.exp(
                    -2j
                    * np.pi
                    * 0.4
                    * (np.indices(SHAPE)[2] - SHAPE[2] // 2)
                    / SHAPE[2]
                )
            )
        )
    )
    copy = np.conj(np.roll(shifted[::-1, ::-1, ::-1], 1, axis=(0, 1, 2)))
    copy *= np.exp(1.3j)
    aligned = PhasingResultAnalyser.phase_retrieval_transfer_function(
        [obj, copy], intensity, nb_shells=12
    )
    raw = PhasingResultAnalyser.phase_retrieval_transfer_function(
        [obj, copy], intensity, nb_shells=12, align=False
    )
    assert np.nanmin(aligned["prtf"][:10]) > 0.9
    # two runs with unrelated phases: mean |1 + exp(i phi)| / 2 = 2 / pi
    assert np.nanmean(raw["prtf"]) < 0.8


def test_directional_prtf_and_q_lab_units():
    """Phases random only for |q_x| beyond q_c: the directions along x
    lose resolution, those along z keep it; a q_lab matrix gives the
    resolution in its length unit."""
    obj = crystal()
    truth = spectrum(obj)
    intensity = np.abs(truth) ** 2
    offsets = np.indices(SHAPE) - np.array(SHAPE)[:, None, None, None] // 2
    q_x = 2 * np.pi * np.abs(offsets[2]) / SHAPE[2]
    q_cut = 1.2
    rng = np.random.default_rng(2)
    runs = []
    for _ in range(12):
        random_phase = np.exp(2j * np.pi * rng.random(SHAPE))
        run = np.where(q_x > q_cut, truth * random_phase, truth)
        runs.append(fftshift(ifftn(ifftshift(run))))
    # 1 pixel = 2 pi / (N * 10 angstrom): lengths in angstrom
    q_lab_matrix = np.diag([2 * np.pi / (n * 10.0) for n in SHAPE])
    result = PhasingResultAnalyser.phase_retrieval_transfer_function(
        runs,
        intensity,
        q_lab_matrix=q_lab_matrix,
        nb_shells=20,
        nb_directions=20,
        align=False,
    )
    directions = result["directions"]
    along_x = np.argmax(np.abs(directions[:, 2]))
    along_z = np.argmax(np.abs(directions[:, 0]))
    resolution_x = result["resolution_directional"][along_x]
    resolution_z = result["resolution_directional"][along_z]
    # pi / q_c in angstrom: q_c (pixel units) / (10 angstrom)
    assert resolution_x == pytest.approx(np.pi / (q_cut / 10), rel=0.15)
    assert resolution_z is None or resolution_z < 0.8 * resolution_x
