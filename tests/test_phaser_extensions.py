"""Tests of the PyNXPhaser extensions: crop/interpolate (coarse-to-fine),
bandwidth PSF, twin alignment, and (with PyNX) seeded, bounded,
homogenised, coarse-to-fine and genetic phasing on a small synthetic
crystal."""

import numpy as np
import pytest

from cdiutils.process.phaser import IS_PYNX_AVAILABLE, PyNXPhaser


def make_crystal(n=48, half=8, phase_amp=1.0, seed=0):
    """Box crystal with a smooth phase, centred in an n^3 array.

    Args:
        n (int): array side.
        half (int): half-size of the box.
        phase_amp (float): amplitude of the smooth phase (rad).
        seed (int): random seed.

    Returns:
        tuple: complex object, support, intensity.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(n) - n // 2
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    support = (
        (np.abs(zz) <= half) & (np.abs(yy) <= half) & (np.abs(xx) <= half)
    )
    phase = (
        phase_amp
        * np.sin(2 * np.pi * xx / (4 * half))
        * np.cos(2 * np.pi * zz / (4 * half))
    )
    obj = support * np.exp(1j * phase)
    intensity = (
        np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))) ** 2
    )
    intensity = rng.poisson(intensity / intensity.max() * 1e6).astype(
        np.float32
    )
    return obj.astype(np.complex64), support, intensity


def test_crop_and_interpolate_keep_the_field_of_view():
    """Cropping reciprocal space by f and interpolating back keeps the
    object at the same relative size."""
    obj, support, intensity = make_crystal()
    cropped = PyNXPhaser.crop_diffraction_data(intensity, 2)
    assert cropped.shape == (24, 24, 24)
    assert cropped.sum() > 0.9 * intensity.sum()
    small = PyNXPhaser.interpolate_object(support.astype(float), cropped.shape)
    back = PyNXPhaser.interpolate_object(small, support.shape) > 0.5
    dice = 2 * (back & support).sum() / (back.sum() + support.sum())
    assert dice > 0.85
    complex_small = PyNXPhaser.interpolate_object(obj, cropped.shape)
    assert np.iscomplexobj(complex_small)


def test_bandwidth_psf_along_q():
    """The bandwidth PSF is normalised, centred and elongated along the
    index direction of Q."""
    b = np.diag([1e-4, 1e-4, 1e-4])  # 1e-4 1/A per pixel
    q = np.array([0.0, 3.0, 0.0])  # Q along axis 1
    psf = PyNXPhaser.bandwidth_psf(b, q, bandwidth=1e-4)
    assert np.isclose(psf.sum(), 1)
    c = np.array(psf.shape) // 2
    assert np.unravel_index(psf.argmax(), psf.shape) == tuple(c)
    rms = []
    for ax in range(3):
        profile = psf.sum(axis=tuple(i for i in range(3) if i != ax))
        x = np.arange(psf.shape[ax]) - c[ax]
        rms.append(np.sqrt((profile * x**2).sum()))
    # |Q| dl/l = 3e-4 1/A = 3 pixels FWHM along axis 1, ~0.3 px elsewhere
    assert rms[1] > 3 * max(rms[0], rms[2])
    assert 1.0 < rms[1] < 1.6


def test_pad_centred_puts_the_kernel_at_the_fft_origin():
    """A padded kernel has its centre at n//2, i.e. at the origin after
    ifftshift, on even and odd axes."""
    kernel = np.zeros((15, 15, 15))
    kernel[7, 7, 7] = 1
    padded = PyNXPhaser.pad_centred(kernel, (312, 200, 201))
    assert np.unravel_index(padded.argmax(), padded.shape) == (156, 100, 100)
    origin = np.unravel_index(np.fft.ifftshift(padded).argmax(), padded.shape)
    assert origin == (0, 0, 0)


def test_align_to_finds_twin_and_shift():
    """_align_to recovers an object from its shifted twin image."""
    obj, support, _ = make_crystal(phase_amp=0.5)
    twin = np.roll(np.conj(obj[::-1, ::-1, ::-1]), (3, -2, 1), axis=(0, 1, 2))
    aligned = PyNXPhaser._align_to(twin, obj)
    assert (
        np.corrcoef(np.abs(aligned).ravel(), np.abs(obj).ravel())[0, 1] > 0.99
    )


@pytest.mark.skipif(not IS_PYNX_AVAILABLE, reason="PyNX not installed")
def test_seeded_bounded_homogenised_run():
    """A run seeded from the truth with obj_noise, a support ceiling and
    homogenisation keeps the support inside the ceiling and converges."""
    obj, support, intensity = make_crystal()
    phaser = PyNXPhaser(
        intensity,
        obj_noise=0.3,
        support_ceiling=support,
        homogenise=0.3,
        support_update_period=20,
        psf=None,
        show_cdi=0,
        calc_llk=0,
    )
    cdi = phaser.init_cdi(
        verbose=False,
        init_main_cdi=False,
        support=support.astype(np.int8),
        obj=obj,
    )
    phaser.run("RAAR**100, ER**60", cdi, init_cdi=False)
    result = cdi.get_obj(shift=True)
    final_support = cdi.get_support(shift=True) > 0
    assert not (final_support & ~support).any()
    corr = np.corrcoef(np.abs(result[support]), np.abs(obj[support]))[0, 1]
    assert np.abs(result[support]).std() / np.abs(result[support]).mean() < 0.5
    assert np.isfinite(corr)


@pytest.mark.skipif(not IS_PYNX_AVAILABLE, reason="PyNX not installed")
def test_coarse_to_fine_and_genetic_run():
    """coarse_to_fine_phasing and the complex, twin-aligned genetic mode
    run end to end and keep objects of the data size."""
    obj, support, intensity = make_crystal()
    phaser = PyNXPhaser(
        intensity, support_update_period=20, psf=None, show_cdi=0, calc_llk=0
    )
    phaser.coarse_to_fine_phasing(
        2,
        "HIO**60, ER**40",
        crop_factors=[2],
        support=support.astype(np.int8),
    )
    assert len(phaser.cdi_list) == 2
    assert phaser.cdi_list[0].get_obj(shift=True).shape == intensity.shape
    phaser2 = PyNXPhaser(
        intensity, support_update_period=20, psf=None, show_cdi=0, calc_llk=0
    )
    phaser2.init_cdi(verbose=False, support=support.astype(np.int8))
    phaser2.genetic_phasing(
        3, 1, "HIO**40, ER**20", mode="complex", twin_align=True
    )
    assert len(phaser2.cdi_list) == 3
