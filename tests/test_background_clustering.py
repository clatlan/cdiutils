"""Tests of the Bragg-signal clustering and the per-pixel background map
(cdiutils.utils)."""

import numpy as np

from cdiutils.utils import background_map, cluster_bragg_signal


def synthetic_scan(seed=0):
    """
    A Gaussian Bragg peak, a weak separate alien peak and a background
    that varies across the detector but not with the rocking angle.

    Args:
        seed: Random seed.

    Returns:
        (data, background): Poisson intensity and the true background
        per pixel and frame.
    """
    shape = (40, 64, 64)
    z, y, x = np.indices(shape)
    peak = 1e4 * np.exp(-((z - 20) ** 2 + (y - 32) ** 2
                          + (x - 32) ** 2) / (2 * 3.0 ** 2))
    alien = 50 * np.exp(-((z - 5) ** 2 + (y - 8) ** 2
                          + (x - 55) ** 2) / (2 * 1.5 ** 2))
    background = 0.2 + 0.3 * (np.arange(64) / 63)[None, :] * np.ones(
        (64, 64))
    expected = peak + alien + background[None]
    rng = np.random.default_rng(seed)
    return rng.poisson(expected).astype(float), background


def test_cluster_separates_bragg_peak_and_alien():
    """The Bragg cluster holds the maximum; the alien is another
    cluster."""
    data, _ = synthetic_scan()
    labels, bragg, region = cluster_bragg_signal(data, threshold=0.3)
    assert bragg > 0
    assert region[20, 32, 32]
    alien_label = labels[5, 8, 55]
    assert alien_label not in (0, bragg)
    assert not region[5, 8, 55]


def test_background_map_recovers_background():
    """The map matches the true background to a few percent away from
    the peak, and is filled under it."""
    data, background = synthetic_scan()
    _, _, region = cluster_bragg_signal(data, threshold=0.4)
    estimate = background_map(data, region, sigma=4)
    error = np.abs(estimate - background) / background
    assert np.median(error) < 0.05
    assert np.isfinite(estimate[32, 32])


def test_merge_size_joins_fringes_to_the_bragg_cluster():
    """A fringe shell detached from the peak joins the Bragg cluster
    with merge_size, not without."""
    shape = (40, 64, 64)
    z, y, x = np.indices(shape)
    radius = np.sqrt((z - 20) ** 2 + (y - 32) ** 2 + (x - 32) ** 2)
    expected = 1e4 * np.exp(-radius ** 2 / 8) + 30 * (np.abs(radius - 12)
                                                      < 1) + 0.2
    data = np.random.default_rng(1).poisson(expected).astype(float)
    labels, bragg, _ = cluster_bragg_signal(data, dilation=0)
    shell = (np.abs(radius - 12) < 0.5)
    assert np.mean(labels[shell] == bragg) < 0.5
    labels, bragg, _ = cluster_bragg_signal(data, dilation=0,
                                            merge_size=5)
    assert np.mean(labels[shell] == bragg) > 0.9


def test_pure_pixels_ignore_fringe_crossings():
    """With fringes crossing some pixels in some frames, the pure-pixel
    map stays at the true background there."""
    data, background = synthetic_scan()
    _, _, region = cluster_bragg_signal(data, threshold=0.3)
    # weak fringe signal outside the region, in a few frames only
    data[30:34, 20:44, 20:44] += 3 * (~region[30:34, 20:44, 20:44])
    leaky = background_map(data, region, sigma=4)
    pure = background_map(data, region, sigma=4, pure_pixels=True)
    centre = (slice(26, 38), slice(26, 38))
    assert np.abs(pure[centre] - background[centre]).mean() < np.abs(
        leaky[centre] - background[centre]).mean()
