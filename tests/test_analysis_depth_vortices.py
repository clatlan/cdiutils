"""Tests of the depth, domain and phase-vortex analysis functions."""
import numpy as np

from cdiutils.analysis import (get_depth_map, get_depth_profile,
                               normalise_domains)
from cdiutils.analysis.dislocation import find_phase_vortices


def box(shape=(20, 30, 20), top=25, bottom=5):
    """Support filled between two heights along axis 1."""
    support = np.zeros(shape, bool)
    support[2:-2, bottom:top, 2:-2] = True
    return support


def test_depth_map_counts_from_the_top():
    support = box()
    depth, height = get_depth_map(support)
    assert np.nanmin(depth) == 0
    assert np.nanmax(depth) == 19
    assert depth[10, 24, 10] == 0 and height[10, 5, 10] == 0
    assert np.isnan(depth[0, 0, 0])


def test_depth_profile_follows_a_gradient():
    support = box()
    strain = np.broadcast_to(np.arange(30, dtype=float)[None, :, None],
                             support.shape).copy()
    profile = get_depth_profile(strain, support, voxel_size=10, min_count=1)
    assert profile["depth"][0] == 0 and profile["depth"][1] == 10
    # the top layer is y = 24, one layer down is y = 23
    np.testing.assert_allclose(profile["mean"][:2], [24, 23])
    np.testing.assert_allclose(profile["std"], 0)


def test_normalise_domains_equalises_two_parts():
    amplitude = np.zeros((20, 20, 40))
    amplitude[4:16, 4:16, 4:18] = 1.0
    amplitude[4:16, 4:16, 22:36] = 0.5
    normalised, domains, plateaus = normalise_domains(amplitude)
    assert len(plateaus) == 2
    np.testing.assert_allclose(normalised[10, 10, 10], 1)
    np.testing.assert_allclose(normalised[10, 10, 30], 1)


def test_screw_line_is_found_and_straight_crystal_is_clean():
    n = 40
    z, y, x = np.indices((n, n, n)) - n // 2
    support = x ** 2 + y ** 2 + z ** 2 < 15 ** 2
    lines, hits = find_phase_vortices(np.zeros(support.shape), support)
    assert lines == [] and not hits.any()
    # screw dislocation along y: the phase winds once around the line
    phase = np.arctan2(z + 0.5, x + 0.5)
    lines, _ = find_phase_vortices(phase, support,
                                   amplitude=support.astype(float))
    assert len(lines) == 1
    assert abs(abs(lines[0]["direction"][1]) - 1) < 0.05
