"""Known-answer tests of find_planar_defects."""

import numpy as np

from cdiutils.analysis.dislocation import find_planar_defects


def _particle(offsets, angle_deg=70.53, n=80, width=1.0):
    """Sphere with low-amplitude planes at `offsets` along a normal
    making `angle_deg` with axis 1."""
    z, y, x = np.mgrid[:n, :n, :n] - n / 2 + 0.5
    sphere = (z**2 + y**2 + x**2) < 30**2
    angle = np.radians(angle_deg)
    normal = np.array([0.0, np.cos(angle), np.sin(angle)])
    along = z * normal[0] + y * normal[1] + x * normal[2]
    amp = np.ones_like(along)
    for offset in offsets:
        amp *= 1 - 0.95 * np.exp(-((along - offset) ** 2) / (2 * width**2))
    return amp * sphere, sphere


def test_lamella_two_boundaries():
    amp, sphere = _particle(offsets=(-3.0, 3.0))
    sheets = find_planar_defects(amp, amp > 0.2)
    top = sheets[0]
    assert abs(top["angle_to_reference"] - 70.53) < 2
    assert len(top["boundaries"]) == 2
    assert abs(top["spacing"][0] - 6.0) < 1.01


def test_single_boundary():
    amp, sphere = _particle(offsets=(0.0,))
    sheets = find_planar_defects(amp, amp > 0.2)
    assert len(sheets[0]["boundaries"]) == 1
    assert sheets[0]["planarity"] < 0.1


def test_no_defect():
    amp, sphere = _particle(offsets=())
    assert find_planar_defects(amp, sphere) == []
