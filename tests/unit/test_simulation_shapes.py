"""Tests of the polyhedron, Winterbottom and hexagonal helpers."""

import numpy as np
import pytest

from cdiutils.simulation import (
    BCDISimulator,
    hexagonal_family_normals,
    make_polyhedron,
    make_winterbottom,
    polyhedron_from_planes,
    winterbottom_geometry,
)

# Ru cell (Anzellini et al., Sci. Rep. 9, 14459 (2019), ambient P, T)
RU_A = 2.705
RU_C = 4.281

CUBE_NORMALS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=float,
)


@pytest.mark.parametrize(
    "hkl, expected",
    [
        ((0, 0, 0, 1), 2),
        ((1, 0, -1, 0), 6),
        ((1, 1, -2, 0), 6),
        ((1, 0, -1, 1), 12),
        ((1, 1, -2, 2), 12),
        ((2, 1, -3, 1), 24),
    ],
)
def test_hexagonal_family_multiplicity(hkl, expected):
    """The 6/mmm family sizes are the textbook multiplicities."""
    normals = hexagonal_family_normals(hkl, RU_C / RU_A)
    assert normals.shape == (expected, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_hexagonal_basal_normal_along_z():
    """(0001) points along +z, i.e. array axis 0 in (z, y, x)."""
    normals = hexagonal_family_normals((0, 0, 1), RU_C / RU_A)
    assert np.allclose(sorted(normals[:, 0]), [-1.0, 1.0])


def test_hexagonal_pyramidal_angle():
    """Angle between (10-11) and (0001) matches the hcp formula."""
    c_over_a = RU_C / RU_A
    normals = hexagonal_family_normals((1, 0, 1), c_over_a)
    angle = np.degrees(np.arccos(np.max(normals[:, 0])))
    # tan(phi) = Q_inplane / Q_z = (2 / (sqrt(3) a)) / (1 / c)
    expected = np.degrees(np.arctan(2 * c_over_a / np.sqrt(3)))
    assert np.isclose(angle, expected)


def test_polyhedron_from_planes_cube():
    """A cube of side 2 has volume 8 and six faces of area 4."""
    geometry = polyhedron_from_planes(CUBE_NORMALS, np.ones(6))
    assert np.isclose(geometry["volume"], 8.0)
    assert len(geometry["faces"]) == 6
    assert np.allclose(geometry["areas"], 4.0)


def test_polyhedron_from_planes_origin_outside():
    """The origin may lie outside the polyhedron."""
    distances = np.array([3.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    geometry = polyhedron_from_planes(CUBE_NORMALS, distances)
    assert np.isclose(geometry["volume"], 2.0 * 2.0 * 2.0)


def test_polyhedron_from_planes_empty_raises():
    """Contradictory half-spaces raise a ValueError."""
    distances = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        polyhedron_from_planes(CUBE_NORMALS, distances)


def test_make_polyhedron_voxel_count():
    """Voxelised cube volume matches the analytic volume."""
    array = make_polyhedron((40, 40, 40), CUBE_NORMALS, np.full(6, 10.0))
    assert array.sum() == pytest.approx(21**3)


def test_winterbottom_limits():
    """W = 0 gives the Wulff shape, W = gamma halves the height."""
    normals = CUBE_NORMALS
    energies = np.ones(6)
    wulff = winterbottom_geometry(normals, energies, 0.0)
    half = winterbottom_geometry(normals, energies, 1.0)
    # cube: height 2, in-plane diameter = face diagonal 2 sqrt(2)
    assert np.isclose(wulff["aspect_ratio"], 1 / np.sqrt(2))
    assert np.isclose(half["height"], 0.5 * wulff["height"])
    # the interface sits at height 0 after the shift
    assert np.isclose(half["vertices"][:, 0].min(), 0.0)


def test_winterbottom_scaling():
    """The in-plane diameter is the requested one."""
    normals = hexagonal_family_normals((0, 0, 1), RU_C / RU_A)
    normals = np.vstack(
        [normals, hexagonal_family_normals((1, 0, 0), RU_C / RU_A)]
    )
    energies = np.array([1.0] * 2 + [1.1] * 6)
    geometry = winterbottom_geometry(normals, energies, 0.5, diameter=100.0)
    assert np.isclose(geometry["diameter"], 100.0)
    assert geometry["height"] > 0


def test_make_winterbottom_is_flat_bottomed():
    """The voxelised crystal touches its interface plane over a disc."""
    normals = hexagonal_family_normals((0, 0, 1), RU_C / RU_A)
    normals = np.vstack(
        [normals, hexagonal_family_normals((1, 0, 0), RU_C / RU_A)]
    )
    energies = np.array([1.0] * 2 + [1.1] * 6)
    array = make_winterbottom((64, 64, 64), normals, energies, 1.0, 40.0)
    occupied_layers = np.where(array.sum(axis=(1, 2)) > 0)[0]
    bottom_area = array[occupied_layers[0]].sum()
    top_area = array[occupied_layers[-1]].sum()
    assert bottom_area > 0.5 * array.sum(axis=(1, 2)).max()
    assert top_area > 0


def test_hexagonal_bragg_angle():
    """Ru(0002) at 9 keV is at theta = asin(lambda / c)."""
    simulator = BCDISimulator(
        energy=9000,
        structure="hexagonal",
        hkl=[0, 0, 0, 2],
        lattice_parameter=(RU_A * 1e-10, RU_C * 1e-10),
    )
    theta = simulator.lattice_parameter_to_bragg_angle()
    expected = np.degrees(np.arcsin(simulator.wavelength / (RU_C * 1e-10)))
    assert np.isclose(theta, expected)
    d_spacing = BCDISimulator.hexagonal_d_spacing([1, 0, 1], RU_A, RU_C)
    assert np.isclose(d_spacing, 2.0550, atol=1e-4)
