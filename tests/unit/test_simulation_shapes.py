"""Tests of the polyhedron, wulffpack and hexagonal helpers."""

import numpy as np
import pytest

from cdiutils.simulation import (
    BCDISimulator,
    get_winterbottom,
    make_from_wulffpack,
    make_polyhedron,
    wulffpack_planes,
)

# Ru cell (Anzellini et al., Sci. Rep. 9, 14459 (2019), ambient P, T)
RU_A = 2.705
RU_C = 4.281
# Ru hcp surface energies, meV/A^2 (Lin et al., JPCC 124, 11005
# (2020), Table 3), four low-energy families
RU_ENERGIES = {
    (0, 0, 0, 1): 163,
    (1, 0, -1, 0): 181,
    (1, 0, -1, 1): 180,
    (1, 0, -1, 2): 189,
}

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


def test_make_polyhedron_voxel_count():
    """Voxelised cube volume matches the analytic volume."""
    array = make_polyhedron((40, 40, 40), CUBE_NORMALS, np.full(6, 10.0))
    assert array.sum() == pytest.approx(21**3)


def test_make_polyhedron_rotation_keeps_volume():
    """A rotated cube keeps (approximately) its volume."""
    rotated = make_polyhedron(
        (64, 64, 64), CUBE_NORMALS, np.full(6, 12.0), rotation=(30, 0, 0)
    )
    # analytic volume of a cube of side 24 pixels
    assert rotated.sum() == pytest.approx(24.0**3, rel=0.06)


@pytest.fixture
def ru_primitive():
    """Primitive hcp Ru structure (ASE)."""
    ase_build = pytest.importorskip("ase.build")
    return ase_build.bulk("Ru", "hcp", a=RU_A, c=RU_C)


def test_get_winterbottom_c_plane(ru_primitive):
    """c-plane hcp interface works (wulffpack 1.5 gives NaN normals)."""
    pytest.importorskip("wulffpack")
    particle = get_winterbottom(
        RU_ENERGIES, (0, 0, -1), 0.5 * 163, primitive_structure=ru_primitive
    )
    normals, distances = wulffpack_planes(particle)
    assert np.all(np.isfinite(normals))
    # interface (outward normal -z in (z, y, x) order) is the lowest
    # plane: z_min = -d for the plane -z = d
    interface = np.isclose(normals[:, 0], -1.0)
    assert np.any(interface)
    vertices = np.vstack(
        [f.vertices for form in particle.forms for f in form.facets]
    )
    assert np.allclose(-distances[interface], vertices[:, 2].min())
    assert 0 < particle.facet_fractions["interface"] < 1


def test_winterbottom_flattens_with_adhesion(ru_primitive):
    """Stronger adhesion (lower interface energy) gives flatter shapes."""
    pytest.importorskip("wulffpack")
    aspect = []
    for interface_energy in (0.5 * 163, -0.5 * 163):
        particle = get_winterbottom(
            RU_ENERGIES,
            (0, 0, -1),
            interface_energy,
            primitive_structure=ru_primitive,
        )
        vertices = np.vstack(
            [f.vertices for form in particle.forms for f in form.facets]
        )
        aspect.append(np.ptp(vertices[:, 2]) / np.ptp(vertices[:, 0]))
    assert aspect[1] < aspect[0]


def test_make_from_wulffpack_diameter(ru_primitive):
    """Voxelised particle has the requested in-plane diameter."""
    pytest.importorskip("wulffpack")
    particle = get_winterbottom(
        RU_ENERGIES, (0, 0, -1), 0.5 * 163, primitive_structure=ru_primitive
    )
    array = make_from_wulffpack((64, 64, 64), particle, diameter=40.0)
    occupied = np.argwhere(array > 0)
    width = np.max(np.ptp(occupied[:, 1:], axis=0)) + 1
    assert 32 <= width <= 41
    # flat interface at the bottom: bottom layer is wider than the top
    layers = np.unique(occupied[:, 0])
    assert array[layers[0]].sum() > array[layers[-1]].sum()


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
