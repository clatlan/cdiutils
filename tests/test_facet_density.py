"""Tests of FacetDensityAnalyser on synthetic faceted crystals."""
import numpy as np
import pytest

from cdiutils.facetanalysis import FacetDensityAnalyser


def truncated_octahedron(size=60, cube=17, octa=24, rotation=None):
    """Support bounded by {100} (cube) and {111} (octahedron) planes.

    Args:
        size (int): array size.
        cube (float): half width of the {100} facets (voxels).
        octa (float): |x| + |y| + |z| limit of the {111} facets.
        rotation (np.ndarray): crystal-to-lab rotation (default identity).

    Returns:
        np.ndarray: boolean support.
    """
    rotation = np.eye(3) if rotation is None else rotation
    grid = np.indices((size, size, size)).reshape(3, -1).T - size / 2
    crystal = grid @ rotation  # lab -> crystal (rotation is orthogonal)
    inside = (np.abs(crystal).max(1) <= cube) & (
        np.abs(crystal).sum(1) <= octa)
    return inside.reshape(size, size, size)


@pytest.fixture(scope="module")
def tilted():
    """Truncated octahedron with [111] along the vertical (axis 1)."""
    top = np.ones(3) / np.sqrt(3)
    rotation = FacetDensityAnalyser._align(top, [0, 1, 0])
    return truncated_octahedron(rotation=rotation), rotation


def test_cubic_families():
    families = FacetDensityAnalyser.cubic_families(1)
    assert set(families) == {"{100}", "{110}", "{111}"}
    assert len(families["{111}"]) == 8
    assert len(FacetDensityAnalyser.cubic_families(3)["{321}"]) == 48


def test_symmetry_of_111_top():
    ops = FacetDensityAnalyser.symmetry_operations((1, 1, 1))
    assert len(ops) == 6  # C3v
    label = FacetDensityAnalyser.orbit_label((1, 1, -1))
    assert label == "(-111)(1-11)(11-1)"


def test_chance_levels_increase_with_index():
    low = FacetDensityAnalyser.chance_level(1, 4.0, n_grid=5000)
    high = FacetDensityAnalyser.chance_level(3, 4.0, n_grid=5000)
    assert low < 0.06 < high


def test_no_gap_in_a_single_part(tilted):
    support, _ = tilted
    assert not FacetDensityAnalyser.planar_gap(support).any()


def test_finds_and_indexes_the_facets(tilted):
    support, _ = tilted
    analyser = FacetDensityAnalyser(support, top=(1, 1, 1), min_count=20)
    reference = analyser.analyse()
    families = {f["family"] for f in reference["facets"]
                if f["kind"] == "facet"}
    assert {"{100}", "{111}"} <= families
    assert reference["orientation_cost_deg"] < 4


def test_gap_faces_are_internal(tilted):
    support, rotation = tilted
    cut = support.copy()
    normal = rotation @ np.array([1, 1, -1]) / np.sqrt(3)
    distance = (np.indices(cut.shape).reshape(3, -1).T - 30) @ normal
    cut[np.abs(distance.reshape(cut.shape)) < 1.2] = False
    analyser = FacetDensityAnalyser(cut, top=(1, 1, 1), min_count=20)
    analyser.analyse()
    _, boundary = analyser.statistics({"one": np.ones(cut.shape)})
    assert set(boundary.side) == {"+n side", "-n side"}
    # with one reflection the in-plane axes are only defined up to the
    # symmetry around the top: any member of the {111} side orbit is right
    plane = tuple(int(v)
                  for v in boundary.plane.iloc[0].strip("()").split(","))
    label = FacetDensityAnalyser.orbit_label(plane)
    side_orbits = (FacetDensityAnalyser.orbit_label((1, 1, -1)),
                   FacetDensityAnalyser.orbit_label((-1, -1, 1)))
    assert label in side_orbits
