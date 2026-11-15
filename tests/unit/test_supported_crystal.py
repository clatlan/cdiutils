"""Tests of SupportedCrystal (Ru(0001) on c-plane sapphire)."""

import numpy as np
import pytest
import xrayutilities as xu

from cdiutils.simulation import SupportedCrystal

# Ru cell (Anzellini et al., Sci. Rep. 9, 14459 (2019), ambient P, T)
RU_A = 2.705
RU_C = 4.281
ENERGY = 9000.0


@pytest.fixture(scope="module")
def ru_on_sapphire():
    """Ru(0001) on c-sapphire, Ru(10-10)* // Al2O3(11-20)*."""
    ruthenium = xu.materials.Crystal(
        "Ru",
        xu.materials.SGLattice(194, RU_A, RU_C, atoms=["Ru"], pos=["2c"]),
    )
    return SupportedCrystal(
        ruthenium,
        inplane_hkl=(1, 0, 0),
        substrate=xu.materials.Al2O3,
        substrate_inplane_hkl=(1, 1, 0),
    )


def test_q_vector_basal(ru_on_sapphire):
    """(0002) is along the surface normal with |Q| = 4 pi / c."""
    q = ru_on_sapphire.q_vector((0, 0, 0, 2))
    assert np.allclose(q, [0.0, 0.0, 4 * np.pi / RU_C], atol=1e-9)


def test_specular_angles(ru_on_sapphire):
    """Specular (0002): eta = theta, delta = 2 theta, nu = 0."""
    angles = ru_on_sapphire.reflection_angles((0, 0, 2), ENERGY)
    theta = np.degrees(np.arcsin(xu.en2lam(ENERGY) / RU_C))
    assert np.isclose(angles["eta"], theta)
    assert np.isclose(angles["delta"], 2 * theta)
    assert np.isclose(angles["nu"], 0.0, atol=1e-6)


@pytest.mark.parametrize("hkl", [(1, 0, 3), (1, 0, 4), (1, 1, 2), (2, 0, 3)])
@pytest.mark.parametrize("incidence", ["symmetric", "coplanar", 10.0])
def test_angles_round_trip(ru_on_sapphire, hkl, incidence):
    """The closed-form angles give back Q through xu QConversion."""
    angles = ru_on_sapphire.reflection_angles(hkl, ENERGY, incidence)
    if np.isnan(angles["eta"]):
        pytest.skip("not reachable in this geometry")
    q = np.ravel(
        ru_on_sapphire.qconversion(
            angles["eta"],
            angles["phi"],
            angles["nu"],
            angles["delta"],
            en=ENERGY,
        )
    )
    assert np.allclose(q, ru_on_sapphire.q_vector(hkl), atol=1e-6)
    if incidence == "coplanar":
        assert np.isclose(angles["nu"], 0.0, atol=1e-4)


def test_nearest_substrate_node(ru_on_sapphire):
    """Ru(10-13) sits next to the allowed sapphire (11-29) node."""
    node, distance = ru_on_sapphire.nearest_substrate_node((1, 0, 3), ENERGY)
    assert node == (1, 1, 9)
    assert distance == pytest.approx(0.066, abs=0.002)


def test_reflection_table_columns(ru_on_sapphire):
    """The planning table has angles and substrate margins."""
    table = ru_on_sapphire.reflection_table(
        [(0, 0, 2), (1, 0, 4)], ENERGY, particle_sizes_nm=(100.0,)
    )
    assert list(table["reflection"]) == ["(0002)", "(10-14)"]
    assert table.loc[1, "coplanar incidence (deg)"] == pytest.approx(
        20.48, abs=0.05
    )
    assert table.loc[0, "margin 100 nm"] < 1 < table.loc[1, "margin 100 nm"]


def test_angles_vs_energy_shape(ru_on_sapphire):
    """One row per reflection and energy."""
    table = ru_on_sapphire.angles_vs_energy(
        [(1, 0, 4), (1, 1, 2)], [8000, 9000, 10000]
    )
    assert len(table) == 6


def test_interface_misfit():
    """Prieto et al. 2022 (p4): 2.706 vs 2.747 A gives ~1.5 %."""
    misfit = SupportedCrystal.interface_misfit(2.706, 2.747)
    assert misfit == pytest.approx(0.01515, abs=1e-4)


def test_winterbottom_aspect_ratio():
    """W_adh = gamma(0001) halves the Wulff height (h/D ~ 0.39)."""
    pytest.importorskip("wulffpack")
    ase_build = pytest.importorskip("ase.build")
    ruthenium = xu.materials.Crystal(
        "Ru",
        xu.materials.SGLattice(194, RU_A, RU_C, atoms=["Ru"], pos=["2c"]),
    )
    sample = SupportedCrystal(
        ruthenium,
        surface_energies={
            (0, 0, 0, 1): 163,
            (1, 0, -1, 0): 181,
            (1, 0, -1, 1): 180,
            (1, 0, -1, 2): 189,
            (1, 1, -2, 2): 195,
        },
        primitive_structure=ase_build.bulk("Ru", "hcp", a=RU_A, c=RU_C),
    )
    metrics = sample.shape_metrics(sample.winterbottom(1.0), diameter=100.0)
    assert metrics["diameter"] == pytest.approx(100.0)
    assert metrics["aspect_ratio"] == pytest.approx(0.39, abs=0.02)
