"""
Unit tests for the operator chain of the phase retrieval GUI.

The chain is described by build_operator_chain_steps (no PyNX needed),
then converted into PyNX operators by PhaseRetrievalGUI and printed by
format_operator_chain.
"""

import pytest

from cdiutils.interactive.phase_retrieval.engine import (
    build_operator_chain_steps,
    format_operator_chain,
)


def expand(steps: list[dict]) -> list[str]:
    """
    Expand the steps into the sequence of applied operators.

    Args:
        steps: the steps from build_operator_chain_steps.

    Returns:
        One entry per algorithm cycle ("HIO", "RAAR", "ER", "ML"), and
        one per support update ("Sup"), PSF initialisation ("InitPSF")
        or detwinning step ("DetwinHIO" or "DetwinRAAR").
    """
    sequence = []
    for step in steps:
        if step["kind"] == "algo":
            block = [step["algo"]] * step["cycles"]
            if step["support_update"]:
                block.append("Sup")
            sequence += block * step["repeat"]
        elif step["kind"] == "init_psf":
            sequence.append("InitPSF")
        elif step["kind"] == "detwin":
            sequence.append(step["algo"])
    return sequence


@pytest.mark.unit
class TestBuildOperatorChainSteps:
    """Test the description of the operator chain."""

    def test_order_without_support_update(self):
        """Test HIO, RAAR then ER are run, without support update."""
        sequence = expand(build_operator_chain_steps(30, 50, 20))
        assert sequence == ["HIO"] * 30 + ["RAAR"] * 50 + ["ER"] * 20

    def test_support_update_blocks(self):
        """Test a support update follows each full block of cycles."""
        steps = build_operator_chain_steps(
            40, 60, 20, support_update_period=20
        )
        block = ["RAAR"] * 20 + ["Sup"]
        assert expand(steps) == (
            (["HIO"] * 20 + ["Sup"]) * 2 + block * 3 + ["ER"] * 20 + ["Sup"]
        )
        # identical consecutive blocks are merged
        assert [step["repeat"] for step in steps] == [2, 3, 1]

    def test_remaining_cycles_are_run(self):
        """Test cycles beyond the last full block are not dropped."""
        sequence = expand(
            build_operator_chain_steps(0, 50, 0, support_update_period=20)
        )
        assert sequence.count("RAAR") == 50
        assert sequence.count("Sup") == 2

    def test_psf_after_half_raar(self):
        """Test the PSF is initialised after half of the RAAR cycles."""
        steps = build_operator_chain_steps(
            30, 60, 20, support_update_period=10, psf=True, update_psf=5
        )
        sequence = expand(steps)
        assert sequence.index("InitPSF") == len(
            expand(
                build_operator_chain_steps(30, 30, 0, support_update_period=10)
            )
        )
        # the PSF is only updated after its initialisation
        algo_steps = [step for step in steps if step["kind"] == "algo"]
        assert algo_steps[0]["update_psf"] == 0
        assert algo_steps[-1]["update_psf"] == 5

    def test_zero_mask_auto(self):
        """Test zero_mask='auto' switches from 1 to 0 at 60 %."""
        steps = build_operator_chain_steps(0, 100, 20, zero_mask="auto")
        switches = [s for s in steps if s["kind"] == "zero_mask_switch"]
        assert len(switches) == 1
        assert switches[0]["start"] == 60
        algo_steps = [step for step in steps if step["kind"] == "algo"]
        assert algo_steps[0]["zero_mask"] == 1
        assert algo_steps[-1]["zero_mask"] == 0

    @pytest.mark.parametrize("zero_mask", [True, False])
    def test_zero_mask_fixed(self, zero_mask):
        """Test a fixed zero_mask is used by all the algorithms."""
        steps = build_operator_chain_steps(10, 10, 10, zero_mask=zero_mask)
        assert all(
            step["zero_mask"] is zero_mask
            for step in steps
            if step["kind"] == "algo"
        )

    @pytest.mark.parametrize(
        "nb_hio, expected", [(100, "DetwinHIO"), (0, "DetwinRAAR")]
    )
    def test_detwin(self, nb_hio, expected):
        """Test detwinning is run once, after 25 % of HIO + RAAR."""
        steps = build_operator_chain_steps(nb_hio, 100, 20, detwin=True)
        detwin_steps = [s for s in steps if s["kind"] == "detwin"]
        assert len(detwin_steps) == 1
        assert detwin_steps[0]["algo"] == expected
        assert detwin_steps[0]["start"] == (nb_hio + 100) // 4

    def test_ml_after_er(self):
        """Test ML is run last, without support update."""
        steps = build_operator_chain_steps(
            0, 20, 20, nb_ml=30, support_update_period=10
        )
        assert steps[-1]["algo"] == "ML"
        assert steps[-1]["cycles"] == 30
        assert not steps[-1]["support_update"]

    def test_ml_skipped_with_psf(self):
        """Test ML is skipped when the PSF is used, as in PyNX."""
        steps = build_operator_chain_steps(0, 20, 20, nb_ml=30, psf=True)
        assert "ML" not in expand(steps)
        assert steps[-1]["kind"] == "skip"


@pytest.mark.unit
def test_format_operator_chain():
    """Test the printed chain lists the operators and the totals."""
    params = {
        "mask_interp": (8, 2),
        "support": "",
        "obj": "",
        "support_autocorrelation_threshold": 0.1,
        "support_threshold": (0.23, 0.30),
        "psf_model": "gaussian",
        "fwhm": 0.5,
    }
    steps = build_operator_chain_steps(
        20, 40, 20, support_update_period=20, psf=True, update_psf=20
    )
    text = format_operator_chain(steps, params)
    assert "InterpIobsMask(d=8, n=2)" in text
    assert "AutoCorrelationSupport(threshold=0.1)" in text
    assert "InitPSF(model=gaussian" in text
    assert "HIO 20, RAAR 40, ER 20, ML 0; 4 support updates" in text
