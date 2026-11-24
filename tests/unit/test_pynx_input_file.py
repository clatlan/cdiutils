"""
Unit tests for the PyNX input file written by BcdiPipeline.phase_retrieval.
"""

import logging

import numpy as np
import pytest

from cdiutils.pipeline.bcdi import BcdiPipeline

# Parameters that pynx-cdi-id01 splits on commas (from the PyNX 2026.1
# CDIRunner.parse_parameters_file)
PYNX_COMMA_KEYS = (
    "support_size",
    "rebin",
    "mask",
    "roi",
    "mask_interp",
    "support_threshold",
    "support_autocorrelation_threshold",
    "support_post_expand",
)


def to_pynx_arguments(lines: list[str]) -> dict:
    """
    Convert input file lines as pynx-cdi-id01 does, before argparse.

    Args:
        lines: the "key = value" lines.

    Returns:
        The command-line words given to each option, by parameter.
    """
    arguments = {}
    for line in lines:
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.replace("'", "").replace('"', "")
        if value in ("false", "False", "none", "None"):
            continue
        if key in PYNX_COMMA_KEYS and "," in value:
            arguments[key] = value.replace(" ", "").split(",")
        else:
            arguments[key] = [value]
    return arguments


def make_lines(**pynx_params) -> list[str]:
    """
    Return the input file lines of the given PyNX parameters.

    Args:
        **pynx_params: the PyNX parameters.

    Returns:
        The "key = value" lines.
    """
    return BcdiPipeline._make_pynx_input_lines(pynx_params)


@pytest.mark.unit
class TestPynxInputLines:
    """Test the parameters are written in a form PyNX can read."""

    @pytest.mark.parametrize(
        "value",
        [[0.23, 0.3], (0.23, 0.3), np.array([0.23, 0.3]), "0.23, 0.3"],
    )
    def test_range_without_brackets(self, value):
        """Test lists, tuples, arrays and strings give two floats."""
        arguments = to_pynx_arguments(make_lines(support_threshold=value))
        assert [float(v) for v in arguments["support_threshold"]] == [
            0.23,
            0.3,
        ]

    def test_tuple_parameters(self):
        """Test integer tuples are written as comma-separated values."""
        arguments = to_pynx_arguments(
            make_lines(
                mask_interp=(8, 2),
                rebin=[1, 1, 1],
                support_post_expand=(1, -2, 1),
            )
        )
        assert arguments["mask_interp"] == ["8", "2"]
        assert arguments["rebin"] == ["1", "1", "1"]
        assert arguments["support_post_expand"] == ["1", "-2", "1"]

    @pytest.mark.parametrize("value", [0.1, [0.1], (0.1,)])
    def test_single_autocorrelation_threshold(self, value):
        """Test a single value is written as a (v, v) range."""
        arguments = to_pynx_arguments(
            make_lines(support_autocorrelation_threshold=value)
        )
        assert arguments["support_autocorrelation_threshold"] == [
            "0.1",
            "0.1",
        ]

    def test_non_integer_support_fraction_skipped(self, caplog):
        """Test a float support fraction is left out, with a warning."""
        logger = logging.getLogger("test_pynx_input_file")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            lines = BcdiPipeline._make_pynx_input_lines(
                {
                    "support_fraction_min": 1e-4,
                    "support_fraction_max": 0.7,
                    "nb_run": 10,
                },
                logger,
            )
        assert lines == ["nb_run = 10"]
        assert "support_fraction_min" in caplog.text

    def test_other_values_unchanged(self):
        """Test scalars, strings, booleans and None are written as is."""
        lines = make_lines(
            nb_raar=600,
            psf="pseudo-voigt,0.5,0.05,20",
            live_plot=False,
            roi=None,
            zero_mask=0,
        )
        assert lines == [
            "nb_raar = 600",
            "psf = pseudo-voigt,0.5,0.05,20",
            "live_plot = False",
            "roi = None",
            "zero_mask = 0",
        ]
