"""
Unit tests for the warning about an old PyNX run on the current machine.
"""

import logging
import types

import pytest

from cdiutils.pipeline.bcdi import BcdiPipeline

COMMAND = "pynx-cdi-id01 pynx-cdi-inputs.txt"


def warnings_for(version, roi, caplog, cmd=COMMAND) -> list[str]:
    """
    Return the warnings logged for a PyNX version and a roi value.

    Args:
        version: the PyNX version found, or None.
        roi: the roi parameter.
        caplog: the pytest log capture fixture.
        cmd: the command run.

    Returns:
        The warning messages.
    """
    logger = logging.getLogger("test_local_pynx_warning")
    pipeline = types.SimpleNamespace(
        logger=logger, _get_pynx_version=lambda executable: version
    )
    with caplog.at_level(logging.WARNING, logger=logger.name):
        BcdiPipeline._warn_old_local_pynx(pipeline, cmd, {"roi": roi})
    return [r.getMessage() for r in caplog.records]


@pytest.mark.unit
class TestLocalPynxWarning:
    """Test when the warnings are given."""

    def test_old_version(self, caplog):
        """Test PyNX 2023 gives the general warning."""
        messages = warnings_for("2023.1.1", None, caplog)
        assert len(messages) == 1
        assert "jump_to_cluster=True" in messages[0]

    @pytest.mark.parametrize("roi", [False, "full", "auto"])
    def test_old_version_bad_roi(self, caplog, roi):
        """Test a roi PyNX 2023 cannot read gives a second warning."""
        messages = warnings_for("2023.1.1", roi, caplog)
        assert len(messages) == 2
        assert "roi=None or 6 integers" in messages[1]

    def test_old_version_integer_roi(self, caplog):
        """Test 6 integers are accepted."""
        assert len(warnings_for("2023.1.1", [0, 2, 0, 2, 0, 2], caplog)) == 1

    @pytest.mark.parametrize("version", ["2026.1", "2024.1", None, "dev"])
    def test_no_warning(self, caplog, version):
        """Test recent, unknown or unreadable versions give no warning."""
        assert warnings_for(version, False, caplog) == []

    def test_other_command(self, caplog):
        """Test a command that is not a PyNX script is not checked."""
        assert warnings_for("2023.1.1", False, caplog, cmd="bash run.sh") == []


@pytest.mark.unit
def test_version_of_missing_script():
    """Test a script that is not found gives None."""
    assert BcdiPipeline._get_pynx_version("no-such-pynx-script") is None
