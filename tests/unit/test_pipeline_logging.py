"""
Unit tests for the process log files of the pipelines.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# Run in a new interpreter with an ASCII locale, as on some cluster
# nodes: the log file must still hold the non-ASCII text (the
# tabulate "fancy_grid" table of the preprocessing summary)
SCRIPT = textwrap.dedent(
    """
    import locale
    import logging
    import sys
    import types

    from cdiutils.pipeline.base import Pipeline

    assert locale.getpreferredencoding(False).lower() not in (
        "utf-8", "utf8"
    ), "the locale should not be UTF-8 for this test"

    logger = logging.getLogger("test_pipeline_logging")
    logger.propagate = False
    pipeline = types.SimpleNamespace(logger=logger)

    # Fail on the "--- Logging error ---" that logging only prints
    logging.raiseExceptions = True
    handler = Pipeline._init_process_logger(pipeline, sys.argv[1])
    handler.handleError = lambda record: sys.exit("logging error")
    logger.warning("Summary table: \\u2552\\u2550\\u2564\\u2550\\u2555")
    handler.close()
    """
)


@pytest.mark.unit
def test_process_log_file_is_utf8(tmp_path):
    """Test non-ASCII messages are written whatever the locale."""
    log_base = tmp_path / "preprocess_output"
    environment = dict(os.environ, LANG="C", LC_ALL="C", PYTHONUTF8="0")
    subprocess.run(
        [sys.executable, "-c", SCRIPT, str(log_base)],
        check=True,
        env=environment,
    )
    text = (tmp_path / "preprocess_output.log").read_text(encoding="utf-8")
    assert "╒═╤═╕" in text
