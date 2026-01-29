"""
Phase retrieval GUI and utilities for BCDI.

This package provides interactive tools for running phase retrieval
algorithms using PyNX.
"""

try:
    import ipywidgets  # noqa: F401

    IS_INTERACTIVE_AVAILABLE = True
except ImportError:
    IS_INTERACTIVE_AVAILABLE = False

try:
    import pynx  # noqa: F401

    IS_PYNX_AVAILABLE = True
except ImportError:
    IS_PYNX_AVAILABLE = False

# Always export list_files (doesn't require PyNX)
from .engine import list_files

# Conditionally import phase retrieval components
if IS_PYNX_AVAILABLE:
    from .engine import (
        initialise_cdi_operator,
        save_cdi_operator_as_cxi,
    )

    if IS_INTERACTIVE_AVAILABLE:
        from .gui import PhaseRetrievalGUI

        __all__ = [
            "PhaseRetrievalGUI",
            "initialise_cdi_operator",
            "save_cdi_operator_as_cxi",
            "list_files",
        ]
    else:
        __all__ = [
            "initialise_cdi_operator",
            "save_cdi_operator_as_cxi",
            "list_files",
        ]
else:
    __all__ = ["list_files"]
