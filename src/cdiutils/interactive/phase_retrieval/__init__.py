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

# Always export the helpers that do not require PyNX
from .engine import (
    build_operator_chain_steps,
    format_operator_chain,
    list_files,
)

# Conditionally import phase retrieval components. __all__ is written
# out in full in each case, so that linters see the exports.
if IS_PYNX_AVAILABLE:
    from .engine import (
        initialise_cdi_operator,
        save_cdi_operator_as_cxi,
    )

    if IS_INTERACTIVE_AVAILABLE:
        from .gui import PhaseRetrievalGUI

        __all__ = [
            "PhaseRetrievalGUI",
            "build_operator_chain_steps",
            "format_operator_chain",
            "initialise_cdi_operator",
            "list_files",
            "save_cdi_operator_as_cxi",
        ]
    else:
        __all__ = [
            "build_operator_chain_steps",
            "format_operator_chain",
            "initialise_cdi_operator",
            "list_files",
            "save_cdi_operator_as_cxi",
        ]
else:
    __all__ = [
        "build_operator_chain_steps",
        "format_operator_chain",
        "list_files",
    ]
