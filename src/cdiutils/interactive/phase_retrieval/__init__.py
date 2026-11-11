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

_PYNX_FREE_EXPORTS = [
    "build_operator_chain_steps",
    "format_operator_chain",
    "list_files",
]

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
        ] + _PYNX_FREE_EXPORTS
    else:
        __all__ = [
            "initialise_cdi_operator",
            "save_cdi_operator_as_cxi",
        ] + _PYNX_FREE_EXPORTS
else:
    __all__ = list(_PYNX_FREE_EXPORTS)
