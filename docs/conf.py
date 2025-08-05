# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the package to the Python path
sys.path.insert(0, os.path.abspath("../src"))

# Enable postponed evaluation of annotations for Python 3.7+
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

# -- Project information -----------------------------------------------------
project = "CDIutils"
copyright = "2025, Clément Atlan"
author = "Clément Atlan"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Extracts docstrings
    "sphinx.ext.napoleon",  # Supports Google-style & NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.mathjax",  # Math support
    "sphinx.ext.intersphinx",  # Cross-references to other docs
    "sphinx.ext.autosummary",  # Auto-generate summary tables
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock imports for optional/heavy dependencies that might not be available during doc building
autodoc_mock_imports = [
    "vtk",
    "pynx",
    "xrayutilities",
    "silx",
    "PyQt5",
    "fabio",
]

# Set autodoc to work with type annotations
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- Theme options -----------------------------------------------------------
html_theme_options = {
    "github_url": "https://github.com/clatlan/cdiutils",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/clatlan/cdiutils",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

html_context = {
    "github_user": "clatlan",
    "github_repo": "cdiutils",
    "github_version": "master",
    "doc_path": "docs",
}
