# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'CdiUtils'
copyright = '2025, Clément Atlan'
author = 'Clément Atlan'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Extracts docstrings
    "sphinx.ext.napoleon",      # Supports Google-style & NumPy-style docstrings
    "sphinx.ext.viewcode",      # Adds links to source code
    "sphinx.ext.mathjax",       # Math support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
