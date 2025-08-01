# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies for documentation building
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'numpy', 'scipy', 'scipy.fft', 'scipy.constants', 'scipy.signal', 
    'scipy.optimize', 'scipy.ndimage', 'matplotlib', 'matplotlib.pyplot',
    'matplotlib.patches', 'matplotlib.gridspec', 'h5py', 'pandas', 
    'scikit-image', 'scikit-learn', 'seaborn', 'silx', 'xrayutilities', 
    'ipyvolume', 'ipython_genutils', 'bokeh', 'panel', 'tornado', 'vtk', 
    'colorcet', 'hdf5plugin', 'tabulate'
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CdiUtils'
copyright = '2025, Clément Atlan'
author = 'Clément Atlan'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",       # Extracts docstrings
    "sphinx.ext.napoleon",       # Supports Google-style & NumPy-style docstrings
    "sphinx.ext.viewcode",       # Adds links to source code
    "sphinx.ext.autosummary",    # Generates summary tables
    "sphinx_autodoc_typehints",  # Shows type hints in docs
    "sphinx.ext.doctest",        # Supports doctest
    "sphinx.ext.coverage",       # Coverage extension
    "sphinx.ext.mathjax",        # Math support
    "sphinx.ext.githubpages",    # GitHub pages support
    "sphinx.ext.todo",           # Todo extension
    "sphinx.ext.intersphinx",    # Cross-reference other docs
    # "nbsphinx",                  # Jupyter notebook support - temporarily disabled
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Automatically generate stub files
autosummary_generate = False  # Temporarily disable to test basic build
autodoc_member_order = "bysource"  # Keeps methods in order of appearance


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
todo_include_todos = True

autoclass_content = 'both'

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 2,
    "navbar_align": "left",
    # "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"]
}
