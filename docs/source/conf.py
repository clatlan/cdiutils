# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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

    'sphinx.ext.doctest',
    # 'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    # 'sphinx.ext.graphviz',
    # 'sphinx.ext.inheritance_diagram',
    'nbsphinx',
    'nbsphinx_link',
    # 'sphinxarg.ext'
]

# Automatically generate stub files
autosummary_generate = True  
autodoc_member_order = "bysource"  # Keeps methods in order of appearance


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
todo_include_todos = True

autoclass_content = 'both'

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 2,
    "navbar_align": "left",
    # "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"]
}

# html_sidebars = {
#     "**": ["globaltoc.html", "sidebar-nav-bs"],
#     # "**": ["localtoc.html"],
#     # "**": ["sidebar-nav-bs"],
#     # "<page_pattern>": ["index", "manual-intro", "tutorials", "manual"]
# }
