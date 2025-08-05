# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "CdiUtils"
copyright = "2025, Clément Atlan"
author = "Clément Atlan"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Extracts docstrings
    "sphinx.ext.napoleon",  # Supports Google-style & NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.mathjax",  # Math support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
