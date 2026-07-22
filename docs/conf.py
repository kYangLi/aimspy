"""Configuration file for the Sphinx documentation builder."""

# -- Path setup --------------------------------------------------------------
# Add the project root (parent of docs/) to sys.path so that autodoc can
# import the aimspy package without requiring an editable install.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
project = "AimsPy"
copyright = "2026, The DeepH team"
author = "The DeepH team"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
#   extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
#   ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

autosummary_generate = True

master_doc = "index"

autodoc_typehints = "none"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/aimspy_theme.css"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "./_image/logo-small.svg"
html_favicon = "./_image/logo-fav.svg"

# title of the website
html_title = ""

html_theme_options = {
    "repository_url": "https://github.com/kYangLi/aimspy",
    "use_repository_button": True,
    "use_issues_button": True,
    "show_prev_next": True,
    "show_navbar_depth": 1,
}

# -- Options for myst ----------------------------------------------
# Notebooks are not executed (AimsPy ships no .ipynb examples currently).
nb_execution_mode = "off"
nb_execution_timeout = 30

myst_enable_extensions = [
    "dollarmath",
]

# Generate anchor IDs for headers so that cross-file references like
# `./key_concepts.md#section-name` resolve correctly.
myst_heading_anchors = 4

# -- Extension configuration -------------------------------------------------

always_document_param_types = True
autosectionlabel_prefix_document = True
