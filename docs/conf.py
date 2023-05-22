# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re
import sys
import sphinx

sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve

# We also add the directory just above to enable local imports
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'meierlab'
copyright = '2023, MCW-Meier-Lab'
author = 'MCW Meier Lab team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "sphinx_design",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "myst_parser",
    "numpydoc",
]

autosummary_generate = True
autodoc_default_options = {
    "imported-members": True,
    "inherited-members": True,
    "undoc-members": True,
    "member-order": "bysource",
}

numpydoc_show_class_members = False

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3.10/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None)
}
intersphinx_disabled_domains = ["std"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = [".rst", ".md"]

plot_gallery = True

# Latest release version
latest_release = re.match(
    r"v?([0-9]+.[0-9]+.[0-9]+).*",
    os.popen("git describe --tags").read().strip(),
).groups()[0]

# The full current version, including alpha/beta/rc tags.
import meierlab

current_version = meierlab.__version__

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = "meierlab"
html_short_title = "meierlab"

copybutton_prompt_text = ">>> "

sphinx_gallery_conf = {
    "doc_module": "meierlab",
    "reference_url": {"meierlab": None},
    "examples_dirs": "../examples/",
    "gallery_dirs": "auto_examples",
}

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "meierlab",
    "https://github.com/mcw-meier-lab/meierlab",
    "meierlab/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)
