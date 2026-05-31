# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AI-Based Handover Prediction in UAV-Assisted 3D Heterogeneous Networks (Beyond 5G / 6G-Oriented)'
copyright = '2026, WASSIM MCHICHI, AOUTMANI MOHAMMED AMIN'
author = 'WASSIM MCHICHI, AOUTMANI MOHAMMED AMIN'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "attrs_inline",
    "html_admonition",
    "html_image",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_extra_path = ['image']
html_theme_options = {
    "body_max_width": "none",
}


html_css_files = [
    "custom.css",
]