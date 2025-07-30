.. meierlab documentation master file, created by
   sphinx-quickstart on Fri May 19 13:27:07 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the MCW Meier Lab documentation!
===========================================

.. container:: index-paragraph

   The **meierlab** project provides functional tools to aid
   lab members in downloading, processing, visualizing,
   and analyzing imaging data.

.. note::

   This project is under active development and is intended
   for academic research only.


.. grid::

   .. grid-item-card:: :fas:`rocket` Quickstart
      :link: readme
      :link-type: ref
      :columns: 12 12 4 4
      :class-card: sd-shadow-md
      :class-title: sd-text-primary
      :margin: 2 2 0 0

      Get started with meierlab.


   .. grid-item-card:: :fas:`th` Examples
      :link: auto_examples/index.html
      :link-type: url
      :columns: 12 12 4 4
      :class-card: sd-shadow-md
      :class-title: sd-text-primary
      :margin: 2 2 0 0

      Discover functionalities by reading examples.


   .. grid-item-card:: :fas:`book` User guide
      :link: user_guide
      :link-type: ref
      :columns: 12 12 4 4
      :class-card: sd-shadow-md
      :class-title: sd-text-primary
      :margin: 2 2 0 0

      Learn about the tools provided.

Featured examples
-----------------

.. grid::

   .. grid-item-card:: :fas:`chart-line` ExploreASL Quality Assessment
      :link: auto_examples/02_preproc/exploreasl_quality_example.html
      :link-type: url
      :columns: 12 12 6 6
      :class-card: sd-shadow-md
      :class-title: sd-text-primary
      :margin: 2 2 0 0

      Comprehensive quality assessment for ExploreASL processed data with configurable metrics and interactive visualizations.

   .. grid-item-card:: :fas:`brain` Atlas Visualization
      :link: auto_examples/01_plotting/plot_atlas.html
      :link-type: url
      :columns: 12 12 6 6
      :class-card: sd-shadow-md
      :class-title: sd-text-primary
      :margin: 2 2 0 0

      Plot and visualize brain atlases and parcellations.


.. toctree::
   :hidden:
   :includehidden:
   :titlesonly:

   quickstart.rst
   auto_examples/index.rst
   user_guide.rst
   modules/index.rst

.. toctree::
   :hidden:
   :caption: Development

   changes/whats_new.rst
   authors.rst
   license.rst
   contrib.rst
   Github Repository <https://github.com/mcw-meier-lab/meierlab>
