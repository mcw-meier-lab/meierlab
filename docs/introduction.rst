Introduction
============


What is ``meierlab``?
=====================

``meierlab`` is a package built by members of the MCW
Meier Lab to help facilitate our day-to-day lab activities
and analyses on neuroimaging data. A lot of our processes 
are common across projects and datasets
It contains tools to download data, visualize data for QC,
perform outlier analyses, process connectivity data, and more.

If you're new to programming and/or imaging data, don't worry.
Here, you'll find plenty of examples and documentation to
help you get started.


.. _quick_start:

Using ``meierlab`` for the first time
=====================================

``meierlab`` is a Python library. If you have never used 
Python before, you may find it useful to explore some other
tutorials and wikis before going through this guide:

`general introduction about Python <http://www.learnpython.org/>`_
`introduction to using Python for science <http://scipy-lectures.github.io/>`_ 

.. note::

    Please check out the internal lab wiki if you're on campus for
    additional information about our data specifically.


First steps with ``meierlab``
-----------------------------

At this stage, you should have :ref:`installed <readme>` ``meierlab`` and 
opened a Jupyter notebook or an IPython / Python session.  
First, load ``meierlab`` with

.. code-block:: default

    import meierlab


Learning from the API references
--------------------------------

All modules are described in the :ref:`API references <modules>`.


Learning from the examples
--------------------------

``meierlab`` comes with a lot of :ref:`examples/tutorials <tutorial_examples>`.
Going through them should give you a precise overview of what you can achieve with this package.


Finding help
------------

We rely pretty heavily on other neuroimaging python and general software
packages that may be of interest, but whose documentation is outside the
scope of this package...

- The documentation of `scikit-learn <https://scikit-learn.org/stable/>`_ explains each method with tips on practical use and examples.  While not specific to neuroimaging, it is often a recommended read.

- (For Python beginners) A quick and gentle introduction to scientific computing with Python with the `scipy lecture notes <http://scipy-lectures.github.io/>`_. Moreover, you can use ``meierlab`` with `Jupyter <http://jupyter.org>`_ notebooks or `IPython <http://ipython.org>`_ sessions. They provide an interactive environment that greatly facilitates debugging and visualisation.

- From the Nipy ecosystem: `Nibabel <https://nipy.org/nibabel/>`_ is a great tool for manipulating imaging data in various formats and `Nilearn <https://nilearn.github.io/stable/index.html>`_ contains many useful analysis and visualization modules.

- Other useful python packages to know: `Numpy <https://numpy.org/doc/stable/>`_ is especially useful for matrix manipulation and general numeric functions, `Pandas <https://pandas.pydata.org/docs/index.html>`_ has many utilities for handling tabular data, and `Networkx <https://networkx.org/>`_ is used for graph creation, analysis, and visualization.

- You will also see references to other neuroimaging packages that aren't necessarily written in Python: `AFNI <https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/index.html>`_, `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_, among others.





