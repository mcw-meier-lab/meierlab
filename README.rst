Meier Lab tools
---------------

The ``meierlab`` package is developed by and for members of
MCW's Meier Lab to help with our day-to-day tasks in our
neuroimaging research. While some tools may only be useful
to those on campus within our lab, others may be helpful to
researchers outside our lab. Therefore, we are happy to make
this code available to others in the hope that it will enable
more efficient and reproducible research practices!


Installation
------------

To install the latest release, we recommend setting up a virtual
environment through ``conda`` or ``venv``.
Conda::

    conda create -n py-meierlab python=3.9
    conda activate py-meierlab

Venv::
    python -m venv py-meierlab
    source py-meierlab/bin/activate  

You can then install ``meierlab`` with pip::

    python -m pip install -U meierlab


Check your installation in a python/iPython session::

    import meierlab


Contributing
------------

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.


License
-------

Copyright (c) 2023, MCW Meier-Lab.
``meierlab`` was created by Lezlie España as part of the MCW Meier Lab. 
It is licensed under the BSD 3-Clause License.


Credits
-------

``meierlab`` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

Many thanks to the python neuroimaging packages that make our work possible, 
especially the `NIPY developers <https://nipy.org/>`.
