Install
=======

Below we assume you have the default Python environment already configured on
your computer and you intend to install ``ardent`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the released version
----------------------------

Install the current release of ``ardent`` with ``pip``::

    $ pip install ardent

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade ardent

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user ardent

Alternatively, you can manually download ``ardent`` from
`GitHub <https://github.com/neurodata/ardent/releases>`_  or
`PyPI <https://pypi.python.org/pypi/ardent>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip install .

Python package dependencies
---------------------------
ARDENT requires the following packages:

- numpy
- matplotlib
- scipy
- scikit-learn
- simpleitk
- nibabel
- nilearn
- pytorch


Hardware requirements
---------------------
`ARDENT` package requires only a standard computer with enough RAM to support the in-memory operations. 

Testing
-------
ARDENT uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.