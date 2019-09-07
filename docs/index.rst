.. ARDENT documentation master file, created by
   sphinx-quickstart on Thu May  2 14:50:03 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ARDENT_: Image Registration Abstracted
======================================

.. _ARDENT: https://github.com/neurodata/ardent

ARDENT is a high-level nonlinear image registration package.

Motivation
----------

Experimental neuroscience produces a stunning amount of imaging data from light or electron microscopy, MRI, and other 3D modalities. 
To be of real use these datasets must be interpreted with respect to each other and to refined standards: well-characterized image datasets called atlases. 
To build these interpretations, dense spatial alignments must be computed. This process is known as image registration, 
in which one image is optimally deformed, or flowed, until it aligns with another. 
Accurate registration is challenged by the large scale of imaging data and the heterogeneity across species scales and modalities.  
The viability of current tools is limited to images acquired in the most standard settings, 
restricting our ability to analyze data from novel experiments performed in a majority of labs.

ARDENT, or Affine and Regularized DEformative Numeric Transform, is an accessible pure-python image registration package in development with these neuroimaging challenges in mind.

Python
------

Python is a powerful programming language that allows concise expressions of network
algorithms.  Python has a vibrant and growing ecosystem of packages that
GraSPy uses to provide more features such as numerical linear algebra and
plotting.  In order to make the most out of GraSPy you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Free software
-------------

ARDENT is free software; you can redistribute and/or modify it 
under the terms of the :doc:`Apache-2.0 </license>`. We welcome contributions.
Join us on `GitHub <https://github.com/neurodata/ardent>`_.

Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   tutorial
   reference/index
   license



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
