Welcome to NEAT-Python's documentation!
=======================================

.. todo::
  This is a **draft** version of the documentation, for Dr. Allen Smith's `multiparam_funcs branch <https://github.com/drallensmith/neat-python/tree/multiparam_funcs>`_,
  and is likely to *rapidly* change. Please see `neat-python.readthedocs.io <http://neat-python.readthedocs.io/en/latest/>`_ for an official version for the
  NEAT-Python `master branch <https://github.com/CodeReclaimers/neat-python/tree/master>`_.


:abbr:`NEAT (NeuroEvolution of Augmenting Topologies)` is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. NEAT-Python is a pure Python implementation of NEAT, with no dependencies other than the Python standard library.

.. index:: TODO
.. index:: examples

.. note::
  Some of the example code has other dependencies; please see each example's README.md file for additional details and installation/setup instructions for
  the main code for each. In addition to dependencies varying with different examples, visualization of the results (via ``visualize.py`` modules) frequently
  requires `graphviz <https://pypi.python.org/pypi/graphviz>`_ and/or `matplotlib <https://matplotlib.org/users/installing.html>`_. TODO: Improve
  README.md file information for the examples.

.. index:: ! TODO

Support for HyperNEAT and other extensions to NEAT is planned once the fundamental NEAT implementation is
more complete and stable.

For further information regarding general concepts and theory, please see `Selected Publications
<http://www.cs.ucf.edu/~kstanley/#publications>`_ on Stanley's website, or his recent `AMA on Reddit
<https://www.reddit.com/r/IAmA/comments/3xqcrk/im_ken_stanley_artificial_intelligence_professor>`_.

If you encounter any confusing or incorrect information in this documentation, please open an issue in the `GitHub project
<https://github.com/CodeReclaimers/neat-python>`_.

.. _toc-label:

Contents:

.. toctree::
   :maxdepth: 2

   neat_overview
   installation
   config_file
   xor_example
   customization
   activation
   ctrnn
   module_summaries
   genome-interface
   reproduction-interface
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

