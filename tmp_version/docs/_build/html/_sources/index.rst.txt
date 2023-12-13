.. FlavorPy documentation master file, created by
   sphinx-quickstart on Wed Dec 13 18:06:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FlavorPy's documentation!
====================================

**FlavorPy** is a Python library for calculations around discrete flavor symmetries. Currently it is split in two parts. The *constructterms* part allows you to calculate group theoretical tensor products and therefore find the invariant terms in the action. The *modelfitting* part is concerned with fitting a model to experimental data. More specifically flavor observables, i.e. masses and mixing, for given mass matrices with an associated parameterspace can be compared and fitted to experimental data. The minimization heavily relies on `lmfit <https://lmfit.github.io/lmfit-py/>`_. The goal of current development is to bring the two parts together, integrate GAP, have quark models, and extend the modelfitting with a MCMC method to study the vicinity of minima.

.. note::
   This project is under active development.

.. toctree::
   :maxdepth: 3
   :caption: Contens:
   
   .. modules
   
   constructterms
   
   modelfitting



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
