.. FlavorPy documentation master file, created by
   sphinx-quickstart on Wed Dec 13 18:06:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Contens:
   
   constructterms
   
   modelfitting
   
   examples/examples_page
   

FlavorPy documentation
======================

FlavorPy is a Python library for calculations around discrete flavor symmetries in particle physics. Currently it is split into two parts: 

.. grid:: 2
   :gutter: 4
   :padding: 2
   
   .. grid-item-card:: ConstructTerms
      
      ..
         The *constructterms* part allows you to calculate group theoretical tensor products and therefore find the invariant terms in the action.
         
      Calculate tensor product and find invariant terms in the action.
      
      .. button-ref:: constructterms
         :click-parent:
         :color: muted
         :align: center
         
         Go to ConstructTerms
      
      
   .. grid-item-card:: ModelFitting
      
      ..
         The *modelfitting* part is concerned with fitting a model to experimental data. More specifically flavor observables, i.e. masses and mixing, for given mass matrices with an associated parameterspace can be compared and fitted to experimental data. The minimization heavily relies on `lmfit <https://lmfit.github.io/lmfit-py/>`_. The goal of current development is to bring the two parts together, integrate GAP, have quark models, and extend the modelfitting with a MCMC method to study the vicinity of minima.
         
      Build a Lepton or Quark model and fit it to the experimental data.
      
      .. button-ref:: modelfitting
         :click-parent:
         :color: muted
         :align: center
         
         Go to ModelFitting

   
Install
-------

You can install FlavorPy from `PyPI <https://pypi.org/project/flavorpy/>`_ with pip by running

.. code-block:: 
   
   pip install flavorpy
      
      
Alternatively, you can:

#. Download the files from the `github repository <https://github.com/FlavorPy/FlavorPy/>`_. 

#. Open python and load the files with:

   .. code-block:: python3

      import os
      dir_to_git_folder = "whereever_you_downloaded_the_files_to/FlavorPy/current_version"  # Adjust this to your case !!
      os.chdir(os.path.expanduser(dir_to_git_folder))
      
      import constructterms as ct
      import modelfitting as mf
      
#. Start using the parts of FlavorPy imported as `ct` and `mf`!

Examples
--------

Introductory examples
~~~~~~~~~~~~~~~~~~~~~

.. grid:: 2
   :gutter: 4
   :padding: 2
   
   .. grid-item-card:: Getting started with ConstructTerms
      
      .. button-ref:: examples/simpleexample_constructterms
         :click-parent:
         :color: muted
         :align: center
         
         Quick start ConstructTerms
         
         
   .. grid-item-card:: Getting started with ModelFitting
      
      .. button-ref:: examples/simpleexample_modelfitting
         :click-parent:
         :color: muted
         :align: center
         
         Quick start ModelFitting
      
      
   .. grid-item-card:: Some more features of ConstructTerms
      
      .. button-ref:: examples/detailedexample_constructterms
         :click-parent:
         :color: muted
         :align: center
         
         Advanced features of ConstructTerms
         
   .. grid-item-card:: Some more features of ModelFitting
      
      .. button-ref:: examples/detailedexample_modelfitting
         :click-parent:
         :color: muted
         :align: center
         
         Advanced features of ModelFitting
         
         
Further examples
~~~~~~~~~~~~~~~~

.. grid:: 2
   :gutter: 4
   :padding: 2
   
   .. grid-item-card:: arXiv:2006.03058
      
      Reproduce the model fitting results of the paper 
      "Double Cover of Modular S4 for Flavour Model Building"
      by P. P. Novichkov, J. T. Penedo, and S. T. Petcov
      
      .. button-ref:: examples/arxiv2006dot03058
         :click-parent:
         :color: muted
         :align: center
         
         Go to 2006.03058 Example


Development
-----------

This project is under active development! 
The objectives of current development are:

* bringing the two parts, ConstructTerms and ModelFitting, together
* integrating `GAP <https://www.gap-system.org/>`_ and its `SmallGroups` library

If you want to contribute, please feel free to contact `Alexander Baur <alexander.baur@tum.de>`_.


Citing FlavorPy
---------------

If FlavorPy contributes to a project that leads to a publication, please acknowledge this fact by citing:

`A. Baur, "FlavorPy", Zenodo, 2024, doi: 10.5281/zenodo.11060597 <https://doi.org/10.5281/zenodo.11060597>`_.

Here is an example of a BibTex entry:

.. code-block:: tex
   
      @software{FlavorPy,
        author        = {Baur, Alexander},
        title         = "{FlavorPy}",
        year          = {2024},
        publisher     = {Zenodo},
        doi           = {10.5281/zenodo.11060597},
        url           = "\url{https://doi.org/10.5281/zenodo.11060597}"
      } 

When using the NuFit experimental data, please also cite:

`I. Esteban, M. C. González-García, M. Maltoni, T. Schwetz, and A. Zhou, The fate of hints: updated global analysis of three-flavor neutrino oscillations, JHEP 09 (2020), 178, arXiv:2007.14792 [hep-ph], https://www.nu-fit.org <https://link.springer.com/article/10.1007/JHEP09(2020)178>`_.


Credit
------

FlavorPy makes use of experimental data obtained by NuFit published in `JHEP 09 (2020) 178 <http://dx.doi.org/10.1007/JHEP09(2020)178>`_, `arXiv:2007.14792 <http://arxiv.org/abs/2007.14792>`_, and their website `www.nu-fit.org <http://www.nu-fit.org/>`_. Please cite NuFit if you use their experimental data.



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
