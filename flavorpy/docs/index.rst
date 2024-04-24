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

FlavorPy is a Python library for calculations around discrete flavor symmetries. Currently it is split into two parts: 

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

#. Download the files from the `github repository <https://github.com/FlavorPy/FlavorPy/>`_. 

#. Open python and load the files with:

   .. code-block:: python3

      import os
      dir_to_git_folder = "whereever_you_downloaded_the_files_to/FlavorPy/current_version"  # Adjust this to your case !!
      os.chdir(os.path.expanduser(dir_to_git_folder))
      
      import constructterms as ct
      import modelfitting as mf
      
#. Start using the parts of FlavorPy imported as `ct` and `mf`!

   

Development
-----------

This project is under active development! 
The objectives of current development are:

* extending ModelFitting by a MCMC method to study the vicinity of minima
* implementing quark models in ModelFitting
* bringing the two parts, ConstructTerms and ModelFitting, together
* integrating `GAP <https://www.gap-system.org/>`_ and its `SmallGroups` library


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
