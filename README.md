# FlavorPy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11060597.svg)](https://doi.org/10.5281/zenodo.11060597)
[![PyPI Latest Release](https://img.shields.io/pypi/v/flavorpy.svg)](https://pypi.org/project/flavorpy/)


What is FlavorPy?
-----------------

**FlavorPy** is a Python library for calculations around discrete flavor symmetries in particle physics. Currently, it is split into two parts:

* The **constructterms** part allows you to calculate group theoretical tensor products and therefore find the invariant terms in the action.

* The **modelfitting** part is concerned with fitting a model to experimental data. More specifically flavor observables, i.e. masses and mixing, for given mass matrices with an associated parameter space can be compared and fitted to experimental data. The minimization heavily relies on [lmfit](https://lmfit.github.io/lmfit-py/).


How to install FlavorPy?
------------------------

You can install FlavorPy from [PyPI](https://pypi.org/project/flavorpy/) with pip by running

```bash

   pip install flavorpy
```

Alternatively, you can:

1. Download the files from the [github repository](https://github.com/FlavorPy/FlavorPy/). 

2. Open python and load the files with:

```python
    import os
    dir_to_git_folder = "home/.../FlavorPy/current_version"  # Adjust this to your case !!
    os.chdir(os.path.expanduser(dir_to_git_folder))

    import constructterms as ct
    import modelfitting as mf
```

3. Start using the FlavorPy packages imported as `ct` and `mf`!


Documentation
-------------

A documentation is hosted on [https://flavorpy.github.io/FlavorPy/](https://flavorpy.github.io/FlavorPy/).
This site also contains examples of how to use the code.


Current development
-------------------

The goal of current development is bringing the two parts together and integrating GAP or SageMath to ConstructTerms.
If you want to contribute, please feel free to contact [Alexander Baur](mailto:alexander.baur@tum.de)


Citing FlavorPy
---------------

If FlavorPy contributes to a project that leads to a publication, please acknowledge this fact by citing 

[A. Baur, "FlavorPy", Zenodo, 2024, doi: 10.5281/zenodo.11060597](https://doi.org/10.5281/zenodo.11060597).

Here is an example of a BibTex entry:

```tex
    @software{FlavorPy,
      author        = {Baur, Alexander},
      title         = "{FlavorPy}",
      year          = {2024},
      publisher     = {Zenodo},
      version       = {v0.1.0},
      doi           = {10.5281/zenodo.11060597},
      url           = "\url{https://doi.org/10.5281/zenodo.11060597}"
    } 
```

When using the NuFit experimental data, please also cite 

[I. Esteban, M. C. González-García, M. Maltoni, T. Schwetz, and A. Zhou, The fate of hints: updated global analysis of three-flavor neutrino oscillations, JHEP 09 (2020), 178, arXiv:2007.14792 [hep-ph], https://www.nu-fit.org](https://link.springer.com/article/10.1007/JHEP09(2020)178).


Credit
------

This package uses experimental data obtained by NuFit published in [JHEP 09 (2020) 178](http://dx.doi.org/10.1007/JHEP09(2020)178), [arXiv:2007.14792](http://arxiv.org/abs/2007.14792), and their website [www.nu-fit.org](http://www.nu-fit.org/).

