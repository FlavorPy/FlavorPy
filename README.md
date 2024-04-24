# FlavorPy


What is FlavorPy?
-----------------

**FlavorPy** is a Python library for calculations around discrete flavor symmetries in particle physics. Currently it is split in two parts:

* The **constructterms** part allows you to calculate group theoretical tensor products and therefore find the invariant terms in the action.

* The **modelfitting** part is concerned with fitting a model to experimental data. More specifically flavor observables, i.e. masses and mixing, for given mass matrices with an associated parameterspace can be compared and fitted to experimental data. The minimization heavily relies on [lmfit](https://lmfit.github.io/lmfit-py/).


How to install FlavorPy?
------------------------

You can install FlavorPy from PyPI with pip by running

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

A documentation is hostet on [https://flavorpy.github.io/FlavorPy/](https://flavorpy.github.io/FlavorPy/).
This site also contains examples on how to use the code.


Current development
-------------------

The goal of current development is to bring the two parts together, integrate GAP, have quark models, and extend the modelfitting with a MCMC method to study the vicinity of minima.
If you want to contribute, please feel free to contact [Alexander Baur](mailto:alexander.baur@tum.de)


Credit
------

This package uses experimental data obtained by NuFit published in [JHEP 09 (2020) 178](http://dx.doi.org/10.1007/JHEP09(2020%29178), [arXiv:2007.14792](http://arxiv.org/abs/2007.14792), and their website [www.nu-fit.org](http://www.nu-fit.org/).

