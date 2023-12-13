import lmfit
from lmfit import Parameters
import numpy as np


class ParameterDimension:
    """
    A parameter dimension, i.e. one direction in a multidimensional parameter space

    :param name: The name of the dimension.
    :type name: str
    :param sample_fct: A function that when evaluated as function() returns a float.
    :type sample_fct: function
    :param vary: Specifies whether the parameter is varied during a fit.
    :type vary: bool, optional
    :param min: Lower bound. Default is -numpy.inf
    :type min: float, optional
    :param max: Upper bound. Default is numpy.inf
    :type max: float, optional
    :param expr: A mathematical expression used to constrain the value during the fit. Default is 'None'
    :type expr: str, optional
    :param brute_step: Step size for grid points, if you use the \'brute\' method when fitting.
    :type brute_step: float, optional
    """
    def __init__(self, name, sample_fct=None, vary=True, min=-np.inf, max=np.inf, expr=None, brute_step=None):

        if sample_fct is None:
            def default_fct():
                return np.random.uniform
            sample_fct = default_fct

        self.name = name
        self.sample_fct = sample_fct
        self.vary = vary
        self.min = min
        self.max = max
        self.expr = expr
        self.brute_step = brute_step

    def __repr__(self):
        return f"Parameter '{self.name}'"


class ParameterSpace(dict):
    """
    A parameter space. This object is a dictionary that contains
    :py:meth:`~modelfitting.parameterspace.ParameterDimension` objects.

    :param name: You can give your parameter space a name.
    :type name: str, optional
    """
    def __init__(self, name='Parameter space'):  # Maybe exclude the option to define the params dict
        super().__init__(self)
        self.name = name

    def __repr__(self):
        return self.name

    def add_dim(self, name, sample_fct=None, vary=True, min=-np.inf, max=np.inf, expr=None, brute_step=None):
        """
        Adds a dimension to your parameter space. Can also be used to update or overwrite an existing dimension.

        :param name: The name of the dimension.
        :type name: str
        :param sample_fct: A function that when evaluated as function() returns a float.
            Default is numpy.random.uniform.
        :type sample_fct: function
        :param vary: Specifies whether the parameter is varied during a fit.
        :type vary: bool, optional
        :param min: Lower bound. Default is -numpy.inf
        :type min: float, optional
        :param max: Upper bound. Default is numpy.inf
        :type max: float, optional
        :param expr: A mathematical expression used to constrain the value during the fit. Default is 'None'
        :type expr: str, optional
        :param brute_step: Step size for grid points, if you use the \'brute\' method when fitting.
        :type brute_step: float, optional
        """
        self[name] = ParameterDimension(name=name, sample_fct=sample_fct, vary=vary,
                                        min=min, max=max, expr=expr, brute_step=brute_step)

    def random_pt(self) -> lmfit.Parameters:
        """
        Draws a sample in your parameter space.

        :return: A lmfit.Parameters object.
        """
        params = Parameters()
        for name in self:
            params.add(name=name,
                       value=self[name].sample_fct(),  # Evaluates the sample_fct, i.e. draws a sample
                       vary=self[name].vary,
                       min=self[name].min,
                       max=self[name].max,
                       expr=self[name].expr,
                       brute_step=self[name].brute_step)

        return params
