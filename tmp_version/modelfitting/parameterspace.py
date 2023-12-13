import lmfit
from lmfit import Parameters
import numpy as np


class ParameterDimension:
    def __init__(self, name, sample_fct=None, vary=True, min=-np.inf, max=np.inf, expr=None, brute_step=None):
        """
        A parameter dimension, i.e. one direction in a multidimensional parameter space
        ----------
        :param name: str
            The name of the dimension.
        :param sample_fct: function
            A function that when evaluated as function() returns a float.
        :param vary: bool, optional
            Specifies whether the parameter is varied during a fit.
        :param min: float, optional
            Lower bound. Default is -numpy.inf
        :param max: float, optional
            Upper bound. Default is numpy.inf
        :param expr: str, optional
            A mathematical expression used to constrain the value during the fit. Default is 'None'
        :param brute_step: float, optional
            Step size for grid points, if you use the \'brute\' method when fitting.
        """
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
    def __init__(self, name='Parameter space'):  # Maybe exclude the option to define the params dict
        """
        A parameter space. This object is a dictionary that contains ParameterDimension objects.
        ----------
        :param name: str, optional
            You can give your parameter space a name.
        """
        super().__init__(self)
        self.name = name

    def __repr__(self):
        return self.name

    def add_dim(self, name, sample_fct=None, vary=True, min=-np.inf, max=np.inf, expr=None, brute_step=None):
        """
        Adds a dimension to your parameter space. Can also be used to update or overwrite an existing dimension.
        ----------
        :param name: str
            The name of the dimension.
        :param sample_fct: function, optional
            A function that when evaluated as function() returns a float.
            Default is numpy.random.uniform
        :param vary: bool, optional
            Specifies whether the parameter is varied during a fit.
        :param min: float, optional
            Lower bound. Default is -numpy.inf
        :param max: float, optional
            Upper bound. Default is numpy.inf
        :param expr: str, optional
            A mathematical expression used to constrain the value during the fit. Default is 'None'
        :param brute_step: float, optional
            Step size for grid points, if you use the \'brute\' method when fitting.
        """
        self[name] = ParameterDimension(name=name, sample_fct=sample_fct, vary=vary,
                                        min=min, max=max, expr=expr, brute_step=brute_step)

    def random_pt(self) -> lmfit.Parameters:
        """
        Draws a sample in your parameter space.
        ----------
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
