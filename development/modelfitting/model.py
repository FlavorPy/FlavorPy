import numpy as np
import pandas as pd
from lmfit import minimize, Parameters


class LeptonModel:
    """
    A model of leptons with its mass matrices, parameters, ordering, and experimental data.
    """
    def __init__(self, mass_matrix_e=np.identity(3), mass_matrix_n=np.identity(3), parameters=Parameters(),
                 ordering='NO', experimental_data= ???,name='', comments=''):
        """
        LeptonModel
        ----------
        :param mass_matrix_e: 3x3 matrix
            The charged lepton mass matrix. Default is \'np.identity(3)\'.
        :param mass_matrix_n: 3x3 matrix
            The light neutrino mass matrix. Default is \'np.identity(3)\'.
        :param parameters: An object of the \'Parameters\' class of the package lmfit
            The parameters of the model. Default is an empty object \'Parameters()\'.
        :param ordering: str
            Specify whether the neutrino spectrum is normal or inverted ordered. Has to be either \'NO\' or \'IO\'.
            Default is \'NO\'.
        :param experimental_data: An object of the \'ExperimentalDataSet\' class
            The experimental data set used when fitting the model. Default is \'???\'.
        :param name: str
            If you want, you can give the model a name.
        :param comments: str
            If you want, you can write some comments here.
        """
        self.parameters = parameters              # Maybe make this a simple dictionary, so you can evaluate one point.
        self.mass_matrix_e = mass_matrix_e
        self.mass_matrix_n = mass_matrix_n
        self.ordering = ordering
        self.experimental_data = experimental_data
        self.name = name
        self.comments = comments

    def get_residual(self, ...):
        # returns [chi(obs1), chi(obs2), ...] just like the lmfit.minimize needs it
        # have an argument like 'as dictionary' that returns the same result as a dictionary, so you can understand the output


class QuarkModel:
    def __int__(self, parameters={}, mass_matrix_u=np.identity(3), mass_matrix_d=np.identity(3),
                parameterization='standard'):
        self.parameters = parameters
        self.mass_matrix_u = mass_matrix_u
        self.mass_matrix_d = mass_matrix_d
        self.parameterization = parameterization




class Model(QuarkModel, LeptonModel):
    def __init__(self, name, comments):
        self.name = name
        self.comments = comments




