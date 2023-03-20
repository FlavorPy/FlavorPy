import numpy as np
import pandas as pd
from lmfit import minimize, Parameters


class LeptonModel:
    def __init__(self, parameters=Parameters(), mass_matrix_e=np.identity(3), mass_matrix_n=np.identity(3),
                 ordering='NO', experimental_data= ,name='', comments=''):
        self.parameters = parameters              # Maybe make this a simple dictionary, so you can evaluate one point.
        self.mass_matrix_e = mass_matrix_e
        self.mass_matrix_n = mass_matrix_n
        self.ordering = ordering
        self.experimental_data = experimental_data
        self.name = name
        self.comments = comments


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




