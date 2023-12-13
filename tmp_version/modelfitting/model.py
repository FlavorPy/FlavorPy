import numpy as np
import pandas as pd
from .experimental_data.NuFit52.nufit52_chisqprofiles import NuFit52_NO
from .parameterspace import ParameterSpace
from .mixingcalculations import calculate_lepton_dimensionless_observables, calculate_lepton_observables
from .fit import Fit
from copy import deepcopy


class FlavorPyError(Exception):
    # Raises an Exception
    pass


class LeptonModel:
    """
    A model of leptons with its mass matrices, parameters, ordering, and experimental data.
    """
    def __init__(self, mass_matrix_e=None, mass_matrix_n=None, parameterspace=None,
                 ordering='NO', experimental_data=NuFit52_NO, fitted_observables='all',
                 name='Lepton Model', comments='', fit_results=None):
        """
        LeptonModel
        ----------
        :param mass_matrix_e: function
            The charged lepton mass matrix M, for Phi_left M Phi_right. This should be a function with one argument,
            namely a dictionary of parameters and the corresponding value.
            E.g. def m_e(params):
                     return params['c']*numpy.identity(3)
                 M = LeptonModel(mass_matrix_e=m_e, ...)
            Default is a function that returns \'numpy.identity(3)\'.
            The program in its current state only gives the dimensionless mass ratios m_e/m_mu and m_mu/m_tau. It is
            advisable to only use dimensionless parameters and simply ignore the overall mass scale, as it would only
            confuse the fit.
            The convention Phi_left M Phi_right is used, where left and right indicates left- and right-handed chiral
            fields, respectively. I.e. L_i^c M_ij e_j, where L refers to the left-handed lepton doublet and e is the
            right-handed charged lepton field, and i,j=1,2,3.
            If you use the other convention, i.e. left-handed fields on the right-hand-side, simply transpose your mass
            matrix.
        :param mass_matrix_n: function
            The light neutrino mass matrix M, for Phi_left M Phi_right.
            Should be a function with one argument, namely a dictionary of parameters.
            Default is a function that returns \'numpy.identity(3)\'.
            It is !strongly! recommended to only use dimensionless parameters and one overall mass scale with the exact
            name \'n_scale\', i.e. mass_matrix_n = params[\'n_scale\']*dimensionless_mass_matrix(params). Otherwise, the
            program will add a parameter \'n_scale\' itself to the ParameterSpace and mass_matrix_n.
        :param parameterspace: An object of the \'ParameterSpace\' class
            The parameterspace of the model. See documentation of \'ParameterSpace\'.
            Default is an empty parameter space.
        :param ordering: str
            Specify whether the neutrino spectrum is normal or inverted ordered. Has to be either \'NO\' or \'IO\'.
            Default is \'NO\'.
        :param experimental_data: An object of the \'ExperimentalData\' class
            The experimental data used when fitting the model. For more information on the structure please confer the
            documentation of the \'ExperimentalData\' class.
            The default is \'NuFit52_NO\', i.e. the data of NuFit v5.2 for Normal Ordering taking into account results
            of SK. Please consider citing NuFit (www.nu-fit.org) when using this data.
        :param fitted_observables: list, optional
            A list of the observables that is considered when calculating chi-square and, hence, when making the fit.
            Default is \'['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'm21^2', 'm3l^2']\'.
        :param name: str, optional
            If you want, you can give the model a name. This name will be used as __repr__.
        :param comments: str, optional
            If you want, you can write some comments here.
        :param fit_results: list, optional
            It is possible to store the results of make_fits() in this list. It is of course also possible to load
            the results from an external calculation into this list.
        """

        def triv_mat(params):
            return np.identity(3)
        if mass_matrix_e is None:
            mass_matrix_e = triv_mat
        if mass_matrix_n is None:
            mass_matrix_n = triv_mat
        if parameterspace is None:
            parameterspace = ParameterSpace()
        if fitted_observables in ['all', 'auto', 'full', 'everything']:
            fitted_observables = ['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'm21^2', 'm3l^2']
        if fit_results is None:
            fit_results = []

        # Create the dimensionless fitted observables
        if 'm21^2' in fitted_observables:
            if 'm3l^2' in fitted_observables:
                if 'r' not in experimental_data.data:
                    # Todo: add a function to ExperimentalData that automatically computes 'r' out of m21^2 and m3l^2
                    raise NameError("""Your experimental data set has no info on \'r\', i.e. r=m21^2/m3l^2.
                                    Please define add the experimental data for 'r' into your experimental dataset.""")
                fitted_observables_dimless = [key for key in fitted_observables if key not in ['m21^2', 'm3l^2']]
                fitted_observables_dimless.append('r')
            else:
                raise NotImplementedError("""I cannot fit only \'m21^2\' without fitting \'m3l^2\'.""")
        elif 'm3l^2' in fitted_observables:
            raise NotImplementedError("""I cannot fit only \'m21^2\' without fitting \'m3l^2\'.""")
        else:
            fitted_observables_dimless = fitted_observables

        self.parameterspace = parameterspace
        self.mass_matrix_e = mass_matrix_e
        self.mass_matrix_n = mass_matrix_n
        self.ordering = ordering
        self.experimental_data = experimental_data
        self.fitted_observables = fitted_observables
        self.fitted_observables_dimless = fitted_observables_dimless
        self.name = name
        self.comments = comments
        self.fit_results = fit_results

        # Add a scale to the neutrino mass matrix, if there is not already one.
        if 'n_scale' not in self.parameterspace:
            # add 'n_scale' to parameterspace
            def const_fct():
                return 1

            self.parameterspace.add_dim(name='n_scale', sample_fct=const_fct, vary=False)

            # add the scale to the mass matrix
            def new_mass_matrix(params):
                return params['n_scale'] * mass_matrix_n(params)

            self.mass_matrix_n = new_mass_matrix

        else:
            self.parameterspace['n_scale'].vary = False  # don't worry we are going to fit it, just

    def __repr__(self):
        return self.name

    def copy(self):
        return deepcopy(self)

    def get_dimless_obs(self, params: dict) -> dict:
        obs = calculate_lepton_dimensionless_observables(mass_matrix_e=self.mass_matrix_e(params),
                                                         mass_matrix_n=self.mass_matrix_n(params),
                                                         ordering=self.ordering)
        return obs

    def get_obs(self, params: dict) -> dict:
        """
        Get a dictionary of all lepton observables for a point in parameterspace.
        ----------
        :param params: dict
            The point in parameter space.
        :return: dict
            All lepton observables, i.e. {'me/mu':0.0048, ...}
        """
        # Get experimental best fit value for squared neutrino mass differences m_21^2 and m_3l^2
        try:
            m21sq_best = self.experimental_data.data_table['m21^2']['best']
        except:
            m21sq_best = None
        try:
            m3lsq_best = self.experimental_data.data_table['m3l^2']['best']
        except:
            m3lsq_best = None

        if 'n_scale' not in params:
            params['n_scale'] = 1

        # Calculate all lepton observables (while simultaneously fitting the neutrino scale)
        obs = calculate_lepton_observables(mass_matrix_e=self.mass_matrix_e(params),
                                           mass_matrix_n=self.mass_matrix_n(params),
                                           ordering=self.ordering, m21sq_best=m21sq_best, m3lsq_best=m3lsq_best)
        return obs

    def residual(self, params: dict) -> list:  # there can be no other arguments than params! Otherwise, you need to adjust fit.py!
        """
        The residual used to make the dimensionless fit.
        ----------
        :param params: dict
            The point in parameterspace.
        :return: list
            A list of values of individual chis (not chi-squares!). Only dimensionless observables are being considered.
        """
        # This is the residual used for the 'fast_fit' that only fits dimensionless observables.
        observables = self.get_dimless_obs(params)
        chisq_list = self.experimental_data.get_chisq_list(values=observables,
                                                           considered_obs=self.fitted_observables_dimless)
        return np.sqrt(np.abs(chisq_list))  # The lmfit minimizer wants chi and not chi-square! It takes a list of single chi.

    def get_chisq(self, params=None, observables=None) -> float:
        """
        Returns the value of chi-square for a given point in parameter space.
        ----------
        :param params: dict
            The point in parameterspace. Alternatively, you can also insert the observables.
            Default is None.
        :param observables: dict
            Alternatively, you can directly insert a dictionary with the observables. Note that the value of \'params\'
            is ignored one you define \'observables\'! Also you need to define either \'params\' or \'observables\'
            Default is None.
        :return: loat
            The value of chi-square.
        """
        if observables is None:
            if params is None:
                raise NameError("""You need to define either params or observables to call LeptonModel.calc_chisq()""")
            observables = self.get_obs(params)

        return self.experimental_data.get_chisq(values=observables, considered_obs=self.fitted_observables)

    def print_chisq(self, params: dict):
        """
        Prints the value of all observables and the associated contribution to chi-square. Also prints total chi-square.
        ----------
        :param params: dict
            The point in parameterspace
        """
        observables = self.get_obs(params)
        chisq_list = self.experimental_data.get_chisq_list(values=observables, considered_obs=self.fitted_observables)
        chisq_dict = {self.fitted_observables[i]: chisq_list[i] for i in range(len(self.fitted_observables))}
        for obs_name in self.fitted_observables:
            print(f"'{obs_name}': {observables[obs_name]},   chisq: {chisq_dict[obs_name]}")
        print(f"Total chi-square: {np.sum(chisq_list)}")

    def make_fit(self, points: int, **fitting_kwargs) -> pd.DataFrame:
        """
        Does the fit for a specific number of random points in parameterspace.
        ----------
        :param points: int
            The number of random points in parameter space you want to fit.
            If you want to fit a specific starting point in parameter space, adjust the \'sampling_fct\' in your
            ParameterSpace.
        :param fitting_kwargs: properties of the Fit class
            You can add keyword arguments that will be passed down to the Fit object used to make the fit.
            Please see the documentation of the Fit class for the specific keyword arguments. Of course, the keywords
            \'model\' and \'params\' can not be passed down to Fit.
        :return: DataFrame object from pandas
            The result of the fit is returned in form of a pandas.DataFrame.
            Note that several (default:4) minimization algorithms are applied consecutively to one random point. Since
            the results of the intermediate steps are also written into the resulting DataFrame, it has more rows than
            the number entered as \'points\'.
        """
        df = self.dimless_fit(points, **fitting_kwargs)
        df = self.complete_fit(df)
        return df

    def dimless_fit(self, points: int, **fitting_kwargs) -> pd.DataFrame:
        """
        Calling this function fits the dimensionless parameters of the model.
        The procedure of Model.make_fit() can be split into the fitting of dimensionless parameters
        (with Model.dimless_fit) and the fitting of the dimensionful ones (with Model.complete_fit), where
        Model.complete_fit() also adds the observables to the resulting pandas.DataFrame.
        This function comes in handy when running fits on an external machine, e.g. a cluster, since the result of
        dimless_fit() uses a smaller amount of memory when stored into a file compared to the result of
        complete_fit(). You can transfer the files from dimless_fit() to your local machine and easily run
        complete_fit() since complete_fit() does not take a lot of time compared to dimless_fit().
        ----------
        :param points: int
            The number of random points in parameter space you want to fit.
            If you want to fit a specific starting point in parameter space, adjust the \'sampling_fct\' in your
            ParameterSpace.
        :param fitting_kwargs: properties of the Fit class
            You can add keyword arguments that will be passed down to the Fit object used to make the fit.
            Please see the documentation of the Fit class for the specific keyword arguments. Of course, the keywords
            \'model\' and \'params\' can not be passed down to Fit.
        :return: DataFrame object from pandas
            The result of the fit is returned in form of a pandas.DataFrame.
            Note that several (default:4) minimization algorithms are applied consecutively to one random point. Since
            the results of the intermediate steps are also written into the resulting DataFrame, it has more rows than
            the number entered as \'points\'.
        """
        # Fast refers to that only the dimensionless observables are fitted, and the resulting pd.DataFrame is also
        # minimal in terms of memory consumption because it does not contain any redundant value.

        if self.parameterspace['n_scale'].vary:
            print("""Please set ParameterSpace[\'n_scale\'].vary = False, unless you have a very good reason.
                  The fit runs faster when it is set to False and the parameter \'n_scale\' will be fitted anyway later
                  when calling Model.complete_fit(). As a rule of thumb, \'n_scale\' was always fitted except
                  you get \'m1_wrong_scaled\'.""")
        df = pd.DataFrame()
        counter_exception = 0
        for i in range(points):
            try:
                SingleFit = Fit(model=self, params=self.parameterspace.random_pt(), **fitting_kwargs)
                single_result = SingleFit.make_fit()
                result_df = SingleFit.fit_results_into_dataframe(single_result)
                df = df.append(result_df, ignore_index=True)
            except:
                counter_exception += 1
                pass
            if counter_exception == points:
                raise FlavorPyError(f"""When calling Model.fast_fit() all fits failed. Try running 
                                    res = Fit(model='{self.name}', params='{self.name}'.parameterspace.random_pt(),
                                              **fitting_kwargs).make_fit() and see what causes the error.
                                    If this runs smoothly, try Fit(...).fit_results_into_dataframe(res).""")
        return df

    def complete_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use this function to add the observables (while simultaneously fitting the dimensionful \'n_scale\' parameter)
        to a pandas.DataFrame containing points in parameterspace.
        ----------
        :param df: pandas.DataFrame
            Points in parameterspace. E.g. the result of Model.dimless_fit().
        :return: pandas.DataFrame
            The as \'df\' entered points in parameterspace plus their corresponding observables and chi-square value.
        """
        # Add all lepton observables and at the same time "fit" the dimensienful neutrino scale
        # Todo: Up to now, the fit of the neutrino scale is part of mixingcalculations.calculate_lepton_observables()
        #       First, extract his step and make it a own Model.dimful_fit() without slowing down the program
        #       Secondly, make this fit an actual fit. Up to now its just an average.
        for key in ['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'r', 'm21^2', 'm3l^2',
                    'm1', 'm2', 'm3', 'eta1', 'eta2', 'J', 'Jmax', 'Sum(m_i)', 'm_b', 'm_bb', 'nscale']:
            if key in df.columns:
                df = df.drop(columns=[key])
        df = df.join(pd.DataFrame([self.get_obs(df.loc[i]) for i in df.index], index=df.index))

        # Add the value of chi-square taking into consideration everything in self.fitted_observables
        df = df.rename(columns={'chisq': 'chisq_dimless'})
        df = df.join(pd.DataFrame([self.experimental_data.get_chisq(values=df.loc[i],
                                                                    considered_obs=self.fitted_observables)
                                   for i in df.index], index=df.index, columns=['chisq']))
        df = df.sort_values(by=['chisq']).reset_index(drop=True)
        df = df.reindex(columns=['chisq'] + list(df.columns)[:-1])  # put 'chisq' at first place in df
        return df

    # def merge_fit_results(self):
        # merge all entries of fit_results, which should be a pd.DataFrame, into one big pd.DataFrame


class QuarkModel:
    def __int__(self, parameterspace=None, mass_matrix_u=np.identity(3), mass_matrix_d=np.identity(3),
                parameterization='standard',
                name='Quark Model', comments=''):
        if parameterspace is None:
            parameterspace = ParameterSpace()

        self.name = name
        self.parameterspace = parameterspace
        self.mass_matrix_u = mass_matrix_u
        self.mass_matrix_d = mass_matrix_d
        self.parameterization = parameterization
        self.comments = comments

    def __repr__(self):
        return self.name


# class Model(QuarkModel, LeptonModel):
#     def __init__(self, name, comments):
#         self.name = name
#         self.comments = comments
#
#     def __repr__(self):
#         return self.name
