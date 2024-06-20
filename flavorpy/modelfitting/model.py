import numpy as np
import pandas as pd
from copy import deepcopy
from .experimental_data.NuFit53.nufit53_chisqprofiles import NuFit53_NO
from .parameterspace import ParameterSpace
from .mixingcalculations import (calculate_lepton_dimensionless_observables, calculate_lepton_observables,
                                 calculate_quark_observables)
from .fit import Fit, LmfitMinimizer


class FlavorModel:
    """
    A flavor model with its mass matrices, parameters, ordering, and experimental data. It can be a model of only
    leptons, only quarks, or a model of both lepton and quark sector, depending on which mass matrices you enter.
    If you enter e.g. only mass_matrix_n, then it will be a lepton model only lepton observables will be considered.
    The information on the sector of the model is stored in FlavorModel.sector.

    :param mass_matrix_e: The charged lepton mass matrix M, for Phi_left M Phi_right.
        This should be a function with one argument, namely a dictionary of parameters and the corresponding value.
        Example::

            def m_e(params):
                return params['c'] * numpy.identity(3)
            M = FlavorModel(mass_matrix_e=m_e, ...)

        Default is a function that returns \'numpy.identity(3)\'.
        The program in its current state only gives the dimensionless mass ratios m_e/m_mu and m_mu/m_tau.
        For fitting, it is advisable to only use dimensionless parameters and simply ignore the overall mass scale,
        as it would only confuse the fit with an additional unnecessary parameter.
        The convention Phi_left M Phi_right is used, where left and right indicates left- and right-handed chiral
        fields, respectively. I.e. L_i^c M_ij e_j, where L refers to the left-handed lepton doublet and e is the
        right-handed charged lepton field, and i,j=1,2,3.
        If you use the other convention, i.e. left-handed fields on the right-hand-side, simply transpose your mass
        matrix.
    :type mass_matrix_e: function
    :param mass_matrix_n: The light neutrino mass matrix M, for Phi_left M Phi_right.
        Should be a function with one argument, namely a dictionary of parameters.
        Default is a function that returns \'numpy.identity(3)\'.
        It is !strongly! recommended to only use dimensionless parameters and one overall mass scale with the exact
        name \'n_scale\', i.e. mass_matrix_n = params[\'n_scale\']*dimensionless_mass_matrix(params). Otherwise, the
        program will add a parameter \'n_scale\' itself to the ParameterSpace and mass_matrix_n.
    :type mass_matrix_n: function
    :param mass_matrix_u: The up-type mass matrix M, for Phi_left M Phi_right.
        Should be a function with one argument, namely a dictionary of parameters.
        Default is a function that returns \'numpy.identity(3)\'.
    :type mass_matrix_u: function
    :param mass_matrix_d: The up-type mass matrix M, for Phi_left M Phi_right.
        Should be a function with one argument, namely a dictionary of parameters.
        Default is a function that returns \'numpy.identity(3)\'.
    :type mass_matrix_d: function
    :param parameterspace: The parameterspace of the model. See documentation of \'ParameterSpace\'.
        Default is an empty parameter space.
    :type parameterspace: :py:meth:`~modelfitting.parameterspace.ParameterSpace`
    :param ordering: Specify whether the neutrino spectrum is normal or inverted ordered. Has to be either \'NO\'
        or \'IO\'.
        Default is \'NO\'.
    :type ordering: str
    :param ckm_parameterization: The parameterization used for the CKM matrix. Either 'standard' for the `standard
        <https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix#%22Standard%22_parameters>`_
        and PDG convention or 'wolfenstein' for the `Wolfenstein  parameterization
        <https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix#Wolfenstein_parameters>`_.
    :type ckm_parameterization: str, either 'standard' or 'wolfenstein'.
    :param experimental_data: The experimental data used when fitting the model. For more information on the structure
        please confer the documentation of :py:meth:`~modelfitting.experimentaldata.ExperimentalData`.
        The default is \'NuFit53_NO\', i.e. the data of `NuFit v5.3 <http://www.nu-fit.org/?q=node/278>`_ for Normal
        Ordering taking into account results of SK. Please consider citing NuFit (www.nu-fit.org) when using this data.
    :type experimental_data: :py:meth:`~modelfitting.experimentaldata.ExperimentalData`
    :param fitted_observables:  A list of the observables that is considered when calculating chi-square and,
        hence, when making the fit. Possible entries are: 'me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi',
        'm21^2', 'm3l^2', 'mu/mc', 'mc/mt', 'md/ms', 'ms/mb', 't12', 't13', 't23', 'dq', 'l', 'A', 'rhobar', 'etabar'.
        Default are all observables of the modeled sector, i.e. all lepton and/or quark observables.
    :type fitted_observables: list, optional
    :param name: If you want, you can give the model a name. This name will be used as __repr__.
    :type name: str, optional
    :param comments: If you want, you can write some comments here.
    :type comments: str, optional
    :param fit_results: A list where you can store the results of :py:meth:`~modelfitting.model.FlavorModel.make_fit`.
        It is of course also possible to load the results from an external calculation into this list.
    :type fit_results: list, optional
    """
    def __init__(self, mass_matrix_e=None, mass_matrix_n=None,
                 mass_matrix_u=None, mass_matrix_d=None,
                 parameterspace=None,
                 ordering='NO', ckm_parameterization='standard',
                 experimental_data=NuFit53_NO, fitted_observables='all',
                 name='FlavorModel', comments='', fit_results=None):

        # Determine which sector this model represents
        if mass_matrix_e is None and mass_matrix_n is None and mass_matrix_u is None and mass_matrix_d is None:
            sector = 'lepton'   # or maybe raise an error that says please define at least one of the mass matrices
        elif mass_matrix_u is None and mass_matrix_d is None:
            sector = 'lepton'
        elif mass_matrix_e is None and mass_matrix_n is None:
            sector = 'quark'
        else:
            sector = 'quark and lepton'

        # Replace all not defined mass matrices with an identity matrix
        def triv_mat(params):
            return np.identity(3)
        if mass_matrix_e is None:
            mass_matrix_e = triv_mat
        if mass_matrix_n is None:
            mass_matrix_n = triv_mat
        if mass_matrix_u is None:
            mass_matrix_u = triv_mat
        if mass_matrix_d is None:
            mass_matrix_d = triv_mat

        # Check if the arguments entered are valid
        if ckm_parameterization not in ['standard', 'wolfenstein']:
            raise NotImplementedError('''The value of \'parameterization\' has to be either 
                                      \'wolfenstein\' or \'standard\'.''')
        if ordering not in ['NO', 'IO']:
            raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')

        # Assign default values
        if parameterspace is None:
            parameterspace = ParameterSpace()
        if fitted_observables in ['all', 'auto', 'full', 'everything']:
            if sector == 'lepton':
                fitted_observables = ['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'm21^2', 'm3l^2']
            elif sector == 'quark':
                if ckm_parameterization == 'standard':
                    fitted_observables = ['mu/mc', 'mc/mt', 'md/ms', 'ms/mb', 't12', 't13', 't23', 'dq']
                elif ckm_parameterization == 'wolfenstein':
                    fitted_observables = ['mu/mc', 'mc/mt', 'md/ms', 'ms/mb', 'l', 'A', 'rhobar', 'etabar']
            elif sector == 'quark and lepton':
                if ckm_parameterization == 'standard':
                    fitted_observables = ['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'm21^2', 'm3l^2',
                                          'mu/mc', 'mc/mt', 'md/ms', 'ms/mb', 't12', 't13', 't23', 'dq']
                elif ckm_parameterization == 'wolfenstein':
                    fitted_observables = ['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'm21^2', 'm3l^2',
                                          'mu/mc', 'mc/mt', 'md/ms', 'ms/mb', 'l', 'A', 'rhobar', 'etabar']
        if fit_results is None:
            fit_results = []

        # Create the dimensionless fitted observables for neutrinos
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

        # Assign as attributes
        self.mass_matrix_e = mass_matrix_e
        self.mass_matrix_n = mass_matrix_n
        self.mass_matrix_u = mass_matrix_u
        self.mass_matrix_d = mass_matrix_d
        self.parameterspace = parameterspace
        self.ordering = ordering
        self.ckm_parameterization = ckm_parameterization
        self.experimental_data = experimental_data
        self.fitted_observables = fitted_observables
        self.fitted_observables_dimless = fitted_observables_dimless
        self.name = name
        self.comments = comments
        self.fit_results = fit_results
        self.sector = sector

        # Add a scale to the neutrino mass matrix, if there is not already one.
        if self.sector in ['lepton', 'quark and lepton'] and 'n_scale' not in self.parameterspace:
            # add 'n_scale' to parameterspace
            def const_fct():
                return 1
            self.parameterspace.add_dim(name='n_scale', sample_fct=const_fct, vary=False)

            # add the scale to the neutrino mass matrix
            def new_mass_matrix(params):
                return params['n_scale'] * mass_matrix_n(params)
            self.mass_matrix_n = new_mass_matrix
        else:
            self.parameterspace['n_scale'].vary = False  # don't worry we are going to fit it, just not in dimless_fit

    def __repr__(self):
        return self.name

    def copy(self):
        """
        Returns a deepcopy of itself.
        """
        return deepcopy(self)

    def get_dimless_obs(self, params: dict) -> dict:
        if self.sector == 'lepton':
            obs = calculate_lepton_dimensionless_observables(mass_matrix_e=self.mass_matrix_e(params),
                                                             mass_matrix_n=self.mass_matrix_n(params),
                                                             ordering=self.ordering)
        elif self.sector == 'quark':
            obs = calculate_quark_observables(mass_matrix_u=self.mass_matrix_u(params),
                                              mass_matrix_d=self.mass_matrix_d(params),
                                              parameterization=self.ckm_parameterization)
        elif self.sector == 'quark and lepton':
            obs_lepton = calculate_lepton_dimensionless_observables(mass_matrix_e=self.mass_matrix_e(params),
                                                                    mass_matrix_n=self.mass_matrix_n(params),
                                                                    ordering=self.ordering)
            obs_quark = calculate_quark_observables(mass_matrix_u=self.mass_matrix_u(params),
                                                    mass_matrix_d=self.mass_matrix_d(params),
                                                    parameterization=self.ckm_parameterization)
            obs = {**obs_lepton, **obs_quark}
        return obs

    def get_obs(self, params: dict) -> dict:
        """
        Get a dictionary of all observables for a point in parameterspace.

        :param params: The point in parameter space.
        :type params: dict
        :return: All observables, e.g. {'me/mu':0.0048, ...}
        :rtype: dict
        """
        # Get experimental best fit value for squared neutrino mass differences m_21^2 and m_3l^2
        if self.sector in ['lepton', 'quark and lepton']:
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

        # Calculate the observables
        if self.sector == 'lepton':
            # Calculate all lepton observables (while simultaneously fitting the neutrino scale)
            obs = calculate_lepton_observables(mass_matrix_e=self.mass_matrix_e(params),
                                               mass_matrix_n=self.mass_matrix_n(params),
                                               ordering=self.ordering, m21sq_best=m21sq_best, m3lsq_best=m3lsq_best)
        elif self.sector == 'quark':
            obs = self.get_dimless_obs(params)
        elif self.sector == 'quark and lepton':
            obs_lep = calculate_lepton_observables(mass_matrix_e=self.mass_matrix_e(params),
                                                   mass_matrix_n=self.mass_matrix_n(params),
                                                   ordering=self.ordering, m21sq_best=m21sq_best, m3lsq_best=m3lsq_best)
            obs_quark = calculate_quark_observables(mass_matrix_u=self.mass_matrix_u(params),
                                                    mass_matrix_d=self.mass_matrix_d(params),
                                                    parameterization=self.ckm_parameterization)
            obs = {**obs_lep, **obs_quark}

        return obs

    def residual(self, params: dict) -> list:  # there can be no other arguments than params! Otherwise, you need to adjust fit.py!
        """
        The residual used to make the dimensionless fit.

        :param params: The point in parameterspace.
        :type params: dict
        :return: A list of values of individual chis (not chi-squares!). Only dimensionless observables are
            being considered.
        :rtype: list
        """
        # This is the residual used for the 'dimless_fit' that only fits dimensionless observables.
        observables = self.get_dimless_obs(params)
        chisq_list = self.experimental_data.get_chisq_list(values=observables,
                                                           considered_obs=self.fitted_observables_dimless)
        return np.sqrt(np.abs(chisq_list))  # The lmfit minimizer wants chi and not chi-square! It takes a list of single chi.

    def get_chisq(self, params=None) -> float:
        """
        Returns the value of chi-square for a given point in parameter space.

        :param params: The point in parameterspace.
            Default is None.
        :type params: dict
        :return: The value of chi-square.
        :rtype: float
        """
        observables = self.get_obs(params)
        return self.experimental_data.get_chisq(values=observables, considered_obs=self.fitted_observables)

    def print_chisq(self, params: dict):
        """
        Prints the value of all observables and the associated contribution to chi-square. Also prints total chi-square.

        :param params: The point in parameterspace
        :type params: dict
        """
        observables = self.get_obs(params)
        chisq_list = self.experimental_data.get_chisq_list(values=observables, considered_obs=self.fitted_observables)
        chisq_dict = {self.fitted_observables[i]: chisq_list[i] for i in range(len(self.fitted_observables))}
        for obs_name in self.fitted_observables:
            print(f"'{obs_name}': {observables[obs_name]},   chisq: {chisq_dict[obs_name]}")
        print(f"Total chi-square: {np.sum(chisq_list)}")

    def make_fit(self, points: int, **fitting_kwargs) -> pd.DataFrame:
        """
        Does a fit for a specific number of random points in parameterspace.

        :param points: The number of random points in parameter space you want to fit.
            If you want to fit a specific starting point in parameter space, adjust the \'sampling_fct\' in your
            ParameterSpace.
        :type points: int

        :param fitting_kwargs: properties of the :py:meth:`~modelfitting.fit.Fit` class.
            You can add keyword arguments that will be passed down to the :py:meth:`~modelfitting.fit.Fit` object used
            to make the fit. Please see the documentation of :py:meth:`~modelfitting.fit.Fit` for the specific keyword
            arguments. Of course, the keywords \'model\' and \'params\' can not be passed down to
            :py:meth:`~modelfitting.fit.Fit`.

        :return: The result of the fit is returned in form of a pandas.DataFrame.
            Note that several (default:4) minimization algorithms are applied consecutively to one random point. Since
            the results of the intermediate steps are also written into the resulting DataFrame, it has more rows than
            the number entered as \'points\'.
        :rtype: pandas.DataFrame
        """
        df = self.dimless_fit(points, **fitting_kwargs)
        df = self.complete_fit(df)
        return df

    def dimless_fit(self, points: int, **fitting_kwargs) -> pd.DataFrame:
        """
        Calling this function fits the dimensionless parameters of the model.
        The procedure of :py:meth:`~modelfitting.model.FlavorModel.make_fit` can be split into the fitting of
        dimensionless parameters (with :py:meth:`~modelfitting.model.FlavorModel.dimless_fit`) and the fitting of the
        dimensionful ones (with :py:meth:`~modelfitting.model.FlavorModel.complete_fit`), where
        Model.complete_fit() also adds the observables to the resulting pandas.DataFrame.
        This function comes in handy when running fits on an external machine, e.g. a cluster, since the result of
        dimless_fit() uses a smaller amount of memory when stored into a file compared to the result of
        complete_fit(). You can more easily transfer the smaller files from dimless_fit() to your local machine and
        there run complete_fit(), which does not take a lot of time compared to dimless_fit().

        :param points: The number of random points in parameter space you want to fit.
            If you want to fit a specific starting point in parameter space, adjust the \'sampling_fct\' in your
            ParameterSpace.
        :type points: int
        :param fitting_kwargs: properties of the :py:meth:`~modelfitting.fit.Fit` class.
            You can add keyword arguments that will be passed down to the :py:meth:`~modelfitting.fit.Fit` object used
            to make the fit. Please see the documentation of :py:meth:`~modelfitting.fit.Fit` for the specific keyword
            arguments. Of course, the keywords \'model\' and \'params\' can not be passed down to
            :py:meth:`~modelfitting.fit.Fit`.
        :return: The result of the fit is returned in form of a pandas.DataFrame.
            Note that several (default:4) minimization algorithms are applied consecutively to one random point. Since
            the results of the intermediate steps are also written into the resulting DataFrame, it has more rows than
            the number entered as \'points\'.
        :rtype: pandas.DataFrame
        """

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
                df = pd.concat([df, result_df], ignore_index=True)
            except:
                counter_exception += 1
                pass
            if counter_exception == points:
                raise FlavorPyError(f"""When calling Model.dimless_fit() all fits failed. Try running 
                                    res = Fit(model='{self.name}', params='{self.name}'.parameterspace.random_pt(),
                                              **fitting_kwargs).make_fit() and see what causes the error.
                                    If this runs smoothly, try Fit(...).fit_results_into_dataframe(res).""")
        return df

    def complete_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use this function to add the observables (while simultaneously fitting the dimensionful \'n_scale\' parameter)
        to a pandas.DataFrame containing points in parameterspace.

        :param df: Points in parameterspace. E.g. the result of :py:meth:`~modelfitting.model.FlavorModel.dimless_fit`.
        :type df: pandas.DataFrame
        :return: The as \'df\' entered points in parameterspace plus their corresponding observables and
            chi-square value.
        :rtype: pandas.DataFrame
        """
        # Add all lepton observables and at the same time "fit" the dimensienful neutrino scale
        # Todo: Up to now, the fit of the neutrino scale is part of mixingcalculations.calculate_lepton_observables()
        #       First, extract his step and make it a own Model.dimful_fit() without slowing down the program
        #       Secondly, make this fit an actual fit. Up to now its just an average.
        for key in ['me/mu', 'mu/mt', 's12^2', 's13^2', 's23^2', 'd/pi', 'r', 'm21^2', 'm3l^2',
                    'm1', 'm2', 'm3', 'eta1', 'eta2', 'J', 'Jmax', 'Sum(m_i)', 'm_b', 'm_bb', 'nscale',
                    'mu/mc', 'mc/mt', 'md/ms', 'ms/mb', 't12', 't13', 't23', 'dq', 'l', 'A', 'rhobar', 'etabar']:
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

    def mcmc_fit(self, df_start, params_not_varied=None, mcmc_steps=1000, burn=300, thin=10, nwalkers=40,
                 nan_policy='omit', progress=True, print_error=False, **emcee_kwargs) -> pd.DataFrame:
        """
        A Markov-Chain-Monte-Carlo (MCMC) sampling is a useful method to explore the parameter space around a minimum.
        A typical workflow would be to first find a minimum using :py:meth:`~modelfitting.model.FlavorModel.make_fit`
        and then further explore the vicinity of this minimum with :py:meth:`~modelfitting.model.FlavorModel.mcmc_fit`
        to get an improved understanding of the probability distribution for the parameters. This method is based on
        the `emcee <https://emcee.readthedocs.io/en/stable/>`_ MCMC sampler.

        :param df_start: Contains the starting points around which further points should be sampled.
        :type df_start: Two-dimensional dict-like, preferably a pandas.DataFrame
        :param params_not_varied: The parameters of self.parameterspace that should not be varied additionally to the
            ones that are already set not to vary in self.parameterspace.
        :type params_not_varied: list, optional
        :param mcmc_steps: The amount of points sampled.
        :type mcmc_steps: int
        :param burn: How many samples from the start of the sampling should be discarded. Default is 300.
        :type burn: int, optional
        :param thin: Only accept 1 in every 'thin' samples. Default is 10.
        :type thin: int, optional
        :param nwalkers: Number of members of the ensemble. Make sure that 'nwalkers' >> number of varied parameters.
            Default is 40.
        :type nwalkers: int
        :param nan_policy: Specify how to handle it if the residual returns NaN values. Default is 'omit'.
        :type nan_policy: possible are 'raise', 'propagate', and 'omit', optional
        :param progress: If True prints a progress bar of the sampling. Default is True.
        :type progress: bool, optional
        :param print_error: If True prints the index of a given row if the sampling didn't work for this row.
            Default is False.
        :type print_error: bool, optional
        :param emcee_kwargs: Further keyword arguments can be passed down to
            `lmfit.Minimizer.emcee <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee>`_
        :return: A pandas.DataFrame containing all the sampled points.
        :rtype: panda.DataFrame
        """

        if params_not_varied is None:
            params_not_varied = []
        df = pd.DataFrame()
        for i in df_start.index:
            if progress:
                print(i, ": ")
            try:
                params = self.parameterspace.random_pt()  # The values of this "random" point are set in the next line
                for param in params:
                    params[param].value = df_start.loc[i][param]
                for par in params_not_varied:
                    params[par].vary = False
                SingleFit = LmfitMinimizer(model=self, params=params, nan_policy=nan_policy)
                out = SingleFit.emcee(steps=mcmc_steps, nwalkers=nwalkers, burn=burn, thin=thin, progress=progress,
                                      **emcee_kwargs)
                flatchain = out.flatchain
                for param in params:
                    if not params[param].vary:
                        flatchain[param] = df_start.loc[i][param]
                df = pd.concat([df, flatchain], ignore_index=True)
            except:
                if print_error:
                    print(f"error with index {i}")
        return df


class FlavorPyError(Exception):
    # Raises an Exception
    pass
