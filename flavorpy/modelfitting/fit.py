#import lmfit
import pandas as pd
from lmfit import minimize, Minimizer
import time
import numpy as np


class Fit:
    """
    This class is supposed to represent the fitting of a single random point.
    
    :param model: The Model whose parameters you want to fit
    :type model: py:meth:`~modelfitting.model.LeptonModel`
    :param params: The parameters you want to fit
    :type params: A lmfit.Parameters object.
    :param methods: A list of all methods that should be used for fitting. Note that only \'nr_methods\' of
        these are actually used. Either the first ones or random ones, depending on \'shuffle_methods\'.
        Default is a mixture of \'least_square\', \'nelder\', and further more advanced algorithms.
    :type methods: list, optional
    :param nr_methods: Default is 4
    :type nr_methods: int, optional
    :param shuffle_methods: Default is True
    :type shuffle_methods: bool, optional
    :param max_time: The amount of time in seconds, that the minimizer-algorithm is allowed to run. After this time
        is elapsed the algorithm is aborted.
    :type max_time: int, optional
    :param retry_time: If the minimization-algorithm is aborted for any reason, as a replacement \'least-squares\'
        is used for \'retry_time\' second.
    :type retry_time: int, optional
    :param dig_deeper: If \'dig_deeper\' is True, then a second round of methods is applied, if the usual
        calculation has lead to a chi-square less than \'dig_deeper_threshold\'.
        Default is False.
    :type dig_deeper: bool, optional
    :param dig_deeper_threshold:
    :type dig_deeper_threshold: float, optional
    """
    def __init__(self, model=None, params=None,
                 methods=None, nr_methods=4, shuffle_methods=True, max_time=45, retry_time=5,
                 dig_deeper=False, dig_deeper_threshold=1000):
        if methods is None:
            methods = ['least_squares', 'least_squares', 'least_squares',
                       'nelder', 'nelder', 'nelder',
                       'powell', 'lbfgsb', 'cg', 'cobyla', 'trust-constr']

        self.model = model
        self.params = params
        self.methods = methods
        self.nr_methods = nr_methods
        self.max_time = max_time
        self.retry_time = retry_time
        self.shuffle_methods = shuffle_methods
        self.dig_deeper = dig_deeper
        self.dig_deeper_threshold = dig_deeper_threshold

    def make_fit(self) -> list:
        """
        Call this function to execute the fit.
        
        :return: A list that contains the results of the fit in form of lmfit.MinimizerResult objects.
        :rtype: list
        """
        if self.shuffle_methods:
            methods = np.random.choice(self.methods, size=self.nr_methods)
        else:
            methods = self.methods[:self.nr_methods]
        fit_results = []
        params = self.params
        for method in methods:
            try:  # Try out the method, but abort if it takes longer than 'max_time'
                fit_results = np.append(fit_results,
                                        minimize(self.model.residual, params, method=method,
                                                 iter_cb=MinimizeStopper(self.max_time)))
            except:  # If the method failed or took to long, use the 'least_squares' method. This always yields a result
                fit_results = np.append(fit_results,
                                        minimize(self.model.residual, params, method='least_squares',
                                                 iter_cb=MinimizeStopper(self.retry_time)))
            params = fit_results[-1].params

        # The 'dig_deeper' option will keep on going for another nr_methods times,
        # but only if the last point's chisq was lower than the 'dig_deeper_threshold'
        if self.dig_deeper and fit_results[-1].chisqr < self.dig_deeper_threshold:
            for method in methods:
                try:
                    fit_results = np.append(fit_results,
                                            minimize(self.model.residual, params, method=method,
                                                     iter_cb=MinimizeStopper(self.max_time)))
                except:
                    fit_results = np.append(fit_results,
                                            minimize(self.model.residual, params, method='least_squares',
                                                     iter_cb=MinimizeStopper(self.retry_time)))
                params = fit_results[-1].params

        return fit_results

    def fit_results_into_dataframe(self, fit_results: list) -> pd.DataFrame:
        """
        Converts the result of Fit.make_fit() into a pandas.DataFrame.
        
        :param fit_results: A list that contains elements of the lmfit.MinimizerResult.
        :type fit_results: list
        :return: A pandas.DataFrame object that contains the best-fit parameters as well as the value of chi-square of
            the elements of fit_results.
        :rtype: pandas.DataFrame
        """
        df = pd.DataFrame()
        for result in fit_results:
            add = {'chisq': result.chisqr}
            for name in result.params:
                add[name] = [result.params[name].value]
            df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
        return df


class LmfitMinimizer(Minimizer):
    """
    A subclass of the `lmfit.Minimizer <https://lmfit.github.io/lmfit-py/fitting.html#using-the-minimizer-class>`_
    class.

    :param model: The Model whose parameters you want to fit
    :type model: :py:meth:`~modelfitting.model.LeptonModel`
    :param params: The parameters you want to fit
    :type params: lmfit.Parameters
    :param **kwargs_Minimizer: Additional keyword arguments to pass to the
        `lmfit.Minimizer <https://lmfit.github.io/lmfit-py/fitting.html#using-the-minimizer-class>`_ superclass.
    """
    def __init__(self, model=None, params=None, nan_policy='omit', **kwargs_Minimizer):
        def residual_nan(p):  # have the residual return 'nan' instead of stopping all calculations
            try:
                return model.residual(p)
            except:
                return np.array([np.nan for i in self.model.fitted_observables])
        super().__init__(residual_nan, params, nan_policy=nan_policy, **kwargs_Minimizer)
        self.model = model


class MinimizeStopper(object):
    """
    This object is able to stop a minimization procedure after max_sec seconds is elapsed.
    """
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()

    # def __call__(self, x1,x2,x3,x4,x5,xk=None, **kwargs):
    def __call__(self, x1, x2, x3):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            return True
