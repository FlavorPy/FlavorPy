import pandas as pd
from lmfit import minimize
import time
import numpy as np


class MinimizeStopper(object):  # This object is able to stop a minimization procedure after max_sec seconds is elapsed.
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()

    # def __call__(self, x1,x2,x3,x4,x5,xk=None, **kwargs):
    def __call__(self, x1, x2, x3):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            return True


class Fit:
    def __init__(self, model=None, params=None,
                 methods=None, nr_methods=4, shuffle_methods=True, max_time=45, retry_time=5,
                 dig_deeper=False, dig_deeper_threshold=1000):
        """
        This Class is supposed to represent the fitting of a single random point.
        ----------
        :param model: Model object of flavorpy
        :param params: Parameters object of lmfit
        :param methods: list, optional
            A list of all methods that should be used for fitting. Note that only \'nr_methods\' of these are actually
            used. Either the first ones or random ones, depending on \'shuffle_methods\'.
            Default is a mixture of \'least_square\', \'nelder\', and further more advanced algorithms.
        :param nr_methods: int, optional
            Default is 4
        :param shuffle_methods: bool, optional
            Default is True
        :param max_time: int, optional
            The amount of time in seconds, that the minimizer-algorithm is allowed to run. After this time is elapsed
            the algorithm is aborted.
        :param retry_time: int, optional
            If the minimization-algorithm is aborted for any reason, as a replacement \'least-squares\' is used for
            \'retry_time\' second.
        :param dig_deeper: bool, optional
            If \'dig_deeper\' is True, then a second round of methods is applied, if the usual calculation has lead to
            a chi-square less than \'dig_deeper_threshold\'.
            Default is False.
        :param dig_deeper_threshold: float, optional
        """
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
        ----------
        :return: list
            A list that contains the results of the fit in form of lmfit.MinimizerResult objects.
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
        Convertes the result of Fit.make_fit() into a pandas.DataFrame
        ----------
        :param fit_results: list
            A list that contains elements of the lmfit.MinimizerResult.
        :return: pandas.DataFrame
            A pandas.DataFrame object that contains the best-fit parameters as well as the value of chi-square of the
            elements of fit_results.
        """
        df = pd.DataFrame()
        for result in fit_results:
            add = {'chisq': result.chisqr}
            for name in result.params:
                add[name] = result.params[name].value
            df = df.append(add, ignore_index=True)
        return df
