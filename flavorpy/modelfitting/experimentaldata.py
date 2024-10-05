import numpy as np
from copy import deepcopy


class ExperimentalData:
    """
    An experimental data set.

    :param name: The name of the set
    :type name: str
    :param data_table: A dictionary that contains the experimental best fit value and 1 sigma errors.
        The dict needs to be in a very specific form! It is very important that you use the keys \'best\',
        \'1sig_min\', and \'1sig_max\'!
        An example is::
        
            data_table = {\'me/mu\': {\'best\':0.0048, \'1sig_min\':0.0046, \'1sig_max\':0.0050}, 'mu/mt': ...}
            
        The data entered in \'data_table\' will be used to calculate a chi-square value assuming a gaussian error,
        i.e.::
        
            chisq = ( (model_value - best) / (0.5*(1sig_max - 1sig_min)) )^2
            
        Alternatively, especially for non-gaussian errors, one can enter the chisquare profiles using the
        'data' parameter. Mixed entries are also possible.
    :type data_table: dict, optional
    :param data: A dictionary, whose keys are the names of the experimental observables and the corresponding values are
        functions, typically chisquare profiles. So for example::
        
            data = {"me/mu": memuchisq, "mu/mt": mumtchisq}
            
        where memuchisq(value) is defined to return ((value - exp_best)/(exp_1sig_error))^2 or something equivalent.
    :type data: dict, optional
    """
    def __init__(self, name="ExpData", data_table=None, data=None):
        if data is None:
            data = {}
        if data_table is None:
            data_table = {}

        self.name = name
        self.data_table = data_table
        self.data = data

        # Warning if no data at all was entered
        if dict(data) == {} and dict(data_table) == {}:
            raise NameError("""You need to define either \'data\' or \'data_table\', otherwise your experimental data
                            is basically empty!""")

        ### Convert data of data_table into chisquare functions and add it to data
        # Raise error if there is an overlap of keys in 'data' and 'data_table'
        keys_overlap = [x for x in self.data_table if x in self.data]
        if len(keys_overlap) > 0:
            raise ValueError(f"""There are overlapping keys in \'data\' and \'data_table\'! I don't know which one 
                             to take. Please remove the overlapping keys either from \'data\' or \'data_table\'.
                             The overlapping keys are: {keys_overlap}""")
        # Conversion and appending to data
        # Todo: Add option to have data_table['1sig_range'] instead of minimum and maximum
        def make_chisq(key: str):
            def chisq(value: float) -> float:
                return ((value - self.data_table[key]['best']) /
                        (1/2*(self.data_table[key]['1sig_max'] - self.data_table[key]['1sig_min']))) ** 2
            return chisq
        for key in self.data_table:
            chisq = make_chisq(key)
            self.data[key] = chisq

    def __repr__(self):
        return self.name

    def copy(self):
        """
        Returns a deep copy.
        """
        return deepcopy(self)

    def get_chisq(self, values: dict, considered_obs="auto") -> float:
        """
        Returns chi-square for a set of values. The values are added quadratically, so chisq =  sum_i chisq_i.

        :param values: The values that you want to compare to the experimental data. \'values\' should be in the form of
            a dictionary like for example \'values = {'me/mu':0.0045, 'mu/mt':0.546, ...}\', or any object that returns
            a callable via a key, i.e. \'values['me/mu']\' returns 0.0045, for example a pandas dataframe.
        :type values: dict
        :param considered_obs: A list that contains the keys of all observables that should be considered to calculate
            the chisquare. So if \'considered_obs=['me/mu', 's12^2']\', then the returned chisquare only contains
            contributions from \'me/mu\' and \'s12^2\', even though values also has an entry \'mu/mt'.
            The default value is \'auto\', which will consider all keys present in \'values\'.
        :type considered_obs: list, optional
        :return: List of chi-square values associated with a specific observable, i.e. [chisq_me/mu, chisq_s12^2, ...].
        :rtype: list
        """
        return sum(self.get_chisq_list(values=values, considered_obs=considered_obs))

    def get_chisq_list(self, values: dict, considered_obs="auto") -> list:
        """
        Returns a list that contains the contributions to chi-square.

        :param values: The values that you want to compare to the experimental data. \'values\' should be in the form of
            a dictionary like for example \'values = {'me/mu':0.0045, 'mu/mt':0.546, ...}\', or any object that returns
            a callable via a key, i.e. \'values['me/mu']\' returns 0.0045, for example a pandas dataframe.
        :type values: dict
        :param considered_obs: A list that contains the keys of all observables that should be considered to calculate
            the chisquare. So if \'considered_obs=['me/mu', 's12^2']\', then the returned chisquare only contains
            contributions from \'me/mu\' and \'s12^2\', even though values also has an entry \'mu/mt'.
            The default value is \'auto\', which will consider all keys present in \'values\'.
        :type considered_obs: list, optional
        :return: List of chi-square values associated with a specific observable, i.e. [chisq_me/mu, chisq_s12^2, ...].
        :rtype: list
        """
        if considered_obs == "auto":
            considered_obs = list(values.keys())

        # Check if there exists experimental data for all keys
        for key in considered_obs:
            if key not in list(self.data.keys()):
                raise ValueError(f"""This experimental data set does not contain any data with the key \'{key}\'!""")

        return np.abs(list([self.data[key](values[key]) for key in considered_obs]))
