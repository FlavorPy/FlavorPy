Detailed example
================

Let us now take a look at a more involved example that better showcases
more advanced features of the modelfitting package. More specifically we
consider the Model 1 described in section 3.1.2 of
`https://arxiv.org/abs/1706.08749 <https://arxiv.org/abs/1706.08749>`_. 
This is a model of modular flavor
symmetries, where modular forms are present in the neutrino mass matrix.

We begin by importing the necessary packages:

.. code:: ipython3

    directory_to_git_folder = "/home/alex/GitRepos/FlavorPy/tmp_version"  # Adjust this to your case!!
    import os
    os.chdir(os.path.expanduser(directory_to_git_folder))
    # Import the modelfitting module of FlavorPy
    import modelfitting as mf
    # We will also need numpy and pandas
    import numpy as np
    import pandas as pd

Mass matrices
-------------

Then we define the mass matrices. There is a subtlety: The modelfitting
package considers a mass matrix :math:`M` for
:math:`\Phi_\mathrm{left} ~M~ \Phi_{\mathrm{right}}`. However the
convention used in the paper is :math:`E^c ~M_\mathrm{e}~L`, meaning
that we need to transpose the mass matrix compared to the paper!
Nevertheless, for this specific case the mass matrices are symmetric
anyway. It is highly recommended to only use dimensionless parameters
and then have one dimensionfull parameter, i.e. an overall scale, in
front of the mass matrix. For the neutrino mass matrix please name this
parameter ‘n_scale’. For the charged lepton mass matrix, simply ignore
it, because we will only fit charged lepton mass ratios.

.. code:: ipython3

    # Charged lepton mass matrix
    def Me(params):
        return np.transpose(np.array([[params['alpha'], 0, 0],
                                      [0, params['beta'], 0],
                                      [0, 0, params['gamma']]]))
    
    # Modular forms
    def Y1(tau, **kwargs):
        q=np.exp(2j*np.pi*tau/3)
        return 1 + 12*np.power(q, 1*3) + 36*np.power(q, 2*3) + 12*np.power(q, 3*3)
    def Y2(tau, **kwargs):
        q=np.exp(2j*np.pi*tau/3)
        return -6*q*(1 + 7*np.power(q,1*3) + 8*np.power(q, 2*3))
    def Y3(tau, **kwargs):
        q=np.exp(2j*np.pi*tau/3)
        return -18*np.power(q, 2)*(1 + 2*np.power(q, 1*3) + 5*np.power(q, 2*3))
    
    # Neutrino mass matrix
    def Mn(params):
        tau = params['Retau']+1j*params['Imtau']
        return params['n_scale']*np.transpose(np.array([[2*Y1(tau), -1*Y3(tau), -1*Y2(tau)],
                                                        [-1*Y3(tau), 2*Y2(tau), -1*Y1(tau)],
                                                        [-1*Y2(tau), -1*Y1(tau), 2*Y3(tau)]], dtype=complex))

Parameter space
---------------

Next we construct the parameter space. We therefore write our own
sampling functions, that when called yield a random point. Note that
especailly for fitting complicated models the sampling heavily impacts
the number of random points (and therefore also the time) needed to find
a good fit. A logarithmic sampling or a mixture of logarithmic and
linear sampling is often a good idea. The ‘lin_sampling(low, high)’
defined here draws a random number between ‘low’ and ‘high’ with a
uniform distribution. The ‘const_sampling(value)’ always yields ‘value’
when called. For our specific model, we can already know what the values
of alpha, beta and gamma will be, since they are directly correlated to
the charged lepton masses. We will therefore use const_sampling to set 
them by hand to a fixed value and prevent the fitting algorithm from 
varying this value by setting ‘vary=False’. For the modulus tau, we will 
choose the ‘lin_sampling’ and restrict the boundaries of the parameter 
space by ‘min’ and ‘max’.

.. code:: ipython3

    # Sampling functions
    def lin_sampling(low=0, high=1):
        def fct():
            return np.random.uniform(low=low, high=high)
        return fct
    def const_sampling(value=0):
        def fct():
            return value
        return fct
    
    # Constructing the parameter space
    ParamSpace = mf.ParameterSpace()
    ParamSpace.add_dim(name='Retau', sample_fct=lin_sampling(low=-0.5, high=0.5), min=-0.5, max=0.5)
    ParamSpace.add_dim(name='Imtau', sample_fct=lin_sampling(low=0.866, high=3), min=0.866, max=4)
    ParamSpace.add_dim(name='n_scale', sample_fct=const_sampling(1.), vary=False)
    ParamSpace.add_dim(name='alpha', sample_fct=const_sampling(0.0048*0.0565), vary=False)
    ParamSpace.add_dim(name='beta', sample_fct=const_sampling(0.0565), vary=False)
    ParamSpace.add_dim(name='gamma', sample_fct=const_sampling(1.), vary=False)

Experimental data
-----------------

We already know from the paper that this model gives better fits with an
invered neutrino mass ordering. We therefore choose the NuFit v5.2
experimental data for inverted ordering including the SK data. The
onedimensional chisqure projections of the NuFit v5.2 data are already
implemented in the modelfitting package and can be simply loaded by

.. code:: ipython3

    mf.NuFit52_IO




.. parsed-literal::

    NuFit v5.2 IO with SK chisquare profiles



If you wanted to compare your model to your own custom experimental 
data, you can create an experimental dataset by

.. code:: ipython3

    my_table = pd.DataFrame(np.array([
        [0.0048, 0.0565, 0.303, 0.02223, 0.569, 0.0741/-2.486, 1.54, 7.41e-05, -2.486e-03],
        [0.0046, 0.0520, 0.292, 0.02165, 0.548, 0.0721/-2.511, 1.38, 7.21e-05, -2.458e-03],
        [0.0050, 0.0610, 0.315, 0.02281, 0.585, 0.0762/-2.458, 1.67, 7.62e-05, -2.511e-03]]),
                                 columns=["me/mu", "mu/mt", "s12^2", "s13^2", "s23^2", "r", "d/pi", "m21^2", "m3l^2"],
                                 index=['best', '1sig_min', '1sig_max'])
    My_ExpData = mf.ExperimentalData(name='my name', data_table=my_table)

The total resudiual is then :math:`\chi^2 = \sum_x \chi^2_x`, where 
:math:`x` represents the observables, e.g. 'me/mu'. The individual 
contributions are determined by
:math:`\chi^2_x = \big(\dfrac{x_\mathrm{model} - x_\mathrm{best}}{1/2\,(x_{1\mathrm{sig}\_\mathrm{max}} - x_\mathrm{1sig\_min}) }\big)^2`,
where :math:`x_\mathrm{model}` is the value of the model and
:math:`x_\mathrm{best}`, :math:`x_{1\mathrm{sig}\_\mathrm{max}}`, and
:math:`x_{1\mathrm{sig}\_\mathrm{min}}` are the experimental values.

Alternatively, if you have a non-gaussian error distribution for lets
say 'me/mu' and you want :math:`\chi^2` to be calculated using a 
specific :math:`\chi^2`-profile, then you can define your experimental 
data set as

.. code:: ipython3

    def memu_profile(memu):  # This is just an example profile
        return 1e11*(memu - 0.003) * (memu - 0.007) * (memu - 0.008) * (memu - 0.001) + 3.0978
    
    My_ExpData_2 = mf.ExperimentalData(name='my name2', 
                                       data_table=my_table[[key for key in my_table.columns if key not in ['me/mu']]],
                                       data={'me/mu':memu_profile})

Constructing the model
----------------------

In the modelfitting module, everything is packed into a class called
Model. The Model object contains the mass matrices, the parameterspace,
the experimental data, the neutrino mass ordering, and even the results
of fits can be stored in this object. Note that the neutrino ordering 
is not (yet) automatically determined by the modelfitting package, nor 
is it checked whether the results of a given random point indeed follow
that ordering. For now there is only the class LeptonModel, however 
models for quark will be implemented soon into the modelfitting package. 
Since the paper does not compare the CP violating phase ‘d/pi’ to the 
experimental data, we will do the same here and only fit the three mixing 
angles as well as the squared neutrino mass differences. It is also not 
necessary to fit the charged lepton masses, since we already fixed them 
to their correct value.

.. code:: ipython3

    Model1 = mf.LeptonModel(name='Feruglios model 1', 
                           comments='''This was the first modular flavor symmetry model.
                                    Unfortunately it is now way outside the experimentally viable region.''',
                           mass_matrix_e=Me,
                           mass_matrix_n=Mn,
                           parameterspace=ParamSpace,
                           ordering='IO',
                           experimental_data=mf.NuFit52_IO,
                           fitted_observables=['s12^2', 's13^2', 's23^2', 'm21^2', 'm3l^2'])

You can now test if the model works, by calculating a random sample
point

.. code:: ipython3

    random_point = Model1.parameterspace.random_pt()
    Model1.get_obs(random_point)




.. parsed-literal::

    {'me/mu': 0.0048,
     'mu/mt': 0.0565,
     's12^2': 0.9989074489211242,
     's13^2': 0.0004515063173869479,
     's23^2': 0.47402369343152484,
     'd/pi': 1.7243267756116443,
     'r': -0.934864550872772,
     'm21^2': 0.0011990866367348556,
     'm3l^2': -0.001282631409668289,
     'm1': 0.021372812096913652,
     'm2': 0.040692551329018854,
     'm3': 0.019319739232105206,
     'eta1': 1.7268565851575932,
     'eta2': 1.6031520425143535,
     'J': -0.0002668890393932839,
     'Jmax': 0.00035035022527843336,
     'Sum(m_i)': 0.08138510265803772,
     'm_b': 0.04070152441097532,
     'm_bb': 0.04064229787720081,
     'nscale': 0.02032799146295054}



If you wanted to see, whether the model can also fit the experimental
data for a normal ordered spectrum simply define

.. code:: ipython3

    Model1_NO = Model1.copy()
    Model1_NO.ordering = 'NO'
    Model1_NO.experimentaldata = mf.NuFit52_NO

and do the following fitting with Model1_NO.

Fitting
-------

We can now fit our model to match experimental data as good as possible.
This is done by simply calling ‘make_fit(points=int)’ on the
LeptonModel. This yields a pandas.DataFrame object, which is very
convenient in data handling. Automatically, it is sorted such that the
lowest :math:`\chi^2` is on top.

The fit is based on the lmfit minimizer,
cf. https://lmfit.github.io/lmfit-py/intro.html. When fitting a
LeptonModel a certain number of random points according to the sample
functions of the parameter space are drawn. Then several minimization
algorithms (methods) implemented in lmfit are applied consecutively
several times (nr_methods) onto every random point. Since minimization
algorithms sometimes get lost and run very long, every applied algorithm
is stopped after a certain amount of seconds (max_time). These and other
arguments used for the fit, can be adjusted to the individual needs.
However, in most cases the default values work very well and it is not
necessary to adjust them. Only for the purpose of demonstation we will 
do it in this case.

.. code:: ipython3

    pd.set_option('display.max_columns', None)  # This pandas setting allows us to see all columns
    
    # Adjusting the default setup for fitting.  Usually this is 
    fitting_kwargs = {'nr_methods':2,
                      'methods':['least_squares', 'least_squares', 'nelder', 'powell', 'cobyla'],
                      'max_time':20}
    
    # Running the fit
    df = Model1.make_fit(points=5, **fitting_kwargs)
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: left;">
          <th></th>
          <th style="min-width: 120px;">chisq</th>
          <th>chisq_dimless</th>
          <th style="min-width: 90px;">Retau</th>
          <th style="min-width: 90px;">Imtau</th>
          <th style="min-width: 90px;">n_scale</th>
          <th style="min-width: 90px;">alpha</th>
          <th style="min-width: 90px;">beta</th>
          <th style="min-width: 90px;">gamma</th>
          <th style="min-width: 90px;">me/mu</th>
          <th style="min-width: 90px;">mu/mt</th>
          <th style="min-width: 90px;">s12^2</th>
          <th style="min-width: 90px;">s13^2</th>
          <th style="min-width: 90px;">s23^2</th>
          <th style="min-width: 50px;">d/pi</th>
          <th style="min-width: 90px;">r</th>
          <th style="min-width: 90px;">m21^2</th>
          <th style="min-width: 90px;">m3l^2</th>
          <th style="min-width: 90px;">m1</th>
          <th style="min-width: 90px;">m2</th>
          <th style="min-width: 90px;">m3</th>
          <th style="min-width: 90px;">eta1</th>
          <th style="min-width: 90px;">eta2</th>
          <th style="min-width: 90px;">J</th>
          <th style="min-width: 90px;">Jmax</th>
          <th style="min-width: 90px;">Sum(m_i)</th>
          <th style="min-width: 90px;">m_b</th>
          <th style="min-width: 90px;">m_bb</th>
          <th style="min-width: 90px;">nscale</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1234.352993</td>
          <td>1234.353829</td>
          <td>-0.011596</td>
          <td>0.994569</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301487</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453327</td>
          <td>-0.029750</td>
          <td>0.000074</td>
          <td>-0.002488</td>
          <td>0.049142</td>
          <td>0.049889</td>
          <td>0.000748</td>
          <td>0.245394</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099778</td>
          <td>0.049266</td>
          <td>0.038487</td>
          <td>0.021932</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1234.352993</td>
          <td>1234.353829</td>
          <td>-0.011596</td>
          <td>0.994569</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301487</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453327</td>
          <td>-0.029750</td>
          <td>0.000074</td>
          <td>-0.002488</td>
          <td>0.049142</td>
          <td>0.049889</td>
          <td>0.000748</td>
          <td>0.245394</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099778</td>
          <td>0.049266</td>
          <td>0.038487</td>
          <td>0.021932</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1234.354697</td>
          <td>1234.353935</td>
          <td>-0.011592</td>
          <td>0.994571</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301476</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453324</td>
          <td>-0.029741</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049146</td>
          <td>0.049893</td>
          <td>0.000747</td>
          <td>0.245403</td>
          <td>1.029627</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099786</td>
          <td>0.049270</td>
          <td>0.038490</td>
          <td>0.021934</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1234.355203</td>
          <td>1234.353743</td>
          <td>-0.011591</td>
          <td>0.994571</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301488</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453328</td>
          <td>-0.029738</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049147</td>
          <td>0.049894</td>
          <td>0.000747</td>
          <td>0.245392</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099788</td>
          <td>0.049271</td>
          <td>0.038491</td>
          <td>0.021934</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1234.355511</td>
          <td>1234.353722</td>
          <td>-0.011591</td>
          <td>0.994572</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301488</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453328</td>
          <td>-0.029737</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049147</td>
          <td>0.049895</td>
          <td>0.000747</td>
          <td>0.245393</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099789</td>
          <td>0.049271</td>
          <td>0.038491</td>
          <td>0.021935</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1234.355512</td>
          <td>1234.353719</td>
          <td>-0.011591</td>
          <td>0.994572</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301487</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453328</td>
          <td>-0.029737</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049147</td>
          <td>0.049895</td>
          <td>0.000747</td>
          <td>0.245393</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099789</td>
          <td>0.049271</td>
          <td>0.038491</td>
          <td>0.021935</td>
        </tr>
        <tr>
          <th>6</th>
          <td>1234.355516</td>
          <td>1234.353722</td>
          <td>0.011591</td>
          <td>0.994572</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301488</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>0.546672</td>
          <td>-0.029737</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049147</td>
          <td>0.049895</td>
          <td>0.000747</td>
          <td>1.754607</td>
          <td>0.970381</td>
          <td>0.043694</td>
          <td>0.044168</td>
          <td>0.099789</td>
          <td>0.049271</td>
          <td>0.038491</td>
          <td>0.021935</td>
        </tr>
        <tr>
          <th>7</th>
          <td>1234.355516</td>
          <td>1234.353719</td>
          <td>0.011591</td>
          <td>0.994572</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301487</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>0.546672</td>
          <td>-0.029737</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049147</td>
          <td>0.049895</td>
          <td>0.000747</td>
          <td>1.754607</td>
          <td>0.970381</td>
          <td>0.043694</td>
          <td>0.044168</td>
          <td>0.099789</td>
          <td>0.049271</td>
          <td>0.038491</td>
          <td>0.021935</td>
        </tr>
        <tr>
          <th>8</th>
          <td>1234.355527</td>
          <td>1234.353723</td>
          <td>-0.011591</td>
          <td>0.994572</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301488</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453328</td>
          <td>-0.029737</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049147</td>
          <td>0.049895</td>
          <td>0.000747</td>
          <td>0.245393</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099789</td>
          <td>0.049271</td>
          <td>0.038491</td>
          <td>0.021935</td>
        </tr>
        <tr>
          <th>9</th>
          <td>1234.356740</td>
          <td>1234.353753</td>
          <td>-0.011589</td>
          <td>0.994572</td>
          <td>1.0</td>
          <td>0.000271</td>
          <td>0.0565</td>
          <td>1.0</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.301488</td>
          <td>0.044683</td>
          <td>0.348875</td>
          <td>1.453329</td>
          <td>-0.029733</td>
          <td>0.000074</td>
          <td>-0.002489</td>
          <td>0.049149</td>
          <td>0.049896</td>
          <td>0.000747</td>
          <td>0.245392</td>
          <td>1.029619</td>
          <td>-0.043694</td>
          <td>0.044168</td>
          <td>0.099793</td>
          <td>0.049273</td>
          <td>0.038493</td>
          <td>0.021935</td>
        </tr>
      </tbody>
    </table>
    </div>



We can also store this result for later purpose in the LeptonModel
object

.. code:: ipython3

    Model1.fit_results.append(df)  # call it with Model1.fit_results[0]

The fitting of this model is rather easy and does not require a lot of
recources since esentially only two parameters are varied. 
However, if one was to fit a more involved model with more
parameters, it can be necessary to run the fit on an external machine,
e.g. a server, and then transfer the result back to your local machine.
To keep the transfer-file as small as possible it is advisable to only
do the dimensionless fit on the external machine, since this is the
computation heavy part. The fitting of the neutrino mass scale and
adding of all lepton observables can then be done on the local machine.
The workflow would be as follows

.. code:: ipython3

    # On the external machine, define the model and then run
    df = Model1.dimless_fit(points=10)
    
    # Then export 'df' to a file, e.g. a csv
    # Transfer this file to your local machine
    # Import is again as 'df'
    
    # On your local machine, to add the lepton observables call
    df = Model1.complete_fit(df)
    
    # And store it in the model
    Model1.fit_results.append(df)

Analysing results
-----------------

You can now analyse the pandas.DataFrame that contains the fit results
conveniently with all the methods that pandas provides. For this
example, let us just look at the :math:`\chi^2`-decomposition of the
best fit point

.. code:: ipython3

    Model1.print_chisq(df.loc[0])


.. parsed-literal::

    's12^2': 0.30149014113020706,   chisq: 6.886414165000344e-05
    's13^2': 0.04468341961654504,   chisq: 1158.346412249292
    's23^2': 0.34887455998636124,   chisq: 76.00431398595319
    'm21^2': 7.402163061361973e-05,   chisq: 0.0015413501327695132
    'm3l^2': -0.002488634807977195,   chisq: 0.0013663894903165749
    Total chi-square: 1234.3537028390099


As also discussed in the paper, the mixing angle :math:`\theta_{13}`
seems not to be in agreement with the experimental data.

Exploring a minimum with Markov Chain Monte Carlo (MCMC) (Not yet implemented)
------------------------------------------------------------------------------

Using the emcee marcov chain monte carlo sampler one can conveniently
explore the neighborhood and hence the confidence level contours of a
specific minimum. This then also yields nice pictures ;)

Unfortunately this is yet to come and still has to be implemented into
the modelfitting module.
So, stay tuned for future development of FlavorPy!


Documentation
=============

The code of FlavorPy is documented here: 
:doc:`../index`.

But of course, as always you can access the documentation inside python with

.. code:: ipython3

    print(mf.LeptonModel.make_fit.__doc__)


.. parsed-literal::

    
            Does the fit for a specific number of random points in parameterspace.
    
            :param points: The number of random points in parameter space you want to fit.
                If you want to fit a specific starting point in parameter space, adjust the 'sampling_fct' in your
                ParameterSpace.
            :type points: int
    
            :param fitting_kwargs: properties of the Fit class.
                You can add keyword arguments that will be passed down to the Fit object used to make the fit.
                Please see the documentation of the Fit class for the specific keyword arguments. Of course, the keywords
                'model' and 'params' can not be passed down to Fit.
    
            :return: The result of the fit is returned in form of a pandas.DataFrame.
                Note that several (default:4) minimization algorithms are applied consecutively to one random point. Since
                the results of the intermediate steps are also written into the resulting DataFrame, it has more rows than
                the number entered as 'points'.
            :rtype: pandas.DataFrame
            

