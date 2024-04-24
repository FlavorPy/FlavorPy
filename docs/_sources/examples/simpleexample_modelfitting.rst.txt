Simple example
==============

This example is supposed to serve as a quick start to the ModelFitting module of FlavorPy.

Import FlavorPy
---------------

After installing flavorpy with 
:code:`pip install flavorpy`,
import the modelfitting module of FlavorPy.

.. code:: ipython3

    # Import the modelfitting module of FlavorPy
    import flavorpy.modelfitting as mf
    
    # We will also need numpy and pandas
    import numpy as np
    import pandas as pd

Defining mass matrices
----------------------

To define a model of leptons, we start by defining its mass matrices

:math:`m_e = \begin{pmatrix}v_1 & v_2 & v_3 \\ v_3 & v_1 & v_2 \\ v_2 & v_3 & v_1\end{pmatrix} \quad`
and
:math:`\quad m_n = \begin{pmatrix}v_1 & v_2 & v_3 \\ v_2 & v_1 & 2 \\ v_3 & 2 & v_1\end{pmatrix}`

For this example we have:

.. code:: ipython3

    # Charged lepton mass matrix
    def Me(params):
        v1, v2, v3 = params['v1'], params['v2'], params['v3']
        return np.array([[v1, v2 ,v3], [v3, v1, v2], [v2, v3, v1]])
    
    # Neutrino mass matrix
    def Mn(params):
        v1, v2, v3 = params['v1'], params['v2'], params['v3']
        return np.array([[v1, v2, v3], [v2, v1, 2], [v3, 2, v1]])

Defining parameterspace
-----------------------

Next, we define the parameterspace of our model. We therefore construct
an empty parameter space and add the parameters to it. When drawing
random points in our parameter space, we will evaluate the ‘sample_fct’,
which in this case is numpys uniform distribution between 0 and 1.

.. code:: ipython3

    ParamSpace = mf.ParameterSpace()
    ParamSpace.add_dim(name='v1', sample_fct=np.random.uniform)
    ParamSpace.add_dim(name='v2', sample_fct=np.random.uniform)
    ParamSpace.add_dim(name='v3', sample_fct=np.random.uniform)

Constructing Model
------------------

Then we can construct the lepton model as follows:

.. code:: ipython3

    Model0 = mf.LeptonModel(mass_matrix_e=Me, mass_matrix_n=Mn, parameterspace=ParamSpace, ordering='NO')

Now we can determine the masses and mixing observables of a given point
in parameter space by:

.. code:: ipython3

    Model0.get_obs({'v1': 1.5, 'v2': 1.1, 'v3': 1.3})




.. parsed-literal::

    {'me/mu': 0.9999999999999992,
     'mu/mt': 0.08882311833686546,
     's12^2': 0.6420066494741999,
     's13^2': 0.008093453559868121,
     's23^2': 0.0012798653500391485,
     'd/pi': 0.0,
     'r': 0.001571483801833027,
     'm21^2': 3.90198549455977e-05,
     'm3l^2': 0.024829944094927194,
     'm1': 0.018287792823374217,
     'm2': 0.019325196539654005,
     'm3': 0.15863287005308152,
     'eta1': 1.0,
     'eta2': 0.0,
     'J': 0.0,
     'Jmax': 0.0015294982440766927,
     'Sum(m_i)': 0.19624585941610972,
     'm_b': 0.02367630520881936,
     'm_bb': 0.020085293881200113,
     'nscale': 0.03548228305985807}



Here, ‘me/mu’ is the mass ratio of electron mass divided by muon mass,
‘sij^2’ refers to the mixing angles :math:`\sin^2(\theta_{ij})`, ‘d/pi’
is the cp violating phase in the PMNS matrix divided by :math:`\pi`, 
‘m21^2’ and ‘m3l^2’ and the squared neutrino mass differences, i.e. 
mij^2 = m_i^2 - m_j^2, ‘r’ is their quotient r = m21^2 / m3l^2, ‘m1’ 
and ‘m2’ and ‘m3’ are the neutrino masses, ‘eta1’ and ‘eta2’ are the 
majorana phases, ‘J’ is the Jarskog determinant, ‘m_b’ and ‘m_bb’ are 
the effective neutrino masses for beta decay and neutrinoless double 
beta decay, respectively.

Fitting model to experimental data
----------------------------------

Let us now fit this model to a specific experimental data set. As a
default the NuFit v5.2 for NO with SK data is used. To fit this model we
choose for example 3 randomly drawn points in the parameter space and
apply minimization algorithms to these points, in order to find a point
that matches the experimental data well. Note that by default 4 
minimization algorithms are applied consecutively to all 3 random points 
such that we get 12 points in the end.

.. code:: ipython3

    pd.set_option('display.max_columns', None)  # This pandas setting allows us to see all columns
    
    df = Model0.make_fit(points=3)
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
          <th style="min-width: 120px;">chisq_dimless</th>
          <th style="min-width: 90px;">v1</th>
          <th style="min-width: 90px;">v2</th>
          <th style="min-width: 90px;">v3</th>
          <th>n_scale</th>
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
          <th>eta1</th>
          <th>eta2</th>
          <th style="min-width: 120px;">J</th>
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
          <td>4.734807e+04</td>
          <td>4.734231e+04</td>
          <td>-1.058010</td>
          <td>1.041775</td>
          <td>0.007423</td>
          <td>1.0</td>
          <td>0.004846</td>
          <td>1.000000</td>
          <td>0.939127</td>
          <td>0.020008</td>
          <td>0.043020</td>
          <td>0.0</td>
          <td>0.031346</td>
          <td>0.000076</td>
          <td>0.002435</td>
          <td>0.016754</td>
          <td>0.018896</td>
          <td>0.052117</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.006725</td>
          <td>0.087767</td>
          <td>0.020031</td>
          <td>0.019447</td>
          <td>0.015745</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.734955e+04</td>
          <td>4.734214e+04</td>
          <td>-1.057276</td>
          <td>1.041025</td>
          <td>0.007412</td>
          <td>1.0</td>
          <td>0.004864</td>
          <td>1.000000</td>
          <td>0.939013</td>
          <td>0.019994</td>
          <td>0.043101</td>
          <td>0.0</td>
          <td>0.031618</td>
          <td>0.000077</td>
          <td>0.002425</td>
          <td>0.016712</td>
          <td>0.018867</td>
          <td>0.052006</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.006734</td>
          <td>0.087585</td>
          <td>0.019997</td>
          <td>0.019416</td>
          <td>0.015717</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.763479e+04</td>
          <td>4.762738e+04</td>
          <td>-1.057276</td>
          <td>1.041023</td>
          <td>0.007410</td>
          <td>1.0</td>
          <td>0.004866</td>
          <td>1.000000</td>
          <td>0.939013</td>
          <td>0.019994</td>
          <td>0.956899</td>
          <td>1.0</td>
          <td>0.031618</td>
          <td>0.000077</td>
          <td>0.002425</td>
          <td>0.016712</td>
          <td>0.018867</td>
          <td>0.052006</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>8.247443e-19</td>
          <td>0.006735</td>
          <td>0.087585</td>
          <td>0.019997</td>
          <td>0.019416</td>
          <td>0.015717</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.455653e+07</td>
          <td>1.508985e+07</td>
          <td>-0.060572</td>
          <td>0.757247</td>
          <td>-0.067181</td>
          <td>1.0</td>
          <td>0.766606</td>
          <td>1.000000</td>
          <td>0.875886</td>
          <td>0.016517</td>
          <td>0.471467</td>
          <td>1.0</td>
          <td>0.856485</td>
          <td>0.001111</td>
          <td>0.001297</td>
          <td>0.000261</td>
          <td>0.033327</td>
          <td>0.036011</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>2.547627e-18</td>
          <td>0.020803</td>
          <td>0.069600</td>
          <td>0.031568</td>
          <td>0.029551</td>
          <td>0.016206</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.476148e+07</td>
          <td>2.476143e+07</td>
          <td>1.116427</td>
          <td>0.933827</td>
          <td>0.761301</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.109402</td>
          <td>0.308190</td>
          <td>0.025502</td>
          <td>0.005890</td>
          <td>1.0</td>
          <td>0.036244</td>
          <td>0.000082</td>
          <td>0.002276</td>
          <td>0.007401</td>
          <td>0.011716</td>
          <td>0.048275</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>6.733970e-19</td>
          <td>0.005499</td>
          <td>0.067392</td>
          <td>0.011819</td>
          <td>0.009761</td>
          <td>0.013125</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2.476176e+07</td>
          <td>2.476165e+07</td>
          <td>1.081985</td>
          <td>0.903959</td>
          <td>0.711235</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.119073</td>
          <td>0.352498</td>
          <td>0.027924</td>
          <td>0.007796</td>
          <td>1.0</td>
          <td>0.042208</td>
          <td>0.000090</td>
          <td>0.002131</td>
          <td>0.007458</td>
          <td>0.012066</td>
          <td>0.046764</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>8.358433e-19</td>
          <td>0.006825</td>
          <td>0.066289</td>
          <td>0.012185</td>
          <td>0.010161</td>
          <td>0.012988</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2.476205e+07</td>
          <td>2.476193e+07</td>
          <td>1.070584</td>
          <td>0.917638</td>
          <td>0.644437</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.142031</td>
          <td>0.521885</td>
          <td>0.026420</td>
          <td>0.012484</td>
          <td>1.0</td>
          <td>0.044094</td>
          <td>0.000092</td>
          <td>0.002094</td>
          <td>0.007832</td>
          <td>0.012396</td>
          <td>0.046423</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.074881e-18</td>
          <td>0.008777</td>
          <td>0.066652</td>
          <td>0.012884</td>
          <td>0.011197</td>
          <td>0.013033</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2.476215e+07</td>
          <td>2.476202e+07</td>
          <td>1.065823</td>
          <td>0.917655</td>
          <td>0.625517</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.148736</td>
          <td>0.554173</td>
          <td>0.026439</td>
          <td>0.013648</td>
          <td>1.0</td>
          <td>0.044925</td>
          <td>0.000093</td>
          <td>0.002078</td>
          <td>0.007942</td>
          <td>0.012508</td>
          <td>0.046274</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.118010e-18</td>
          <td>0.009129</td>
          <td>0.066724</td>
          <td>0.013071</td>
          <td>0.011445</td>
          <td>0.013044</td>
        </tr>
        <tr>
          <th>8</th>
          <td>2.477452e+07</td>
          <td>2.477439e+07</td>
          <td>2.665004</td>
          <td>-0.107629</td>
          <td>4.486246</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.568848</td>
          <td>0.627061</td>
          <td>0.025983</td>
          <td>0.085722</td>
          <td>0.0</td>
          <td>0.044634</td>
          <td>0.000093</td>
          <td>0.002084</td>
          <td>0.014539</td>
          <td>0.017446</td>
          <td>0.047906</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.000000e+00</td>
          <td>0.021255</td>
          <td>0.079891</td>
          <td>0.018020</td>
          <td>0.017207</td>
          <td>0.006356</td>
        </tr>
        <tr>
          <th>9</th>
          <td>2.477452e+07</td>
          <td>2.477439e+07</td>
          <td>2.665004</td>
          <td>-0.107629</td>
          <td>4.486246</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.568848</td>
          <td>0.627061</td>
          <td>0.025983</td>
          <td>0.085722</td>
          <td>0.0</td>
          <td>0.044634</td>
          <td>0.000093</td>
          <td>0.002084</td>
          <td>0.014539</td>
          <td>0.017446</td>
          <td>0.047906</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.000000e+00</td>
          <td>0.021255</td>
          <td>0.079891</td>
          <td>0.018020</td>
          <td>0.017207</td>
          <td>0.006356</td>
        </tr>
        <tr>
          <th>10</th>
          <td>2.477857e+07</td>
          <td>2.499999e+07</td>
          <td>0.283663</td>
          <td>0.041842</td>
          <td>0.679834</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.554919</td>
          <td>0.884184</td>
          <td>0.000612</td>
          <td>0.133847</td>
          <td>1.0</td>
          <td>0.563388</td>
          <td>0.000743</td>
          <td>0.001319</td>
          <td>0.003915</td>
          <td>0.027542</td>
          <td>0.036532</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>3.299055e-19</td>
          <td>0.002694</td>
          <td>0.067989</td>
          <td>0.025949</td>
          <td>0.024819</td>
          <td>0.015164</td>
        </tr>
        <tr>
          <th>11</th>
          <td>2.477902e+07</td>
          <td>2.478546e+07</td>
          <td>2.169190</td>
          <td>-0.118765</td>
          <td>5.225849</td>
          <td>1.0</td>
          <td>1.000000</td>
          <td>0.638308</td>
          <td>0.605378</td>
          <td>0.015513</td>
          <td>0.119236</td>
          <td>1.0</td>
          <td>0.127448</td>
          <td>0.000197</td>
          <td>0.001544</td>
          <td>0.011954</td>
          <td>0.018431</td>
          <td>0.041074</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>2.378544e-18</td>
          <td>0.019422</td>
          <td>0.071459</td>
          <td>0.016947</td>
          <td>0.016302</td>
          <td>0.005316</td>
        </tr>
      </tbody>
    </table>
    </div>



Well, from the high value of :math:`\chi^2`, we see that this model
doesn’t seem to be able to replicate the experimentally measured values.
Let us take a look at the individual contributions to :math:`\chi^2` for
the first point by

.. code:: ipython3

    Model0.print_chisq(df.loc[0])


.. parsed-literal::

    'me/mu': 0.0048458380255911905,   chisq: 0.05252811475246681
    'mu/mt': 1.0,   chisq: 43960.11111111111
    's12^2': 0.9391272761692583,   chisq: 2810.124385323049
    's13^2': 0.020008064149810444,   chisq: 15.032334445663338
    's23^2': 0.04301966448870213,   chisq: 543.503523800527
    'd/pi': 0.0,   chisq: 10.98525
    'm21^2': 7.634175414433096e-05,   chisq: 1.115587066597515
    'm3l^2': 0.002435482734326478,   chisq: 7.142800422396115
    Total chi-square: 47348.067520284094


It looks like several observables are not in agreement with the
experimental data. Note that :math:`\chi^2=x` is often interpreted as the
specific point lying in the :math:`\sqrt{x}\,\sigma` confidence level
region.

All in all, the model was probably to simple or we needed to widen the
boundaries of our parameter space.
