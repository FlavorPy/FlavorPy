Simple example
==============

This example is supposed to serve as a quick start to the ModelFitting
module of FlavorPy.

Import FlavorPy
---------------

After installing FlavorPy with 
:code:`pip install flavorpy`, 
import the modelfitting module.

.. code:: ipython3

    # Import the modelfitting module of FlavorPy
    import flavorpy.modelfitting as mf
    
    # We will also need numpy and pandas
    import numpy as np
    import pandas as pd

Defining mass matrices
----------------------

To define a model of leptons, we start by defining its mass matrices

:math:`M_e = \begin{pmatrix} 0.0048\times0.0565 & 0 & 0 \\ 0 & 0.0565 & 0 \\ 0 & 0 & 1\end{pmatrix} \quad`
and
:math:`\quad M_n = \begin{pmatrix} 1 & v_2 & v_3 \\ v_1 & 1 & v_3 \\ v_1 & v_2 & 1\end{pmatrix}`

For this example we have:

.. code:: ipython3

    # Charged lepton mass matrix
    def Me(params):
        v1, v2, v3 = params['v1'], params['v2'], params['v3']
        return np.array([[0.0048*0.0565, 0 ,0], [0, 0.0565, 0], [0, 0, 1]])
    
    # Neutrino mass matrix
    def Mn(params):
        v1, v2, v3 = params['v1'], params['v2'], params['v3']
        return np.array([[1, v2, v3], [v1, 1, v3], [v1, v2, 1]])

Defining parameterspace
-----------------------

Next, we define the parameterspace of our model. We therefore construct
an empty parameter space and add the parameters to it. When fitting, we will 
draw random points within this parameter space.

.. code:: ipython3

    ParamSpace = mf.ParameterSpace()
    ParamSpace.add_dim(name='v1')
    ParamSpace.add_dim(name='v2')
    ParamSpace.add_dim(name='v3')

Constructing Model
------------------

Then we can construct the lepton model as follows:

.. code:: ipython3

    MyModel = mf.FlavorModel(mass_matrix_e=Me, mass_matrix_n=Mn, parameterspace=ParamSpace, ordering='NO')

Now we can determine the masses and mixing observables of a given point
in parameter space by:

.. code:: ipython3

    MyModel.get_obs({'v1': 1.5, 'v2': 1.1, 'v3': 1.3})




.. parsed-literal::

    {'me/mu': 0.0048,
     'mu/mt': 0.0565,
     's12^2': 0.9053720503789906,
     's13^2': 0.2893377761696659,
     's23^2': 0.5280465699834944,
     'd/pi': 0.0,
     'r': 0.010306101829667999,
     'm21^2': 4.9968698643488846e-05,
     'm3l^2': 0.0048484576874298696,
     'm1': 0.0033969202435270816,
     'm2': 0.007842688683377208,
     'm3': 0.06971367695488995,
     'eta1': 1.0,
     'eta2': 1.0,
     'J': 0.0,
     'Jmax': 0.05585663171203268,
     'Sum(m_i)': 0.08095328588179423,
     'm_b': 0.03822289118358137,
     'm_bb': 0.025548760738177294,
     'nscale': 0.019258907884260646}



Here, ‘me/mu’ is the mass ratio of electron mass divided by muon mass,
‘sij^2’ refers to the mixing angles :math:`\sin^2(\theta_{ij})`, ‘d/pi’
is the cp violating phase in the PMNS matrix divided by :math:`\pi`,
‘m21^2’ and ‘m3l^2’ and the squared neutrino mass differences,
i.e. mij^2 = m_i^2 - m_j^2, ‘r’ is their quotient r = m21^2 / m3l^2,
‘m1’ and ‘m2’ and ‘m3’ are the neutrino masses, ‘eta1’ and ‘eta2’ are
the majorana phases, ‘J’ is the Jarskog determinant, ‘m_b’ and ‘m_bb’
are the effective neutrino masses for beta decay and neutrinoless double
beta decay, respectively.

Fitting model to experimental data
----------------------------------

Let us now fit this model to a specific experimental data set. As a
default the NuFit v5.3 for NO with SK data is used. To fit this model we
choose for example 3 randomly drawn points in the parameter space and
apply minimization algorithms to these points, in order to find a point
that matches the experimental data well. Note that by default 4
minimization algorithms are applied consecutively to all 3 random points
such that we get 12 points in the end.

.. code:: ipython3

    pd.set_option('display.max_columns', None)  # This pandas setting allows us to see all columns
    
    df = MyModel.make_fit(points=5)
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
          <td>5.651948</td>
          <td>5.651702</td>
          <td>5.036863</td>
          <td>-0.528403</td>
          <td>0.701039</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022200</td>
          <td>0.499957</td>
          <td>1.0</td>
          <td>0.029596</td>
          <td>0.000074</td>
          <td>0.002505</td>
          <td>0.002326</td>
          <td>0.008920</td>
          <td>0.050108</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.107236e-18</td>
          <td>0.033538</td>
          <td>0.061353</td>
          <td>0.009208</td>
          <td>0.005369</td>
          <td>0.006852</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5.651948</td>
          <td>5.651702</td>
          <td>5.036863</td>
          <td>-0.528403</td>
          <td>0.701039</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022200</td>
          <td>0.499957</td>
          <td>1.0</td>
          <td>0.029596</td>
          <td>0.000074</td>
          <td>0.002505</td>
          <td>0.002326</td>
          <td>0.008920</td>
          <td>0.050108</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.107236e-18</td>
          <td>0.033538</td>
          <td>0.061353</td>
          <td>0.009208</td>
          <td>0.005369</td>
          <td>0.006852</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.651948</td>
          <td>5.651702</td>
          <td>5.036863</td>
          <td>-0.528403</td>
          <td>0.701039</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022200</td>
          <td>0.499957</td>
          <td>1.0</td>
          <td>0.029596</td>
          <td>0.000074</td>
          <td>0.002505</td>
          <td>0.002326</td>
          <td>0.008920</td>
          <td>0.050108</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.107236e-18</td>
          <td>0.033538</td>
          <td>0.061353</td>
          <td>0.009208</td>
          <td>0.005369</td>
          <td>0.006852</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.651948</td>
          <td>5.651702</td>
          <td>5.036863</td>
          <td>-0.528403</td>
          <td>0.701039</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022200</td>
          <td>0.499957</td>
          <td>1.0</td>
          <td>0.029596</td>
          <td>0.000074</td>
          <td>0.002505</td>
          <td>0.002326</td>
          <td>0.008920</td>
          <td>0.050108</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.107236e-18</td>
          <td>0.033538</td>
          <td>0.061353</td>
          <td>0.009208</td>
          <td>0.005369</td>
          <td>0.006852</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.655930</td>
          <td>5.655555</td>
          <td>5.049672</td>
          <td>-0.534924</td>
          <td>0.713045</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022200</td>
          <td>0.500035</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002402</td>
          <td>0.008939</td>
          <td>0.050113</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.124460e-18</td>
          <td>0.033679</td>
          <td>0.061455</td>
          <td>0.009246</td>
          <td>0.005457</td>
          <td>0.006835</td>
        </tr>
        <tr>
          <th>5</th>
          <td>5.655930</td>
          <td>5.655555</td>
          <td>5.049672</td>
          <td>-0.534924</td>
          <td>0.713045</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022200</td>
          <td>0.500035</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002402</td>
          <td>0.008939</td>
          <td>0.050113</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.124460e-18</td>
          <td>0.033679</td>
          <td>0.061455</td>
          <td>0.009246</td>
          <td>0.005457</td>
          <td>0.006835</td>
        </tr>
        <tr>
          <th>6</th>
          <td>5.655930</td>
          <td>5.655555</td>
          <td>5.049672</td>
          <td>-0.534924</td>
          <td>0.713045</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022200</td>
          <td>0.500035</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002402</td>
          <td>0.008939</td>
          <td>0.050113</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.124460e-18</td>
          <td>0.033679</td>
          <td>0.061455</td>
          <td>0.009246</td>
          <td>0.005457</td>
          <td>0.006835</td>
        </tr>
        <tr>
          <th>7</th>
          <td>5.657581</td>
          <td>5.657194</td>
          <td>5.041598</td>
          <td>-0.533464</td>
          <td>0.714117</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022280</td>
          <td>0.500068</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002413</td>
          <td>0.008942</td>
          <td>0.050114</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.131556e-18</td>
          <td>0.033737</td>
          <td>0.061469</td>
          <td>0.009260</td>
          <td>0.005469</td>
          <td>0.006845</td>
        </tr>
        <tr>
          <th>8</th>
          <td>5.657581</td>
          <td>5.657194</td>
          <td>5.041598</td>
          <td>-0.533464</td>
          <td>0.714117</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022280</td>
          <td>0.500068</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002413</td>
          <td>0.008942</td>
          <td>0.050114</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.131556e-18</td>
          <td>0.033737</td>
          <td>0.061469</td>
          <td>0.009260</td>
          <td>0.005469</td>
          <td>0.006845</td>
        </tr>
        <tr>
          <th>9</th>
          <td>5.657581</td>
          <td>5.657194</td>
          <td>5.041598</td>
          <td>-0.533464</td>
          <td>0.714117</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022280</td>
          <td>0.500068</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002413</td>
          <td>0.008942</td>
          <td>0.050114</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.131556e-18</td>
          <td>0.033737</td>
          <td>0.061469</td>
          <td>0.009260</td>
          <td>0.005469</td>
          <td>0.006845</td>
        </tr>
        <tr>
          <th>10</th>
          <td>5.657581</td>
          <td>5.657194</td>
          <td>5.041598</td>
          <td>-0.533464</td>
          <td>0.714117</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022280</td>
          <td>0.500068</td>
          <td>1.0</td>
          <td>0.029592</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002413</td>
          <td>0.008942</td>
          <td>0.050114</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.131556e-18</td>
          <td>0.033737</td>
          <td>0.061469</td>
          <td>0.009260</td>
          <td>0.005469</td>
          <td>0.006845</td>
        </tr>
        <tr>
          <th>11</th>
          <td>5.658516</td>
          <td>5.658316</td>
          <td>5.050711</td>
          <td>-0.535313</td>
          <td>0.712793</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.309623</td>
          <td>0.022189</td>
          <td>0.500027</td>
          <td>1.0</td>
          <td>0.029604</td>
          <td>0.000074</td>
          <td>0.002505</td>
          <td>0.002400</td>
          <td>0.008940</td>
          <td>0.050108</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>4.123487e-18</td>
          <td>0.033671</td>
          <td>0.061447</td>
          <td>0.009243</td>
          <td>0.005455</td>
          <td>0.006833</td>
        </tr>
        <tr>
          <th>12</th>
          <td>14.019848</td>
          <td>14.013724</td>
          <td>5.028969</td>
          <td>0.702363</td>
          <td>-0.526542</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022280</td>
          <td>0.500002</td>
          <td>0.0</td>
          <td>0.029570</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050119</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033596</td>
          <td>0.061379</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>13</th>
          <td>14.019853</td>
          <td>14.013724</td>
          <td>5.028969</td>
          <td>0.702363</td>
          <td>-0.526542</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022280</td>
          <td>0.500002</td>
          <td>0.0</td>
          <td>0.029570</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050119</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033596</td>
          <td>0.061379</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>14</th>
          <td>14.019853</td>
          <td>14.013724</td>
          <td>5.028969</td>
          <td>0.702363</td>
          <td>-0.526542</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022280</td>
          <td>0.500002</td>
          <td>0.0</td>
          <td>0.029570</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050119</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033596</td>
          <td>0.061379</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>15</th>
          <td>14.019853</td>
          <td>14.013724</td>
          <td>5.028969</td>
          <td>0.702363</td>
          <td>-0.526542</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022280</td>
          <td>0.500002</td>
          <td>0.0</td>
          <td>0.029570</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050119</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033596</td>
          <td>0.061379</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>16</th>
          <td>14.020580</td>
          <td>14.014997</td>
          <td>5.028979</td>
          <td>0.702446</td>
          <td>-0.526597</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305033</td>
          <td>0.022281</td>
          <td>0.500001</td>
          <td>0.0</td>
          <td>0.029571</td>
          <td>0.000074</td>
          <td>0.002506</td>
          <td>0.002340</td>
          <td>0.008921</td>
          <td>0.050119</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033597</td>
          <td>0.061380</td>
          <td>0.009224</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>17</th>
          <td>14.021824</td>
          <td>14.013749</td>
          <td>5.028994</td>
          <td>0.702405</td>
          <td>-0.526469</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022280</td>
          <td>0.500000</td>
          <td>0.0</td>
          <td>0.029565</td>
          <td>0.000074</td>
          <td>0.002507</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050121</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033596</td>
          <td>0.061382</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>18</th>
          <td>14.021824</td>
          <td>14.013749</td>
          <td>5.028994</td>
          <td>0.702405</td>
          <td>-0.526469</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.305000</td>
          <td>0.022280</td>
          <td>0.500000</td>
          <td>0.0</td>
          <td>0.029565</td>
          <td>0.000074</td>
          <td>0.002507</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050121</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033596</td>
          <td>0.061382</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
        <tr>
          <th>19</th>
          <td>14.022732</td>
          <td>14.014400</td>
          <td>5.029001</td>
          <td>0.702364</td>
          <td>-0.526447</td>
          <td>1</td>
          <td>0.0048</td>
          <td>0.0565</td>
          <td>0.304985</td>
          <td>0.022280</td>
          <td>0.500000</td>
          <td>0.0</td>
          <td>0.029565</td>
          <td>0.000074</td>
          <td>0.002507</td>
          <td>0.002339</td>
          <td>0.008921</td>
          <td>0.050121</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000000e+00</td>
          <td>0.033595</td>
          <td>0.061381</td>
          <td>0.009223</td>
          <td>0.005383</td>
          <td>0.006864</td>
        </tr>
      </tbody>
    </table>
    </div>


The fit yields a point with :math:`\chi^2` arround 5. Since
:math:`\chi^2=x` can be interpreted as the specific point lying in the
:math:`\sqrt{x}\,\sigma` confidence level region, this means that our
point is outside the 2\ :math:`\sigma` but inside the 3\ :math:`\sigma`
region of the experimental data. Let us take a look at the individual
contributions to :math:`\chi^2` for the best-fit point by

.. code:: ipython3

    MyModel.print_chisq(df.loc[0])


.. parsed-literal::

    'me/mu': 0.0048,   chisq: 0.0
    'mu/mt': 0.0565,   chisq: 0.0
    's12^2': 0.30500000121100895,   chisq: 4.2032172992145e-08
    's13^2': 0.02219999995440627,   chisq: 1.0523577958707064e-08
    's23^2': 0.4999568484745416,   chisq: 2.9383389552791437
    'd/pi': 1.0,   chisq: 2.71318
    'm21^2': 7.414848837981547e-05,   chisq: 0.0003285239267222246
    'm3l^2': 0.0025053616533515753,   chisq: 0.00010020584059472139
    Total chi-square: 5.651947737602212


It looks like the :math:`\sin^2\theta_{12}`, :math:`\sin^2\theta_{13}`,
:math:`\Delta m_{21}^2`, and :math:`\Delta m_{3\ell}^2` are within their
experimental 1\ :math:`\sigma` intervall. However,
:math:`\sin^2\theta_{23}` and the CP phase
:math:`\delta_{\mathrm{CP}}/\pi` are only within their experimental
2\ :math:`\sigma` intervall. On a side note, it is no surprise, that the
CP phase is off, since it is always CP conserving due to the mass
matrices being real. All errors then add up to our best-fit point of the
model lying within the 3\ :math:`\sigma` confidence level region.

MCMC fit to explore the minimum of the model
--------------------------------------------

To explore the neighborhood of the minimum we use the emcee marcov chain
monte carlo sampler.

.. code:: ipython3

    df_mcmc = MyModel.mcmc_fit(df.loc[[0]], mcmc_steps=2000)
    df_mcmc = MyModel.complete_fit(df_mcmc)


.. parsed-literal::

    0 : 100%|██████████| 2000/2000 [00:30<00:00, 65.16it/s]


Plotting the results
--------------------

We can plot the parameterspace and see the contour of the
3\ :math:`\sigma` CL region. The colormap is scaled in such a way that
the 1\ :math:`\sigma` region, i.e. \ :math:`\chi^2<1`, is green, the
2\ :math:`\sigma` region with :math:`1<\chi^2<4` is yellow, the
3\ :math:`\sigma` region with :math:`4<\chi^2<9` is orange, and anything
white lies outside of the :math:`5\sigma` region.

.. code:: ipython3

    mf.plot(df_mcmc, x='v1', y='v2', vmin=0, vmax=25);



.. image:: simpleexample_modelfitting_v1_v2.png


We can also plot the observables and their corresponding
1\ :math:`\sigma` and 3\ :math:`\sigma` CL bounds from NuFit v5.3,
i.e. http://www.nu-fit.org/?q=node/278,

.. code:: ipython3

    mf.plot(df_mcmc, x='s12^2', y='m21^2', show_exp='1dim', vmin=0, vmax=25);



.. image:: simpleexample_modelfitting_s12_m21.png


but also the two dimensional :math:`\chi^2`-profiles to get a better
estimate of the CL regions for correlated, non-gaussian errors, e.g.

.. code:: ipython3

    mf.plot(df_mcmc, x='s23^2', y='d/pi', show_exp='2dim', vmin=0, vmax=25, gridsize=8);



.. image:: simpleexample_modelfitting_s23_dpi.png


Note that our model spreads out the full 1\ :math:`\sigma` region of the
well measured :math:`\sin^2\theta_{12}` and :math:`\sin^2\theta_{13}`
while having a clear prediction for :math:`\sin^2\theta_{23}` and the
CP-angle :math:`\delta_{\mathrm{CP}}^\ell/\pi`, which have not yet been
measured that precisely.

Also from the lobster plot we see a prediction for the lightest neutrino
mass and the effective neutrion mass for neutrinoless double beta decay:

.. code:: ipython3

    ax = mf.plot(df_mcmc, x='m1', y='m_bb', 
                 ordering='NO', show_exp='2dim', xscale='log', yscale='log', vmin=0, vmax=25)
    ax.axvspan(0.037,10, facecolor='gray',alpha=0.21); # arXiv: 2009.03287
    ax.axhspan(0.156,10, facecolor='gray',alpha=0.21);  # arXiv: 2203.02139
    ax.axhspan(0.036,0.156, facecolor='gray',alpha=0.13);  # arXiv: 2203.02139
    ax.text(0.1,0.0012,'Cosmology', color='gray');
    ax.text(0.0006,0.3,'KamLAND-Zen', color='gray');



.. image:: simpleexample_modelfitting_lobster.png

