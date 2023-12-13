import numpy as np
import pandas as pd
from ...experimentaldata import ExperimentalData

###########################################################################################
# Mixing-values taken from nu-fit 5.2 [2007.14792] see also www.nu-fit.org                #
# Mixing-values with SK atmospheric data                                                  #
# Charged lepton masses see arxiv:1706.08749                                              #
###########################################################################################

Lexpdata_NO = pd.DataFrame(np.array([
    [0.0048, 0.0565, 0.303, 0.02225, 0.451, 0.0741/2.507, 1.29, 7.41e-05, 2.507e-03],
    [0.0046, 0.0520, 0.291, 0.02166, 0.435, 0.0721/2.533, 1.14, 7.21e-05, 2.480e-03],
    [0.0050, 0.0610, 0.315, 0.02281, 0.470, 0.0762/2.480, 1.49, 7.62e-05, 2.533e-03],
    [0.0042, 0.0430, 0.270, 0.02052, 0.408, 0.0682/2.590, 0.80, 6.82e-05, 2.427e-03],
    [0.0054, 0.0700, 0.341, 0.02398, 0.603, 0.0803/2.427, 1.94, 8.03e-05, 2.590e-03]]),
                             columns=["me/mu", "mu/mt", "s12^2", "s13^2", "s23^2", "r", "d/pi", "m21^2", "m3l^2"],
                             index=['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])

Lexpdata_IO = pd.DataFrame(np.array([
    [0.0048, 0.0565, 0.303, 0.02223, 0.569, 0.0741/-2.486, 1.54, 7.41e-05, -2.486e-03],
    [0.0046, 0.0520, 0.292, 0.02165, 0.548, 0.0721/-2.511, 1.38, 7.21e-05, -2.458e-03],
    [0.0050, 0.0610, 0.315, 0.02281, 0.585, 0.0762/-2.458, 1.67, 7.62e-05, -2.511e-03],
    [0.0042, 0.0430, 0.270, 0.02048, 0.411, 0.0682/-2.406, 1.08, 6.82e-05, -2.570e-03],
    [0.0054, 0.0700, 0.341, 0.02416, 0.613, 0.0803/-2.570, 1.92, 8.03e-05, -2.406e-03]]),
                             columns=["me/mu", "mu/mt", "s12^2", "s13^2", "s23^2", "r", "d/pi", "m21^2", "m3l^2"],
                             index=['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])

NuFit52_NO_gauss = ExperimentalData(name='NuFit v5.2 NO with SK Gaussian errors', data_table=Lexpdata_NO)
NuFit52_IO_gauss = ExperimentalData(name='NuFit v5.2 IO with SK Gaussian errors', data_table=Lexpdata_IO)
