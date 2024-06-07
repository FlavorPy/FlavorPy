import numpy as np
import pandas as pd
from ...experimentaldata import ExperimentalData

###########################################################################################
# Mixing-values taken from nu-fit 5.3 [2007.14792] see also www.nu-fit.org                #
# Mixing-values with SK atmospheric data                                                  #
# Charged lepton masses see arxiv:1706.08749                                              #
###########################################################################################

Lexpdata_NO = pd.DataFrame(np.array([
    [0.0048, 0.0565, 0.307, 0.02224, 0.454, 0.0741/2.505, 1.289, 7.41e-05, 2.505e-03],
    [0.0046, 0.0520, 0.296, 0.02167, 0.438, 0.0721/2.529, 1.15, 7.21e-05, 2.479e-03],
    [0.0050, 0.0610, 0.319, 0.0228, 0.473, 0.0762/2.479, 1.506, 7.62e-05, 2.529e-03],
    [0.0042, 0.0430, 0.275, 0.02047, 0.411, 0.0681/2.586, 0.772, 6.81e-05, 2.426e-03],
    [0.0054, 0.0700, 0.344, 0.02397, 0.606, 0.0803/2.426, 1.944, 8.03e-05, 2.586e-03]]),
                             columns=["me/mu", "mu/mt", "s12^2", "s13^2", "s23^2", "r", "d/pi", "m21^2", "m3l^2"],
                             index=['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])
Lexpdata_IO = pd.DataFrame(np.array([
    [0.0048, 0.0565, 0.307, 0.02222, 0.568, 0.0741/-2.487, 1.517, 7.41e-05, -2.487e-03],
    [0.0046, 0.0520, 0.296, 0.02165, 0.547, 0.0721/-2.511, 1.372, 7.21e-05, -2.46e-03],
    [0.0050, 0.0610, 0.319, 0.02291, 0.584, 0.0762/-2.46, 1.65, 7.62e-05, -2.511e-03],
    [0.0042, 0.0430, 0.275, 0.02049, 0.412, 0.0681/-2.407, 1.083, 6.81e-05, -2.566e-03],
    [0.0054, 0.0700, 0.344, 0.02420, 0.611, 0.0803/-2.566, 1.9, 8.03e-05, -2.407e-03]]),
                             columns=["me/mu", "mu/mt", "s12^2", "s13^2", "s23^2", "r", "d/pi", "m21^2", "m3l^2"],
                             index=['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])

NuFit53_NO_gauss = ExperimentalData(name='NuFit v5.3 NO with SK Gaussian errors', data_table=Lexpdata_NO)
NuFit53_IO_gauss = ExperimentalData(name='NuFit v5.3 IO with SK Gaussian errors', data_table=Lexpdata_IO)
