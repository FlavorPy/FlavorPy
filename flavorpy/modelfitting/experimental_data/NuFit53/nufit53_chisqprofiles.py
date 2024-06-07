import numpy as np
import pandas as pd
import scipy.interpolate
import pkgutil
from io import StringIO
from ...experimentaldata import ExperimentalData

###########################################################################################
# Mixing-values taken from nu-fit 5.3 [2007.14792] see also www.nu-fit.org                #
# Mixing-values with SK atmospheric data                                                  #
# Charged lepton masses see arxiv:1706.08749                                              #
###########################################################################################

### import the 1-dimensional chi^2 projections
# Normal ordering
data_no = StringIO(pkgutil.get_data(__name__, "v53.release-SKyes-NO.txt").decode())
exps13_NO = pd.read_csv(data_no, delimiter='\s+', skiprows=1524981, nrows=102, index_col=False)
data_no.seek(0)
exps12_NO = pd.read_csv(data_no, delimiter='\s+', skiprows=1525084, nrows=133, index_col=False)
data_no.seek(0)
exps23_NO = pd.read_csv(data_no, delimiter='\s+', skiprows=1525220, nrows=101, index_col=False)
data_no.seek(0)
expdcp_NO = pd.read_csv(data_no, delimiter='\s+', skiprows=1525323, nrows=73, index_col=False)
data_no.seek(0)
expm21_NO = pd.read_csv(data_no, delimiter='\s+', skiprows=1525398, nrows=312, index_col=False)
data_no.seek(0)
expm3l_NO = pd.read_csv(data_no, delimiter='\s+', skiprows=1525712, nrows=480, index_col=False)
expdcp_NO['d/pi'] = np.mod(expdcp_NO['Delta_CP/deg']/180, 2)
expdcp_NO = expdcp_NO.sort_values(by=['d/pi'])
expdcp_NO = expdcp_NO.drop_duplicates(subset=['d/pi'])
expm21_NO['m21'] = np.power(10, expm21_NO['Log10(Delta_m21^2/[eV^2])'])
expm3l_NO['m3l'] = expm3l_NO['Delta_m31^2/[1e-3_eV^2]']*1e-03
# Inverted ordering
data_io = StringIO(pkgutil.get_data(__name__, "v53.release-SKyes-IO.txt").decode())
exps13_IO = pd.read_csv(data_io, delimiter='\s+', skiprows=1524981, nrows=102, index_col=False)
data_io.seek(0)
exps12_IO = pd.read_csv(data_io, delimiter='\s+', skiprows=1525084, nrows=133, index_col=False)
data_io.seek(0)
exps23_IO = pd.read_csv(data_io, delimiter='\s+', skiprows=1525220, nrows=101, index_col=False)
data_io.seek(0)
expdcp_IO = pd.read_csv(data_io, delimiter='\s+', skiprows=1525323, nrows=73, index_col=False)
data_io.seek(0)
expm21_IO = pd.read_csv(data_io, delimiter='\s+', skiprows=1525398, nrows=312, index_col=False)
data_io.seek(0)
expm3l_IO = pd.read_csv(data_io, delimiter='\s+', skiprows=1525712, nrows=480, index_col=False)
expdcp_IO['d/pi'] = np.mod(expdcp_IO['Delta_CP/deg']/180, 2)
expdcp_IO = expdcp_IO.sort_values(by=['d/pi'])
expdcp_IO = expdcp_IO.drop_duplicates(subset=['d/pi'])
expm21_IO['m21'] = np.power(10, expm21_IO['Log10(Delta_m21^2/[eV^2])'])
expm3l_IO['m3l'] = expm3l_IO['Delta_m32^2/[1e-3_eV^2]']*1e-03
# Shift the Delta_chi^2 such that it reaches zero:
exps12_IO['Delta_chi^2'] = exps12_IO['Delta_chi^2'] - np.min(exps12_IO['Delta_chi^2'])
exps13_IO['Delta_chi^2'] = exps13_IO['Delta_chi^2'] - np.min(exps13_IO['Delta_chi^2'])
exps23_IO['Delta_chi^2'] = exps23_IO['Delta_chi^2'] - np.min(exps23_IO['Delta_chi^2'])
expdcp_IO['Delta_chi^2'] = expdcp_IO['Delta_chi^2'] - np.min(expdcp_IO['Delta_chi^2'])
expm21_IO['Delta_chi^2'] = expm21_IO['Delta_chi^2'] - np.min(expm21_IO['Delta_chi^2'])
expm3l_IO['Delta_chi^2'] = expm3l_IO['Delta_chi^2'] - np.min(expm3l_IO['Delta_chi^2'])

### Make an interpolation to get a continuous function out of the discrete datapoints
# Normal ordering
chisqs12Spline_NO = scipy.interpolate.make_interp_spline(exps12_NO['sin^2(theta12)'], exps12_NO['Delta_chi^2'])
chisqs13Spline_NO = scipy.interpolate.make_interp_spline(exps13_NO['sin^2(theta13)'], exps13_NO['Delta_chi^2'])
chisqs23Spline_NO = scipy.interpolate.make_interp_spline(exps23_NO['sin^2(theta23)'], exps23_NO['Delta_chi^2'])
chisqdcpSpline_NO = scipy.interpolate.make_interp_spline(expdcp_NO['d/pi'], expdcp_NO['Delta_chi^2'])
chisqm21Spline_NO = scipy.interpolate.make_interp_spline(expm21_NO['m21'], expm21_NO['Delta_chi^2'])
chisqm3lSpline_NO = scipy.interpolate.make_interp_spline(expm3l_NO['m3l'], expm3l_NO['Delta_chi^2'])
# Inverted Ordering
chisqs12Spline_IO = scipy.interpolate.make_interp_spline(exps12_IO['sin^2(theta12)'], exps12_IO['Delta_chi^2'])
chisqs13Spline_IO = scipy.interpolate.make_interp_spline(exps13_IO['sin^2(theta13)'], exps13_IO['Delta_chi^2'])
chisqs23Spline_IO = scipy.interpolate.make_interp_spline(exps23_IO['sin^2(theta23)'], exps23_IO['Delta_chi^2'])
chisqdcpSpline_IO = scipy.interpolate.make_interp_spline(expdcp_IO['d/pi'], expdcp_IO['Delta_chi^2'])
chisqm21Spline_IO = scipy.interpolate.make_interp_spline(expm21_IO['m21'], expm21_IO['Delta_chi^2'])
chisqm3lSpline_IO = scipy.interpolate.make_interp_spline(expm3l_IO['m3l'], expm3l_IO['Delta_chi^2'])

### Extend the chisq profiles at their ends, i.e. where the dataset of nufit is trucated, by a gaussian chisq profile
# Gaussian data
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


def chisq_simple(value: float, data, name: str) -> float:
    return ((value - data[name]['best']) / (data[name]['1sig_max'] - data[name]['1sig_min']) * 2) ** 2


# Extension
def s12_profile_NO(value: float) -> float:
    if value < 0.17 or value > 0.8:
        return chisq_simple(value, Lexpdata_NO, "s12^2")
    else:
        return chisqs12Spline_NO(value)


def s13_profile_NO(value: float) -> float:
    if value > 0.07:
        return chisq_simple(value, Lexpdata_NO, "s13^2")
    else:
        return chisqs13Spline_NO(value)


def s23_profile_NO(value: float) -> float:
    if value < 0.25 or value > 0.7:
        return chisq_simple(value, Lexpdata_NO, "s23^2")
    else:
        return chisqs23Spline_NO(value)


def m21_profile_NO(value: float) -> float:
    if value < 0.000001 or value > 0.01:
        return chisq_simple(value, Lexpdata_NO, "m21^2")
    else:
        return chisqm21Spline_NO(value)


def m3l_profile_NO(value: float) -> float:
    if value < 0.0002 or value > 0.007:
        return chisq_simple(value, Lexpdata_NO, "m3l^2")
    else:
        return chisqm3lSpline_NO(value)


def s12_profile_IO(value: float) -> float:
    if value < 0.17 or value > 0.8:
        return chisq_simple(value, Lexpdata_IO, "s12^2")
    else:
        return chisqs12Spline_IO(value)


def s13_profile_IO(value: float) -> float:
    if value > 0.07:
        return chisq_simple(value, Lexpdata_IO, "s13^2")
    else:
        return chisqs13Spline_IO(value)


def s23_profile_IO(value: float) -> float:
    if value < 0.25 or value > 0.7:
        return chisq_simple(value, Lexpdata_IO, "s23^2")
    else:
        return chisqs23Spline_IO(value)


def m21_profile_IO(value: float) -> float:
    if value < 0.000001 or value > 0.01:
        return chisq_simple(value, Lexpdata_IO, "m21^2")
    else:
        return chisqm21Spline_IO(value)


def m3l_profile_IO(value: float) -> float:
    if value > -0.0002 or value < -0.007:
        return chisq_simple(value, Lexpdata_IO, "m3l^2")
    else:
        return chisqm3lSpline_IO(value)


### Make profiles for charged lepton masses and the ratio of neutrino masses (they are gaussian!)
# It would in principle be possible to generate a "realistic" chisq profile for the neutrino masses ratio r
# out of the profiles for m21^2 and m3l^2, but I did not had the time to do that...
def memu_profile(value: float) -> float:
    return chisq_simple(value, Lexpdata_NO, "me/mu")


def mumt_profile(value: float) -> float:
    return chisq_simple(value, Lexpdata_NO, "mu/mt")


def r_profile_NO(value: float) -> float:
    return chisq_simple(value, Lexpdata_NO, "r")


def r_profile_IO(value: float) -> float:
    return chisq_simple(value, Lexpdata_IO, "r")


### Finally the output dataset
chisq_profiles_NO = {"me/mu": memu_profile, "mu/mt": mumt_profile, "s12^2": s12_profile_NO, "s13^2": s13_profile_NO,
                     "s23^2": s23_profile_NO, "d/pi": chisqdcpSpline_NO, "r": r_profile_NO,
                     "m21^2": m21_profile_NO, "m3l^2": m3l_profile_NO}

chisq_profiles_IO = {"me/mu": memu_profile, "mu/mt": mumt_profile, "s12^2": s12_profile_IO, "s13^2": s13_profile_IO,
                     "s23^2": s23_profile_IO, "d/pi": chisqdcpSpline_IO, "r": r_profile_IO,
                     "m21^2": m21_profile_IO, "m3l^2": m3l_profile_IO}

NuFit53_NO = ExperimentalData(name="NuFit v5.3 NO with SK chisquare profiles", data=chisq_profiles_NO)

NuFit53_IO = ExperimentalData(name="NuFit v5.3 IO with SK chisquare profiles", data=chisq_profiles_IO)
