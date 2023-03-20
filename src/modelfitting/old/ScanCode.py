# Math and dataset
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
from mpmath import *   # for dedekind eta function
import scipy  # for Bspline of Jarlskog data and chisq profiles

# Minimizers
from lmfit import minimize, Parameters, fit_report
import lmfit
import emcee
from iminuit import Minuit

# Clustering and Data generation in cluster shape
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Miscellaneous
import time

#Plotting
import alphashape
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes


########### Experimental Values ##############

### Leptons

# Values taken from nu-fit 5.1 [2007.14792] see also www.nu-fit.org
# The values are with SK data!
# ordering of the dataset: best, 1_sigma_min, 1_sigma_max, 3_sigma_min, 3_sigma_max  (expept for 'd/pi')
Lexpdata_NO = pd.DataFrame(np.array([
    [0.0048,0.0565,0.304,0.02246,0.450,0.0742/2.510,1.28,7.42e-05,2.510e-03],
    [0.0046,0.0565-0.0045,0.292,0.02184,0.434,0.0722/2.537,1.14,7.22e-05,2.483e-03],
    [0.0050,0.0565+0.0045,0.316,0.02308,0.469,0.0763/2.483,1.48,7.63e-05,2.537e-03],
    [0.0042,0.0565-3*0.0045,0.269,0.02060,0.408,0.0682/2.593,0.8,6.82e-05,2.430e-03],
    [0.0054,0.0565+3*0.0045,0.343,0.02435,0.603,0.0804/2.430,1.94,8.04e-05,2.593e-03]])
                             ,columns = ["me/mu","mu/mt","s12^2","s13^2","s23^2","r","d/pi","m21^2","m3l^2"]
                             ,index = ['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])
                             #,index=['best', '1sig upper', '1sig lower', ... ])
Lexpdata_IO = pd.DataFrame(np.array([
    [0.0048,0.0565,0.304,0.02241,0.570,0.0742/-2.490,1.54,7.42e-05,-2.490e-03],
    [0.0046,0.0565-0.0045,0.292,0.02179,0.548,0.0722/-2.516,1.38,7.22e-05,-2.462e-03],
    [0.0050,0.0565+0.0045,0.317,0.02315,0.586,0.0763/-2.462,1.67,7.63e-05,-2.516e-03],
    [0.0042,0.0565-3*0.0045,0.269,0.02055,0.410,0.0682/-2.410,1.08,6.82e-05,-2.574e-03],
    [0.0054,0.0565+3*0.0045,0.343,0.02457,0.613,0.0804/-2.574,1.92,8.04e-05,-2.410e-03]])
                             ,columns = ["me/mu","mu/mt","s12^2","s13^2","s23^2","r","d/pi","m21^2","m3l^2"]
                             ,index = ['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])
                             
# Use Antusch with M_SUSY=10TeV and tan beta = 10 just like in 2103.16311 so that it matches with the quark sector instead of the standard feruglio method values                     
Lexpdata_NO['me/mu'] = Lexpdata_NO['me/mu'].replace(Lexpdata_NO['me/mu'].tolist(), 
                                                    [4.737e-03,4.737e-03+0.04e-03,4.737e-03-0.04e-03, 
                                                     4.737e-03+3*0.04e-03,4.737e-03-3*0.04e-03])
Lexpdata_NO['mu/mt'] = Lexpdata_NO['mu/mt'].replace(Lexpdata_NO['mu/mt'].tolist(), 
                                                    [0.05857,0.05857+0.00047,0.05857-0.00047, 
                                                     0.05857+3*0.00047,0.05857-3*0.00047])
                             

# 1-dimensional chi^2 projections
exps13_NO = pd.read_csv('~/v51.release-SKyes-NO1.txt', delimiter='\s+', skiprows=1524978, nrows=102, index_col=False);
exps12_NO = pd.read_csv('~/v51.release-SKyes-NO1.txt', delimiter='\s+', skiprows=1525082, nrows=133, index_col=False);
exps23_NO = pd.read_csv('~/v51.release-SKyes-NO1.txt', delimiter='\s+', skiprows=1525217, nrows=101, index_col=False);
expdcp_NO = pd.read_csv('~/v51.release-SKyes-NO1.txt', delimiter='\s+', skiprows=1525320, nrows=73, index_col=False);
expm21_NO = pd.read_csv('~/v51.release-SKyes-NO1.txt', delimiter='\s+', skiprows=1525395, nrows=312, index_col=False);
expm3l_NO = pd.read_csv('~/v51.release-SKyes-NO1.txt', delimiter='\s+', skiprows=1525709, nrows=480, index_col=False);
expdcp_NO['d/pi'] = np.mod(expdcp_NO['Delta_CP/deg']/180,2);
expdcp_NO = expdcp_NO.sort_values(by=['d/pi'])
expdcp_NO = expdcp_NO.drop_duplicates(subset=['d/pi'])
expm21_NO['m21'] = np.power(10, expm21_NO['Log10(Delta_m21^2/[eV^2])']);
expm3l_NO['m3l'] = expm3l_NO['Delta_m31^2/[1e-3_eV^2]']*1e-03

exps13_IO = pd.read_csv('~/v51.release-SKyes-IO1.txt', delimiter='\s+', skiprows=1524978, nrows=102, index_col=False);
exps12_IO = pd.read_csv('~/v51.release-SKyes-IO1.txt', delimiter='\s+', skiprows=1525082, nrows=133, index_col=False);
exps23_IO = pd.read_csv('~/v51.release-SKyes-IO1.txt', delimiter='\s+', skiprows=1525217, nrows=101, index_col=False);
expdcp_IO = pd.read_csv('~/v51.release-SKyes-IO1.txt', delimiter='\s+', skiprows=1525320, nrows=73, index_col=False);
expm21_IO = pd.read_csv('~/v51.release-SKyes-IO1.txt', delimiter='\s+', skiprows=1525395, nrows=312, index_col=False);
expm3l_IO = pd.read_csv('~/v51.release-SKyes-IO1.txt', delimiter='\s+', skiprows=1525709, nrows=480, index_col=False);
expdcp_IO['d/pi'] = np.mod(expdcp_IO['Delta_CP/deg']/180,2);
expdcp_IO = expdcp_IO.sort_values(by=['d/pi'])
expdcp_IO = expdcp_IO.drop_duplicates(subset=['d/pi'])
expm21_IO['m21'] = np.power(10, expm21_IO['Log10(Delta_m21^2/[eV^2])']);
expm3l_IO['m3l'] = expm3l_IO['Delta_m32^2/[1e-3_eV^2]']*1e-03
# Shift the Delta_chi^2 such that it reaches zero:
exps12_IO['Delta_chi^2'] = exps12_IO['Delta_chi^2'] - np.min(exps12_IO['Delta_chi^2'])
exps13_IO['Delta_chi^2'] = exps13_IO['Delta_chi^2'] - np.min(exps13_IO['Delta_chi^2'])
exps23_IO['Delta_chi^2'] = exps23_IO['Delta_chi^2'] - np.min(exps23_IO['Delta_chi^2'])
expdcp_IO['Delta_chi^2'] = expdcp_IO['Delta_chi^2'] - np.min(expdcp_IO['Delta_chi^2'])
expm21_IO['Delta_chi^2'] = expm21_IO['Delta_chi^2'] - np.min(expm21_IO['Delta_chi^2'])
expm3l_IO['Delta_chi^2'] = expm3l_IO['Delta_chi^2'] - np.min(expm3l_IO['Delta_chi^2'])



# Experimental values taken from [2003.08511]
# ordering: best, 1_sigma_min, 1_sigma_max, 3_sigma_min, 3_sigma_max  (exept for 'd/pi', where min and max was interchanged)
Lexpdata2 = pd.DataFrame(np.array([
    [0.0048,0.0565,0.305,0.0222,0.545,0.0734/2.485,1.28,7.34e-05,2.485e-03],
    [0.0046,0.0565-0.0045,0.292,0.0214,0.498,0.0720/2.514,1.10,7.20e-05,2.453e-03],
    [0.0050,0.0565+0.0045,0.319,0.0228,0.565,0.0751/2.453,1.66,7.51e-05,2.514e-03],
    [0.0042,0.0565-3*0.0045,0.265,0.0201,0.436,0.0692/2.578,0.07,6.92e-05,2.389e-03],
    [0.0054,0.0565+3*0.0045,0.347,0.0241,0.595,0.0790/2.389,0.81,7.90e-05,2.578e-03]])
                             ,columns=["me/mu","mu/mt","s12^2","s13^2","s23^2","r","d/pi","dm^2","Dm^2"]
                             ,index = ['best', '1sig_min', '1sig_max', '3sig_min', '3sig_max'])
                             #,columns=["me/mu","mu/mt","s12^2","s13^2","s23^2","r","d/pi","m21^2","m31^2"])

    
# Choose experimental data used for fitting  (I think this is obsolete and can be deleted...)
dataL_NO = np.array(Lexpdata_NO)[0][:7]
data_errorL_NO = np.abs((np.array(Lexpdata_NO)[1][:7]-np.array(Lexpdata_NO)[2][:7])/2)
dataLnu_NO = Lexpdata_NO
dataL_IO = np.array(Lexpdata_IO)[0][:7]
data_errorL_IO = np.abs((np.array(Lexpdata_IO)[1][:7]-np.array(Lexpdata_IO)[2][:7])/2)
dataLnu_IO = Lexpdata_IO

### Quarks

# Masses from [2103.16311] and angles from https://pdg.lbl.gov/2020/reviews/rpp2020-rev-ckm-matrix.pdf   
# Wolfenstein parameters are not at GUT scale, we assume their running to be negligible !!!
Qexpdata_Wolfenstein = pd.DataFrame(np.array([
    [1.9286e-03,2.8213e-03,5.0523e-02,1.8241e-02,0.22650,0.790,0.141,0.357],
    [0.6017e-03,0.1195e-03,0.6191e-02,0.1005e-02,0.00048,0.015,0.017,0.011]
]), columns=["mu/mc","mc/mt","md/ms","ms/mb","l","A","rhobar","etabar"], index=['best','1sig_range'])

# Experimental values taken from [2103.16311]
Qexpdata_Standard = pd.DataFrame(np.array([
    [1.9286e-03,2.8213e-03,5.0523e-02,1.8241e-02,13.0268,0.199962,2.30043,69.2133,np.power(np.tan(13.0268*2*np.pi/360),2)],
    [0.6017e-03,0.1195e-03,0.6191e-02,0.1005e-02,0.04183,0.007448,0.03667,3.1146,np.power(np.tan((13.0268+0.04182)*2*np.pi/360),2)-np.power(np.tan(13.0268*2*np.pi/360),2)]
]), columns=["mu/mc","mc/mt","md/ms","ms/mb","t12","t13","t23","dq","tan(t12)^2"], index=['best','1sig_range'])

# Combined Wolfenstein and Standard
Qexpdata = Qexpdata_Wolfenstein.join(Qexpdata_Standard[[x for x in Qexpdata_Standard.columns 
                                                        if x not in Qexpdata_Wolfenstein.columns]])


# Choose experimental data used for fitting
dataQ = np.array(Qexpdata)[0]
data_errorQ = np.array(Qexpdata)[1]


########### Realistic Chisq computation ##############

chisqs12Spline_NO = scipy.interpolate.make_interp_spline(exps12_NO['sin^2(theta12)'], exps12_NO['Delta_chi^2'])
chisqs13Spline_NO = scipy.interpolate.make_interp_spline(exps13_NO['sin^2(theta13)'], exps13_NO['Delta_chi^2'])
chisqs23Spline_NO = scipy.interpolate.make_interp_spline(exps23_NO['sin^2(theta23)'], exps23_NO['Delta_chi^2'])
chisqdcpSpline_NO = scipy.interpolate.make_interp_spline(expdcp_NO['d/pi'], expdcp_NO['Delta_chi^2'])
chisqm21Spline_NO = scipy.interpolate.make_interp_spline(expm21_NO['m21'], expm21_NO['Delta_chi^2'])
chisqm3lSpline_NO = scipy.interpolate.make_interp_spline(expm3l_NO['m3l'], expm3l_NO['Delta_chi^2'])

chisqs12Spline_IO = scipy.interpolate.make_interp_spline(exps12_IO['sin^2(theta12)'], exps12_IO['Delta_chi^2'])
chisqs13Spline_IO = scipy.interpolate.make_interp_spline(exps13_IO['sin^2(theta13)'], exps13_IO['Delta_chi^2'])
chisqs23Spline_IO = scipy.interpolate.make_interp_spline(exps23_IO['sin^2(theta23)'], exps23_IO['Delta_chi^2'])
chisqdcpSpline_IO = scipy.interpolate.make_interp_spline(expdcp_IO['d/pi'], expdcp_IO['Delta_chi^2'])
chisqm21Spline_IO = scipy.interpolate.make_interp_spline(expm21_IO['m21'], expm21_IO['Delta_chi^2'])
chisqm3lSpline_IO = scipy.interpolate.make_interp_spline(expm3l_IO['m3l'], expm3l_IO['Delta_chi^2'])

        
def SingleChi(obs, key='me/mu', ordering='NO', **kwargs):
    if ordering=='NO':
            Lexpdata = Lexpdata_NO
            chisqs12Spline = chisqs12Spline_NO
            chisqs13Spline = chisqs13Spline_NO
            chisqs23Spline = chisqs23Spline_NO
            chisqdcpSpline = chisqdcpSpline_NO
            chisqm21Spline = chisqm21Spline_NO
            chisqm3lSpline = chisqm3lSpline_NO
    if ordering=='IO':
            Lexpdata = Lexpdata_IO 
            chisqs12Spline = chisqs12Spline_IO
            chisqs13Spline = chisqs13Spline_IO
            chisqs23Spline = chisqs23Spline_IO
            chisqdcpSpline = chisqdcpSpline_IO
            chisqm21Spline = chisqm21Spline_IO
            chisqm3lSpline = chisqm3lSpline_IO
            
    if key in ['me/mu', 'mu/mt', 'r']:  
        return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
        
    elif key=='s12^2':
        if obs[key]<0.17 or obs[key]>0.8:
            return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
        else:
            return np.sqrt(np.abs(chisqs12Spline(obs[key])))
        
    elif key=='s13^2':
        if obs[key]>0.07:
            return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
        else:
            return np.sqrt(np.abs(chisqs13Spline(obs[key])))
        
    elif key=='s23^2':
        if obs[key]<0.25 or obs[key]>0.7:
            return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
        else:
            return np.sqrt(np.abs(chisqs23Spline(obs[key])))
        
    elif key=='d/pi':
            return np.sqrt(np.abs(chisqdcpSpline(obs[key])))
     
    elif key=='m21^2':
        if obs[key]<0.000001 or obs[key]>0.001:
            return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
        else:
            return np.sqrt(np.abs(chisqm21Spline(obs[key])))
        
    elif key=='m3l^2':
        if ordering=='NO':
            if obs[key]<0.0002 or obs[key]>0.007:
                return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
            else:
                return np.sqrt(np.abs(chisqm3lSpline(obs[key])))
        if ordering=='IO':
            if obs[key]>-0.0002 or obs[key]<-0.007:
                return np.abs(obs[key]-Lexpdata[key]['best'])/(Lexpdata[key]['1sig_max']-Lexpdata[key]['1sig_min'])*2
            else:
                return np.sqrt(np.abs(chisqm3lSpline(obs[key])))
        
    else: 
        print('I dont know an experimental value for ', key)
        
        
def ChisqL(df_row, error_type='gaussian'):
    considered_obs = ["me/mu","mu/mt","s12^2","s13^2","s23^2","d/pi","m21^2","m3l^2"]
    if error_type=='gaussian':
        if df_row['ordering']=='NO':
            Lexpdata = Lexpdata_NO
        if df_row['ordering']=='IO':
            Lexpdata = Lexpdata_IO
        return np.sum(np.power(np.array(
                            [(Lexpdata[key]['best'] - df_row[key]) / (Lexpdata[key]['1sig_max'] - Lexpdata[key]['1sig_min']) *2
                             for key in considered_obs]), 2))
    if error_type=='realistic':
        return np.sum(np.power(np.array([SingleChi(df_row, key=key, ordering=df_row['ordering']) for key in considered_obs]) ,2))
        
def ChisqQ(df_row):
    try:
        FittedObservablesQ = ['mu/mc','mc/mt','md/ms','ms/mb','l','A','rhobar','etabar']
        return np.sum(np.power(np.array([(Qexpdata[i]['best'] - df_row[i]) / Qexpdata[i]['1sig_range'] for i in FittedObservablesQ]), 2))
    except:
        FittedObservablesQ = ["mu/mc","mc/mt","md/ms","ms/mb","t12","t13","t23","dq"]
        return np.sum(np.power(np.array([(Qexpdata[i]['best'] - df_row[i]) / Qexpdata[i]['1sig_range'] for i in FittedObservablesQ]), 2))


def Chisq(df_row, error_type='gaussian', sector='lepton'):
    if sector=='lepton':
        return ChisqL(df_row, error_type=error_type)
    elif sector=='quark':
        return ChisqQ(df_row)
    elif sector in ['both', 'quark and lepton', 'lepton and quark']:
        return ChisqL(df_row, error_type=error_type) + ChisqQ(df_row)


########### Define Modular Forms and Mass Matrices ##############

def Y1(tau, modform='full', **kwargs):
    if modform=='full':
        return -3*sqrt(2)*eta(3*tau)**3/eta(tau)
    if modform=='q-exp':
        q=np.exp(2j*np.pi*tau/3)
        return -3*np.sqrt(2)*q*(1 + np.power(q, 1*3) + 2*np.power(q, 2*3) + 2*np.power(q, 4*3))
    if modform=='q-exp3':
        q=np.exp(2j*np.pi*tau/3)
        return -3*np.sqrt(2)*q*(1 + np.power(q, 1*3) + 2*np.power(q, 2*3))

def Y2(tau, modform='full', **kwargs):
    if modform=='full':
        return 3*eta(3*tau)**3/eta(tau) + eta(tau/3)**3/eta(tau)
    if modform=='q-exp':
        q=np.exp(2j*np.pi*tau)
        return 1 + 6*q + 6*np.power(q, 3) + 6*np.power(q, 4)
    if modform=='q-exp3':
        q=np.exp(2j*np.pi*tau)
        return 1 + 6*q + 6*np.power(q, 3)

# The superpotential mass matrix
def M(Y1, Y2, s1, s2, s3):
    return np.array(-Y1/np.sqrt(2)*np.array([[0,s3,s2],[s3,0,s1],[s2,s1,0]], dtype=complex)
                    +Y2*np.array([[s1,0,0],[0,s2,0],[0,0,s3]], dtype=complex), dtype=complex)
    

# The Kahlerpotential terms
# From varphi varphi_c phi phi_c
def K(s1, s2, s3):         # In the draft called A_ij(Phi)
    return np.array([[np.conj(s1)*s1, np.conj(s2)*s1, np.conj(s3)*s1],
                     [np.conj(s1)*s2, np.conj(s2)*s2, np.conj(s3)*s2], 
                     [np.conj(s1)*s3, np.conj(s2)*s3, np.conj(s3)*s3]], dtype=complex)

# From Y Y_c varphi varphi_c phi phi_c (there are 3 invariant terms, one is prop to identity, one is prop to K, and the third is: )
def K2pp(y1, y2, s1, s2, s3):             # In the draft called B_ij(Phi)
    Y11 = y1*np.conj(y1)
    Y21 = y2*np.conj(y1)
    Y12 = y1*np.conj(y2)
    Y22 = y2*np.conj(y2)
    return np.array([[Y11*s1*np.conj(s1)-2*Y22*s1*np.conj(s1),  #Y11*s1*np.conj(s1)+Y22*(-s1*np.conj(s1)+s2*np.conj(s2)+s3*np.conj(s3)),   the commented one still contains a part prop to identity
                      -Y11*s1*np.conj(s2)+np.sqrt(2)*Y12*s2*np.conj(s3)+np.sqrt(2)*Y21*s3*np.conj(s1),
                      -Y11*s1*np.conj(s3)+np.sqrt(2)*Y12*s3*np.conj(s2)+np.sqrt(2)*Y21*s2*np.conj(s1)],
                     [-Y11*s2*np.conj(s1)+np.sqrt(2)*Y12*s1*np.conj(s3)+np.sqrt(2)*Y21*s3*np.conj(s2),
                      Y11*s2*np.conj(s2)-2*Y22*s2*np.conj(s2),    #Y11*s2*np.conj(s2)+Y22*(s1*np.conj(s1)-s2*np.conj(s2)+s3*np.conj(s3)),  the commented one still contains a part prop to identity
                      -Y11*s2*np.conj(s3)+np.sqrt(2)*Y12*s3*np.conj(s1)+np.sqrt(2)*Y21*s1*np.conj(s2)],
                     [-Y11*s3*np.conj(s1)+np.sqrt(2)*Y12*s1*np.conj(s2)+np.sqrt(2)*Y21*s2*np.conj(s3),
                      -Y11*s3*np.conj(s2)+np.sqrt(2)*Y12*s2*np.conj(s1)+np.sqrt(2)*Y21*s1*np.conj(s3),
                      Y11*s3*np.conj(s3)-2*Y22*s3*np.conj(s3)]   #Y11*s3*np.conj(s3)+Y22*(s1*np.conj(s1)+s2*np.conj(s2)-s3*np.conj(s3))]     the commented one still contains a part prop to identity
                    ], dtype=complex)
                    
# Assuming that all flavons sum up to an effective flavon the full kähler metric for a field is:
def effK(lambdaK, kappa, y1, y2, s1A, s2A, s3A, s1B, s2B, s3B):
    # Kähler pot = some scale ( K^(id) + lambdaK* (1*K(...) + kappa * K2pp(...)))
    # that means: K^(id) contains the (im T)^(2/3)*Y^*YPsi^*Psi term as well as some terms of the sum of all terms prop to Phi^*PhiPsi^*Psi. Hence, lambdaK and lambdaK*kappa shouldnot be much bigger than 1
    # and: kappa = ( (im tau)^(-1/3) kappa_2^(Y Phi) ) / ( (im tau)^(-4/3)*kappa^(Phi) + (im tau)^(-1/3) |Y^(1)(T)|^2 kappa_1^(Y Phi) )  \simeq kappa_2^(Y Phi) / kappa_1^(Y Phi), since |Y|^2 \simeq 1 then roughly 1/10 < kappa < 10 as both kappa_i should be of order 1
    return  np.identity(3) + lambdaK*(K(s1A,s2A,s3A) + kappa*K2pp(y1,y2,s1B,s2B,s3B))

# The Delta(54) u T' eclectic model
def M_ecl_quark(params, KahlerCorrection=False, LooseKahler=False, DiagMethod='eigh', withPhases=True, LKdown=False, PhiAisPhiB=False, RadialKahler=False, **kwargs):
    tau = params['Retau']+1j*params['Imtau']
    y1 = Y1(tau, **kwargs)
    y2 = Y2(tau, **kwargs)
    if withPhases==True:
        s1u = params['s1u']*np.exp(1j*params['p1u'])
        s2u = params['s2u']*np.exp(1j*params['p2u'])
        s3u = 1.
        s1d = params['s1d']*np.exp(1j*params['p1d'])
        s2d = params['s2d']*np.exp(1j*params['p2d'])
        s3d = 1.
    else:
        s1u = params['s1u'] + 0.0j
        s2u = params['s2u'] + 0.0j
        s3u = 1.
        s1d = params['s1d'] + 0.0j
        s2d = params['s2d'] + 0.0j
        s3d = 1.
    
    Mu = M(y1, y2, s1u, s2u, s3u,)
    Md = M(y1, y2, s1d, s2d, s3d,)
    
    if KahlerCorrection==True:
        if LooseKahler==False:
            s1n = params['s1n']*np.exp(1j*params['p1n'])
            s2n = params['s2n']*np.exp(1j*params['p2n'])
            s3n = 1.
        
            Ku = (np.power(np.imag(tau), -1/3) * (params['buu2pp']*K2pp(y1, y2, s1u, s2u, s3u) +   
                                                  params['bdu2pp']*K2pp(y1, y2, s1d, s2d, s3d) +    
                                                  params['bnu2pp']*K2pp(y1, y2, s1n, s2n, s3n)) +
                  np.power(np.imag(tau), -4/3) * (params['buu']*K(s1u, s2u, s3u) +                    
                                                  params['bdu']*K(s1d, s2d, s3d) + 
                                                  params['bnu']*K(s1n, s2n, s3n)) +
                  np.power(np.imag(tau), -2/3) * np.identity(3))                     
            Kd = (np.power(np.imag(tau), -1/3) * (params['bud2pp']*K2pp(y1, y2, s1u, s2u, s3u) + 
                                                  params['bdd2pp']*K2pp(y1, y2, s1d, s2d, s3d) + 
                                                  params['bnd2pp']*K2pp(y1, y2, s1n, s2n, s3n)) +
                  np.power(np.imag(tau), -4/3) * (params['bud']*K(s1u, s2u, s3u) + 
                                                  params['bdd']*K(s1d, s2d, s3d) + 
                                                  params['bnd']*K(s1n, s2n, s3n)) +
                  np.power(np.imag(tau), -2/3) * np.identity(3))
            Kq = (np.power(np.imag(tau), -1/3) * (params['buq2pp']*K2pp(y1, y2, s1u, s2u, s3u) + 
                                                  params['bdq2pp']*K2pp(y1, y2, s1d, s2d, s3d) + 
                                                  params['bnq2pp']*K2pp(y1, y2, s1n, s2n, s3n)) +
                  np.power(np.imag(tau), -4/3) * (params['buq']*K(s1u, s2u, s3u) + 
                                                  params['bdq']*K(s1d, s2d, s3d) + 
                                                  params['bnq']*K(s1n, s2n, s3n)) +
                  np.power(np.imag(tau), -2/3) * np.identity(3))
              
        if LooseKahler==True:
            s1k = params['s1k']*np.exp(1j*params['p1k'])
            s2k = params['s2k']*np.exp(1j*params['p2k'])
            s3k = params['s3k']*np.exp(1j*params['p3k'])
            Ku = params['bk']*K(s1k, s2k , s3k) + np.identity(3)
            Kd = params['bk']*K(s1k, s2k , s3k) + np.identity(3)
            Kq = params['bk']*K(s1k, s2k , s3k) + np.identity(3)
            
        if LooseKahler=='down':
            s1k = params['s1k']
            s2k = params['s2k']
            s3k = params['s3k']
            s1kd = params['s1kd']
            s2kd = params['s2kd']
            s3kd = params['s3kd']
            Ku = params['bk']*K(s1k, s2k , s3k) + np.identity(3)
            Kd = params['bkd']*K(s1kd, s2kd , s3kd) + np.identity(3)
            Kq = params['bk']*K(s1k, s2k , s3k) + np.identity(3)
            
        if LooseKahler=='new':
            s1kuA = params['s1kqA']
            s2kuA = params['s2kqA']
            s3kuA = 1. 
            s1kuB = params['s1kqB']
            s2kuB = params['s2kqB']
            s3kuB = 1. 
            s1kdA = params['s1kqA']
            s2kdA = params['s2kqA']
            s3kdA = 1. 
            s1kdB = params['s1kqB']
            s2kdB = params['s2kqB']
            s3kdB = 1. 
            s1kqA = params['s1kqA']
            s2kqA = params['s2kqA']
            s3kqA = 1. 
            s1kqB = params['s1kqB']
            s2kqB = params['s2kqB']
            s3kqB = 1. 
            lambdaKu = params['lambdaKq']
            lambdaKd = params['lambdaKq']
            lambdaKq = params['lambdaKq']
            kappau = params['kappaq']
            kappad = params['kappaq']
            kappaq = params['kappaq']
            
            if PhiAisPhiB==True:
                s1kuB, s2kuB, s1kdB, s2kdB, s1kqB, s2kqB = s1kuA, s2kuA, s1kdA, s2kdA, s1kqA, s2kqB
                kappau, kappad, kappaq = 1., 1., 1.
            
            if LKdown==True:
                s1kuA, s2kuA, s3kuA = params['s1kqA'], params['s2kqA'], 1.0
                s1kuB, s2kuB, s3kuB = s1kuA, s2kuA, s3kuA
                s1k1A, s2kqA, s3kqA = s1kuA, s2kuA, s3kuA
                s1k1B, s2kqB, s3kqB = s1kuA, s2kuA, s3kuA
                s1kdA, s2kdA, s3kdA = params['s1kqB'], params['s2kqB'], 1.0
                s1kdB, s2kdB, s3kdB = s1kdA, s2kdA, s3kdA
                lambdaKu = params['lambdaKq']
                lambdaKq = params['lambdaKq']
                lambdaKd = params['kappaq']
                kappau, kappaq, kappad = 1., 1., 1.
                           
            Ku = effK(lambdaKu, kappau, y1, y2, s1kuA, s2kuA, s3kuA, s1kuB, s2kuB, s3kuB)
            Kd = effK(lambdaKd, kappad, y1, y2, s1kdA, s2kdA, s3kdA, s1kdB, s2kdB, s3kdB)
            Kq = effK(lambdaKq, kappaq, y1, y2, s1kqA, s2kqA, s3kqA, s1kqB, s2kqB, s3kqB)
            
            
        if LooseKahler=='all36':
            s3k=0.1
            Ku = effK(params['lambdaKu'], params['kappau'], y1, y2, params['s1kuA'], params['s2kuA'], s3k, params['s1kuB'], params['s2kuB'], s3k)
            Kd = effK(params['lambdaKd'], params['kappad'], y1, y2, params['s1kdA'], params['s2kdA'], s3k, params['s1kdB'], params['s2kdB'], s3k)
            Kq = effK(params['lambdaKq'], params['kappaq'], y1, y2, params['s1kqA'], params['s2kqA'], s3k, params['s1kqB'], params['s2kqB'], s3k)
            
        if LooseKahler=='newchoice':
            s3k=1.
            Kq = effK(params['lambdaKq'], params['kappaq'], y1, y2, params['s1kq'], params['s2kq'], s3k, params['s1kq'], params['s2kq'], s3k)
            Ku = Kq
            Kd = effK(params['lambdaKd'], params['kappad'], y1, y2, params['s1kd'], params['s2kd'], s3k, params['s1kd'], params['s2kd'], s3k)
            
        if LooseKahler=='June':
            if RadialKahler==False:
                s3k=1.
                Kq = effK(params['lambdaKq'], params['kappaq'], y1, y2, params['s1kq'], params['s2kq'], s3k, params['s1kq'], params['s2kq'], s3k)
                Ku = effK(params['lambdaKu'], params['kappau'], y1, y2, params['s1ku'], params['s2ku'], s3k, params['s1ku'], params['s2ku'], s3k)
                Kd = effK(params['lambdaKd'], params['kappad'], y1, y2, params['s1kd'], params['s2kd'], s3k, params['s1kd'], params['s2kd'], s3k)
            if RadialKahler==True:
                s1kq = np.sin(params['phiq'])*np.cos(params['thetaq'])
                s2kq = np.sin(params['phiq'])*np.sin(params['thetaq'])
                s3kq = np.cos(params['thetaq'])
                s1ku = np.sin(params['phiu'])*np.cos(params['thetau'])
                s2ku = np.sin(params['phiu'])*np.sin(params['thetau'])
                s3ku = np.cos(params['thetau'])
                s1kd = np.sin(params['phid'])*np.cos(params['thetad'])
                s2kd = np.sin(params['phid'])*np.sin(params['thetad'])
                s3kd = np.cos(params['thetad'])
                Kq = effK(params['lambdaKq'], params['kappaq'], y1, y2, s1kq, s2kq, s3kq, s1kq, s2kq, s3kq)
                Ku = effK(params['lambdaKu'], params['kappau'], y1, y2, s1ku, s2ku, s3ku, s1ku, s2ku, s3ku)
                Kd = effK(params['lambdaKd'], params['kappad'], y1, y2, s1kd, s2kd, s3kd, s1kd, s2kd, s3kd)
            
                
        
        # Diagonalize the Kahler terms
        if DiagMethod=='eigh':
            Du, Uu = np.linalg.eigh(Ku)  # Uu * Du * Du^dagger = Ku
            Dd, Ud = np.linalg.eigh(Kd)
            Dq, Uq = np.linalg.eigh(Kq)
        if DiagMethod=='svd':
            Uu, Du, Uur = np.linalg.svd(Ku)
            Ud, Dd, Udr = np.linalg.svd(Kd)
            Uq, Dq, Uqr = np.linalg.svd(Kq)
        
        # Match notation of 1909.06910:  (still a discrepancy since U_i here is U_i^dagger in the paper)
        Du = np.sqrt(Du)
        Dd = np.sqrt(Dd)
        Dq = np.sqrt(Dq)
        
        # 
        Mu = np.dot(np.dot(np.dot(1/Dq*np.identity(3), Uq.T), Mu), Uu*1/Du)
        Md = np.dot(np.dot(np.dot(1/Dq*np.identity(3), Uq.T), Md), Ud*1/Dd)
    
    return {'Mu':Mu, 'Md':Md}

def M_ecl_lepton(params, withPhases=True, KahlerCorrection=False, LooseKahler=False, DiagMethod='eigh', PhiAisPhiB=False, **kwargs):
    tau = params['Retau']+1j*params['Imtau']
    y1 = Y1(tau, **kwargs)
    y2 = Y2(tau, **kwargs)
    if withPhases==True:
        s1e = params['s1d']*np.exp(1j*params['p1d'])
        s2e = params['s2d']*np.exp(1j*params['p2d'])
        s3e = 1.
        s1D = params['s1n']*np.exp(1j*params['p1n'])
        s2D = params['s2n']*np.exp(1j*params['p2n'])
        s3D = params['s3n']*np.exp(1j*params['p3n'])
    else:
        s1e = params['s1d'] + 0.0j
        s2e = params['s2d'] + 0.0j
        s3e = 1.
        s1D = params['s1n'] + 0.0j
        s2D = params['s2n'] + 0.0j
        s3D = params['s3n'] + 0.0j
    s1R = s1e                  # This is because the model assumes that \varphi_e and \varphi_R are the same flavon.
    s2R = s2e                  # This is because the model assumes that \varphi_e and \varphi_R are the same flavon.
    s3R = s3e                  # This is because the model assumes that \varphi_e and \varphi_R are the same flavon.
    
    
    Me = M(y1, y2, s1e, s2e, s3e,)
    MD = M(y1, y2, s1D, s2D, s3D,)
    MR = 2*M(y1, y2, s1R, s2R, s3R,)
    Mn = -1*np.dot(MD, np.dot(np.linalg.inv(MR), np.transpose(MD)))
    
    if KahlerCorrection==True:
        if LooseKahler==False:
            print('Sorry, I only do LooseKahler=\'new\' Kahler Corrections for the lepton sector!')
        if LooseKahler=='new':
            s1keA = params['s1klA']
            s2keA = params['s2klA']
            s3keA = 1.
            s1keB = params['s1klB']
            s2keB = params['s2klB']
            s3keB = 1. 
            s1klA = params['s1klA']
            s2klA = params['s2klA']
            s3klA = 1.
            s1klB = params['s1klB']
            s2klB = params['s2klB']
            s3klB = 1. 
            s1knA = params['s1klA']
            s2knA = params['s2klA']
            s3knA = 1.
            s1knB = params['s1klB']
            s2knB = params['s2klB']
            s3knB = 1.
            lambdaKe = params['lambdaKl']
            lambdaKl = params['lambdaKl']
            lambdaKn = params['lambdaKl']
            kappae = params['kappal']
            kappal = params['kappal']
            kappan = params['kappal']
            
            if PhiAisPhiB==True:
                s1keB, s2keB, s1klB, s2klB, s1knB, s2knB = s1keA, s2keA, s1klA, s2klA, s1knA, s2knB
                kappae, kappal, kappan = 1., 1., 1.
            
            Ke = effK(lambdaKe, kappae, y1, y2, s1keA, s2keA, s3keA, s1keB, s2keB, s3keB)
            Kl = effK(lambdaKl, kappal, y1, y2, s1klA, s2klA, s3klA, s1klB, s2klB, s3klB)
            Kn = effK(lambdaKn, kappan, y1, y2, s1knA, s2knA, s3knA, s1knB, s2knB, s3knB)
            
        if LooseKahler=='newchoice':
            s3k=1.
            Kl = effK(params['lambdaKq'], params['kappaq'], y1, y2, params['s1kq'], params['s2kq'], s3k, params['s1kq'], params['s2kq'], s3k)
            Kn = Kl
            Ke = effK(params['lambdaKe'], params['kappae'], y1, y2, params['s1ke'], params['s2ke'], s3k, params['s1ke'], params['s2ke'], s3k)
            
        if LooseKahler=='June':  #should not be necessary, but just to be sure
            Kl = np.identity(3)
            Kn = np.identity(3)
            Ke = np.identity(3)
           
        
        # Diagonalize the Kahler terms
        if DiagMethod=='eigh':
            De, Ue = np.linalg.eigh(Ke)  # Uu * Du * Du^dagger = Ku
            Dl, Ul = np.linalg.eigh(Kl)
            Dn, Un = np.linalg.eigh(Kn)
        if DiagMethod=='svd':
            Ue, De, Uer = np.linalg.svd(Ke)
            Ul, Dl, Ulr = np.linalg.svd(Kl)
            Un, Dn, Unr = np.linalg.svd(Kn)
        
        # Match notation of 1909.06910:  (still a discrepancy since U_i here is U_i^dagger in the paper)
        De = np.sqrt(De)
        Dl = np.sqrt(Dl)
        Dn = np.sqrt(Dn)
        
        # 
        Me = np.dot(np.dot(np.dot(1/Dl*np.identity(3), Ul.T), Me), Ue*1/De)
        MD = np.dot(np.dot(np.dot(1/Dl*np.identity(3), Ul.T), MD), Un*1/Dn)
        MR = np.dot(np.dot(np.dot(1/Dn*np.identity(3), Un.T), MR), Un*1/Dn)
        Mn = -1*np.dot(MD, np.dot(np.linalg.inv(MR), np.transpose(MD)))
    
    return {'Me':Me, 'Mn':Mn}

########### The Feruglio Model ##############

def Y1_f(tau, **kwargs):
    q=np.exp(2j*np.pi*tau/3)
    return 1 + 12*np.power(q, 1*3) + 36*np.power(q, 2*3) + 12*np.power(q, 3*3)
def Y2_f(tau, **kwargs):
    q=np.exp(2j*np.pi*tau/3)
    return -6*q*(1 + 7*np.power(q,1*3) + 8*np.power(q, 2*3))
def Y3_f(tau, **kwargs):
    q=np.exp(2j*np.pi*tau/3)
    return -18*np.power(q, 2)*(1 + 2*np.power(q, 1*3) + 5*np.power(q, 2*3))

def K1(params, **kwargs):
    Y1 = Y1_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y2 = Y2_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y3 = Y3_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y1c = np.conj(Y1)
    Y2c = np.conj(Y2)
    Y3c = np.conj(Y3)
    return 1/3*np.array([[4*Y1c*Y1 + Y2c*Y2 + Y3c*Y3, -2*Y1c*Y3 + Y2c*Y1 -2*Y3c*Y2, -2*Y1c*Y2 -2*Y2c*Y3 + Y3c*Y1], 
                         [-2*Y3c*Y1 + Y1c*Y2 -2*Y2c*Y3, 4*Y2c*Y2 + Y1c*Y1 + Y3c*Y3, Y3c*Y2 -2*Y1c*Y3 -2*Y2c*Y1], 
                         [-2*Y2c*Y1 -2*Y3c*Y2 + Y1c*Y3, Y2c*Y3 -2*Y3c*Y1 -2*Y1c*Y2, 4*Y3c*Y3 + Y1c*Y1 + Y2c*Y2]],
                    dtype=complex)

def K2(params, **kwargs):
    Y1 = Y1_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y2 = Y2_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y3 = Y3_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y1c = np.conj(Y1)
    Y2c = np.conj(Y2)
    Y3c = np.conj(Y3)
    return np.array([[Y2c*Y2 + Y3c*Y3, -Y2c*Y1, -Y3c*Y1], 
                     [-Y1c*Y2, Y3c*Y3 + Y1c*Y1, -Y3c*Y2], 
                     [-Y1c*Y3, -Y2c*Y3, Y2c*Y2 + Y1c*Y1]], dtype=complex)

def K3(params, **kwargs):
    Y1 = Y1_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y2 = Y2_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y3 = Y3_f(params['Retau']+1j*params['Imtau'], **kwargs)
    Y1c = np.conj(Y1)
    Y2c = np.conj(Y2)
    Y3c = np.conj(Y3)
    K3 = np.array([[Y2c*Y2 - Y3c*Y3, -2*Y1c*Y3 - Y2c*Y1, 2*Y1c*Y2 + Y3c*Y1], 
                   [Y1c*Y2 + 2*Y2c*Y3, Y3c*Y3 - Y1c*Y1, -Y3c*Y2 -2*Y2c*Y1], 
                   [-2*Y3c*Y2 - Y1c*Y3, Y2c*Y3 + 2*Y3c*Y1, -Y2c*Y2 + Y1c*Y1]], dtype=complex)
    return 1/np.sqrt(3)*(K3 + np.conj(K3).T)


def M_fer_lepton(params, KahlerCorrection=False,
                 KahlerReal=False, DiagMethod='eigh', leftright='left', permutation=False,
                 **kwargs):
    tau = params['Retau']+1j*params['Imtau']
    gamma = 1.
    beta = Lexpdata_NO['mu/mt']['best'] * gamma
    alpha = Lexpdata_NO['me/mu']['best'] * beta
    
    Me = np.diag([alpha, gamma, beta])  # Feruglio says he interchanged tau and mu mass
    Mn = np.array([[2*Y1_f(tau, **kwargs), -1*Y3_f(tau, **kwargs), -1*Y2_f(tau, **kwargs)],
                   [-1*Y3_f(tau, **kwargs), 2*Y2_f(tau, **kwargs), -1*Y1_f(tau, **kwargs)],
                   [-1*Y2_f(tau, **kwargs), -1*Y1_f(tau, **kwargs), 2*Y3_f(tau, **kwargs)]], dtype=complex)
    
    if KahlerCorrection==True:
        # Diagonalize the Kahler potential
        Kl = params['alpha1']*K1(params, **kwargs) + params['alpha2']*K2(params, **kwargs) + params['alpha3']*K3(params, **kwargs) + np.identity(3)
        
        if KahlerReal==True:
            Kl = np.real(Kl)
        
        if DiagMethod=='eigh':
            Dl, Ul = np.linalg.eigh(Kl)  # Ul * Dl * Dl^dagger = Kl
        if DiagMethod=='svd':
            Ul, Dl, Ulr = np.linalg.svd(Kl)
        
        # Match notation of 1909.06910:
        Dl = np.sqrt(Dl)
        #Ul = np.conj(Ul).T
        
        # Correct the mass matrices
        #Me = np.dot(1/Dl*np.conj(Ul), Me)
        #Mn = np.dot(np.dot(1/Dl*np.conj(Ul), Mn), np.conj(Ul).T*1/Dl)
        if leftright=='left':
            Me = np.dot(np.dot(1/Dl*np.identity(3), Ul.T), Me)
        if leftright=='right':
            Me = np.dot(Me, Ul*1/Dl)
            
        if permutation==True:
            Me = np.dot(Me, [[0,0,1],[0,1,0],[1,0,0]])
        
        Mn = np.dot(np.dot(np.dot(1/Dl*np.identity(3), Ul.T), Mn), Ul*1/Dl)

        #print('Sorry, Kahler Corrections are yet to come!')
     
    # This should be the correct scaling factor \Lambda
    Mn = 1/(3*np.sqrt(2)) * 0.05071/0.5360019372564414 * Mn
        
    return {'Me':Me, 'Mn':Mn}


    


########### Generate CKM Matrix ##############

def FetchWolf(CKM):
    l = np.abs(CKM[0,1])/np.sqrt(np.power(np.abs(CKM[0,0]),2) + np.power(np.abs(CKM[0,1]),2))
    A = 1/l*np.abs(CKM[1,2]/CKM[0,1])
    rhobar = np.real(-CKM[0,0]*np.conj(CKM[0,2])/CKM[1,0]/np.conj(CKM[1,2]))
    etabar = np.imag(-CKM[0,0]*np.conj(CKM[0,2])/CKM[1,0]/np.conj(CKM[1,2]))
    
    return {'l':l, 'A':A, 'rhobar':rhobar, 'etabar':etabar}

def FetchStandard(CKM):
    t13 = np.arcsin(np.abs(CKM[0,2]))
    t12 = np.arctan(np.abs(CKM[0,1])/np.abs(CKM[0,0]))
    t23 = np.arctan(np.abs(CKM[1,2])/np.abs(CKM[2,2]))
    d = np.mod(-1*np.angle(
        ((np.conjugate(CKM[0,0])*CKM[0,2]*CKM[2,0]*np.conjugate(CKM[2,2]))/
         (np.cos(t12)*np.power(np.cos(t13),2)*np.cos(t23)*np.sin(t13)) + np.cos(t12)*np.cos(t23)*np.sin(t13))/
        (np.sin(t12)*np.sin(t23)))/np.pi, 2)*180
    return {'t12':t12*180/np.pi, 't13':t13*180/np.pi, 't23':t23*180/np.pi, 'dq':d}


def CKMobs(Mu=np.identity(3), Md=np.identity(3), CKM_par='Wolfenstein', **kwargs):#, leftright='left'):
    
    # Singular Value decomposition for digonalization
    tVul, mu, tVurh = np.linalg.svd(Mu)    # Mu = tVul * mu * tVurh
    tVdl, md, tVdrh = np.linalg.svd(Md)
    
    # correct the fact that svd sorts its singular values descending
    permut = np.array([[0,0,1],[0,1,0],[1,0,0]]) 
    mu = np.dot( permut, mu)
    md = np.dot( permut, md)  
    Vul = np.dot( permut, np.linalg.inv(tVul))
    Vdl = np.dot( permut, np.linalg.inv(tVdl))
    
    # construct CKM Matrix and fetch parameters of it
    CKM = np.dot( Vul, Vdl.conj().T )
    
    if CKM_par=='Wolfenstein':
        CKM_p = FetchWolf(CKM)
    if CKM_par=='Standard':
        CKM_p = FetchStandard(CKM)
            
    obs = {'mu/mc':mu[0]/mu[1], 'mc/mt':mu[1]/mu[2], 'md/ms':md[0]/md[1], 'ms/mb':md[1]/md[2], **CKM_p}
    
    return obs
    
    
def CKMresidual(params, CKM_par='Wolfenstein',
                FittedObservablesQ=['mu/mc','mc/mt','md/ms','ms/mb','l','A','rhobar','etabar'],
                WeightFactorForChiQ=None, **kwargs):
    
    if WeightFactorForChiQ==None:
        WeightFactorForChiQ = np.array([1 for i in FittedObservablesQ])
    if CKM_par=='Wolfenstein':
        expdata=Qexpdata_Wolfenstein
    if CKM_par=='Standard':
        expdata=Qexpdata_Standard
        
    try:   # if calculating the CKM matrix fails, because SVD didn't converge or something, the minimizer is "punished" with a high chisq, e.g. 10^(15)
        MassMatrices = M_ecl_quark(params, **kwargs)
        obs = CKMobs(Mu=MassMatrices['Mu'], Md=MassMatrices['Md'], CKM_par=CKM_par, **kwargs) 
        #use np.sum to get a single chi value
        return WeightFactorForChiQ * np.array([(expdata[i]['best'] - obs[i]) / expdata[i]['1sig_range'] 
                                               for i in FittedObservablesQ])
    except:
        return np.full(8, 1e40) 
        
        
def MyResidual(params, preWeightFactorForChiQ=[1,1,1,1,1/10,1/100,1/100,1/5], ConKSinKratio=(1,3), **kwargs):
	ChiConK = CKMresidual(params, **kwargs)
	kwargs['WeightFactorForChiQ'] = preWeightFactorForChiQ
	kwargs['KahlerCorrection']=False
	ChiSinK = CKMresidual(params, **kwargs)
	return np.concatenate((ConKSinKratio[0]*ChiConK, ConKSinKratio[1]*ChiSinK))
	


########### Generate PMNS Matrix ##############

### Get the parametrization parameters of the PMNS matrix
def FetchParametrization(PMNS):
    t13 = np.arcsin(np.abs(PMNS[0,2]))
    t12 = np.arctan(np.abs(PMNS[0,1])/np.abs(PMNS[0,0]))
    t23 = np.arctan(np.abs(PMNS[1,2])/np.abs(PMNS[2,2]))
    d = np.mod(-1*np.angle(
        ((np.conjugate(PMNS[0,0])*PMNS[0,2]*PMNS[2,0]*np.conjugate(PMNS[2,2]))/
         (np.cos(t12)*np.power(np.cos(t13),2)*np.cos(t23)*np.sin(t13)) + np.cos(t12)*np.cos(t23)*np.sin(t13))/
        (np.sin(t12)*np.sin(t23)))/np.pi, 2)
    eta1 = np.mod(np.angle(PMNS[0,0]/PMNS[0,2])/np.pi - d, 2)
    eta2 = np.mod(np.angle(PMNS[0,1]/PMNS[0,2])/np.pi - d, 2)
    return [t12, t13, t23, d, eta1, eta2]

### Determine the dimensionless observables
def PMNSobs(Me=np.identity(3), Mn=np.identity(3), ordering='NO', **kwargs):  
    
    # Singular Value decomposition for digonalization
    tVel, me, tVerh = np.linalg.svd(Me)  #tVel * diag(me) * tVerh = Me
    tVnl, mn, tVnrh = np.linalg.svd(Mn)  #tVnl * diag(mn) * tVnrh = Mn
    
    # correct the fact that svd sorts its singular values (=physical masses) in descending order
    permutE = np.array([[0,0,1],[0,1,0],[1,0,0]])
    if ordering=='NO':
        permutN = np.array([[0,0,1],[0,1,0],[1,0,0]])
    if ordering=='IO':
        permutN = np.array([[0,1,0],[1,0,0],[0,0,1]])
    me = np.dot( permutE, me)
    mn = np.dot( permutN, mn)  
    Vel = np.dot( permutE, np.linalg.inv(tVel))
    Vnl = np.dot( permutN, np.linalg.inv(tVnl))
    
    # construct PMNS Matrix and fetch its parametrization
    PMNS = np.dot( Vel.conj(), Vnl.T)  #PMNS = np.dot( Vel.conj().T, Vnl)
    t12, t13, t23, d, eta1, eta2 = FetchParametrization(PMNS)
    m21sq = np.power(mn[1], 2) - np.power(mn[0], 2)
    if ordering=='NO':
        m3lsq = np.power(mn[2], 2) - np.power(mn[0], 2)
    if ordering=='IO':
        m3lsq = np.power(mn[2], 2) - np.power(mn[1], 2)
    
    # observables ordered as ["me/mu","mu/mt","t12","t13","t23","r","d/pi", "m1","m2","m3","eta1","eta2"]
    #obs = np.array([me[0]/me[1], me[1]/me[2], np.sin(t12)**2, np.sin(t13)**2, np.sin(t23)**2, m21sq/m3lsq, d,
    #                mn[0], mn[1], mn[2], eta1, eta2])
    obs = {'me/mu':me[0]/me[1], 'mu/mt':me[1]/me[2], 
           's12^2':np.sin(t12)**2, 's13^2':np.sin(t13)**2, 's23^2':np.sin(t23)**2,
           'r':m21sq/m3lsq, 'd/pi':d,
           'm1':mn[0], 'm2':mn[1], 'm3':mn[2], 'eta1':eta1, 'eta2':eta2}
            # Carefull!! the neutrino-masses mn[i] are not yet scaled correctly!
    
    return obs


### Determine the remaining observables of the Lepton-sector and return all observables
# Change 'M_ecl_lepton' if you do another model that the eclectic D54 u  T' model
def AllLeptonObs(params, model='eclectic', **kwargs): # 'params' should be a dictionary or a pandas.DataFrame but with only one row
    if model=='eclectic':
        MassMatrices = M_ecl_lepton(params, KahlerCorrection=params['KahlerCorrection'], **kwargs)
    if model=='feruglio':
        MassMatrices = M_fer_lepton(params, KahlerCorrection=params['KahlerCorrection'], **kwargs)
    DimLessObs = PMNSobs(Me=MassMatrices['Me'], Mn=MassMatrices['Mn'], ordering=params['ordering'], KahlerCorrection=params['KahlerCorrection'], **kwargs)
    s12sq = DimLessObs['s12^2']
    s13sq = DimLessObs['s13^2']
    s23sq = DimLessObs['s23^2']
    c12sq = np.cos(np.arcsin(np.sqrt(s12sq)))**2
    c13sq = np.cos(np.arcsin(np.sqrt(s13sq)))**2
    c23sq = np.cos(np.arcsin(np.sqrt(s23sq)))**2
    eta1 = DimLessObs['eta1']
    eta2 = DimLessObs['eta2']
    d = DimLessObs['d/pi']
    ordering = params['ordering']
    
    Jmax = np.sqrt(c12sq*s12sq*c23sq*s23sq*s13sq)*c13sq
    J = Jmax*np.sin(d*np.pi)
    
    mn = np.array([DimLessObs['m1'], DimLessObs['m2'], DimLessObs['m3']])
    m21sq = np.power(mn[1], 2) - np.power(mn[0], 2)
    if ordering=='NO':
        m3lsq = np.power(mn[2], 2) - np.power(mn[0], 2)
    if ordering=='IO':
        m3lsq = np.power(mn[2], 2) - np.power(mn[1], 2)
    
    if ordering=='NO':
        Lexpdata=Lexpdata_NO
    if ordering=='IO':
        Lexpdata=Lexpdata_IO
    
    nscale = np.sqrt((Lexpdata['m21^2']['best'] / m21sq + Lexpdata['m3l^2']['best'] / m3lsq )/2)
    
    mn = nscale*mn
    m21sq = nscale*nscale*m21sq
    m3lsq = nscale*nscale*m3lsq
    
    if ordering=='NO':
        mbeta = np.sqrt(mn[0]**2 + m21sq*(1-c13sq*c12sq) + m3lsq*s13sq)
        mbetabeta = np.abs(mn[0]*c12sq*c13sq + 
                           np.sqrt(m21sq+mn[0]**2)*s12sq*c13sq*np.exp(2j*np.pi*(eta2-eta1)) + 
                           np.sqrt(m3lsq+m21sq+mn[0]**2)*s13sq*np.exp(-2j*np.pi*(d+eta1)))
    if ordering=='IO':
        mbeta = np.sqrt(mn[2]**2 + m21sq*c13sq*c12sq - m3lsq*c13sq)
        mbetabeta = np.abs(mn[2]*s13sq + 
                           np.sqrt(mn[2]**2-m3lsq)*s12sq*c13sq*np.exp(2j*np.pi*(eta2+d)) + 
                           np.sqrt(mn[2]**2-m3lsq-m21sq)*c12sq*c13sq*np.exp(2j*np.pi*(eta1+d)))
    
    
    obs = {'me/mu':DimLessObs['me/mu'], 'mu/mt':DimLessObs['mu/mt'], 
           's12^2':DimLessObs['s12^2'], 's13^2':DimLessObs['s13^2'], 's23^2':DimLessObs['s23^2'],
           'd/pi':DimLessObs['d/pi'], 'r':DimLessObs['r'], 'm21^2':m21sq, 'm3l^2':m3lsq,
           'm1':mn[0], 'm2':mn[1], 'm3':mn[2], 'eta1':eta1, 'eta2':eta2,
           'J':J, 'Jmax':Jmax, 'Sum(m_i)':np.sum(mn), 'm_b':mbeta, 'm_bb':mbetabeta, 'nscale':nscale}
     
    return obs


### This function is used in the optimizer
def PMNSresidual(params, ordering='NO', error_type='gaussian', model='eclectic',
                 FittedObservablesL=['me/mu','mu/mt','s12^2','s13^2','s23^2','r','d/pi'], **kwargs):
    try:  # if calculating the PMNS matrix fails, because SVD didn't converge or something, the minimizer is "punished" with a high chisq, e.g. 10^(15)
        if model=='eclectic':
            MassMatrices = M_ecl_lepton(params, **kwargs)
        if model=='feruglio':
            MassMatrices = M_fer_lepton(params, **kwargs)
        obs = PMNSobs(Me=MassMatrices['Me'], Mn=MassMatrices['Mn'], ordering=ordering, **kwargs)
        
        if error_type=='gaussian':
            if ordering=='NO':
                Lexpdata = Lexpdata_NO
            if ordering=='IO':
                Lexpdata = Lexpdata_IO
            return np.abs(np.array([(Lexpdata[key]['best'] - obs[key]) / 
                                    (Lexpdata[key]['1sig_max'] - Lexpdata[key]['1sig_min']) *2 
                                    for key in FittedObservablesL]))
            #return (data - obs) / data_error
            #use np.sum( ) to give the optimizer a single chi-value instead of a list of them
        if error_type=='realistic':
            return np.abs(np.array([SingleChi(obs, key=key, ordering=ordering, **kwargs) 
                                    for key in FittedObservablesL]))
    except:
        return np.full(len(FittedObservablesL), 1e50) 
    #I have no idea why, but the scipy-optimizer wants a list of chi-values not chi^2-values.
    
    
    
##### This funtion is used for the optimizer to fit quarks and lepton simultaneously #####

def weight(chi):
    if abs(chi) > 5:
        chi = 2*chi
    if abs(chi) > 3:
        chi = 2*chi
    return chi

def CKMandPMNSresidual(params, QuarkLeptonChiRatio=(1,1), LKahlerCorrection=False, weight35region=False, CheckKahler=False, lambdamax=1, RadialKahler=False, **kwargs):
    kwargs['RadialKahler']=RadialKahler
    ChiQuark = CKMresidual(params, **kwargs)
    kwargs['KahlerCorrection']=LKahlerCorrection
    ChiLepton = PMNSresidual(params, **kwargs)

    if CheckKahler==True:
        if RadialKahler==False:
            CorQ = np.abs(params['lambdaKq']*(np.power(params['s1kq'], 2) + np.power(params['s2kq'], 2) + 1.))
            CorD = np.abs(params['lambdaKd']*(np.power(params['s1kd'], 2) + np.power(params['s2kd'], 2) + 1.))
            try:
                CorE = np.abs(params['lambdaKe']*(np.power(params['s1ke'], 2) + np.power(params['s2ke'], 2) + 1.))
            except:
                pass
            try:
                CorU = np.abs(params['lambdaKu']*(np.power(params['s1ku'], 2) + np.power(params['s2ku'], 2) + 1.))
            except:
                pass

        if RadialKahler==True:
            CorQ = np.abs(params['lambdaKq'])
            CorD = np.abs(params['lambdaKd'])
            try:
                CorE = np.abs(params['lambdaKe'])
            except:
                pass
            try:
                CorU = np.abs(params['lambdaKu'])
            except:
                pass

        if  CorQ > lambdamax or CorQ*params['kappaq'] > lambdamax:
            ChiQuark = 1e10*CorQ*ChiQuark
            #ChiLepton = 1e10*CorQ*ChiLepton
        if  CorD > lambdamax or CorD*params['kappad'] > lambdamax:
            ChiQuark = 1e10*CorD*ChiQuark
        try:
            if  CorE > lambdamax or CorE*params['kappae'] > lambdamax:
                ChiLepton = 1e10*CorE*ChiLepton
        except:
            pass
        try:
            if  CorU > lambdamax or CorU*params['kappau'] > lambdamax:
                ChiQuark = 1e10*CorU*ChiQuark
        except:
            pass
        

    if weight35region==True:
        ChiQuark = np.array([weight(chi) for chi in ChiQuark])
        ChiLepton = np.array([weight(chi) for chi in ChiLepton])

    return np.concatenate((QuarkLeptonChiRatio[0]*ChiQuark, QuarkLeptonChiRatio[1]*ChiLepton))



########### Generate Random parameters ##############

### My random distribution 'MyRandomS()'. It takes values between 0 and 1, where it strongly favors very small values and slightly favors big values.
def MyRandomS():
    return np.random.choice(np.concatenate((
                                     np.random.power(7, 10),  # big values
                                     #1-np.random.power(80, 20),      # very small values
                                     10**(np.random.uniform(low=-6.0,high=0.0, size=20)),
                                     np.random.uniform(low=0.0, high=1.0, size=10)  # flat backgroud
                                    )))
                                    
def MyRandomSk():
    return np.random.choice(np.concatenate((
        10**(np.random.uniform(low=-4.0,high=0.0, size=20)),
        (np.random.uniform(low=0.0,high=5.0, size=15)),
        np.abs(np.random.normal(1,2,size=10))
                                    )))                                    

def MyRandomB():
    return np.random.choice(np.concatenate((
                                  #   np.random.power(7, 10),  # big values
                                     10**(np.random.uniform(low=-6.0,high=0.0, size=10)),
                                  #   np.random.uniform(low=0.0, high=1.0, size=10)  # backgroud
                                    )))

# Use this to plot a histogram of these distributions:
#def plot_loghist(x, bins):
#  hist, bins = np.histogram(x, bins=bins)
#  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#  plt.hist(x, bins=logbins)
#  plt.xscale('log')
#plot_loghist([MyRandomS() for i in range(10000)], 20)
                                    

def RandomParameterSet(withPhases=False, Imtau_region=(0.866,5.0), LeptonFixed=False, KahlerCorrection=False, LooseKahler=False, s3n=False,
                       SetZero=[], SetValue={}, SetRange={}, Fixate=[], bmax=0.1, PhasesOnlyU=False, NeighborhoodC=False, June=False, secondLepMin=False, 
                       RadialKahler=False, **kwargs):
    params = Parameters()
    
    params.add('Retau', value=np.random.uniform(low=-1.5, high=1.5), min=-1.5, max=1.5)
    params.add('Imtau', value=np.random.uniform(low=Imtau_region[0], high=Imtau_region[1]), min=0.001, max=6)
    
    # In case you want s1e and s2e and their corresponding phases
    #sList = ['s1u', 's2u', 's1d', 's2d', 's1e', 's2e', 's1n', 's2n']
    #pList = ['p1u', 'p2u', 'p1d', 'p2d', 'p1e', 'p2e', 'p1n', 'p2n']
    sList = ['s1u', 's2u', 's1d', 's2d', 's1n', 's2n']
    pList = ['p1u', 'p2u', 'p1d', 'p2d', 'p1n', 'p2n']   # plist is already defined as a kwarg
    bList = ['buu', 'bdu', 'bnu', 'bud', 'bdd', 'bnd', 'buq', 'bdq', 'bnq']
    bList2pp = ['buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp']
    
    for s in sList:
        params.add(s, value=np.random.choice([-1,1])*MyRandomS(), min=-1, max=1)
    
    if withPhases==False:
        for p in pList:
            params.add(p, value=0.0, min=-np.pi, max=np.pi, vary=False)
    elif withPhases==True and PhasesOnlyU==False:
        for s in sList:
            params[s].set(value = np.abs(params[s].value), min=0)
        for p in pList:
            params.add(p, value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
    elif withPhases==True and PhasesOnlyU==True:
        for s in ['s1u','s2u']:
            params[s].set(value = np.abs(params[s].value), min=0)
        for p in ['p1u','p2u']:
            params.add(p, value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        for p in ['p1d', 'p2d', 'p1n', 'p2n']:
            params.add(p, value=0.0, min=-np.pi, max=np.pi, vary=False)
            
    if KahlerCorrection==True and LooseKahler==False:
        for b in bList:
            params.add(b, value=np.random.choice([-1,1])*MyRandomB(), min=-bmax, max=bmax)
        for b in bList2pp:
            params.add(b, value=np.random.choice([-1,1])*MyRandomB(), min=-bmax, max=bmax)
            
    if LooseKahler==True:
        params.add('bk', value=np.random.choice([-1,1])*MyRandomB(), min=-bmax, max=bmax)
        params.add('s1k', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s2k', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s3k', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('p1k', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        params.add('p2k', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        params.add('p3k', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        
    if LooseKahler=='down':
        params.add('bk', value=np.random.choice([-1,1])*MyRandomB(), min=-bmax, max=bmax)
        params.add('s1k', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s2k', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s3k', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('bkd', value=np.random.choice([-1,1])*MyRandomB(), min=-bmax, max=bmax)
        params.add('s1kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s2kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s3kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
     
    if LooseKahler=='new':
        params.add('lambdaKq', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('lambdaKl', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('kappaq', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=10), min=-8, max=8)
        params.add('kappal', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=10), min=-8, max=8)
        params.add('s1kqA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kqA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kqB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kqB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1klA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2klA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kl', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1klB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2klB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kl', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        
    if LooseKahler=='all36':
        params.add('lambdaKq', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('lambdaKu', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('lambdaKd', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('lambdaKl', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('kappaq', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=10), min=-8, max=8)
        params.add('kappau', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=10), min=-8, max=8)
        params.add('kappad', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=10), min=-8, max=8)
        params.add('kappal', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=10), min=-8, max=8)
        params.add('s1kqA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kqA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kqB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kqB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kuA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kuA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kuB', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kuB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kuB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kuB', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kdA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kdA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kdA', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kdB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2kdB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kdB', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1klA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2klA', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kl', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1klB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        params.add('s2klB', value=np.random.choice([-1,1])*MyRandomSk(), min=-8, max=8)
        #params.add('s3kl', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        
    if LooseKahler=='newchoice':
        params.add('lambdaKq', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('lambdaKd', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('lambdaKe', value=np.random.choice([-1,1])*MyRandomS(), min=-0.5, max=0.5)
        params.add('kappaq', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=5), min=-5, max=5)
        params.add('kappad', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=5), min=-5, max=5)
        params.add('kappae', value=np.random.choice([-1,1])*np.random.uniform(low=1/10, high=5), min=-5, max=5)
        params.add('s1kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        params.add('s2kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        #params.add('s3kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        params.add('s2kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        #params.add('s3kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        params.add('s1ke', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        params.add('s2ke', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        #params.add('s3kuB', value=np.random.choice([-1,1])*MyRandomSk(), min=-1, max=1)
        
    if LooseKahler=='June':
        params = Parameters()
        params.add('Retau', value=np.random.normal(loc=0.0222, scale=0.01), min=-1.5, max=1.5)
        params.add('Imtau', value=np.random.normal(loc=3.21, scale=0.05), min=0.001, max=6)
        params.add('kappad', value=1.0, min=-5, max=5, vary=False)
        params.add('kappau', value=1.0, min=-5, max=5, vary=False)
        params.add('kappaq', value=1.0, min=-5, max=5, vary=False)
        params.add('lambdaKd', value=np.random.choice([-1,1])*MyRandomS(), min=-bmax, max=bmax)
        params.add('lambdaKu', value=np.random.choice([-1,1])*MyRandomS(), min=-bmax, max=bmax)
        params.add('lambdaKq', value=np.random.choice([-1,1])*MyRandomS(), min=-bmax, max=bmax)
        params.add('p1d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1u', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        params.add('p2d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2u', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        params.add('p3n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('s1d', value=np.random.normal(loc=-0.388e-04, scale=0.06e-04), min=-1, max=1)
        params.add('s1n', value=np.random.normal(loc=0.00120, scale=0.0001), min=-1, max=1)
        params.add('s1u', value=MyRandomS(), min=0, max=1)
        params.add('s2d', value=np.random.normal(loc=0.0566, scale=0.004), min=-1, max=1)
        params.add('s2n', value=np.random.normal(loc=-0.984, scale=0.04), min=-2, max=2)
        params.add('s2u', value=MyRandomS(), min=0, max=1)
        params.add('s3n', value=1.0, min=-1, max=1, vary=False)
        if RadialKahler==False:
            params.add('s1kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s1ku', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s1kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s2kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s2ku', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s2kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        if RadialKahler==True:
            params.add('phiq', value=np.random.uniform(low=0, high=2*np.pi), min=0, max=2*np.pi)
            params.add('phiu', value=np.random.uniform(low=0, high=2*np.pi), min=0, max=2*np.pi)
            params.add('phid', value=np.random.uniform(low=0, high=2*np.pi), min=0, max=2*np.pi)
            params.add('thetaq', value=np.random.uniform(low=0, high=np.pi), min=0, max=np.pi)
            params.add('thetau', value=np.random.uniform(low=0, high=np.pi), min=0, max=np.pi)
            params.add('thetad', value=np.random.uniform(low=0, high=np.pi), min=0, max=np.pi)
        
    if LooseKahler=='June' and secondLepMin==True:
        params = Parameters()
        params.add('Retau', value=np.random.normal(loc=-0.041, scale=0.01), min=-1.5, max=1.5)
        params.add('Imtau', value=np.random.normal(loc=3.157, scale=0.05), min=0.001, max=6)
        params.add('kappad', value=1.0, min=-5, max=5, vary=False)
        params.add('kappau', value=1.0, min=-5, max=5, vary=False)
        params.add('kappaq', value=1.0, min=-5, max=5, vary=False)
        params.add('lambdaKd', value=np.random.choice([-1,1])*MyRandomS(), min=-bmax, max=bmax)
        params.add('lambdaKu', value=np.random.choice([-1,1])*MyRandomS(), min=-bmax, max=bmax)
        params.add('lambdaKq', value=np.random.choice([-1,1])*MyRandomS(), min=-bmax, max=bmax)
        params.add('p1d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1u', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        params.add('p2d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2u', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
        params.add('p3n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('s1d', value=np.random.normal(loc=-0.21e-04, scale=0.03e-04), min=-1, max=1)
        params.add('s1n', value=np.random.normal(loc=0.00122, scale=0.0001), min=-1, max=1)
        params.add('s1u', value=MyRandomS(), min=0, max=1)
        params.add('s2d', value=np.random.normal(loc=0.0557, scale=0.004), min=-1, max=1)
        params.add('s2n', value=np.random.normal(loc=-0.982, scale=0.05), min=-2, max=2)
        params.add('s2u', value=MyRandomS(), min=0, max=1)
        params.add('s3n', value=1.0, min=-1, max=1, vary=False)
        if RadialKahler==False:
            params.add('s1kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s1ku', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s1kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s2kd', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s2ku', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
            params.add('s2kq', value=np.random.choice([-1,1])*MyRandomSk(), min=-100, max=100)
        if RadialKahler==True:
            params.add('phiq', value=np.random.uniform(low=0, high=2*np.pi), min=0, max=2*np.pi)
            params.add('phiu', value=np.random.uniform(low=0, high=2*np.pi), min=0, max=2*np.pi)
            params.add('phid', value=np.random.uniform(low=0, high=2*np.pi), min=0, max=2*np.pi)
            params.add('thetaq', value=np.random.uniform(low=0, high=np.pi), min=0, max=np.pi)
            params.add('thetau', value=np.random.uniform(low=0, high=np.pi), min=0, max=np.pi)
            params.add('thetad', value=np.random.uniform(low=0, high=np.pi), min=0, max=np.pi)
        
        
    if s3n==True:
        params.add('s3n', value=np.random.choice([-1,1])*MyRandomS(), min=-1, max=1)
        params.add('p3n', value=np.random.uniform(low=-np.pi, high=np.pi), min=-np.pi, max=np.pi)
    else:
        params.add('s3n', value=1.0, min=-1, max=1, vary=False)
        params.add('p3n', value=0.0, min=-np.pi, max=np.pi, vary=False)

    if LeptonFixed==True:
        params.add('Imtau', value=3.15, min=0.01, max=15, vary=False)
        params.add('s1d', value=2.130622461948839e-05, min=-1, max=1, vary=False)
        params.add('s2d', value=0.05599153207496821, min=-1, max=1, vary=False)
        params.add('s1n', value=-0.0012076287800633079, min=-1, max=1, vary=False)
        params.add('s2n', value=0.9999989743583448, min=-1, max=1, vary=False)
        params.add('p1d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2n', value=0.0, min=-np.pi, max=np.pi, vary=False)
    
    for el in SetZero:
        params.add(el, value=0.0, vary=False)
        
    for el in SetRange:
        params[el].min = SetRange[el][0]
        params[el].max = SetRange[el][1]
        
    for el in SetValue:
        params[el].value = SetValue[el]
        
    for el in Fixate:
        params[el].vary=False
        
    if NeighborhoodC==True:
        params = Parameters()
        params.add('Retau', value=np.random.normal(loc=-1.089464, scale=0.2), min=-1.5, max=1.5)
        params.add('Imtau', value=np.random.normal(loc=2.062096, scale=0.5), min=0.001, max=6)
        params.add('kappad', value=1.0, min=-5, max=5, vary=False)
        params.add('kappae', value=1.0, min=-5, max=5, vary=False)
        params.add('kappaq', value=1.0, min=-5, max=5, vary=False)
        params.add('lambdaKd', value=np.random.normal(loc=0.393663, scale=0.2), min=-0.5, max=0.5, vary=False)
        params.add('lambdaKe', value=np.random.normal(loc=0.375900, scale=0.2), min=-0.5, max=0.5, vary=False)
        params.add('lambdaKq', value=np.random.normal(loc=0.106140, scale=0.2), min=-0.5, max=0.5, vary=False)
        params.add('p1d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1u', value=np.random.normal(loc=-3.141593, scale=1), min=-np.pi, max=np.pi)
        params.add('p2d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2u', value=np.random.normal(loc=-1.373457, scale=1.), min=-np.pi, max=np.pi)
        params.add('p3n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('s1d', value=np.random.normal(loc=0.018338, scale=0.01), min=-1, max=1)
        params.add('s1kd', value=np.random.normal(loc=1.189557, scale=1.), min=-100, max=100)
        params.add('s1ke', value=np.random.normal(loc=0.086623, scale=0.06), min=-100, max=100)
        params.add('s1kq', value=np.random.normal(loc=0.144204, scale=0.1), min=-100, max=100)
        params.add('s1n', value=np.random.normal(loc=0.269684, scale=0.1), min=-1, max=1)
        params.add('s1u', value=np.random.normal(loc=0.025977, scale=0.01), min=0, max=1)
        params.add('s2d', value=np.random.normal(loc=-0.203037, scale=0.1), min=-1, max=1)
        params.add('s2kd', value=np.random.normal(loc=-0.353835, scale=0.1), min=-100, max=100)
        params.add('s2ke', value=np.random.normal(loc=-1.285605, scale=0.6), min=-100, max=100)
        params.add('s2kq', value=np.random.normal(loc=-2.898115, scale=1), min=-100, max=100)
        params.add('s2n', value=np.random.normal(loc=-0.165203, scale=0.1), min=-1, max=1)
        params.add('s2u', value=np.random.normal(loc=0.060844, scale=0.02), min=0, max=1)
        params.add('s3n', value=1.0, min=-1, max=1, vary=False)
        
    if NeighborhoodC=='double':
        params = Parameters()
        params.add('Retau', value=np.random.normal(loc=-1.047934, scale=0.1), min=-1.5, max=1.5)
        params.add('Imtau', value=np.random.normal(loc=2.059187, scale=0.1), min=0.001, max=6)
        params.add('kappad', value=1.0, min=-5, max=5, vary=False)
        params.add('kappae', value=1.0, min=-5, max=5, vary=False)
        params.add('kappaq', value=1.0, min=-5, max=5, vary=False)
        params.add('lambdaKd', value=np.random.normal(loc=0.396571, scale=0.05), min=-0.5, max=0.5, vary=False)
        params.add('lambdaKe', value=np.random.normal(loc=0.374715, scale=0.05), min=-0.5, max=0.5, vary=False)
        params.add('lambdaKq', value=np.random.normal(loc=0.105741, scale=0.05), min=-0.5, max=0.5, vary=False)
        params.add('p1d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p1u', value=np.random.normal(loc=-3.134713, scale=0.1), min=-np.pi, max=np.pi)
        params.add('p2d', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('p2u', value=np.random.normal(loc=-1.214336, scale=0.2), min=-np.pi, max=np.pi)
        params.add('p3n', value=0.0, min=-np.pi, max=np.pi, vary=False)
        params.add('s1d', value=np.random.normal(loc=0.025928, scale=0.01), min=-1, max=1)
        params.add('s1kd', value=np.random.normal(loc=1.159149, scale=0.1), min=-100, max=100)
        params.add('s1ke', value=np.random.normal(loc=0.011006, scale=0.05), min=-100, max=100)
        params.add('s1kq', value=np.random.normal(loc=0.184207, scale=0.1), min=-100, max=100)
        params.add('s1n', value=np.random.normal(loc=0.288237, scale=0.05), min=-1, max=1)
        params.add('s1u', value=np.random.normal(loc=0.025006, scale=0.005), min=0, max=1)
        params.add('s2d', value=np.random.normal(loc=-0.216246, scale=0.02), min=-1, max=1)
        params.add('s2kd', value=np.random.normal(loc=0.141004, scale=0.05), min=-100, max=100)
        params.add('s2ke', value=np.random.normal(loc=-1.291561, scale=0.1), min=-100, max=100)
        params.add('s2kq', value=np.random.normal(loc=-2.902265, scale=0.5), min=-100, max=100)
        params.add('s2n', value=np.random.normal(loc=-0.172432, scale=0.05), min=-1, max=1)
        params.add('s2u', value=np.random.normal(loc=0.06337, scale=0.01), min=0, max=1)
        params.add('s3n', value=1.0, min=-1, max=1, vary=False)
        
    return params


def RandomParameterSet_fer(Imtau_region=(0.866,5.0), KahlerCorrection=False, 
                           SetZero=[], SetValue={}, **kwargs):
    params = Parameters()
    params.add('Retau', value=np.random.uniform(low=-1.5, high=1.5), min=-1.5, max=1.5)
    params.add('Imtau', value=np.random.uniform(low=Imtau_region[0], high=Imtau_region[1]), min=0.001, max=6)
    
    if KahlerCorrection==True:
        bList = ['a1', 'a2', 'a3']
        for b in bList:
            params.add(b, value=np.random.choice([-1,1])*(10**(np.random.uniform(low=-6.0,high=-1.0))), 
                       min=-0.1, max=0.1)
    
    for el in SetZero:
        params.add(el, value=0.0, vary=False)
        
    for el in SetValue:
        params[el].value = SetValue[el]
        
    return params

########### Set a time limit on each minimization ##############

# the time limit is specified by using "iter_cb=MinimizeStopper(max_time_in_seconds)" in the minimizer
# When the time limit is reached, the 'least_squares' method returns it current values, all other methods simply give an error that stops all calculations (this can be repared by using 'try')
class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    #def __call__(self, x1,x2,x3,x4,x5,xk=None):
    def __call__(self, x1,x2,x3,xk=None,**kwargs):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            return True


########### My minimizer ##############

# I scan in the following manner:
#   - choose some random parameters
#   - minimize 'nr_methods'-times and thereby choose randomly one of the listed methods
#   - always take the preceeding best fit value as start value for the next method
#   - If the time limit 'max_time' is reached, abort and use least_squares method (least_squares can give you its current value when the time limit is reached)


def myscan(sector='lepton', nr_methods=4, max_time=45, retry_time=10, model='eclectic', methods=[], DigDeeper=False, DigDeeperThreshold=1000, **kwargs):
    # Generate random parameterset
    if model=='eclectic':
        params = RandomParameterSet(**kwargs)
    if model=='feruglio':
        params = RandomParameterSet_fer(**kwargs)
        
    if sector=='lepton':
        residual=PMNSresidual
    elif sector=='quark':
        residual=CKMresidual
    elif sector in ['both', 'quark and lepton', 'lepton and quark']:
        residual=CKMandPMNSresidual
    elif sector=='myquark':
    	residual=MyResidual
    else:
        print('I dont know any sector called ', sector)
    
    # Choose the methods you want to try for the optimizer
    if methods==[]:
        methods = ['least_squares','least_squares','least_squares',#'least_squares',
                   'nelder','nelder','nelder',#'nelder','nelder','nelder','nelder','nelder',
                   'differential_evolution',
                   'powell',
                   'lbfgsb',
                   'cg',
                   'cobyla',
                   'trust-constr']
                   
    methods = np.random.choice(methods, size=nr_methods)

    outs=[];
    
    tmp = []

    for i in range(len(methods)):
        try:  # Try out the method, but abort if it takes longer than 'max_time' in seconds
            outs = np.append(outs,
                             minimize(residual, params, kws={'model':model, **kwargs}, 
                                      method=methods[i], iter_cb=MinimizeStopper(max_time))
                            )
            print(outs[-1].chisqr,"  :  ",methods[i])
            tmp.append([methods[i], outs[-1].chisqr, outs[-1].chisqr/outs[-2].chisqr])
        except:  # If the method failed or took to long use 'least_squares' for maximum retry_time seconds
            outs = np.append(outs,
                             minimize(residual, params, kws={'model':model, **kwargs}, 
                                      method='least_squares', iter_cb=MinimizeStopper(retry_time))
                            )
        params=outs[i].params
            
    if DigDeeper==True and outs[-1].chisqr < DigDeeperThreshold:
        for i in range(len(methods)):
            try:  # Try out the method, but abort if it takes longer than 'max_time' in seconds
                outs = np.append(outs,
                                 minimize(residual, params, kws={'model':model, **kwargs}, 
                                          method=methods[i], iter_cb=MinimizeStopper(max_time))
                                )
            except:  # If the method failed or took to long use 'least_squares' for maximum 5 seconds
                outs = np.append(outs,
                                 minimize(residual, params, kws={'model':model, **kwargs}, 
                                          method='least_squares', iter_cb=MinimizeStopper(10))
                                )
            params=outs[i+len(methods)].params
            
    print("\n")
    print(tmp)

    return outs

########### Gather all minima found with my minimizer in a pandas DataFrame ##############

def generate_df(outs, KahlerCorrection=False, ordering='NO', sector='lepton', **kwargs):
    df = pd.DataFrame()
    for i in range(len(outs)):
        add = {}
        add['chisq'] = outs[i].chisqr
        add['KahlerCorrection'] = KahlerCorrection
        add['ordering'] = ordering
        add['sector'] = sector
        for name in outs[i].params:
            add[name] = outs[i].params[name].value
        df = df.append(add, ignore_index=True)

    return df


########### Finally, the actual scan ##############

def TheScan(num_rand_pts, **kwargs):
    df = pd.DataFrame()
    for i in range(num_rand_pts):
        try:
            outs = myscan(**kwargs);
            outdf = generate_df(outs, **kwargs);
            df = df.append(outdf, ignore_index=True)
        except:
            pass
    return df
    
    
########### Sorting DataFrame such that s1x < s2x ##############
 
def interchange1and2(df):
    for i in df.index:
        if np.abs(df.loc[i]['s1d']) > np.abs(df.loc[i]['s2d']):
            df.at[i, 's1u'], df.at[i, 's2u'] = df.loc[i]['s2u'], df.loc[i]['s1u']
            df.at[i, 's1d'], df.at[i, 's2d'] = df.loc[i]['s2d'], df.loc[i]['s1d']
            df.at[i, 's1n'], df.at[i, 's2n'] = df.loc[i]['s2n'], df.loc[i]['s1n']
    return df


########### Adding observables to the pandas-DataFrame that TheScan spits out ##############

def addLeptonObs(df, **kwargs):
    try: del kwargs['KahlerCorrection']
    except: pass
    df = df.join(pd.DataFrame([AllLeptonObs(df.loc[i], **kwargs) for i in df.index]))
    return df

def addChi(df, error_type='gaussian'):
    try: del kwargs['KahlerCorrection']
    except: pass
    df['chisq'] = np.array([Chisq(df.loc[i], error_type=error_type, sector=df['sector'].loc[i]) for i in df.index]);
    return df

def addCKMobs(df, **kwargs):
    try: del kwargs['KahlerCorrection']
    except: pass
    df = df.join(pd.DataFrame([CKMobs(**{**M_ecl_quark(df.loc[i],
                                                     KahlerCorrection=df['KahlerCorrection'].loc[i],
                                                     **kwargs),
                                       **kwargs})
                               for i in df.index]))
    return df
    
        
########### Random scan in good region ##############

def StripFundDomain(dfnew):
    dfnew = dfnew[((np.abs(dfnew['s1u'])<1.0) & (np.abs(dfnew['s2u'])<1.0) & 
                   (np.abs(dfnew['s1d'])<1.0) & (np.abs(dfnew['s2d'])<1.0) & 
                   (np.abs(dfnew['s1n'])<1.0) & (np.abs(dfnew['s2n'])<1.0))] 
    dfnew = dfnew[((dfnew['Retau']<1.5) & (dfnew['Retau']>-1.5))]
    dfnew = dfnew[((dfnew['p1u']<np.pi) & (dfnew['p1u']>-np.pi) & 
                   (dfnew['p2u']<np.pi) & (dfnew['p2u']>-np.pi) &
                   (dfnew['p1d']<np.pi) & (dfnew['p1d']>-np.pi) & 
                   (dfnew['p2d']<np.pi) & (dfnew['p2d']>-np.pi) & 
                   (dfnew['p1n']<np.pi) & (dfnew['p1n']>-np.pi) & 
                   (dfnew['p2n']<np.pi) & (dfnew['p2n']>-np.pi))]
    dfnew = dfnew.reset_index(drop=True)
    return dfnew

# Generate a random set of inputparameters that are in the neighborhood of 'df'
def GenRandomNearCluster(df, params, fullparams, n, n_components=30, broaden_factor=4):
    X = np.array(df[params])
    # Identify 'n_components' clusters in guassian shape in the parameter space of the parameters in 'params'
    gmm = GaussianMixture(n_components=n_components, reg_covar=1e-10)
    gmm.fit(X).predict(X)
    # Artificially broaden the clusters to get a wider spread of the random points
    gmm.covariances_ = broaden_factor*gmm.covariances_
    # Generate the random points
    Xnew,_ = gmm.sample(n)
    dfnew = pd.DataFrame(Xnew)
    dfnew.columns = params
    # Fill up the parameters not in 'param' with zeros
    for par in fullparams:
        if par not in params:
            dfnew[par] = np.zeros(n)
    # throw away points that are not within the boundaries of the parameterspace
    dfnew = StripFundDomain(dfnew)
    
    return dfnew


def PlotClusters(df, x, y, n, axis=None, title=None,
                 params = ['Retau', 'Imtau', 's1d', 's2d', 's1n', 's2n'], 
                 fullparams = ['Retau', 'Imtau', 's1u', 's2u', 's1d', 's2d', 's1n', 's2n', 'p1u', 'p2u', 'p1d', 'p2d', 'p1n', 'p2n'],
                 n_components=5, max_iter=100, weight_concentration_prior=1/100,
                 broaden_factor=1.0):
    gmm = BayesianGaussianMixture(n_components=n_components, reg_covar=1e-10, max_iter=max_iter, weight_concentration_prior=weight_concentration_prior)
    X = np.array(df[params])

    labels = gmm.fit(X).predict(X)
    gmm.covariances_ = broaden_factor*gmm.covariances_

    df['c'] = labels
    
    Xnew,_ = gmm.sample(n)
    dfnew = pd.DataFrame(Xnew)
    dfnew.columns = params
    for par in fullparams:
        if par not in params:
            dfnew[par] = np.zeros(n)
    dfnew = StripFundDomain(dfnew)

    plt.figure(figsize=(10,6))
    if not axis==None:
        plt.axis(axis)
    plt.hexbin(dfnew[x], dfnew[y], cmap='ocean_r', alpha=0.7, extent=axis)
    scatter = plt.scatter(df[x], df[y], c=df['c'], s=4, cmap='viridis', label=[str(i) for i in range(n_components)])#, zorder=2, edgecolor='k')

    legend = plt.legend(*scatter.legend_elements(),
                        title="Clusters", bbox_to_anchor=(1.2, 1.))
    if title==None:
    	pass
    else:
    	plt.title(title);
    plt.xlabel(x);
    plt.ylabel(y);
    
    
    
    
print('\nLoaded functions:')
print(' Error calculation')
print('   SingleChi(obs, key=\'me/mu\', ordering=\'NO\', **kwargs):  returns Chi (not Chi^2!) of obs[key] according to Nufit5.1')
print('   Chisq(df_row, error_type=\'gaussian\', sector=\'lepton\'):  returns the Chisquare value of the \'sector\'-observables in the dictionary df_row')
print(' Mass matrices of models')
print('   Y1(tau, modform=\'full\', **kwargs): and Y2(tau, modform=\'full\', **kwargs):  return the vlaues of the modular forms of T\' of weight 1')
print('   M(Y1, Y2, s1, s2, s3, **kwargs):  returns the Mass matrix that comes from the superpotential term Y^(1) ( Phi_(2/3) )^3')
print('   K(s1, s2, s3):  returns the Kahler metric coming from ( Phi_(2/3) Phi^*_(2/3) ) ^2')
print('   K2pp(y1, y2, s1, s2, s3):  returns the Kahler metric coming from Y^(1) Y^(1)^* ( Phi_(3/2) Phi^*_(3/2) ) ^2')
print('   M_ecl_quark(params, KahlerCorrection=False, DiagMethod=\'eigh\', **kwargs):  returns \'{\'Mu\':Mu, \'Md\':Md}\', where Mu and Md are the quark masses of the eclectic model')
print('   M_ecl_lepton(params, KahlerCorrection=False, **kwargs):  returns\'{\'Me\':Me, \'Mn\':Mn}\', where Me and Mn are the lepton mass matrices of the eclectic model')
print('   M_fer_lepton(params, KahlerCorrection=False, KahlerReal=False, DiagMethod=\'eigh\', leftright=\'left\', permutation=False, **kwargs):  returns the mass matrices of feruglios model 1')
print(' Calculation of mass and mixing parameters')
print('   FetchWolf(CKM):  returns \'{\'l\':l, \'A\':A, \'rhobar\':rhobar, \'etabar\':etabar}\' of CKM-matrix')
print('   FetchStandard(CKM):  returns \'{\'t12\':t12, \'t13\':t13, \'t23\':t23, \'dq\':d}\' of CKM-matrix')
print('   CKMobs(Mu=np.identity(3), Md=np.identity(3), CKM_par=\'Wolfenstein\', **kwargs):  returns a dictionary with the quark mass and mixing parameters')
print('   CKMresidual(params, CKM_par=\'Wolfenstein\', FittedObservablesQ=[\'mu/mc\',\'mc/mt\',\'md/ms\',\'ms/mb\',\'l\',\'A\',\'rhobar\',\'etabar\'], WeightFactorForChiQ=None, **kwargs):  use this function for the optimizer')
print('   FetchParametrization(PMNS):  returns the array \'[t12, t13, t23, d, eta1, eta2]\' of PMNS-matrix')
print('   PMNSobs(Me=np.identity(3), Mn=np.identity(3), ordering=\'NO\', **kwargs):  returns a dictionary with the lepton mass and mixing parameters (Carefull, mn[i] not scaled correctly!)')
print('   AllLeptonObs(params, modform=\'full\', model=\'eclectic\'):  returns a dictionary with the lepton mass and mixing parameters')
print('   PMNSresidual(params, ordering=\'NO\', error_type=\'gaussian\', model=\'eclectic\',FittedObservablesL=[\'me/mu\',\'mu/mt\',\'s12^2\',\'s13^2\',\'s23^2\',\'r\',\'d/pi\'], **kwargs):  use this function for the optimizer')
print('   PMNSandCKMresidual( ... ):  I still have to write this function!!')
print(' Generate random parameters')
print('   MyRandomS(): and MyRandomB():  these are the distributions used to randomly get a flavon vev or kahler potential parameter')
print('   RandomParameterSet(withPhases=False, Imtau_region=(0.866,5.0), LeptonFixed=False, KahlerCorrection=False, SetZero=[], SetValue={}, SetRange={}, Fixate=[], bmax=0.1, **kwargs):  returns a set of random parameters that can be used for the optimizer')
print('   RandomParameterSet_fer(Imtau_region=(0.866,5.0), KahlerCorrection=False, SetZero=[], SetValue={}, **kwargs):  same but for feruglio model1 ')
print(' The optimization')
print('   myscan(sector=\'lepton\', nr_methods=4, max_time=45, model=\'eclectic\', **kwargs):  returns a list that contains the outputs of the optimizers')
print('   generate_df(outs, KahlerCorrection=False, ordering=\'NO\', sector=\'lepton\', **kwargs):  returns a DataFrame that contains the outputs of the optimizers \'outs\' ')
print('   TheScan(num_rand_pts, **kwargs):  returns a DataFrame that contains the scanresults')
print(' Amendments to a DataFrame')
print('   addLeptonObs(df, modform=\'full\', model=\'eclectic\'):  returns the same df but with all lepton observables')
print('   addCKMobs(df, **kwargs):  returns the same df but with all quark observables')
print('   addChi(df, error_type=\'gaussian\'):  returns the same df but with the Chi^2 value corresponding to df[\'sector\']')
print(' Generate new random point near existing points')
print('   GenRandomNearCluster(df, params, fullparams, n, n_components=30, broaden_factor=4):  returns a new df with random points near the points in df')

print('\n\n')
    
print('\nPossible kwargs (first one is the default):')
print(' General setup:')
print('   sector = \'lepton\' or \'quark\' or \'lepton and quark\' ')
print('   model = \'eclectic\' or \'feruglio\' ')
print('   KahlerCorrection = False or True')
print('   ordering = \'NO\' or \'IO\' ')
print(' Parameter space:')
print('   withPhases = False or True')
print('   Imtau_region = (0.866,5.0) or (Imtau_min, Imtau_max)')
print('   LeptonFixed = False or True')
print('   SetZero = [] or a list with e.g. \'buu\', \'s1u\', or \'Retau\' ')
print('   SetValue = {} or a dictionary like e.g. {\'Retau\':0.4, \'Imtau\':0.3} ')
print('   SetRange = {} or a dictionary like e.g. {\'Retau\':(-1.5, 1.5), ...} ')
print('   Fixate = [] or a list with e.g. \'s1u\' (sets the \'vary\' of this parameter to \'False\') ')
print('   bmax = 0.1 or any float (is the max for the Kahlercorrection parameters)')
print(' Error derivation:')
print('   error_type = \'gaussian\' or \'realistic\' ')
print('   CKM_par = \'Wolfenstein\' or \'Standard\' (dont forget to adjust \'FittedObservablesQ\' accordingly!) ')
print('   FittedObservablesQ = [\'mu/mc\',\'mc/mt\',\'md/ms\',\'ms/mb\',\'l\',\'A\',\'rhobar\',\'etabar\'] or any subset or \'t12\' ...')
print('   WeightFactorForChiQ = None or [1, 1/10, ..., 100] ordering as in \'FittedObservablesQ\'')
print('   FittedObservablesL=[\'me/mu\',\'mu/mt\',\'s12^2\',\'s13^2\',\'s23^2\',\'r\',\'d/pi\'] or any subset')
print(' Calculation details:')
print('   DiagMethod = \'eigh\' or \'svd\' ')
print('   modform = \'full\' or \'q-exp\' or \'q-exp3\' ')
print('   nr_methods = 4 or any integer')
print('   max_time = 45 or any number measured in seconds')


def DoThis(loops, n, filename='~/test.csv', chimax=100, Kscan=False, numbKparams=5, sort12df=False, **kwargs):
	start_time = time.time()
	for l in range(loops):
		if Kscan==True:
			blist = ['buu', 'bdu', 'bnu', 'bud', 'bdd', 'bnd', 'buq', 'bdq', 'bnq', 'buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp']
			thezeros = np.random.choice(blist, replace=False, size=len(blist) - numbKparams)
			kwargs['SetZero'] = np.concatenate((kwargs['SetZero'], thezeros))
		try:
			df = TheScan(n, **kwargs)
			df = df.sort_values(by=['chisq']).reset_index(drop=True)
			df = df[(df['chisq']<chimax)]
			df = df[(df['Imtau']>0.5)]
			if l==0:
				dffull = df
			else:
				dffull = dffull.append(df, ignore_index=True)
		except:
			pass
		print('end of step ', l)
		
	if sort12df == True:
	    dffull = interchange1and2(dffull)
		
	dffull = dffull.sort_values(by=['chisq']).reset_index(drop=True)
	end_time = time.time()
	print('It took: ', (end_time-start_time)/60/60,' hours')
	if filename==None:
		return dffull
	else:
		dffull.to_csv(filename, index=False)
	
	
def KahlerPictures(df_sinK, Correctionslist=['buu'], Consideredobs=['mu/mc','mc/mt','md/ms','ms/mb'], 
                   errors=False, bmax=0.5, i=0, clist=None, size=100, **kwargs):
    
    plt.figure(figsize=(12,8))
    
    if clist==None:
        clist = {'t12':'blue', 't13':'green', 't23':'orange', 'dq':'red', 
                 'l':'blue', 'A':'green', 'rhobar':'orange', 'etabar':'red',
                 'mu/mc':'blue', 'mc/mt':'green', 'md/ms':'orange', 'ms/mb':'red'}
    
    for b_name in Correctionslist:
        df = {}
        for el in ['Retau', 'Imtau', 's1u', 's2u', 's1d', 's2d', 's1n', 's2n',
            'p1u', 'p2u', 'p1d', 'p2d', 'p1n', 'p2n']:
            df[el] = np.full(size, df_sinK[el][i])
        for el in ['buu', 'bdu', 'bnu', 'bud', 'bdd', 'bnd', 'buq', 'bdq', 'bnq']:
            df[el] = np.full(size, 0.0)
        for el in ['buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp']:
            df[el] = np.full(size, 0.0)
        b_max=bmax
        b_min=-1*bmax
        df[b_name] = np.linspace(b_min, b_max, size)
        df['KahlerCorrection'] = np.full(size, True)
        df = pd.DataFrame(df)
        df = addCKMobs(df, **kwargs)
        
        plt.title(b_name)
        
        for name in Consideredobs:
            plt.plot(df[b_name], df[name], label=name, c=clist[name])
            plt.axhline(Qexpdata[name]['best'], c=clist[name], ls='dotted')
            if errors==True:
                plt.axhspan(Qexpdata[name]['best']+3/2*Qexpdata[name]['1sig_range'], 
                            Qexpdata[name]['best']-3/2*Qexpdata[name]['1sig_range'], 
                            facecolor=clist[name], alpha=0.2)
           
    plt.axvline(0, lw=1, c='gray', alpha=0.5)
    plt.legend()
    plt.xlabel('beta_i');
    plt.ylabel('obs_i');
    plt.yscale('log')
    
    
def plot_minimum(df, key, ran, nr_pts=100, **kwargs):
    df = pd.DataFrame([out.params.valuesdict() for i in range(nr_pts)])
    df['KahlerCorrection'], df['ordering'], df['sector'] = kwargs['KahlerCorrection'], kwargs['ordering'], kwargs['sector']
    best = df[key][0]
    df[key] = np.linspace(best - ran[0], best + ran[1], num=len(df))
    
    if myargs['sector'] in ['both', 'quark and lepton', 'lepton and quark']:
        df = addCKMobs(df, **kwargs)
        df['KahlerCorrection']=False
        df = addLeptonObs(df)
        df['KahlerCorrection']=True
    if myargs['sector'] == 'lepton':
        df = addLeptonObs(df)
    if myargs['sector'] == 'quark':
        df = addCKMobs(df, **kwargs)
    df = addChi(df, error_type='realistic')
    
    plt.figure(figsize=(15,3))
    plt.plot(df[key], df['chisq'])
    plt.axvline(best, c='lightgray')
    plt.axhline(np.min(df['chisq']), c='lightgray', ls='--')
    plt.axhline(np.min(df['chisq'])+1, c='lightgray', ls='--')
    plt.xlabel(key)
    plt.ylabel(r'$\chi^2$')
    idx = np.argwhere(np.diff(np.sign(df['chisq'] - np.array([np.min(df['chisq'])+1 for i in range(nr_pts)])))).flatten()
    err = np.linspace(best - ran[0], best + ran[1], num=len(df))[idx]
    d = np.abs(best/(err[1]-err[0]))
    plt.plot(err, [np.min(df['chisq'])+1 for i in err], 'ro')
    plt.title(r'$\mathrm{d}_i$ = ' + str(d))
    
    
def ExploreWithMCMC(df, filename=None, length='Full', param_list=['Retau', 'Imtau', 's1d', 's2d', 's1n', 's2n'], 
                    burn=300, steps=50000, thin=50, nwalkers=20, **kwargs):
    if 'SetValue' in kwargs:
        print('there is a \'SetValue\' in the arguments you gave to \'ExploreWithMCMC\'')
    if length=='Full':
        length=len(df.index)
        
    for i in df.index[:length]:
        try:
            thepoint = df.loc[i][param_list].to_dict()
            MCMCargs = {'SetValue':thepoint, **kwargs}
            params = RandomParameterSet(**MCMCargs) # this is not so "random" since we 'SetValue' the relevant parameters
            print(i,": ")
            out = lmfit.minimize(PMNSresidual, params=params, kws={**MCMCargs}, method='emcee',
                                 nan_policy='omit', burn=burn, steps=steps, thin=thin, nwalkers=nwalkers,
                                 is_weighted=True)
            if i==0:
                dfout = out.flatchain
            else:
                dfout = dfout.append(out.flatchain, ignore_index=True)
        except: pass
    
    if filename==None:
        return dfout
    else:
        dfout.to_csv(filename, index=False)
        
        
def mycompletion(df, chimax=25):
    df['KahlerCorrection'], df['ordering'], df['sector'] = False, 'NO', 'lepton'
    df['s1u'], df['s2u'], df['s3n'], df['p3n'] = 0.0, 0.0, 1.0, 0.0
    df['p1n'], df['p2n'], df['p1d'], df['p2d'], df['p1u'], df['p2u'] = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    df = addLeptonObs(df, modform='full')
    df = addChi(df, error_type='realistic')
    df = df.sort_values(by=['chisq']).reset_index(drop=True)
    df = df[(df['chisq']<chimax)]
    # Sort out the ones that are not within bounds
    df = df[(df['m1']<0.037)]
    df = df[(df['m_bb']<0.061)]
    return df
    
    
myargs23={'sector':'lepton and quark','CKM_par':'Standard', 'ordering':'NO',
         'modform':'q-exp',
         'error_type':'realistic',
         'DigDeeper':True, 'DigDeeperThreshold':1000,
         'SetZero':['p1u', #'s1u',
                    'p2u', #'s2u',
                    'p1d', #'s1d',
                    'p2d', #'s2d',
                    'p1n','p2n',#'s1n','s2n',
                    'p1k','p2k','p3k',
                    'buu', 'bdu', 'bnu', 'bud', 'bdd', 'bnd', 'buq', 'bdq', 'bnq',
                    'buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp'],
         'SetRange':{'s1k':(-6,6), 's2k':(-6,6),# 's3k':(-3,3), 
                     's1kd':(-6,6), 's2kd':(-6,6),# 's3kd':(-3,3), 
                     's1u':(-1,1), 's2u':(-1,1), 's1d':(-1,1), 's2d':(-1,1), 's1n':(-2,2), 's2n':(-2,2)},
         'SetValue':{'s3k':1, 's3kd':1},
         'Fixate':['s3k', 's3kd'],
         'withPhases':False,
         'KahlerCorrection':True,
         'LooseKahler':'down',
         'bmax':3,
         'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'max_time':70, 'retry_time':15, 'nr_methods':6,
         #'Kscan':True, 'numbKparams':6,   #'SetZero':['buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp'],
         'chimax':1000}
         
myargs21={'sector':'lepton and quark','CKM_par':'Standard', 'ordering':'NO',
         'modform':'q-exp',
         'error_type':'realistic',
         'DigDeeper':True, 'DigDeeperThreshold':1000,
         'SetZero':['p1u', #'s1u',
                    'p2u', #'s2u',
                    'p1d', #'s1d',
                    'p2d', #'s2d',
                    'p1n','p2n',#'s1n','s2n',
                    'p1k','p2k','p3k',
                    'buu', 'bdu', 'bnu', 'bud', 'bdd', 'bnd', 'buq', 'bdq', 'bnq',
                    'buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp'],
         'SetRange':{'s2k':(-6,6), 's3k':(-6,6),# 's1k':(-3,3), 
                     's2kd':(-6,6), 's3kd':(-6,6),# 's1kd':(-3,3), 
                     's1u':(-1,1), 's2u':(-1,1), 's1d':(-1,1), 's2d':(-1,1), 's1n':(-2,2), 's2n':(-2,2)},
         'SetValue':{'s1k':1, 's1kd':1},
         'Fixate':['s1k', 's1kd'],
         'withPhases':False,
         'KahlerCorrection':True,
         'LooseKahler':'down',
         'bmax':3,
         'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'max_time':70, 'retry_time':15, 'nr_methods':6,
         #'Kscan':True, 'numbKparams':6,   #'SetZero':['buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp'],
         'chimax':1000}
         
myargs1={'sector':'lepton and quark','CKM_par':'Standard', 'ordering':'NO',
         'modform':'q-exp',
         'error_type':'realistic',
         'SetZero':['p1u', #'s1u',
                    'p2u', #'s2u',
                    'p1d', #'s1d',
                    #'p2d', #'s2d',
                    'p1n','p2n',#'s1n','s2n',
                    'p1k','p2k','p3k',
                    'buu', 'bdu', 'bnu', 'bud', 'bdd', 'bnd', 'buq', 'bdq', 'bnq',
                    'buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp'],
         'SetRange':{'s2k':(-3,3), 's3k':(-3,3),# 's1k':(-3,3), 
                     's1u':(-1,1), 's2u':(-1,1), 's1d':(-1,1), 's2d':(-1,1), 's1n':(-2,2), 's2n':(-2,2)},
         'SetValue':{'s1k':1},
         'Fixate':['s1k'],
         'withPhases':True,
         'KahlerCorrection':True,
         'LooseKahler':True,
         'bmax':5,
         'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'max_time':70, 'retry_time':15, 'nr_methods':5,
         #'Kscan':True, 'numbKparams':6,   #'SetZero':['buu2pp', 'bdu2pp', 'bnu2pp', 'bud2pp', 'bdd2pp', 'bnd2pp', 'buq2pp', 'bdq2pp', 'bnq2pp'],
         'chimax':10000}
         
leptonargs = {'sector':'lepton', 'ordering':'NO', 
              'modform':'q-exp', 'error_type':'realistic',
              'SetZero':['s1u','s2u'], 'withPhases':False, 'Imtau_region':(1.7,4.5),
              'SetRange':{'s1n':(-2,2), 's2n':(-2,2), 's1d':(-0.005,0.005)}, 
              'DigDeeper':True, 'DigDeeperThreshold':100, 
              'max_time':30, 'retry_time':5, 'nr_methods':4, 
              'sort12df':True, 'chimax':25}
              
quarkargs = {'sector':'quark', 
             'KahlerCorrection':True, 'LooseKahler':'new',
             'modform':'q-exp', 
             'max_time':50, 'retry_time':5, 'nr_methods':4, 'chimax':10000}
             
quarkleptonargs = {'sector':'quark and lepton', 'withPhases':False, 
                   'KahlerCorrection':True, 'LooseKahler':'new',
                   'modform':'q-exp', 'error_type':'realistic', 
                   'max_time':100, 'retry_time':10, 'nr_methods':6, 'chimax':5000,
                   'SetZero':['lambdaKl','kappal','s1klA','s2klA','s1klB','s2klB'],
                   'SetRange':{'lambdaKq':(-0.1, 0.1), 
                               'Imtau':(3.17, 3.24), 'Retau':(0.0140, 0.0304), 
                               's1d':(-0.0000443, -0.0000339), 's2d':(0.0528, 0.0605), 
                               's1n':(0.00111, 0.00128), 's2n':(-1.010, -0.9943)}}
                               
newargs = {'sector':'both', 'chimax':2000,
           'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
           'modform':'q-exp', 'error_type':'realistic',
           'max_time':60, 'retry_time':20, 'nr_methods':4,
           'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5)},
           'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
           'Fixate':['kappaq','kappad','kappae'],
           'CKM_par':'Standard', 'FittedObservablesQ':["mu/mc","mc/mt","md/ms","ms/mb","t12","t13","t23","dq"]}
           
newargs2 = {'sector':'both', 'chimax':2000,
           'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
           'modform':'q-exp', 'error_type':'realistic',
           'max_time':60, 'retry_time':20, 'nr_methods':4,
           'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5), 'kappad':(-3,3), 'Imtau':(0.886,4.0)},
           'SetValue':{'kappaq':1., 'kappae':1.},
           'Fixate':['kappaq','kappae'],
           'CKM_par':'Standard', 'FittedObservablesQ':["mu/mc","mc/mt","md/ms","ms/mb","t12","t13","t23","dq"]}
           
newargsNoCPl = {'sector':'both', 'chimax':2000,
                'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
                'modform':'q-exp', 'error_type':'realistic',
                'max_time':60, 'retry_time':20, 'nr_methods':4,
                'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5), 'Imtau':(0.886,4.0)},
                'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
                'Fixate':['kappaq','kappad','kappae'],
                'CKM_par':'Standard', 'FittedObservablesQ':["mu/mc","mc/mt","md/ms","ms/mb","t12","t13","t23","dq"],
                'FittedObservablesL':['me/mu','mu/mt','s12^2','s13^2','s23^2','r']}
                
                
ArgsA = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5)},
         'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
         'Fixate':['kappaq','kappad','kappae']}
         
ArgsB = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5)},
         'SetValue':{'kappaq':1., 'kappae':1.},
         'Fixate':['kappaq','kappae']}
         
ArgsC = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'plist':['p1u','p2u'],
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5)},
         'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
         'Fixate':['kappaq','kappad','kappae']}
         
ArgsCc = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True, 'NeighborhoodC':True,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':5,
         'plist':['p1u','p2u'],
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5)},
         'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
         'Fixate':['kappaq','kappad','kappae']}
         
ArgsCcdouble = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True, 'NeighborhoodC':'double', 'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':5,
         'plist':['p1u','p2u'],
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5)},
         'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
         'Fixate':['kappaq','kappad','kappae']}
         
ArgsD = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5), 'Imtau':(0.886,4.0)},
         'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
         'Fixate':['kappaq','kappad','kappae'],
         'CKM_par':'Standard', 'FittedObservablesQ':["mu/mc","mc/mt","md/ms","ms/mb","t12","t13","t23"],
         'FittedObservablesL':['me/mu','mu/mt','s12^2','s13^2','s23^2','r']}
         
ArgsE = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'newchoice', 'LKahlerCorrection':True, 'CheckKahler':True,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetRange':{'lambdaKq':(-0.5, 0.5), 'lambdaKd':(-0.5,0.5), 'lambdaKe':(-0.5,0.5), 's1kq':(-1,1), 's2kq':(-1,1), 's1kd':(-1,1), 's2kd':(-1,1), 's1ke':(-1,1), 's2ke':(-1,1), 'Retau':(-0.5,0.5), 'Imtau':(0.8,4)},
         'SetValue':{'kappaq':1., 'kappad':1., 'kappae':1.},
         'Fixate':['kappaq','kappad','kappae']}
         
         
JuneA = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':False, 'withPhases':True, 'PhasesOnlyU':True,
         'bmax':5,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4}
         
JuneB = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'bmax':0.6,    #bmax is an error it should be 'lambdamax'!!
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4}
         
JuneC = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True, 'secondLepMin':True,
         'bmax':0.6,    #bmax is an error it should be 'lambdamax'!!
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4}
         
JuneD = {'sector':'both', 'chimax':100,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'bmax':1.0,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.20555, 'Retau':0.022221, 's1d':-0.0000388, 's2d':0.056553, 's1n':0.001194, 's2n':-0.984153},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}
         
JuneE = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'bmax':0.6,    #bmax is an error it should be 'lambdamax'!!
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.205, 'Retau':0.0222, 's1d':-0.0000388, 's2d':0.0566, 's1n':0.0012, 's2n':-0.984, 'p1u':0.0, 'p2u':0.0},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n', 'p1u', 'p2u']}
         
JuneF = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'bmax':0.5,    #bmax is an error it should be 'lambdamax'!!
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.20555, 'Retau':0.022221, 's1d':-0.0000388, 's2d':0.056553, 's1n':0.001194, 's2n':-0.984153},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}
         
JuneG = {'sector':'both', 'chimax':2000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'bmax':0.3,   #bmax is an error it should be 'lambdamax'!!
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.20555, 'Retau':0.022221, 's1d':-0.0000388, 's2d':0.056553, 's1n':0.001194, 's2n':-0.984153},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}

JuneH = {'sector':'both', 'chimax':1000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'bmax':0.1,  #bmax is an error it should be 'lambdamax'!!
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.205, 'Retau':0.0222, 's1d':-0.0000388, 's2d':0.0566, 's1n':0.00120, 's2n':-0.984},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}
         
JuneI = {'sector':'both', 'chimax':5000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'lambdamax':0.8, 'bmax':0.8,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.205, 'Retau':0.0222, 's1d':-0.0000388, 's2d':0.0566, 's1n':0.00120, 's2n':-0.984},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}

JuneJ = {'sector':'both', 'chimax':5000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True, 'RadialKahler':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'lambdamax':0.8, 'bmax':0.8,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.205, 'Retau':0.0222, 's1d':-0.0000388, 's2d':0.0566, 's1n':0.00120, 's2n':-0.984},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}

JuneK = {'sector':'both', 'chimax':1000,
         'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True, 'RadialKahler':True,
         'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
         'lambdamax':0.9, 'bmax':0.9,
         'modform':'q-exp', 'error_type':'realistic',
         'max_time':70, 'retry_time':20, 'nr_methods':4,
         'SetValue':{'Imtau':3.205, 'Retau':0.0222, 's1d':-0.0000388, 's2d':0.0566, 's1n':0.00120, 's2n':-0.984},
         'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}
         
         
leptonargsIO = {'sector':'lepton', 'ordering':'IO', 
              'modform':'q-exp', 'error_type':'realistic',
              'SetZero':['s1u','s2u'], 'withPhases':False, 'Imtau_region':(0.3,5),
              'SetRange':{'s1n':(-2,2), 's2n':(-2,2), 's2d':(-0.1,0.1), 's1d':(-0.005,0.005)}, 
              'DigDeeper':True, 'DigDeeperThreshold':200, 
              'max_time':30, 'retry_time':5, 'nr_methods':4, 
              'sort12df':True, 'chimax':300}
              
leptonargsAntusch = {'sector':'lepton', 'ordering':'NO', 
                     'modform':'q-exp', 'error_type':'realistic',
                     'SetZero':['s1u','s2u'], 'withPhases':False, 'Imtau_region':(1.7,4.5),
                     'SetRange':{'s1n':(-2,2), 's2n':(-2,2), 's1d':(-0.005,0.005), 's2d':(-0.07,0.07)}, 
                     'DigDeeper':True, 'DigDeeperThreshold':100, 
                     'max_time':30, 'retry_time':5, 'nr_methods':4, 
                     'sort12df':True, 'chimax':25}
                     
                     
JuneAntusch1 = {'sector':'both', 'chimax':5000,
                'KahlerCorrection':True, 'LooseKahler':'June', 'LKahlerCorrection':False, 'CheckKahler':True, 'withPhases':True, 'PhasesOnlyU':True,
                'CKM_par':'Standard', 'FittedObservablesQ':['mu/mc','mc/mt','md/ms','ms/mb','t12','t13','t23','dq'],
                #'lambdamax':0.8, 'bmax':0.8,
                'modform':'q-exp', 'error_type':'realistic',
                'max_time':70, 'retry_time':20, 'nr_methods':4,
                'SetValue':{'Imtau':3.195, 'Retau':0.02279, 's1d':-0.00004069, 's2d':0.05833, 's1n':0.001224, 's2n':-0.9857},
                'Fixate':['Imtau', 'Retau', 's1d', 's2d', 's1n', 's2n']}






# Wolfram language
from wolframclient.evaluation import WolframLanguageSession


session = WolframLanguageSession()


class F_term_equation:
    def __init__(self, parameters, equation):
        self.parameters = parameters
        self.equation = equation


def solve_Fterms(params, sols, equation_list, **kwargs):
    wolfram_code = 'TimeConstrained['
    if all([equation.parameters == equation_list[1].parameters for equation in equation_list]):
        wolfram_code = wolfram_code + 'thePoint={'
        for parameter in equation_list[1].parameters:
            wolfram_code = wolfram_code + str(parameter) + '->' + ("%.18f" % params[str(parameter)].real) + '+I*' + (
                        "%.18f" % params[str(parameter)].imag) + ','
        wolfram_code = wolfram_code[:-1] + '}; '
    else:
        raise TypeError('''Not all F-term equations have the same parameters. Please list all parameters for every
                           F-term equation, even though this specific equation depends only on a subset.''')

    wolfram_code = wolfram_code + 'solution = NSolve['
    for equation in equation_list:
        wolfram_code = wolfram_code + '((' + equation.equation + ')/.thePoint)&&'
    wolfram_code = wolfram_code[:-2]
    wolfram_code = wolfram_code + ',{'
    for sol in sols:
        wolfram_code = wolfram_code + str(sol) + ','

    wolfram_code = wolfram_code[:-1] + '}]; '

    wolfram_code = wolfram_code + 'result = {'
    for sol in sols:
        wolfram_code = wolfram_code + 'Re[' + str(sol) + '],'
        wolfram_code = wolfram_code + 'Im[' + str(sol) + '],'

    wolfram_code = wolfram_code[:-1] + '}/.solution[[Mod[' + str(
        int(params['select_f_solution'] + 1)) + ',Length[solution]]]]'

    wolfram_code = wolfram_code + ',0.9]'

    wolfram_result = session.evaluate(wolfram_code)

    if str(wolfram_result) == '$Aborted':
        result = {sols[i]: 0.0 for i in range(len(sols))}
    else:
        result = {sols[i]: float(wolfram_result[2 * i]) + 1j * float(wolfram_result[2 * i + 1]) for i in
                  range(len(sols))}

    return result