# Mixing-values taken from nu-fit 5.1 [2007.14792] see also www.nu-fit.org
# Mixing-values with SK atmospheric data
# Charged lepton masses see arxiv:1706.08749
Lexpdata_NO = pd.DataFrame(np.array([
    [0.0048, 0.0565, 0.304, 0.02246, 0.450, 0.0742/2.510, 1.28, 7.42e-05, 2.510e-03],
    [0.0046, 0.0565-0.0045, 0.292, 0.02184, 0.434, 0.0722/2.537, 1.14, 7.22e-05, 2.483e-03],
    [0.0050, 0.0565+0.0045, 0.316, 0.02308, 0.469, 0.0763/2.483, 1.48, 7.63e-05, 2.537e-03],
    [0.0042, 0.0565-3*0.0045, 0.269, 0.02060, 0.408, 0.0682/2.593, 0.8, 6.82e-05, 2.430e-03],
    [0.0054, 0.0565+3*0.0045, 0.343, 0.02435, 0.603, 0.0804/2.430, 1.94, 8.04e-05, 2.593e-03]])
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