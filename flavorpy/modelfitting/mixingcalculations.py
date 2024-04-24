import numpy as np


def calculate_quark_observables(mass_matrix_u=np.identity(3), mass_matrix_d=np.identity(3),
                                parameterization='standard') -> dict:
    """
    Calculates the quark observables of up and down-type mass matrices.

    :param mass_matrix_u: The up-type quark mass matrix M, for Phi_left M Phi_right.
    :type mass_matrix_u: 3x3 matrix
    :param mass_matrix_d: The down-type quark mass matrix M, for Phi_left M Phi_right.
    :type mass_matrix_d: 3x3 matrix
    :param parameterization: Specify whether you want the result in standard or wolfenstein parametrization.
        Has to be either \'standard\' or \'wolfenstein\'.
    :type parameterization: str, default:\'standard\'
    :return: dict
        Contains the standard or wolfenstein parameters as well as the quark mass ratios.
    """
    # Singular Value decomposition for digonalization
    Vul_tilde, mu, Vurh_tilde = np.linalg.svd(mass_matrix_u)  # Mu = tVul * mu * tVurh
    Vdl_tilde, md, Vdrh_tilde = np.linalg.svd(mass_matrix_d)

    # correct the fact that svd sorts its singular values descending
    permut = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    mu, md = np.dot(permut, mu), np.dot(permut, md)
    Vul, Vdl = np.dot(permut, np.linalg.inv(Vul_tilde)), np.dot(permut, np.linalg.inv(Vdl_tilde))

    # construct CKM Matrix and calculate its parameterization
    CKM = np.dot(Vul, Vdl.conj().T)
    if parameterization == 'wolfenstein':
        CKM_p = get_wolfenstein_parameters(CKM)
    elif parameterization == 'standard':
        CKM_p = get_standard_parameters_ckm(CKM)
    else:
        raise NotImplementedError('''The value of \'parameterization\' has to be either
                                  \'wolfenstein\' or \'standard\'.''')

    return {'mu/mc': mu[0] / mu[1], 'mc/mt': mu[1] / mu[2], 'md/ms': md[0] / md[1], 'ms/mb': md[1] / md[2], **CKM_p}


def calculate_lepton_dimensionless_observables(mass_matrix_e=np.identity(3), mass_matrix_n=np.identity(3),
                                               ordering='NO') -> dict:
    """
    Calculates the dimensionless observables of a charged lepton and neutrino mass matrix.

    :param mass_matrix_e: The charged lepton mass matrix M, for Phi_left M Phi_right, where left and right indicates
        left- and right-handed chiral fields, respectively. I.e. L_i^c M_ij e_j, where L refers to the left-handed
        lepton doublet and e is the right-handed charged lepton field, and i,j=1,2,3.
        If you use the other convention, i.e. left-handed fields on the right-hand-side, simply transpose your mass
        matrix.
    :type mass_matrix_e: 3x3 matrix
    :param mass_matrix_n: The neutrino mass matrix M, for Phi_left M Phi_right.
    :type mass_matrix_n: 3x3 matrix
    :param ordering: Specify whether the neutrino spectrum is normal or inverted ordered. Has to be either \'NO\'
        or \'IO\'. Default is \'NO\'.
    :type ordering: str
    :return: Contains the PMNS-parameters and the charged lepton mass matrices. It also contains wrongly scaled neutrino
        masses, that are needed for the function \'calculate_lepton_observables\'.
    :rtype: dict
    """
    # Singular Value decomposition for digonalization.   There are also some takagi decompositions in the utils.py file.
    tVel, me, tVerh = np.linalg.svd(mass_matrix_e)  # tVel * diag(me) * tVerh = Me
    tVnl, mn, tVnrh = np.linalg.svd(mass_matrix_n)  # tVnl * diag(mn) * tVnrh = Mn

    # correct the fact that svd sorts its singular values (=physical masses) in descending order.
    # Idea: one could add some code here that automatically detects, whether the spectrum is normal or inverted ordered
    #       by checking if mn[0]-mn[1] >< mn[1]-mn[2]. The ordering would then be an observable. I don't know if this
    #       would be useful for fitting, because the experimental data is specific for normal or inverted ordering.
    #       But maybe one could take the absolute value of m21^2/mel^2 and simply fit to the NO data to get a rough
    #       estimate because the exp data for NO and IO are very similar.
    #       Or the experimental data set simply needs to contain both, the data for NO and for IO.
    permutE = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    if ordering == 'NO':
        permutN = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    elif ordering == 'IO':
        permutN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')
    me, mn = np.dot(permutE, me), np.dot(permutN, mn)
    Vel, Vnl = np.dot(permutE, np.linalg.inv(tVel)), np.dot(permutN, np.linalg.inv(tVnl))

    # construct PMNS Matrix and calculate its parameterization
    PMNS = np.dot(Vel.conj(), Vnl.T)  # PMNS = np.dot( Vel.conj().T, Vnl)
    pmns_parameters = get_standard_parameters_pmns(PMNS)

    # calculate neutrino mass-squared-differences.
    m21sq = np.power(mn[1], 2) - np.power(mn[0], 2)
    if ordering == 'NO':
        m3lsq = np.power(mn[2], 2) - np.power(mn[0], 2)
    elif ordering == 'IO':
        m3lsq = np.power(mn[2], 2) - np.power(mn[1], 2)
    else:
        raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')

    return {'me/mu': me[0] / me[1], 'mu/mt': me[1] / me[2], 'r': m21sq / m3lsq, **pmns_parameters,
            'm1_wrong_scaled': mn[0], 'm2_wrong_scaled': mn[1], 'm3_wrong_scaled': mn[2],
            'm21^2_wrong_scaled': m21sq, 'm3l^2_wrong_scaled': m3lsq}


def calculate_lepton_observables(mass_matrix_e=np.identity(3), mass_matrix_n=np.identity(3), ordering='NO',
                                 m21sq_best=None, m3lsq_best=None) -> dict:
    """
    Calculates the observables of the lepton sector from charged lepton and neutrino mass matrices.
    
    :param mass_matrix_e: The charged lepton mass matrix M, for Phi_left M Phi_right, where left and right indicates
        left- and right-handed chiral fields, respectively. I.e. L_i^c M_ij e_j, where L refers to the left-handed
        lepton doublet and e is the right-handed charged lepton field, and i,j=1,2,3.
        If you use the other convention, i.e. left-handed fields on the right-hand-side, simply transpose your mass
        matrix.
    :type mass_matrix_e: 3x3 matrix
    :param mass_matrix_n: The neutrino mass matrix M, for Phi_left M Phi_right.
    :type mass_matrix_n: 3x3 matrix
    :param ordering: Specify whether the neutrino spectrum is normal or inverted ordered. Has to be either \'NO\'
        or \'IO\'. Default is \'NO\'.
    :type ordering: str
    :param m21sq_best: The best fit value for the squared neutrino mass difference m_1^2 - m_2^2.
        Default is None, which will yield 7.41e-05.
    :type m21sq_best: float
    :param m3lsq_best: The best fit value for the squared neutrino mass difference m_3^2 - m_l^2, where l=1 for NO
        and l=2 for IO. Default is None, yielding 2.507e-03 for NO and -2.486e-03 for IO.
    :type m3lsq_best: float
    :return: Contains the PMNS parameters as well as the neutrino masses and charged lepton mass ratios.
    :rtype: dict
    """
    # Get dimensionless observables from calculate_lepton_dimensionless_observables()
    dimless_obs = calculate_lepton_dimensionless_observables(mass_matrix_e, mass_matrix_n, ordering)
    s12sq = dimless_obs['s12^2']
    s13sq = dimless_obs['s13^2']
    s23sq = dimless_obs['s23^2']
    c12sq = np.cos(np.arcsin(np.sqrt(s12sq))) ** 2
    c13sq = np.cos(np.arcsin(np.sqrt(s13sq))) ** 2
    c23sq = np.cos(np.arcsin(np.sqrt(s23sq))) ** 2
    eta1 = dimless_obs['eta1']
    eta2 = dimless_obs['eta2']
    d = dimless_obs['d/pi']

    # Jarlskog
    Jmax = np.sqrt(c12sq * s12sq * c23sq * s23sq * s13sq) * c13sq
    J = Jmax * np.sin(d * np.pi)

    # Correctly scale neutrino masses
    if m21sq_best is None:
        m21sq_best = 7.41e-05
    if m3lsq_best is None:
        if ordering == 'NO':
            m3lsq_best = 2.507e-03
        elif ordering == 'IO':
            m3lsq_best = -2.486e-03
        else:
            raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')
    mn = np.array([dimless_obs['m1_wrong_scaled'], dimless_obs['m2_wrong_scaled'], dimless_obs['m3_wrong_scaled']])
    m21sq, m3lsq = dimless_obs['m21^2_wrong_scaled'], dimless_obs['m3l^2_wrong_scaled']
    nscale = np.sqrt((m21sq_best / m21sq + m3lsq_best / m3lsq) / 2)
    mn = nscale * mn
    m21sq = nscale * nscale * m21sq
    m3lsq = nscale * nscale * m3lsq

    # Calculate effective neutrino mass for beta-decay and for neutrinoless double beta-decay.
    if ordering == 'NO':
        m_beta = np.sqrt(mn[0] ** 2 + m21sq * (1 - c13sq * c12sq) + m3lsq * s13sq)
        m_betabeta = np.abs(mn[0] * c12sq * c13sq +
                            np.sqrt(m21sq + mn[0] ** 2) * s12sq * c13sq * np.exp(2j * np.pi * (eta2 - eta1)) +
                            np.sqrt(m3lsq + m21sq + mn[0] ** 2) * s13sq * np.exp(-2j * np.pi * (d + eta1)))
    elif ordering == 'IO':
        m_beta = np.sqrt(mn[2] ** 2 + m21sq * c13sq * c12sq - m3lsq * c13sq)
        m_betabeta = np.abs(mn[2] * s13sq +
                            np.sqrt(mn[2] ** 2 - m3lsq) * s12sq * c13sq * np.exp(2j * np.pi * (eta2 + d)) +
                            np.sqrt(mn[2] ** 2 - m3lsq - m21sq) * c12sq * c13sq * np.exp(2j * np.pi * (eta1 + d)))
    else:
        raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')

    return {'me/mu': dimless_obs['me/mu'], 'mu/mt': dimless_obs['mu/mt'],
            's12^2': dimless_obs['s12^2'], 's13^2': dimless_obs['s13^2'], 's23^2': dimless_obs['s23^2'],
            'd/pi': dimless_obs['d/pi'], 'r': dimless_obs['r'], 'm21^2': m21sq, 'm3l^2': m3lsq,
            'm1': mn[0], 'm2': mn[1], 'm3': mn[2], 'eta1': dimless_obs['eta1'], 'eta2': dimless_obs['eta2'],
            'J': J, 'Jmax': Jmax, 'Sum(m_i)': np.sum(mn), 'm_b': m_beta, 'm_bb': m_betabeta, 'nscale': nscale}


def get_wolfenstein_parameters(CKM) -> dict:
    """
    Get values of the Wolfenstein-parameterization of CKM matrix, according to PDG.

    :param CKM: The CKM matrix in a 3x3-matrix-shape.
    :type CKM: 3x3 matrix
    :return: Dictionary that contains the parameters
    :rtype: dict
    """
    l = np.abs(CKM[0, 1]) / np.sqrt(np.power(np.abs(CKM[0, 0]), 2) + np.power(np.abs(CKM[0, 1]), 2))
    A = 1 / l * np.abs(CKM[1, 2] / CKM[0, 1])
    rhobar = np.real(-CKM[0, 0] * np.conj(CKM[0, 2]) / CKM[1, 0] / np.conj(CKM[1, 2]))
    etabar = np.imag(-CKM[0, 0] * np.conj(CKM[0, 2]) / CKM[1, 0] / np.conj(CKM[1, 2]))
    return {'lambda': l, 'A': A, 'rhobar': rhobar, 'etabar': etabar}


def get_standard_parameters_ckm(CKM) -> dict:
    """
    Get the values of the standard-parametrization of CKM matrix.

    :param CKM: The CKM matrix in a 3x3-matrix-shape.
    :type CKM: 3x3 matrix
    :return: Dictionary that contains the parameters
    :rtype: dict
    """
    t13 = np.arcsin(np.abs(CKM[0, 2]))
    t12 = np.arctan(np.abs(CKM[0, 1]) / np.abs(CKM[0, 0]))
    t23 = np.arctan(np.abs(CKM[1, 2]) / np.abs(CKM[2, 2]))
    d = np.mod(-1 * np.angle(
        ((np.conjugate(CKM[0, 0]) * CKM[0, 2] * CKM[2, 0] * np.conjugate(CKM[2, 2])) /
         (np.cos(t12) * np.power(np.cos(t13), 2) * np.cos(t23) * np.sin(t13)) + np.cos(t12) * np.cos(t23) * np.sin(
                    t13)) /
        (np.sin(t12) * np.sin(t23))) / np.pi, 2) * 180
    return {'t12': t12 * 180 / np.pi, 't13': t13 * 180 / np.pi, 't23': t23 * 180 / np.pi, 'd': d}


def get_standard_parameters_pmns(PMNS) -> dict:
    """
    Get the values of the standard-parameterization of PMNS matrix.

    :param PMNS: The PMNS matrix in a 3x3-matrix-shape.
    :type PMNS: 3x3 matrix
    :return: Dictionary that contains the parameters.
    :rtype: dict
    """
    t13 = np.arcsin(np.abs(PMNS[0, 2]))
    t12 = np.arctan(np.abs(PMNS[0, 1])/np.abs(PMNS[0, 0]))
    t23 = np.arctan(np.abs(PMNS[1, 2])/np.abs(PMNS[2, 2]))
    d = np.mod(-1*np.angle(
        ((np.conjugate(PMNS[0, 0])*PMNS[0, 2]*PMNS[2, 0]*np.conjugate(PMNS[2, 2])) /
         (np.cos(t12)*np.power(np.cos(t13), 2)*np.cos(t23)*np.sin(t13)) + np.cos(t12)*np.cos(t23)*np.sin(t13)) /
        (np.sin(t12)*np.sin(t23)))/np.pi, 2)
    eta1 = np.mod(np.angle(PMNS[0, 0]/PMNS[0, 2])/np.pi - d, 2)
    eta2 = np.mod(np.angle(PMNS[0, 1]/PMNS[0, 2])/np.pi - d, 2)
    return {'s12^2': np.sin(t12)**2, 's13^2': np.sin(t13)**2, 's23^2': np.sin(t23)**2,
            'd/pi': d, 'eta1': eta1, 'eta2': eta2}


def calculate_ckm(Mu=np.identity(3), Md=np.identity(3)) -> np.ndarray:
    """
    Calculates the CKM matrix out of the up and down-type quark mass matrices.

    :param Mu: The up-type quark mass matrix M, for Phi_left M Phi_right.
    :type Mu: 3x3 matrix
    :param Md: The down-type quark mass matrix M, for Phi_left M Phi_right.
    :type Md: 3x3 matrix
    :return: The CKM matrix
    :rtype: 3x3 matrix
    """
    # This function is not necessarily needed for the package. It's just here in case someone might find it useful.

    # Singular Value decomposition for digonalization
    tVul, mu, tVurh = np.linalg.svd(Mu)  # Mu = tVul * mu * tVurh
    tVdl, md, tVdrh = np.linalg.svd(Md)

    # correct the fact that svd sorts its singular values descending
    permut = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    mu, md = np.dot(permut, mu), np.dot(permut, md)
    Vul, Vdl = np.dot(permut, np.linalg.inv(tVul)), np.dot(permut, np.linalg.inv(tVdl))

    # construct CKM Matrix
    CKM = np.dot(Vul, Vdl.conj().T)

    return CKM


def calculate_pmns(Me=np.identity(3), Mn=np.identity(3), ordering='NO') -> np.ndarray:
    """
    Calculates the PMNS matrix out of the charged lepton and light neutrino mass matrices.

    :param Me: The charged lepton mass matrix M, for Phi_left M Phi_right.
    :type Me: 3x3 matrix
    :param Mn: The light neutrino mass matrix M, for Phi_left M Phi_right.
    :type Mn: 3x3 matrix
    :param ordering: Specify whether the neutrino spectrum is normal or inverted ordered. Has to be either
        \'NO\' or \'IO\'.
    :type ordering: str
    :return: The PMNS matrix.
    :rtype: 3x3 matrix
    """
    # This function is not necessarily needed for the package. It's just here in case someone might find it useful.

    # Singular Value decomposition for digonalization
    tVel, me, tVerh = np.linalg.svd(Me)  # tVel * diag(me) * tVerh = Me
    tVnl, mn, tVnrh = np.linalg.svd(Mn)  # tVnl * diag(mn) * tVnrh = Mn

    # correct the fact that svd sorts its singular values (=physical masses) in descending order
    permutE = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    if ordering == 'NO':
        permutN = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    elif ordering == 'IO':
        permutN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')
    me, mn = np.dot(permutE, me), np.dot(permutN, mn)
    Vel, Vnl = np.dot(permutE, np.linalg.inv(tVel)), np.dot(permutN, np.linalg.inv(tVnl))

    # construct PMNS Matrix
    PMNS = np.dot(Vel.conj(), Vnl.T)  # PMNS = np.dot( Vel.conj().T, Vnl)

    return PMNS

