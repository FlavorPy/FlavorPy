from .experimental_data.NuFit53.nufit53_chisqprofiles import Lexpdata_NO, Lexpdata_IO
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import pkgutil
from io import StringIO


def flavorpy_cmap() -> matplotlib.colors.Colormap:
    """
    A `matplotlib.colormap
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap>`_ where the
    first 1/25 is green, the next 4/25 are yellow, the next 9/25 are orange, and the remaining 16/25 are an opaque red
    that fades out to white, i.e.

    .. image:: /images/flavorpy_cmap.png

    Particularly useful colormap for representing values of chisq, whose square-root can be
    compared to confidence levels.

    :return: A `matplotlib.colormap
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap>`_
    """
    chimin, chimax = 0, 25
    chiperbit = (chimax - chimin) / 256
    sig1 = np.rint(1 / chiperbit).astype(int)
    sig2 = np.rint(4 / chiperbit).astype(int)
    sig3 = np.rint(9 / chiperbit).astype(int)

    greenyellow = np.array([156 / 256, 255 / 256, 47 / 256, 1])
    yellowgreen = np.array([231 / 256, 255 / 256, 100 / 256, 1])
    yellow = np.array([255 / 256, 255 / 256, 51 / 256])
    yelloworange = np.array([236 / 256, 216 / 256, 3 / 256, 1])
    orangeyellow = np.array([255 / 256, 191 / 256, 66 / 256, 1])
    orangered = np.array([255 / 256, 69 / 256, 0 / 256, 1])
    redorange = np.array([253 / 256, 141 / 256, 109 / 256, 1])

    nodes = [0.0, sig1 / 256, (sig1 + 1) / 256, (sig1 + sig2) / 2 / 256, sig2 / 256, (sig2 + 1) / 256, sig3 / 256,
             (sig3 + 1) / 256, 1.0]
    colors = ['green', greenyellow, yellowgreen, yellow, yelloworange, orangeyellow, orangered, redorange, 'white']
    cmap = LinearSegmentedColormap.from_list("flavorpy_cmap", list(zip(nodes, colors)))
    return cmap


def plot(df, x='me/mu', y='mu/mt', cmap=flavorpy_cmap(), ordering='NO', show_exp=None, xylabels=None, exp_colors='gray',
         **hexbin_plt_kwargs):
    """
    Quickly plot the data resulting of a fit. Gives a hexbin plot where the color is specified by chisq. Plots can
    include the latest experimental data for charged leptons from `NuFit v5.3 <http://www.nu-fit.org/?q=node/278>`_.


    :param df: The data you want to plot. E.g. the output of :py:meth:`~modelfitting.model.FlavorModel.complete_fit`.
    :type df: pandas.DataFrame
    :param x: The column of df plotted on the x-Axis.
    :type x: str
    :param y: The column of df plotted on the y-Axis.
    :type y: str
    :param cmap: The `matplotlib.colormap
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap>`_ or a
        `registered colormap <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_ name that will be
        used as the colormap for the hexbin plot.
    :type cmap: str or matplotlib.colormap
    :param ordering: Specify whether the neutrino spectrum is normal or inverted ordered.
    :type ordering: str, either \'NO\' or \'IO\', default is \'NO\'
    :param show_exp: Whether to plot `NuFit v5.3 <http://www.nu-fit.org/?q=node/278>`_ experimental data. Chose \'1dim\'
        for plotting the boundaries of the 1-dimensional projections of chisq or \'2dim\' for plotting the 2-dimensional
        ones. Please consider citing `NuFit <http://www.nu-fit.org/?q=node/278>`_, when using this data.
    :type show_exp: str, either \'1dim\' or \'2dim\', optional
    :param xylabels: The labels used for x and y-axis.
    :type xylabels: tuple or list, optional
    :param exp_colors: The color of the experimental boundaries. See
        `named colors in matplotlib <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
    :type exp_colors: str, optional
    :param hexbin_plt_kwargs: Keyword arguments that will be passed down to
        `matplotlib.pyplot.hexbin <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hexbin.html>`_.
    :type hexbin_plt_kwargs: dict, optional
    :return: `matplotlib.axes
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
    """
    # initialize plot
    fig, ax = plt.subplots(figsize=(6.18, 3.82))

    # plot the points in df
    im = ax.hexbin(df[x], df[y], C=df['chisq'], reduce_C_function=np.min, cmap=cmap, **hexbin_plt_kwargs)
    fig.colorbar(im).set_label(r'$\qquad\chi^2$', rotation=0)

    # plot experimental data
    if show_exp == '1dim':
        if ordering == 'NO':
            Lexpdata = Lexpdata_NO
        elif ordering == 'IO':
            Lexpdata = Lexpdata_IO
        else:
            raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')
        ax.axvline(Lexpdata[x]['1sig_min'], ls='solid', c=exp_colors)
        ax.axvline(Lexpdata[x]['1sig_max'], ls='solid', c=exp_colors)
        ax.axvline(Lexpdata[x]['3sig_min'], ls='dotted', c=exp_colors)
        ax.axvline(Lexpdata[x]['3sig_max'], ls='dotted', c=exp_colors)
        ax.axhline(Lexpdata[y]['1sig_min'], ls='solid', c=exp_colors)
        ax.axhline(Lexpdata[y]['1sig_max'], ls='solid', c=exp_colors)
        ax.axhline(Lexpdata[y]['3sig_min'], ls='dotted', c=exp_colors)
        ax.axhline(Lexpdata[y]['3sig_max'], ls='dotted', c=exp_colors)
        plot_range_x = Lexpdata[x]['3sig_max'] - Lexpdata[x]['3sig_min']
        plot_range_y = Lexpdata[y]['3sig_max'] - Lexpdata[y]['3sig_min']
        plt.xlim((Lexpdata[x]['3sig_min'] - 0.1*plot_range_x, Lexpdata[x]['3sig_max'] + 0.1*plot_range_x))
        plt.ylim((Lexpdata[y]['3sig_min'] - 0.1*plot_range_y, Lexpdata[y]['3sig_max'] + 0.1*plot_range_y))
    if show_exp == '2dim':
        if ordering == 'NO':
            data = StringIO(pkgutil.get_data(__name__, "experimental_data/NuFit53/v53.release-SKyes-NO.txt").decode())
            Lexpdata = Lexpdata_NO
        elif ordering == 'IO':
            data = StringIO(pkgutil.get_data(__name__, "experimental_data/NuFit53/v53.release-SKyes-IO.txt").decode())
            Lexpdata = Lexpdata_IO
        else:
            raise NotImplementedError('''The value of \'ordering\' has to be either \'NO\' or \'IO\'.''')
        row_in_file = {('s12^2', 's13^2'): {'skiprows': 1216548, 'nrows': 12566},
                       ('s13^2', 'm21^2'): {'skiprows': 1230116, 'nrows': 31824},
                       ('s12^2', 'm21^2'): {'skiprows': 1261942, 'nrows': 41496},
                       ('s13^2', 's23^2'): {'skiprows': 1303440, 'nrows': 10302},
                       ('s13^2', 'm3l^2'): {'skiprows': 1313744, 'nrows': 16830},
                       ('s23^2', 'm3l^2'): {'skiprows': 1330576, 'nrows': 16665},
                       ('s13^2', 'd/pi'): {'skiprows': 1347243, 'nrows': 7446},
                       ('s23^2', 'd/pi'): {'skiprows': 1354692, 'nrows': 7373},
                       ('m3l^2', 'd/pi'): {'skiprows': 1362066, 'nrows': 12045},
                       ('s12^2', 's23^2'): {'skiprows': 1374113, 'nrows': 13433},
                       ('s12^2', 'd/pi'): {'skiprows': 1387548, 'nrows': 9709},
                       ('s12^2', 'm3l^2'): {'skiprows': 1397259, 'nrows': 21945},
                       ('m21^2', 's23^2'): {'skiprows': 1419206, 'nrows': 31512},
                       ('m21^2', 'd/pi'): {'skiprows': 1450720, 'nrows': 22776},
                       ('m21^2', 'm3l^2'): {'skiprows': 1473498, 'nrows': 51480},
                       }
        for combination in row_in_file:
            if combination == (x, y) or combination == (y, x):
                expdata = pd.read_csv(data, delimiter='\s+', index_col=False, **row_in_file[combination])
                expdata = expdata.rename(columns={'sin^2(theta12)': 's12^2',
                                                  'sin^2(theta13)': 's13^2',
                                                  'sin^2(theta23)': 's23^2'})
                expdata['chisq'] = expdata['Delta_chi^2'] - np.min(expdata['Delta_chi^2'])
                if x == 'd/pi' or y == 'd/pi':
                    expdata['d/pi'] = np.mod(expdata['Delta_CP/deg'] / 180, 2)
                if x == 'm21^2' or y == 'm21^2':
                    expdata['m21^2'] = np.power(10, expdata['Log10(Delta_m21^2/[eV^2])'])
                if x == 'm3l^2' or y == 'm3l^2':
                    if ordering == 'NO':
                        expdata['m3l^2'] = 1e-03 * expdata['Delta_m31^2/[1e-3_eV^2]']
                ax.tricontour(expdata[x], expdata[y], expdata['chisq'],
                              (1, 4, 9), linestyles=('solid', 'dashed', 'dotted'), colors=exp_colors)
                plot_range_x = Lexpdata[x]['3sig_max'] - Lexpdata[x]['3sig_min']
                plot_range_y = Lexpdata[y]['3sig_max'] - Lexpdata[y]['3sig_min']
                plt.xlim((Lexpdata[x]['3sig_min'] - 0.1 * plot_range_x, Lexpdata[x]['3sig_max'] + 0.1 * plot_range_x))
                plt.ylim((Lexpdata[y]['3sig_min'] - 0.1 * plot_range_y, Lexpdata[y]['3sig_max'] + 0.1 * plot_range_y))

        # lobster-plot
        if y == 'm_bb':

            # Inverted neutrino mass ordering
            if ordering == 'IO' and x == 'm3':
                pts = 100000
                pts_m3 = 1000
                # generate random points
                s12sq = np.random.uniform(low=Lexpdata['s12^2']['3sig_min'], high=Lexpdata['s12^2']['3sig_max'],
                                          size=pts)
                t12 = np.arcsin(np.sqrt(s12sq))
                c12sq = np.power(np.cos(t12), 2)
                s13sq = np.random.uniform(low=Lexpdata['s13^2']['3sig_min'], high=Lexpdata['s13^2']['3sig_max'],
                                          size=pts)
                t13 = np.arcsin(np.sqrt(s13sq))
                c13sq = np.power(np.cos(t13), 2)
                s23sq = np.random.uniform(low=Lexpdata['s23^2']['3sig_min'], high=Lexpdata['s23^2']['3sig_max'],
                                          size=pts)
                t23 = np.arcsin(np.sqrt(s23sq))
                c23sq = np.power(np.cos(t23), 2)
                m21sq = np.random.uniform(low=Lexpdata['m21^2']['3sig_min'], high=Lexpdata['m21^2']['3sig_max'],
                                          size=pts)
                m3lsq = np.random.uniform(low=Lexpdata['m3l^2']['3sig_min'], high=Lexpdata['m3l^2']['3sig_max'],
                                          size=pts)
                phi1 = np.random.uniform(low=0.0, high=2.0, size=pts)
                phi2 = np.random.uniform(low=0.0, high=2.0, size=pts)
                # a list of all m_bb values for one m3 and for all above generated ranodm points
                def mbb(m3):
                    return np.abs(m3 * s13sq +
                                  np.sqrt(np.power(m3, 2) - m3lsq) * s12sq * c13sq * np.exp(2j * np.pi * phi1) +
                                  np.sqrt(np.power(m3, 2) - m3lsq - m21sq) * c12sq * c13sq * np.exp(2j * np.pi * phi2))
                # determine the min and max of the m_bb-list for all m3 values
                m3_list = np.logspace(-4, 1, num=pts_m3)
                expm3mbb = pd.DataFrame(np.transpose(m3_list))
                expm3mbb.columns = ['m3']
                # np.max(gentest(0.1)['mbb'])
                expm3mbb['mbb_high'] = [np.max(mbb(expm3mbb['m3'][i])) for i in range(len(expm3mbb['m3']))]
                expm3mbb['mbb_low'] = [np.min(mbb(expm3mbb['m3'][i])) for i in range(len(expm3mbb['m3']))]
                # plot it
                ax.plot(expm3mbb['m3'], expm3mbb['mbb_high'], c=exp_colors, ls='--')
                ax.plot(expm3mbb['m3'], expm3mbb['mbb_low'], c=exp_colors, ls='--')
                plt.xlim((1e-04, 1))
                plt.ylim((1e-04, 1))

            # Normal neutrino mass ordering
            if y == 'm_bb':
                if ordering == 'NO' and x == 'm1':
                    pts = 100000
                    pts_m1 = 1000
                    # generate random points
                    s12sq = np.random.uniform(low=Lexpdata['s12^2']['3sig_min'], high=Lexpdata['s12^2']['3sig_max'],
                                              size=pts)
                    t12 = np.arcsin(np.sqrt(s12sq))
                    c12sq = np.power(np.cos(t12), 2)
                    s13sq = np.random.uniform(low=Lexpdata['s13^2']['3sig_min'], high=Lexpdata['s13^2']['3sig_max'],
                                              size=pts)
                    t13 = np.arcsin(np.sqrt(s13sq))
                    c13sq = np.power(np.cos(t13), 2)
                    s23sq = np.random.uniform(low=Lexpdata['s23^2']['3sig_min'], high=Lexpdata['s23^2']['3sig_max'],
                                              size=pts)
                    t23 = np.arcsin(np.sqrt(s23sq))
                    c23sq = np.power(np.cos(t23), 2)
                    m21sq = np.random.uniform(low=Lexpdata['m21^2']['3sig_min'], high=Lexpdata['m21^2']['3sig_max'],
                                              size=pts)
                    m3lsq = np.random.uniform(low=Lexpdata['m3l^2']['3sig_min'], high=Lexpdata['m3l^2']['3sig_max'],
                                              size=pts)
                    phi1 = np.random.uniform(low=0.0, high=2.0, size=pts)
                    phi2 = np.random.uniform(low=0.0, high=2.0, size=pts)
                    # a list of all m_bb values for one m3 and for all above generated ranodm points
                    def mbb(m1):
                        return np.abs(m1 * c12sq * c13sq +
                                      np.sqrt(np.power(m1, 2) + m21sq) * s12sq * c13sq * np.exp(2j*np.pi * phi1) +
                                      np.sqrt(np.power(m1, 2) + m21sq + m3lsq) * s13sq * np.exp(-2j * np.pi * phi2))
                    # determine the min and max of the m_bb-list for all m3 values
                    m1_list = np.logspace(-4, 1, num=pts_m1)
                    expm1mbb = pd.DataFrame(np.transpose(m1_list))
                    expm1mbb.columns = ['m1']
                    # np.max(gentest(0.1)['mbb'])
                    expm1mbb['mbb_high'] = [np.max(mbb(expm1mbb['m1'][i])) for i in range(len(expm1mbb['m1']))]
                    expm1mbb['mbb_low'] = [np.min(mbb(expm1mbb['m1'][i])) for i in range(len(expm1mbb['m1']))]
                    # plot it
                    ax.plot(expm1mbb['m1'], expm1mbb['mbb_high'], c=exp_colors, ls='--')
                    ax.plot(expm1mbb['m1'], expm1mbb['mbb_low'], c=exp_colors, ls='--')
                    plt.xlim((1e-04,1))
                    plt.ylim((1e-04, 1))

    # put x and y labels
    label_dict = {'s12^2': r'$\mathrm{sin}^2\,\theta_{12}$', 's13^2': r'$\mathrm{sin}^2\,\theta_{13}$',
                  's23^2': r'$\mathrm{sin}^2\,\theta_{23}$', 'd/pi': r'$\delta^\mathrm{\ell}_\mathrm{CP}/\pi$',
                  'm21^2': r'$\Delta\,m_{21}^2~[\mathrm{eV}^2]$', 'm3l^2': r'$\Delta\,m_{3\ell}^2~[\mathrm{eV}^2]$',
                  'Retau': r'$\mathrm{Re}\,\tau$', 'Imtau': r'$\mathrm{Im}\,\tau$',
                  'm1': r'$m_1$', 'm2': r'$m_2$', 'm3': r'$m_3$', 'm_bb': r'$m_{\beta\beta}$', 'm_b': r'$m_\beta$',
                  'Sum(m_i)': r'$\sum\,m_i$', 'nscale': r'$\Lambda_\nu$', 'Jmax': r'$J_\mathrm{max}$',
                  'eta1': r'$\eta_1/\pi$', 'eta2': r'$\eta_2/\pi$',
                  'me/mu': r'$m_\mathrm{e}/m_\mu$', 'mu/mt': r'$m_\mu/m_\tau$'}
    if xylabels is None:
        try:
            ax.set_xlabel(label_dict[x])
        except:
            ax.set_xlabel(x)
        try:
            ax.set_ylabel(label_dict[y])
        except:
            ax.set_ylabel(y)
    else:
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])

    return ax

