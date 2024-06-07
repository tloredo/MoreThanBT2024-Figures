"""
Plot ingredients for understanding the typical set for a
standard multivariate normal and its relationship to
the chi-squared distribution.

Created 2024-06-05 by Tom Loredo, for iid22 paper
"""

import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from numpy import *
from scipy import stats, special

from matplotlib import rc

rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125
rc('font', size=14)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=14)
rc('ytick.major', pad=8)
rc('ytick', labelsize=14)
rc('savefig', dpi=300)
rc('axes.formatter', limits=(-4,4))
rcParams['text.usetex'] = True

ion()


d = 30  # dimension
norm = stats.norm()  # single coordinate PDF
chi = stats.chi(d)  # PDF for coord. vector magnitude; mode @ sqrt(d-1)


# Set range depending on d.
x_u = sqrt(d-1) + 5*chi.std()
xvals = linspace(0., x_u, 300)  # doubles as coord value & vec distance

coord_pdf = 2*norm.pdf(xvals)  # double b/c plotting vs. abs. value
chi_pdf = chi.pdf(xvals)
area = 2*pi**(d/2.) * xvals**(d-1) / special.gamma(d/2.)

typ_l = chi.ppf(.05)
typ_u = chi.ppf(.95)
x_typ = linspace(typ_l, typ_u, 200)
chi_typ = chi.pdf(x_typ)

# Borrow axes setup from a matplolib example:
# https://matplotlib.org/stable/gallery/spines/multiple_yaxis_with_spines.html

# The main axis has a shared abscissa (x_i, distance), ord. for chi^2
fig, ax = plt.subplots(figsize=(8,5))
fig.subplots_adjust(bottom=.135, right=0.75)

ax_coord = ax.twinx()  # for single coord PDF
ax_shell = ax.twinx()  # for shell area vs. distance

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
ax_shell.spines.right.set_position(("axes", 1.2))

a = .7
p1, = ax.plot(xvals, chi_pdf, 'C0', lw=3, alpha=a, label=r'$\chi$ PDF')
p2, = ax_coord.plot(xvals, coord_pdf, 'C1', lw=3, alpha=a, label='Coord. PDF')
p3, = ax_shell.plot(xvals, area, 'C2', lw=3, alpha=a, label='Shell area')

# ax.axvline(typ_l, ls='--', alpha=.6)
# ax.axvline(typ_u, ls='--', alpha=.6)
ax.fill_between(x_typ, 0, chi_typ, color='C0', ec='none', alpha=0.3)

# Make some space for the legend above the curves.
ax.set_xlim(0., 1.05*x_u)
ax.set_ylim(0., 1.4*chi_pdf.max())
ax_coord.set_ylim(0., 1.4*coord_pdf.max())
ax_shell.set_ylim(0., 1.1*area.max())

ax.set(xlabel=r'$|\epsilon_i|,\, \chi = |\vec{\epsilon}|$', ylabel=r'$\chi$ PDF')
ax_coord.set(ylabel='Coord. PDF')
ax_shell.set(ylabel='Shell area')

ax.yaxis.label.set_color(p1.get_color())
ax_coord.yaxis.label.set_color(p2.get_color())
ax_shell.yaxis.label.set_color(p3.get_color())

ax.tick_params(axis='y', colors=p1.get_color())
ax_coord.tick_params(axis='y', colors=p2.get_color())
ax_shell.tick_params(axis='y', colors=p3.get_color())

ax.legend(handles=[p1, p2, p3], loc='upper left')

# savefig('MVNTypicalSet-30.pdf')

# Experiment with random draws.
draws = [norm.rvs(d) for i in range(100000)]
chis = [sqrt(sum(draw*draw)) for draw in draws]
cmax = [abs(draw).max() for draw in draws]

fig = figure(figsize=(7,5))
fig.subplots_adjust(bottom=.135)
xlabel(r'$\max(|\epsilon_i|),\, \chi = |\vec{\epsilon}|$')
ylabel('Estimated PDF')
ax.set_xlim(0., 1.05*x_u)

a = .6
hist(chis, 40, density=True, color='C0', alpha=a, label='Distance')
hist(cmax, 40, density=True, color='C5', alpha=a, label='Max coord.')

cdf = norm.cdf(xvals) - norm.cdf(-xvals)
xpdf = d*(2*norm.pdf(xvals))*cdf**(d-1)
plot(xvals, chi_pdf, 'C0', lw=1.5, alpha=a)
plot(xvals, xpdf, c='C5', lw=1.5, alpha=a)

legend()

# savefig('MVNTypicalSet-DistMax-30.pdf')
