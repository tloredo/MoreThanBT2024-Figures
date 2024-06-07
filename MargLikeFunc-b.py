"""
Illustration of ingredients for approximating the marginal likelihood
function for a background parameter in a background "subtraction"
problem.

Created 2012-03-15 by Tom Loredo
2024-06-02ff Modified for iid22 paper
"""

from numpy import *
import matplotlib as mpl
from matplotlib.pyplot import *

# import myplot
# from scipy.stats import norm

# myplot.tex_on()

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



hpi = pi/2.
A_l = .95  # likelihood height
x_l = 7.  # likelihood peak loc'n
A_pri = 0.15  # prior height

def like(x, A=A_l, mu=x_l, sig=.5):
    return A*exp(-0.5*(x-mu)**2/sig**2)

# flat prior with smooth boundaries:
def prior(x, A=A_pri, l=0, u=10., s=.75):
    if (x-l) <= s:  # left transition
        return A*sin(hpi*(x-l)/s)
    elif (u-x) <= s:  # right transition
        return A*sin(hpi*(u-x)/s)
    else:  #  middle
        return A

# slowly varying bounded prior, starting w/ finite slope:
def prior(x, A=A_pri, l=0, u=10., s=.75):
    return A*(sin(pi*(x-l)/(u-l)))**.65

# slowly varying bounded prior, more smoothly bounded:
def prior(x, A=A_pri, l=0, u=10., s=.75):
    w = u - l
    return A*(.5 + .5*sin(2*pi*(x-l)/w - hpi))**.2



xvals = linspace(0, 10, 301)
lvals = like(xvals)
pvals = array([prior(x) for x in xvals])

fig = figure(figsize=(10,6), linewidth=2, frameon=False)
# Left & bottom spines only; from spine_placement_demo.html
ax = gca()
for loc, spine in ax.spines.items():
    if loc in ['right','top']:
        spine.set_color('none') # don't draw spine
    elif loc in ['left','bottom']:
        spine.set_linewidth(2)

# Colors for like, prior content:
pc = 'b'
#pc = '0.3'  # dark gray
# pc = (.5,0,0)  # dark red
lc = (0,.4,0)  # dark green

# Plot curves:
plot(xvals, lvals, ls='-', c=lc, lw=3)
plot(xvals, pvals, ls='--', c=pc, lw=3)

# Plot points:
# vlines([7], 0, .95, color='.5')  # gray line through max like
plot([x_l], [A_l], marker='o', mfc=lc, mew=0, ms=10)
plot([x_l], [prior(x_l)], marker='o', mfc=pc, mew=0, ms=10)
plot([x_l], [0], 'ko', mew=0, ms=10, clip_on=False)

# Optional arrows:
# lw = Arrow(6, .4, 1, 0., .1)
# pw = Arrow(.5, .2, .9, 0., .1)
# ax = gca()
# ax.add_patch(lw)
# ax.add_patch(pw)

# Annotate likelihood width:
annotate('', (6.4, .425), (7.6, .425), arrowprops=dict(arrowstyle='<->', edgecolor=lc))
text(7, .415, r'$\delta b_{s_1}$', ha='center', va='top', fontsize=20, color=lc)
#text(7.1, .405, r'$\delta b$', ha='left', va='top', fontsize=20, color=lc)

# Annotate prior width:
# annotate('',(.375, .1),(9.625, .1), arrowprops=dict(arrowstyle='<->'))
# text(5, .075, r'$\Delta b$', ha='center', va='top', fontsize=16)
# above curve:
# annotate('', (.6, .2), (9.4, .2), arrowprops=dict(arrowstyle='<->'))
#text(5, .22, r'$\Delta b$', ha='center', va='bottom', fontsize=20)
# below curve:
# (leave this out for marginal likelihood function; useful for Ockham)
# annotate('', (.6, .11), (9.4, .11), arrowprops=dict(arrowstyle='<->', edgecolor=pc))
# text(5, .1, r'$\Delta b_s$', ha='center', va='top', fontsize=20, color=pc)

# Label curves:
#text(6.5, .8, r'${\cal L}( b)$', ha='right', va='bottom', fontsize=20, color='b')
#text(1.8, .125, r'$\pi( b)$', ha='left', va='top', fontsize=20, color='k')

# Label points:
text(6.85, .96, r'${\cal L}_p(s_1) \equiv {\cal L}(s_1,\hat{b}_{s_1})$', ha='right', va='center', fontsize=20, color=lc)
#text(7.1, .16, r'$\pi(\hat{ b})$', ha='left', va='bottom', fontsize=20, color='k')
# This uses pi i/o p:
# text(7, .165, r'$\pi(\hat{b}_{s_1})$', ha='center', va='bottom', fontsize=20, color=pc)
text(7, .165, r'$p(\hat{b}_{s_1})$', ha='center', va='bottom', fontsize=20, color=pc)
text(7, -.03, r'$\hat{b}_{s_1}$', ha='center', va='top', fontsize=20, color='k')

xlabel(r'$ b$', labelpad=15, fontsize=32)
ylabel(r'$\pi,\;{\cal L}$', labelpad=15, fontsize=32)
ylim(0, 1.)
xticks([])
yticks([])

# savefig('MargLikeFunc-b.pdf')

