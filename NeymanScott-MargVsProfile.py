"""
Contrast handling nuisance parameters via marginalization vs. profile likelihood
using the Neyman-Scott problem as an example.

Created 2024-06-03 by Tom Loredo, translated from Fortran code from 1993-03-03
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from scipy import stats

# try:
#     import myplot
#     from myplot import close_all, csavefig
#     myplot.tex_on()
#     csavefig.save = False
# except ImportError:
#     pass

from matplotlib import rc

# Customize plot properties:
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

rt2 = sqrt(2.)
twopi = 2*pi

# Setup a NumPy new-style RNG to make plot reproducible.
# rng = np.random.default_rng(seed=19271937)
rng = np.random.default_rng(seed=19371927)  # used for iid22 paper


def sigmas2p(nu):
    """
    Return the probability within `nu` standard deviations of the mean
    for a normal distribution.  This is the complement of `sigmas2tp`.
    """
    # return 1. - sigmas2tp(nu)
    return 2.*stats.norm.cdf(nu) - 1.

# Probabilities vs. umbers of std dev'ns:
p_sigmas = [sigmas2p(n) for n in range(1,6)]
# Simple fractions, percentages:
p_simp = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def dll_wilks(ndim, c):
    """
    Calculate the delta-log-likelihood value defining an asymptotic
    `ndim`-dimensional confidence region with confidence level c,
    based on Wilks's theorem.

    The delta-log-likelihood value is the magnitude of the delta;
    it is positive.
    """
    c = asarray(c)
    return 0.5*stats.chi2.ppf(c, ndim)

dll_sigmas = dll_wilks(2, p_sigmas)
dll_simp = dll_wilks(2, p_simp)


class Grid2D:
    """
    Utility class for computing and plotting a function of 2 variables 
    using a grid.
    """
 
    # Contour levels for a bivariate normal log(PDF) enclosing probabilities
    # 68.3%, 95.4%, 99.73%, 99.99% (in reverse):
    bvn_levels = - flip(dll_simp)

    def __init__(self, xrange, nx, yrange, ny):
        """
        Define the grid.

        Parameters
        ----------
        xvals : 2-element float sequence
            Grid locations for the first variable.

        yvals : 2-element float sequence
            Grid locations for the second variable.
        """
        self.x_l, self.x_u = xrange
        self.y_l, self.y_u = yrange
        self.nx, self.ny = nx, ny

        # Vectors of x and y values:
        self.x = linspace(self.x_l, self.x_u, nx)
        self.y = linspace(self.y_l, self.y_u, ny)

        # 2-D float arrays giving the x or y values over the grid;
        # note that these use 'xy' indexing, so grid element [j,i]
        # corresponds to (x[i], y[j]), i.e., a row is over x values.
        self.xg, self.yg = np.meshgrid(self.x, self.y, indexing='xy')
 
        # 2-D array of 2-vectors giving (x,y) over the grid:
        self.xyg = empty(self.xg.shape + (2,))
        self.xyg[:,:,0] = self.xg
        self.xyg[:,:,1] = self.yg

    def contour(self, ax, func, levels, sub_max=False, div_max= False, 
        clab=False, aspect=None):
        """
        Plot contours of the 2-vector argument function `func` on the
        specified axes.

        Parameters
        ----------
        ax : mpl axes instance
            The axes to use for the plot

        func : function
            Function of (x, y) (where the arguments may be arrays of matching
            shape)

        levels : sequence of floats OR string
            Contour levels.  If `levels` = 'bvn' or 'BVN' (strings, for
            bivariate normal), use a default set of levels corresponding
            to regions of a bivariate normal with probabilities
            0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999.

        sub_max : boolean
            If not False, interpret `levels` as offsets with respect to the
            maximum value of the function.  If `sub_max` is a float, it is
            used as the maximum value; otherwise, the max on the grid is used.

        div_max : boolean
            If not False, interpret `levels` as offsets with respect to the
            the function divided by its maximum value.  If `div_max` is a
            float, it is used as the maximum value; otherwise, the max on the
            grid is used.

        aspect : float
            If provided, use to specify the aspect ratio.  This can be useful
            for preserving slopes of contours when the function is a PDF.
            E.g., for a bivariate normal, set aspect = sig_x/sig_y to ensure
            that the eliptical contours appear at a 45 deg angle, and
            regression lines exhibit the right relationshipt to the elipses.
            This choice will make an independent BVN have circular contours,
            regardless of the values of sig_x and sig_y.
        """
        fvals = func(self.xg, self.yg)
        if sub_max is not False:
            if isinstance(sub_max, float):
                fvals = fvals - sub_max
            else:
                fvals = fvals - fvals.max()
        if div_max is not False:
            if isinstance(div_max, float):
                fvals = fvals/div_max
            else:
                fvals = fvals/fvals.max()

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

        if aspect is not None:
            ax.set_aspect(aspect)

        if levels in ['bvn', 'BVN']:
            levels = self.bvn_levels
        cont = ax.contour(self.xg, self.yg, fvals, levels,
            colors=['gray', 'b', 'g', 'r'])
        if clab:
            clabel(cont, inline=1, fontsize=10)


# True normal dist'n parameters for the example:
mu_t =  2.5
sig_t =  1.
norm_t = stats.norm(0., sig_t)  # offset from mu_t

n_pairs = 50

# Grid info for bivariate plots:
mu_l, mu_u =  0., 5.
n_mu = 100
sig_l, sig_u = .01, 3.
n_sig =  100

# Plot the joint likelihood for a single pair (random or hard-wired):
if False:  # random case
    x, y = norm_t.rvs(2, random_state=rng) + mu_t
else:  # symmetric about the true mean
    x = mu_t - sig_t/rt2
    y = mu_t + sig_t/rt2

# Moment estimators (marginal modes for flat priors?):
m = 0.5 * (x + y)
ss = 0.5 * (x - y)**2

# Likelihood at MLE:
sig2 = 0.5 * ss
like_max = exp(-0.5*ss/sig2) / (sig2 * twopi)

# Likelihood function for single pair over (sig, mu):
grid = Grid2D((sig_l, sig_u), n_sig, (mu_l, mu_u), n_mu)

def pair_llike(sigs, mus, mu_hat=m, ss=ss):
    sigs2 = sigs**2
    like = exp(-(mus - mu_hat)**2/sigs2) * exp(-0.5*ss/sigs2) / (twopi*sigs2)
    return log(like)

# fig, axl = subplots(figsize=(8,6))
fig, (axl, axr) = subplots(1, 2, figsize=(13,5))
# subplots_adjust(left=0.1)

sca(axl)
xlabel(r'$\sigma$')
ylabel(r'$\mu$, $\;(x,y)$')

grid.contour(axl, pair_llike, 'bvn', sub_max=log(like_max))
axvline(sig_t, ls='--', c='k', alpha=.3)
xlim(0, 3)

# Show the (x,y) pair near the mu axis.  This isn't really the right
# space, but the scale and units match.
plot([.1,.1], [x,y], 'ko')


# Marginal and profile likelihood functions:
mlike_max = exp(-0.5) / (2.*sqrt(twopi)*sqrt(ss))
sigs = linspace(sig_l, sig_u, 300)
plike = exp(-0.5*ss/sigs**2) / (twopi * sigs**2) / like_max
mlike = exp(-0.5*ss/sigs**2) / (2.*sqrt(twopi)*sigs) / mlike_max

# fig, ax = subplots(figsize=(8,5))
sca(axr)
xlim(0, 3)
xlabel(r'$\sigma$')
ylabel(r'$\mathcal{L}_m, \mathcal{L}_p$')

plot(sigs, mlike, ls='-', c='C0', lw=2, alpha=.8, label='Marginal')
plot(sigs, plike, ls='-', c='C1', lw=2, alpha=.8, label='Profile')
legend()


# Simulate multiple pairs; compare marginal vs. profile.

# Generate additional mus from a broad normal population dist'n or
# broad uniform; it doesn't really matter for inference of sigma.
if False:
    mus = stats.norm(3., 10.).rvs(n_pairs-1, random_state=rng)
else:
    mus = stats.uniform(loc=-5., scale=15.).rvs(n_pairs-1, random_state=rng)

# Generate pairs, keeping track of the sufficient statistics for sigma.
for mu in mus:
    x, y = mu + norm_t.rvs(2, random_state=rng)
    ss += 0.5*(x-y)**2

# Compute marginal, profile.
sig_hat = sqrt(ss / n_pairs)
mlike_max = exp(-0.5*ss/sig_hat**2) / (2.*sqrt(twopi)*sig_hat)**n_pairs
sig_hat = sqrt(0.5*ss / n_pairs)
like_max = exp(-0.5*ss/sig_hat**2) / (twopi * sig_hat**2)**n_pairs
# Scale these down for visibility.
mlike = .5*exp(-0.5*ss/sigs**2) / (2.*sqrt(twopi)*sigs)**n_pairs / mlike_max
plike = .5*exp(-0.5*ss/sigs**2) / (twopi * sigs**2)**n_pairs / like_max
plot(sigs, mlike, ls='--', c='C0', lw=2, alpha=.8, label='Marginal')
plot(sigs, plike, ls='--', c='C1', lw=2, alpha=.8, label='Profile')

axvline(sig_t, ls='--', c='k', alpha=.3)

annot = dict(horizontalalignment='left', verticalalignment='center',
    fontsize=18)
axr.text(1.2, .8, '1 pair', **annot)
axr.text(.66, .57, '%i pairs' % n_pairs, **annot)


fig.tight_layout()
# savefig('NeymanScott-JointMargProfile.pdf')


# BVN test of Grid2D:
if False:
    fig, ax = subplots(figsize=(8,8))

    def bvn(x, y):
        logp = ((x - 4)/2.)**2 + ((y - 2)/.5)**2
        return -0.5*logp

    means = array([4., 2.])
    sigs = array([2., .5])
    def bvn2(xy, means=means, sigs=sigs):
        dxy = (xy - means)/sigs
        return -0.5*dxy**2

    grid = Grid2D((0., 10.), 100, (-2., 6.), 100)
    grid.contour(ax, bvn, 'bvn', aspect=None)

