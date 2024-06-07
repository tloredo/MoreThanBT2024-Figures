"""
Plot a collection of bivariate normal sampling distributions and the associated 
likelihood function.

2021-06-02 Created by Tom Loredo, for CASt Summer School
    (based on bvn_lab.py from BDA2020 course lab)
2024-06-07:  Modified for portability for iid22 paper
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from numpy import *
from numpy.random import rand
from scipy import stats
from scipy.stats import multivariate_normal

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# try:
#     import myplot
#     from myplot import close_all, csavefig
#     # myplot.tex_on()
#     csavefig.save = False
# except ImportError:
#     pass

# Customize plot properties:
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


class BivariateNormal:
    """
    Bivariate normal dist'n, including specification of conditionals and
    marginals.
    """

    def __init__(self, means, sigs, rho):
        """
        Define the BVN via its joint dist'n description in terms of marginal
        means, marginal std deviations, and the correlation coefficient.

        Parameters
        ----------
        means : 2-element float sequence
            Marginal means

        sigs : 2-element float sequence
            Marginal standard deviations

        rho : float
            Correlation coefficient
        """
        self.means = asarray(means)
        self.sigs = asarray(sigs)
        self.rho = rho
        cross = rho*sigs[0]*sigs[1]
        self.cov = array([[sigs[0]**2, cross], [cross, sigs[1]**2]])
        self.bvn = multivariate_normal(self.means, self.cov)

        # Conditional for y|x:
        self.slope_y = rho*sigs[1]/sigs[0]
        self.int_y = means[1] - self.slope_y*means[0]
        self.csig_y = sigs[1]*sqrt(1.-rho**2)

        # Conditional for x|y:
        self.slope_x = rho*sigs[0]/sigs[1]
        self.int_x = means[0] - self.slope_x*means[1]
        self.csig_x = sigs[0]*sqrt(1.-rho**2)

        # Standard normal, for samplers.
        self.std_norm = stats.norm()

    def y_x(self, x):
        """
        Return the conditional expectation for `y` given `x`.
        """
        return self.int_y + self.slope_y*x

    def x_y(self, y):
        """
        Return the conditional expectation for `x` given `y`.
        """
        return self.int_x + self.slope_x*y

    def pdf(self, xy):
        """
        Evaluate the PDF at the point(s) `xy`.
        """
        return self.bvn.pdf(asarray(xy))

    def log_pdf(self, xy):
        """
        Evaluate the log PDF at the point(s) `xy`.
        """
        return self.bvn.logpdf(asarray(xy))

    def sample(self, n=1):
        """
        Return an array of `n` samples from the BVN.
        """
        return self.bvn.rvs(n)

    def y_x_sample(self, x):
        """
        Return a sample of y from the conditional distribution for y
        given `x`.
        """
        return self.y_x(x) + self.csig_y*self.std_norm.rvs()

    def x_y_sample(self, y):
        """
        Return a sample of x from the conditional distribution for x
        given `y`.
        """
        return self.x_y(y) + self.csig_x*self.std_norm.rvs()

    def xy_grid(self, n, fac=5.):
        """
        Define vectors and 2-D arrays useful for plotting a BVN and related
        functions.
        """
        # Vectors of x and y values:
        x = linspace(self.means[0]-fac*self.sigs[0],
                     self.means[0]+fac*self.sigs[0], n)
        y = linspace(self.means[1]-fac*self.sigs[1],
                     self.means[1]+fac*self.sigs[1], n)
        # 2-D float arrays giving the x or y values over the grid;
        # note that these use 'xy' indexing, so grid element [j,i]
        # corresponds to (x[i], y[j]), i.e., a row is over x values.
        xg, yg = np.meshgrid(x, y)
        # 2-D array of 2-vectors giving (x,y) over the grid:
        xyg = empty(xg.shape + (2,))
        xyg[:,:,0] = xg
        xyg[:,:,1] = yg
        return x, y, xg, yg, xyg

    # Eli's conditionals from 2018:
    def y_x_pdf(self, x, yvals):
        """
        Evaluate the conditional PDF for y given x.
        """
        p_x = exp((-(x-self.means[0])**2)/(2*self.sigs[0]**2)) / (sqrt(2*pi*self.sigs[0]**2))
        points = [(x, i) for i in yvals]
        p_xy = self.pdf(points)
        return p_xy / p_x

    def x_y_pdf(self, y, xvals):
        """
        Evaluate the conditional PDF for x given y.
        """
        p_y = exp((-(y-self.means[1])**2)/(2*self.sigs[1]**2)) / (sqrt(2*pi*self.sigs[1]**2))
        points = [(i, y) for i in xvals]
        p_xy = self.pdf(points)
        return p_xy / p_y


ion()


# Sampling distribuions fig:
sdfig = figure(figsize=(8,6))
ax = sdfig.add_subplot(111, projection='3d')

ccolors = ('blue', 'darkgreen', 'firebrick', 'saddlebrown', 'dimgray',
           'k', 'k', 'k', 'k', 'k')

muvals = linspace(-3, 3., 5)
logz = -log(2*pi)  # log norm constant
# Relative levels (wrt mode) for CL = 68.3%, 95.4%, 99.73%, 99.994%, 99.99994%:
rlevels = [-14.372, -9.667, -5.915, -3.090, -1.148]
for mu in muvals:
    # Shifted, standard bivar normal (uncorrelated):
    bvn = BivariateNormal([mu,mu], [1.,1.], 0.)
    x, y, xg, yg, xyg = bvn.xy_grid(50, 6.)
    lpdf = bvn.log_pdf(xyg) - logz
    # cset = ax.contour(xg, yg, lpdf, offset=mu, levels=rlevels,
    #                   cmap=cm.coolwarm, alpha=.9)
    cset = ax.contour(xg, yg, lpdf, offset=mu, levels=rlevels,
                      colors=ccolors, alpha=.9)

x1, x2 = 0.8, 1.1
ax.scatter([x1], [x2], muvals[:1], marker='o', s=30)
ax.plot([x1, x1], [x2, x2], [muvals[0], muvals[-1]], '-k', lw=2)

ax.set_xlabel(r'$x_1$', labelpad=10)
ax.set_ylabel(r'$x_2$', labelpad=10)
ax.set_zlabel(r'$\mu$', labelpad=10)

ticks = [-8, -4., 0., 4., 8.]
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xlim3d(-8, 8)
ax.set_ylim3d(-8, 8)
ax.set_zlim3d(muvals[0], muvals[-1])


# Likelihood function fig:
lffig = figure(figsize=(8,5))
ax = lffig.add_subplot(111)

data = array([x1,x2])
muvals = linspace(muvals[0], muvals[-1], 100)
lvals = empty_like(muvals)
for i, mu in enumerate(muvals):
    bvn = BivariateNormal([mu,mu], [1.,1.], 0.)
    lvals[i] = bvn.pdf(data)

ax.plot(muvals, lvals, '-', lw=2)

ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\mathcal{L}(\mu)$')
