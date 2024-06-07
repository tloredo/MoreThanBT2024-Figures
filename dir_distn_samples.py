"""
Plot stacks of histograms depicting draws from a Dirichlet distribution.

Created 2010-06-25 by Tom Loredo
2024-06-07:  Update to Py-3 for iid22 paper
"""
from pylab import *
from scipy import *
from numpy.random.mtrand import dirichlet
from myplot import hline, shelves

ion()

# Set the number of bins and the Dirichlet concentration parameter:
nbins = 30
# 1 for flat, 0 for divisible
if 0:
    alphas = ones(nbins)
    #alphas = .5*ones(nbins)  # a case with mode at corners
else:
    alphas = 2 * ones(nbins) / nbins
centers = arange(nbins, dtype=float)/nbins + .5/nbins

ns = 10
samps = dirichlet(alphas, ns)

fmts = ['b-', 'g-', 'r-', 'c-', 'm-', 'k-']
dy = 0.
#delta = 2.5/nbins
delta = 1.
nf = len(fmts)


# Plot a stack of samples.
if 1:
    for i in range(ns):
        shelves(samps[i], fmt=fmts[i % nf], dy=dy)
        hline(dy)
        dy += delta
    
    xlim(0, 1)
    #savefig('Dir-Flat30.pdf')
