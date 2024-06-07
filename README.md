# MoreThanBT2024-Figures
This repo hosts Python and R scripts to produce figures from Loredo &amp; Wolpert (2024), "Bayesian inference: More than Bayes's theorem."

Scripts:

* BinomialDistnLikelihoodFunc.R — 3D plot showing a binomial distribution sampling distribution (PMF) and likelihood functions associated with various possible observed counts
* MVNTypicalSet.py — Plots elucidating the nature of typical sets for a 30-D multivariate normal distribution
* MargLikeFunc-b.py — Depicts ingredients for approximating the marginal likelihood
  function for a background parameter in a background "subtraction"
  problem
* NeymanScott-MargVsProfile.py — Produces two plots illustrating the Neyman-Scott problem, comparing the behavior of profile and marginal likelihood functions
* dir_distn_samples.py — Plot stacks of histograms depicting draws from a Dirichlet distribution (the paper uses three such plots, with different choices for the concentration parameter)
* norm_samp_like.py — Plot a collection of bivariate normal sampling distributions and the associated 
  likelihood function



The Python scripts were developed using a `py310astro` Conda environment defined as follows (this is a general environment for astronomical computation; not all of these packages are necessary for the scripts):

```python
$ mamba create --name py310astro -c conda-forge -c defaults python=3.10 ipython jupyter scipy matplotlib \
  h5py beautifulsoup4 html5lib bleach pandas sortedcontainers \
  pytz setuptools mpmath bottleneck jplephem healpy asdf pyarrow colorcet hypothesis
```

