from diffcv.mcmc import HMCSampler
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx

from tqdm import tqdm
import typing as tp
import pandas as pd

import sys
sys.path.append("/home/korolevki/mcmc-control-variates/diffcv")

import optax

sns.set_style("darkgrid")

import numpyro.distributions as D

DIST_DIM = 2 # Dimension of a Gaussian
N_CHAINS = 1 # Number of parallel chains for MCMC estimates
MCMC_N_RUNS = 1000 # Number of MCMC estimates to calculate CI
BATCH_SIZE = 128 # Batch size for CV training

dist = D.MultivariateNormal(loc=0, covariance_matrix=jnp.eye((DIST_DIM)))
grad_log_prob = jax.jit(jax.grad(dist.log_prob))

rng = jax.random.PRNGKey(50)
rng, key = jax.random.split(rng)
sampler = HMCSampler(log_p=dist.log_prob, dim=DIST_DIM, n_samples=10000, gamma=1e-1, burnin_steps=500)
samples = sampler(key, n_chains=1)