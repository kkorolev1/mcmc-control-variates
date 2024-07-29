import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp

from diffcv.utils import fill_diagonal

class GaussianMixture:
    
    def __init__(self, means: jnp.ndarray, covs: jnp.ndarray, coeffs=None):
        if coeffs is None:
            coeffs = jnp.full((means.shape[0]), 1 / means.shape[0], dtype=jnp.float32)

        assert means.shape[0] == covs.shape[0] == coeffs.shape[0]
        assert jnp.allclose(sum(coeffs), 1)
        
        if means.ndim == 1:
            means = means[:, None]
        
        if covs.ndim == 1:
            covs = covs.reshape(means.shape[0], 1).repeat(means.shape[1], axis=1)
        
        if covs.ndim < 3:
            new_covs = jnp.zeros((means.shape[0], means.shape[1], means.shape[1]), dtype=jnp.float32)
            covs = fill_diagonal(new_covs, covs)
        
        self.means = means
        self.covs = covs
        self.coeffs = coeffs

    def log_p(self, x):
        assert x.shape == self.means.shape[1:], f"{x.shape=} != {self.means.shape[1:]=}"
        log_ps = multivariate_normal.logpdf(x, self.means, self.covs)
        return logsumexp(log_ps, b=self.coeffs)
    
    def sample(self, n_samples, key):
        mixture_size = len(self.coeffs)
        mixture_component = random.randint(key, (n_samples, ), 0, mixture_size)
        key, subkey = random.split(key)
        samples = random.multivariate_normal(subkey, self.means, self.covs, shape=(n_samples, mixture_size))
        return jnp.take_along_axis(samples, mixture_component[:, None, None], axis=1).squeeze(axis=1)
        
    def mean(self):
        return jnp.dot(self.coeffs, self.means)
    
    def cov(self):
        mean = self.mean()
        mean_var = (self.coeffs.reshape((-1, 1, 1)) * self.covs).sum(axis=0)
        var_mean = (self.coeffs.reshape((-1, 1, 1)) * jnp.einsum("ij,ik->ijk", self.means - mean, self.means - mean)).sum(axis=0)
        return mean_var + var_mean
        
        