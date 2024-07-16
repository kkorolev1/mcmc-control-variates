import jax.numpy as np
from jax import random
from jax.scipy.stats import multivariate_normal

from diffcv.utils import fill_diagonal

class GaussianMixture:
    
    def __init__(self, means: np.ndarray, covs: np.ndarray, coeffs=None):
        if coeffs is None:
            coeffs = np.full((means.shape[0]), 1 / means.shape[0], dtype=np.float32)

        assert means.shape[0] == covs.shape[0] == coeffs.shape[0]
        assert np.allclose(sum(coeffs), 1)
        
        if means.ndim == 1:
            means = means[:, None]
        
        if covs.ndim == 1:
            covs = covs.reshape(means.shape[0], 1).repeat(means.shape[1], axis=1)
        
        if covs.ndim < 3:
            new_covs = np.zeros((means.shape[0], means.shape[1], means.shape[1]))
            covs = fill_diagonal(new_covs, covs)
        
        self.means = means
        self.covs = covs
        self.coeffs = coeffs

    def log_p(self, x):
        assert x.shape == self.means.shape[1:], f"{x.shape=} != {self.means.shape[1:]=}"
        ps = multivariate_normal.pdf(x, self.means, self.covs)
        return np.log(np.dot(ps, self.coeffs))
    
    def sample(self, n_samples, key):
        mixture_size = len(self.coeffs)
        mixture_component = random.randint(key, (n_samples, ), 0, mixture_size)
        samples = random.multivariate_normal(key, self.means, self.covs, shape=(n_samples, mixture_size))
        return np.take_along_axis(samples, mixture_component[:, None, None], axis=1).squeeze(axis=1)
        
