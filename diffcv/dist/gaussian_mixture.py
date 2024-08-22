import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp

from diffcv.utils import fill_diagonal


class GaussianMixture:

    def __init__(
        self,
        component_means: jnp.ndarray,
        component_covs: jnp.ndarray,
        mixin_coeffs=None,
    ):
        if mixin_coeffs is None:
            mixin_coeffs = jnp.full(
                (component_means.shape[0]),
                1 / component_means.shape[0],
                dtype=jnp.float32,
            )

        assert (
            component_means.shape[0] == component_covs.shape[0] == mixin_coeffs.shape[0]
        )
        assert jnp.allclose(sum(mixin_coeffs), 1)

        if component_means.ndim == 1:
            component_means = component_means[:, None]

        if component_covs.ndim == 1:
            component_covs = component_covs.reshape(component_means.shape[0], 1).repeat(
                component_means.shape[1], axis=1
            )

        if component_covs.ndim < 3:
            new_covs = jnp.zeros(
                (
                    component_means.shape[0],
                    component_means.shape[1],
                    component_means.shape[1],
                ),
                dtype=jnp.float32,
            )
            component_covs = fill_diagonal(new_covs, component_covs)

        self.component_means = component_means
        self.component_covs = component_covs
        self.mixin_coeffs = mixin_coeffs

    def log_prob(self, x):
        log_ps = multivariate_normal.logpdf(
            x, self.component_means, self.component_covs
        )
        return logsumexp(log_ps, b=self.mixin_coeffs)

    def sample(self, key: random.PRNGKey, n_samples: int):
        mixture_size = len(self.mixin_coeffs)
        key1, key2 = random.split(key)
        mixture_component = random.randint(key1, (n_samples,), 0, mixture_size)
        samples = random.multivariate_normal(
            key2,
            self.component_means,
            self.component_covs,
            shape=(n_samples, mixture_size),
        )
        return jnp.take_along_axis(
            samples, mixture_component[:, None, None], axis=1
        ).squeeze(axis=1)

    @property
    def mean(self):
        return jnp.dot(self.mixin_coeffs, self.component_means)

    @property
    def cov(self):
        mean = self.mean
        mean_var = (self.mixin_coeffs.reshape((-1, 1, 1)) * self.component_covs).sum(
            axis=0
        )
        var_mean = (
            self.mixin_coeffs.reshape((-1, 1, 1))
            * jnp.einsum(
                "ij,ik->ijk", self.component_means - mean, self.component_means - mean
            )
        ).sum(axis=0)
        return mean_var + var_mean

    @property
    def variance(self):
        return jnp.diag(self.cov)
