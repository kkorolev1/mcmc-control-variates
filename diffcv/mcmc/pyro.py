import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp

from numpyro.infer import MCMC, HMC
from numpyro.infer.mcmc import MCMCKernel

from .base import Sampler


class PyroSampler(Sampler):
    def __init__(
        self,
        kernel: MCMCKernel,
        dim: int,
        n_samples: int = 1000,
        burnin_steps: int = 0,
        init_std: float = 1.0,
    ):
        super().__init__(
            dim=dim, n_samples=n_samples, burnin_steps=burnin_steps, init_std=init_std
        )
        self.kernel = kernel

    def __call__(self, key: jax.random.PRNGKey, n_chains: int = 1000):
        starter_points = (
            jax.random.normal(
                key, shape=(n_chains, self.dim) if n_chains > 1 else (self.dim,)
            )
            * self.init_std
        )
        mcmc = MCMC(
            self.kernel,
            num_samples=self.n_samples,
            num_warmup=self.burnin_steps,
            num_chains=n_chains,
            jit_model_args=True,
            progress_bar=False,
        )
        _, subkey = jax.random.split(key)
        mcmc.run(subkey, init_params=starter_points)
        samples = mcmc.get_samples(group_by_chain=True)
        return samples


class HMCSampler(PyroSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        n_samples: int = 1000,
        burnin_steps: int = 0,
        init_std: float = 1.0,
    ):
        potential_fn = jax.jit(lambda x: -log_prob(x).squeeze())
        kernel = HMC(
            potential_fn=potential_fn,
            step_size=gamma,
            adapt_step_size=True,
            dense_mass=False,
        )
        super().__init__(
            kernel=kernel,
            dim=dim,
            n_samples=n_samples,
            burnin_steps=burnin_steps,
            init_std=init_std,
        )
