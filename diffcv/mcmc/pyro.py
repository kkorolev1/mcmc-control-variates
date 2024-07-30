import jax
import jax.numpy as jnp
from jax import random
import typing as tp

import pyro.infer
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

from .base import Sampler


class PyroSampler(Sampler):
    
    
    def __init__(self, kernel: MCMCKernel, dim: int, n_samples: int = 1000, burnin_steps: int = 0):
        super().__init__(dim=dim, n_samples=n_samples, burnin_steps=burnin_steps)
        self.kernel = kernel
    
    
    def __call__(self, key: jax.random.PRNGKey, n_chains: int = 1000, init_std: float = 5):
        starter_points = jax.random.normal(key, shape=(n_chains, self.dim)) * init_std
        initial_params = {"points": starter_points}
        mcmc = pyro.infer.MCMC(self.kernel, num_samples=self.n_samples, warmup_steps=self.burnin_steps,
                    num_chains=n_chains, initial_params=initial_params, mp_context="spawn")
        mcmc.run()
        chains = mcmc.get_samples(group_by_chain=True)

        return chains["points"].squeeze()
    

class HMC(PyroSampler):
    def __init__(self, log_p: tp.Callable, dim: int,
                gamma: float = 5e-3, n_samples: int = 1000, burnin_steps: int = 0):
        energy_func = jax.jit(lambda x: -log_p(x))
        kernel = pyro.infer.HMC(
            potential_fn=energy_func, step_size=gamma,
            num_steps=5, adapt_step_size=True, full_mass=False
        )
        super().__init__(kernel=kernel, dim=dim, n_samples=n_samples, burnin_steps=burnin_steps)
