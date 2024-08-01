import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp

from numpyro.infer import MCMC, HMC
from numpyro.infer.mcmc import MCMCKernel

from .base import Sampler


class PyroSampler(Sampler):
    
    
    def __init__(self, kernel: MCMCKernel, dim: int, n_samples: int = 1000, burnin_steps: int = 0):
        super().__init__(dim=dim, n_samples=n_samples, burnin_steps=burnin_steps)
        self.kernel = kernel
    
    @eqx.filter_jit
    def __call__(self, key: jax.random.PRNGKey, n_chains: int = 1000, init_std: float = 5):
        starter_points = jax.random.normal(key, shape=(n_chains, self.dim)) * init_std
        mcmc = MCMC(self.kernel, num_samples=self.n_samples, num_warmup=self.burnin_steps,
                    num_chains=n_chains, jit_model_args=True)
        _, subkey = jax.random.split(key)
        mcmc.run(subkey, init_params=starter_points)
        chains = mcmc.get_samples(group_by_chain=True)
        return chains["points"]
    

class HMCSampler(PyroSampler):
    def __init__(self, log_p: tp.Callable, dim: int,
                gamma: float = 5e-3, n_samples: int = 1000, burnin_steps: int = 0):
        energy_func = jax.jit(lambda x: -log_p(x))
        kernel = HMC(
            potential_fn=energy_func, step_size=gamma,
            num_steps=5, adapt_step_size=True, dense_mass=False
        )
        super().__init__(kernel=kernel, dim=dim, n_samples=n_samples, burnin_steps=burnin_steps)
