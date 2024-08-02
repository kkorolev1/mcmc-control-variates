import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import typing as tp

from .base import Sampler


class ULASampler(Sampler):  
    
      
    def __init__(self, grad_log_prob: tp.Callable, dim: int, gamma: float = 5e-3, n_samples: int = 1000, burnin_steps: int = 0, init_std: float = 1.0):
        super().__init__(dim=dim, n_samples=n_samples, burnin_steps=burnin_steps, init_std=init_std)
        self.grad_log_prob = grad_log_prob
        self.gamma = gamma
        
    
    @eqx.filter_jit
    def sample_chain(self, x, key: random.PRNGKey):
        """
        Args:
            x (jax.ndarray): Data of shape (batch, ...)
            key (random.PRNGKey): PRNGKey for random draws
        """
        def langevin_step(prev_x, key: random.PRNGKey):
            """Scannable langevin dynamics step.

            Args:
                prev_x (jax.ndarray): Previous value of x in langevin dynamics step
                key (random.PRNGKey): PRNGKey for random draws
            """
            z = random.normal(key, shape=x.shape)
            new_x = prev_x + self.gamma * jax.vmap(self.grad_log_prob)(prev_x) + jnp.sqrt(2 * self.gamma) * z
            return new_x, prev_x
        keys = random.split(key, self.n_samples + self.burnin_steps)
        final_xs, xs = jax.lax.scan(langevin_step, init=x, xs=keys)
        xs = jnp.vstack(xs)
        return final_xs, xs[self.burnin_steps:]

    @eqx.filter_jit
    def __call__(self, key: jax.random.PRNGKey, n_chains: int = 1):
        starter_points = jax.random.normal(key, shape=(n_chains, 1, self.dim)) * self.init_std
        starter_keys = jax.random.split(key, n_chains)
        _, samples = jax.vmap(self.sample_chain)(starter_points, starter_keys)
        return samples
