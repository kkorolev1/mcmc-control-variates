import jax
import jax.numpy as jnp
from jax import random


class Sampler:
    def __init__(self, dim: int, n_samples: int = 1000, burnin_steps: int = 0, init_std: float = 1.0):
        self.dim = dim
        self.n_samples = n_samples
        self.burnin_steps = burnin_steps
        self.init_std = init_std


    def __call__(self, key: jax.random.PRNGKey, n_chains: int = 1):
        raise NotImplementedError
    
