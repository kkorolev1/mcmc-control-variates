import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp
from tqdm.auto import tqdm

from .base import Sampler

class Estimator:
    def __init__(self, fn: tp.Callable, sampler: Sampler):
        self.fn = fn
        self.sampler = sampler
        
    def __call__(self, key: jax.random.PRNGKey, n_chains: int = 1000, n_estimates: int = 1, progress: bool = True):
        keys = jax.random.split(key, n_estimates)
        if progress:
            keys = tqdm(keys)
        estimates = []
        for exp_key in keys:
            samples = self.sampler(exp_key, n_chains=n_chains).squeeze(0)
            estimates.append(jax.vmap(self.fn)(samples).mean())
        estimates = jnp.stack(estimates)
        return estimates