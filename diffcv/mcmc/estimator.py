import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp
from tqdm.auto import tqdm
from dataclasses import asdict
import math

from .base import Sampler
from diffcv.config import EstimatorConfig


class Estimator:
    def __init__(self, fn: tp.Callable, sampler: Sampler):
        self.fn = fn
        self.sampler = sampler

    def __call__(
        self,
        key: jax.random.PRNGKey,
        config: EstimatorConfig,
        progress: bool = True,
    ):
        keys = jax.random.split(key, config.n_estimates)
        if progress:
            keys = tqdm(keys)
        estimates = []
        n_chains = math.ceil(config.total_samples / config.sampling_config.steps)
        for exp_key in keys:
            samples = self.sampler(
                exp_key, **asdict(config.sampling_config), n_chains=n_chains
            )
            samples = samples.reshape(-1, samples.shape[-1])[: config.total_samples]
            estimates.append(jax.vmap(self.fn)(samples).mean())
        estimates = jnp.stack(estimates)
        return estimates

    @staticmethod
    def bias(true_pi, estimates):
        return (true_pi - estimates.mean()).item()

    @staticmethod
    def std(estimates):
        return estimates.std().item()
