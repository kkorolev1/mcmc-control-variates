import jax
import jax.numpy as jnp
from jax import random


class Sampler:
    def __init__(
        self,
        dim: int,
        init_std: float = 1.0,
    ):
        self.dim = dim
        self.init_std = init_std

    def __call__(
        self,
        key: jax.random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        n_chains: int = 1,
        skip_steps: int = 1,
    ):
        raise NotImplementedError
