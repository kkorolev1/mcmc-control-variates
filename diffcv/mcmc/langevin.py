import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import typing as tp

from .base import Sampler


class LangevinSampler(Sampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        init_std: float = 1.0,
        step: str = "ula",
    ):
        super().__init__(
            dim=dim,
            init_std=init_std,
        )
        self.log_prob = log_prob
        self.grad_log_prob = jax.jit(jax.grad(log_prob))
        self.gamma = gamma
        self.step = step

    @eqx.filter_jit
    def sample_chain(
        self,
        x: jnp.ndarray,
        key: random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        skip_steps: int = 1,
    ):
        def ula_step(prev_x, key: random.PRNGKey):
            z = random.normal(key, shape=x.shape)
            new_x = (
                prev_x
                + self.gamma * self.grad_log_prob(prev_x)
                + jnp.sqrt(2 * self.gamma) * z
            )
            return new_x, prev_x

        def mala_step(prev_x, key: random.PRNGKey):
            step_key, proposal_key = jax.random.split(key, 2)
            new_x, _ = ula_step(prev_x, step_key)
            log_pi_diff = self.log_prob(new_x) - self.log_prob(prev_x)
            log_new_prev = (
                (new_x - prev_x - self.gamma * self.grad_log_prob(prev_x)) ** 2
            ).sum()
            log_prev_new = (
                (prev_x - new_x - self.gamma * self.grad_log_prob(new_x)) ** 2
            ).sum()
            p = jnp.exp(log_pi_diff + (-log_prev_new + log_new_prev) / (4 * self.gamma))
            u = jax.random.uniform(proposal_key)
            return jax.lax.cond(
                u <= p,
                lambda new_x, prev_x: (new_x, prev_x),
                lambda _, prev_x: (prev_x, prev_x),
                new_x,
                prev_x,
            )

        keys = random.split(key, skip_steps * steps + burnin_steps)
        if self.step == "ula":
            step_fn = ula_step
        elif self.step == "mala":
            step_fn = mala_step
        else:
            raise NotImplementedError(f"Unknown step function {self.step}")
        _, xs = jax.lax.scan(step_fn, init=x, xs=keys)
        xs = jnp.vstack(xs)
        return xs[burnin_steps:][::skip_steps]

    @eqx.filter_jit
    def __call__(
        self,
        key: jax.random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        n_chains: int = 1,
        skip_steps: int = 1,
    ):
        key1, key2 = jax.random.split(key, 2)
        starter_points = (
            jax.random.normal(key1, shape=(n_chains, self.dim)) * self.init_std
        )
        starter_keys = jax.random.split(key2, n_chains)
        samples = jax.vmap(self.sample_chain, in_axes=(0, 0, None, None, None))(
            starter_points, starter_keys, steps, burnin_steps, skip_steps
        )
        return samples


class ULASampler(LangevinSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        init_std: float = 1.0,
    ):
        super().__init__(
            log_prob=log_prob,
            dim=dim,
            gamma=gamma,
            init_std=init_std,
            step="ula",
        )


class MALASampler(LangevinSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        init_std: float = 1.0,
    ):
        super().__init__(
            log_prob=log_prob,
            dim=dim,
            gamma=gamma,
            init_std=init_std,
            step="mala",
        )
