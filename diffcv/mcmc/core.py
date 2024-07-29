import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import typing as tp


def sample_multichain(sampler: eqx.Module, dim: int, key: jax.random.PRNGKey, n_chains: int = 1000, init_std: float = 5):
    starter_points = jax.random.normal(key, shape=(n_chains, 1, dim)) * init_std
    starter_keys = jax.random.split(key, n_chains)

    _, samples = jax.vmap(sampler)(starter_points, starter_keys)
    samples = samples.reshape(-1, dim)
    return samples


def estimate_mcmc(fn: tp.Callable, sampler: eqx.Module, dim: int, key: jax.random.PRNGKey, n_chains: int = 1000, init_std: float = 5):
    samples = sample_multichain(sampler, dim, key, n_chains=n_chains, init_std=init_std)
    return jax.vmap(fn)(samples).mean()


def estimate_n_mcmc(fn: tp.Callable, sampler: eqx.Module, dim: int, key: jax.random.PRNGKey, n_chains: int = 100, init_std: float = 5, n_runs: int = 1000):
    estimates = []
    for exp_key in jax.random.split(key, n_runs):
        estimates.append(estimate_mcmc(fn, sampler, dim, exp_key, n_chains=n_chains, init_std=init_std))   
    estimates = jnp.stack(estimates)
    return estimates