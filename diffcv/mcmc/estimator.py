import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp
from tqdm.auto import tqdm


def estimate_mcmc(fn: tp.Callable, sampler: eqx.Module, key: jax.random.PRNGKey, n_chains: int = 1000, init_std: float = 5):
    samples = sampler(key, n_chains=n_chains, init_std=init_std)
    return jax.vmap(fn)(samples).mean()


def estimate_n_mcmc(fn: tp.Callable, sampler: eqx.Module, key: jax.random.PRNGKey,
                    n_chains: int = 100, init_std: float = 5, n_runs: int = 1000, progress: bool = True):
    estimates = []
    keys = jax.random.split(key, n_runs)
    if progress:
        keys = tqdm(keys)
    for exp_key in keys:
        estimates.append(estimate_mcmc(fn, sampler, exp_key, n_chains=n_chains, init_std=init_std))   
    estimates = jnp.stack(estimates)
    return estimates