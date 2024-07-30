import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl

from diffcv.mcmc import sample_multichain
from diffcv.utils import inf_loop


def get_data_from_sampler(batch_size: int, sampler: eqx.Module, key: jax.random.PRNGKey, n_chains: int, init_std: float):
    training_samples = sample_multichain(sampler, key, n_chains=n_chains, init_std=init_std)
    dataset = jdl.ArrayDataset(training_samples)
    dataloader = jdl.DataLoader(
        dataset,
        backend="jax",
        batch_size=batch_size,
        shuffle=True
    )
    dataloader = inf_loop(dataloader)
    return dataset, dataloader
