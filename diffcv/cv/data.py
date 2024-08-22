import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl
from dataclasses import asdict

from diffcv.mcmc.base import Sampler
from diffcv.config import SamplingConfig


def get_data_from_sampler(
    key: jax.random.PRNGKey,
    batch_size: int,
    sampler: Sampler,
    total_samples: int,
    sampling_config: SamplingConfig,
    shuffle: bool = True,
):
    n_chains = int(total_samples / sampling_config.steps)
    samples = sampler(key, **asdict(sampling_config), n_chains=n_chains)
    samples = samples.reshape(-1, samples.shape[-1])[:total_samples]
    dataset = jdl.ArrayDataset(samples)
    dataloader = jdl.DataLoader(
        dataset, backend="jax", batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return dataloader
