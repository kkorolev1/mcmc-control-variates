import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl


def get_data_from_sampler(
    batch_size: int,
    sampler: eqx.Module,
    key: jax.random.PRNGKey,
    n_chains: int,
    shuffle: bool = True,
):
    samples = sampler(key, n_chains=n_chains)
    samples = samples.reshape(-1, samples.shape[-1])
    dataset = jdl.ArrayDataset(samples)
    dataloader = jdl.DataLoader(
        dataset, backend="jax", batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return dataloader
