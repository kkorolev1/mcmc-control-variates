import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl


def get_data_from_sampler(
    batch_size: int, sampler: eqx.Module, key: jax.random.PRNGKey, n_chains: int
):
    training_samples = sampler(key, n_chains=n_chains)
    training_samples = training_samples.reshape(-1, training_samples.shape[-1])
    dataset = jdl.ArrayDataset(training_samples)
    dataloader = jdl.DataLoader(
        dataset, backend="jax", batch_size=batch_size, shuffle=True, drop_last=True
    )
    return dataset, dataloader
