import equinox as eqx
import jax
import jax.numpy as np
import typing as tp


def cv_loss(cv_model: eqx.Module, fn: tp.Callable, data: np.ndarray) -> float:
    gx = jax.vmap(cv_model)(data)
    fx = jax.vmap(fn)(data)
    f_wave = fx - fx.mean()
    grad_g = jax.vmap(jax.jacfwd(cv_model))(data)
    loss = -2 * gx * f_wave + (grad_g ** 2).sum(axis=-1)
    return loss.mean()
