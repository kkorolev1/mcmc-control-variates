import equinox as eqx
import jax
import jax.numpy as np
import typing as tp


def l2_loss(x):
    return (x ** 2).mean()

def l2_reg(model: eqx.Module, alpha=0.1):
    return alpha * sum(l2_loss(w) for w in jax.tree_leaves(eqx.filter(model, eqx.is_array)))

def cv_loss(model: eqx.Module, fn: tp.Callable, data: np.ndarray) -> float:
    gx = jax.vmap(model)(data)
    fx = jax.vmap(fn)(data)
    f_wave = fx - fx.mean()
    grad_g = jax.vmap(jax.jacfwd(model))(data)
    struct_risk = -2 * gx * f_wave + (grad_g ** 2).sum(axis=-1)
    return struct_risk.mean() + l2_reg(model)
