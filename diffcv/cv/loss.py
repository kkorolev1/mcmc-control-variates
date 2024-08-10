import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp

from .generator import ScalarGenerator


def l2_loss(x, y):
    return ((x - y) ** 2).mean()


def l2_reg(model: eqx.Module, alpha=0.1):
    """Regularization for linear layers"""
    return alpha * sum(
        [
            (w**2).sum()
            for w in jax.tree.leaves(eqx.filter(model, eqx.is_array))
            if w.ndim > 1
        ]
    )


class DiffusionLoss:
    def __init__(self, fn: tp.Callable, l2_alpha: float = 0.0):
        self.fn = fn
        self.l2_alpha = l2_alpha

    def __call__(
        self,
        model: eqx.Module,
        data: jnp.ndarray,
        key: jax.random.PRNGKey,
        fn_mean: jnp.ndarray,
    ):
        g_vals = jax.vmap(model)(data)
        f_vals = jax.vmap(self.fn)(data)
        f_wave = f_vals - fn_mean  # f_vals.mean(axis=-1)
        grad_g = jax.vmap(jax.grad(model))(data)
        loss = -2 * g_vals * f_wave + (grad_g**2).sum(axis=-1)
        return loss.mean()  # + l2_reg(model, alpha=self.l2_alpha)


class VarLoss:
    def __init__(
        self, fn: tp.Callable, grad_log_prob: tp.Callable, generator_cls=ScalarGenerator
    ):
        self.fn = fn
        self.grad_log_prob = grad_log_prob
        self.generator_cls = generator_cls

    def __call__(self, model: eqx.Module, data: jnp.ndarray, key: jax.random.PRNGKey):
        generator = self.generator_cls(self.grad_log_prob, model)
        batch_size = data.shape[0]
        residual = jax.vmap(self.fn)(data) + jax.vmap(generator)(data) - model.c
        return (residual**2).sum() / (batch_size - 1)


class DiffLoss:
    def __init__(
        self,
        fn: tp.Callable,
        grad_log_prob: tp.Callable,
        noise_std: float = 1.0,
        generator_cls=ScalarGenerator,
    ):
        self.fn = fn
        self.grad_log_prob = grad_log_prob
        self.noise_std = noise_std
        self.generator_cls = generator_cls

    def __call__(self, model: eqx.Module, data: jnp.ndarray, key: jax.random.PRNGKey):
        generator = self.generator_cls(self.grad_log_prob, model)
        noise = self.noise_std * jax.random.normal(key, shape=data.shape)
        perturbed_data = data + noise
        return l2_loss(
            jax.vmap(self.fn)(perturbed_data) + jax.vmap(generator)(perturbed_data),
            jax.vmap(self.fn)(data) + jax.vmap(generator)(data),
        )
