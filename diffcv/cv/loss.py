import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp
from functools import partial

from .generator import Generator


def l2_loss(x):
    return (x ** 2).mean()

def l2_reg(model: eqx.Module, alpha=0.1):
    return alpha * sum(l2_loss(w) for w in jax.tree_leaves(eqx.filter(model, eqx.is_array)))


class CVLoss:
    def __init__(self,  fn: tp.Callable, l2_alpha: float = 0.0):
        self.fn = fn
        self.l2_alpha = l2_alpha
    
    def __call__(self, model: eqx.Module, data: jnp.ndarray, batch_index: jnp.ndarray, key: jax.random.PRNGKey):
        g_vals = jax.vmap(model)(data)
        f_vals = jax.vmap(self.fn)(data)
        f_wave = f_vals - f_vals.mean()
        grad_g = jax.vmap(jax.jacfwd(model))(data)
        loss = -2 * g_vals * f_wave + (grad_g ** 2).sum(axis=-1)
        return loss.mean() + l2_reg(model, alpha=self.l2_alpha)


class DiffLoss:
    def __init__(self, fn: tp.Callable, grad_log_p: tp.Callable, noise_std: float = 1.0):
        self.fn = fn
        self.grad_log_p = grad_log_p
        self.noise_std = noise_std
    
    def __call__(self, model: eqx.Module, data: jnp.ndarray, batch_index: jnp.ndarray, key: jax.random.PRNGKey):
        generator = Generator(self.grad_log_p, model)        
        noise = self.noise_std * jax.random.normal(key, shape=data.shape)
        perturbed_data = data + noise
        return l2_loss(jax.vmap(self.fn)(perturbed_data) + jax.vmap(generator)(perturbed_data) - jax.vmap(self.fn)(data) - jax.vmap(generator)(data))


class CVALSLoss:    
    def __init__(self, fn: tp.Callable, grad_log_p: tp.Callable, switch_steps: int = 100, l2_alpha: float = 0.0):
        self.fn = fn
        self.grad_log_p = grad_log_p
        self.switch_steps = switch_steps
        self.l2_alpha = l2_alpha
    
    @staticmethod
    def grad_loss(inputs):
        data, generator, fn = inputs
        return l2_loss(jax.vmap(jax.jacfwd(fn))(data) + jax.vmap(jax.jacfwd(generator))(data))

    @staticmethod
    def cv_loss(inputs):
        model, data, generator, fn, l2_alpha = inputs
        g_vals = jax.vmap(model)(data)
        f_vals = jax.vmap(fn)(data)
        f_wave = f_vals - (f_vals + jax.vmap(generator)(jax.lax.stop_gradient(data))).mean()
        grad_g = jax.vmap(jax.jacfwd(model))(data)
        loss = -2 * g_vals * f_wave + (grad_g ** 2).sum(axis=-1)
        
        return loss.mean() + l2_reg(model, alpha=l2_alpha)
    
    def __call__(self, model: eqx.Module, data: jnp.ndarray, batch_index: jnp.ndarray):
        generator = Generator(self.grad_log_p, model)
        condition = (batch_index // self.switch_steps) % 2 == 1
        
        loss_value = jax.lax.cond(
            jnp.all(condition),
            lambda: CVALSLoss.grad_loss((data, generator, self.fn)),
            lambda: CVALSLoss.cv_loss((model, data, generator, self.fn, self.l2_alpha)),
        )
        
        return loss_value
