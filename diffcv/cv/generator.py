import jax
import jax.numpy as jnp
import equinox as eqx
import typing as tp
#from folx import forward_laplacian


class Generator:
    def __call__(self, x):
        raise NotImplementedError


class ScalarGenerator(Generator):
    """NN parametrises a scalar function, which requires evaluating trace of a hessian for a NN
    """
    grad_log_p: tp.Callable
    grad_g: tp.Callable
    hessian_g: tp.Callable
        
    def __init__(self, grad_log_p, g):
        self.grad_log_p = grad_log_p
        self.grad_g = jax.jacfwd(g)
        self.hessian_g = jax.hessian(g)
    
    @eqx.filter_jit
    def __call__(self, x):
        return jnp.dot(self.grad_log_p(x), self.grad_g(x)) + jnp.trace(self.hessian_g(x))


class VectorGenerator(Generator):
    """NN parametrises a gradient
    """
    grad_log_p: tp.Callable
    g: tp.Callable
    jacob_g: tp.Callable
        
    def __init__(self, grad_log_p, g):
        self.grad_log_p = grad_log_p
        self.g = g
        self.jacob_g = jax.jacfwd(g)
    
    @eqx.filter_jit
    def __call__(self, x):
        return jnp.dot(self.grad_log_p(x), self.g(x)) + jnp.trace(self.jacob_g(x))