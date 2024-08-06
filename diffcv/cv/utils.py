import jax
from jax import nn


@jax.jit
def requ(x):
    return nn.relu(x) ** 2


@jax.jit
def recu(x):
    return nn.relu(x) ** 3
