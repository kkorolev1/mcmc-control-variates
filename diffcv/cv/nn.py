import equinox as eqx
import jax
from jax import nn, random
import jax.numpy as jnp

import typing as tp


@jax.jit
def identity(x):
    return x


def normal_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    return jax.random.normal(key, shape=weight.shape)


def init_params(
    model: eqx.Module,
    get_params: tp.Callable,
    init_fn: tp.Callable,
    key: jax.random.PRNGKey,
):
    params = get_params(model)
    new_params = [
        init_fn(param, subkey)
        for param, subkey in zip(params, jax.random.split(key, len(params)))
    ]
    new_model = eqx.tree_at(get_params, model, new_params)
    return new_model


def init_linear(
    model: eqx.Module,
    key: jax.random.PRNGKey,
    init_weights_fn: tp.Callable,
    init_biases_fn: tp.Callable = None,
):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x) and hasattr(x, "bias")
    ]
    wkey, bkey = jax.random.split(key, 2)
    new_model = init_params(model, get_weights, init_weights_fn, wkey)
    if init_biases_fn is None:
        init_biases_fn = init_weights_fn
    new_model = init_params(model, get_biases, init_biases_fn, bkey)
    return new_model


class CVMLP(eqx.Module):
    mlp: eqx.Module

    def __init__(
        self,
        in_size: int = 1,
        out_size: int = 1,
        width_size: int = 0,
        depth: int = 0,
        activation: tp.Callable = nn.elu,
        final_activation: tp.Callable = identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: random.PRNGKey,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, x):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        """
        return self.mlp(x).squeeze()


class CVLinear(CVMLP):
    def __init__(
        self,
        in_size=1,
        out_size=1,
        use_bias=True,
        *,
        key: random.PRNGKey,
    ):
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            width_size=0,
            depth=0,
            key=key,
            use_final_bias=use_bias,
        )
