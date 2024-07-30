import equinox as eqx
import jax
from jax import nn, random
import jax.numpy as jnp

class CVMLP(eqx.Module):

    mlp: eqx.Module

    def __init__(
        self,
        in_size=1,
        out_size=1,
        width_size=4096,
        depth=1,
        activation=nn.elu,
        use_final_bias=True,
        key=random.PRNGKey(50),
    ):
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
            use_final_bias=use_final_bias
        )

    @eqx.filter_jit
    def __call__(self, x):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        """
        y = self.mlp(x).squeeze()
        return y


class CVLinear(CVMLP):
    def __init__(
        self,
        in_size=1,
        out_size=1,
        bias=True,
        key=random.PRNGKey(50),
    ):
        super().__init__(in_size=in_size, out_size=out_size, depth=0, key=key, use_final_bias=bias)

