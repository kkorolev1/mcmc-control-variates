import equinox as eqx
import jax
import jax.numpy as np
from jax import random


class LangevinDynamics(eqx.Module):
    gradient_func: eqx.Module
    n_samples: int = 1000
    gamma: float = 5e-3
    burnin_steps: int = 0
    
    @eqx.filter_jit
    def __call__(self, x, key: random.PRNGKey):
        """Callable implementation for sampling.

        Args:
            x (jax.ndarray): Data of shape (batch, ...)
            key (random.PRNGKey): PRNGKey for random draws
        """
        def langevin_step(prev_x, key: random.PRNGKey):
            """Scannable langevin dynamics step.

            Args:
                prev_x (jax.ndarray): Previous value of x in langevin dynamics step
                key (random.PRNGKey): PRNGKey for random draws
            """
            z = random.normal(key, shape=x.shape)
            new_x = prev_x + self.gamma * jax.vmap(self.gradient_func)(prev_x) + np.sqrt(2 * self.gamma) * z
            return new_x, prev_x
        keys = random.split(key, self.n_samples)
        final_xs, xs = jax.lax.scan(langevin_step, init=x, xs=keys)
        xs = np.vstack(xs)
        return final_xs, xs[self.burnin_steps:]
    
    @eqx.filter_jit
    def sample(self, x, key: random.PRNGKey):
        return self(x, key)
    
    