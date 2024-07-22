import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typing as tp
import jax_dataloader as jdl
from tqdm.auto import tqdm


def fit_cv(model: eqx.Module, 
           dataloader: jdl.DataLoader, 
           optimizer: optax.GradientTransformation, 
           loss: tp.Callable, 
           n_steps: int = 1000):
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    dloss = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    
    @eqx.filter_jit
    def step(model, data, opt_state, batch_index):
        loss_score, grads = dloss(model, data, batch_index)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_score
    
    loss_history = []
    for batch_index, batch in tqdm(zip(range(n_steps), dataloader), total=n_steps):
        if batch_index >= n_steps:
            break
        batch = batch[0] # dataloader returns tuple of size (1,)
        model, opt_state, loss_score = step(model, batch, opt_state, jnp.asarray(batch_index))
        loss_history.append(loss_score)
    return model, loss_history
