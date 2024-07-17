import equinox as eqx
import jax
import jax.numpy as np
import optax
import typing as tp
import jax_dataloader as jdl
from tqdm.auto import tqdm


def fit_cv(cv_model: eqx.Module, 
           fn: tp.Callable, 
           dataloader: jdl.DataLoader, 
           optimizer: optax.GradientTransformation, 
           loss: tp.Callable, 
           n_steps: int = 1000):
    
    opt_state = optimizer.init(eqx.filter(cv_model, eqx.is_array))
    dloss = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    
    @eqx.filter_jit
    def step(cv_model, fn, data, opt_state):
        loss_score, grads = dloss(cv_model, fn, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        cv_model = eqx.apply_updates(cv_model, updates)
        return cv_model, opt_state, loss_score
    
    loss_history = []
    for batch_idx, batch in enumerate(tqdm(dataloader, total=n_steps)):
        if batch_idx >= n_steps:
            break
        batch = batch[0] # dataloader returns tuple of size (1,)
        cv_model, opt_state, loss_score = step(cv_model, fn, batch, opt_state)
        loss_history.append(loss_score)
    return cv_model, loss_history