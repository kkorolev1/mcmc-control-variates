import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typing as tp
import jax_dataloader as jdl
from tqdm.auto import tqdm

from diffcv.logger import Logger


class CVTrainer:
    def __init__(self, model: eqx.Module, dataloader: jdl.DataLoader, optimizer: optax.GradientTransformation, 
                 loss: tp.Callable, logger: Logger, n_steps: int = 1000, log_every_n_steps: int = 100):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.logger = logger
        
        self.n_steps = n_steps
        self.log_every_n_steps = log_every_n_steps

    def train(self, key: jax.random.PRNGKey):
        model = self.model
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        dloss = eqx.filter_jit(eqx.filter_value_and_grad(self.loss))
        
        @eqx.filter_jit
        def step(model, data, opt_state, batch_index, key):
            loss_score, grads = dloss(model, data, batch_index, key)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score
        
        for batch_index, batch in enumerate(tqdm(self.dataloader, total=self.n_steps)):
            if batch_index >= self.n_steps:
                break
            batch = batch[0] # dataloader returns tuple of size (1,)
            self.logger.set_step(batch_index)
            model, opt_state, loss_score = step(model, batch, opt_state, jnp.asarray(batch_index), key)
            if batch_index % self.log_every_n_steps == 0:
                self.logger.add_scalar("loss", loss_score.item())
                self.logger.add_scalar("learning_rate", opt_state.hyperparams["learning_rate"].item())
        return model
