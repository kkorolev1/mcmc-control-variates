import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typing as tp
import jax_dataloader as jdl
from tqdm.auto import tqdm

from diffcv.cv.generator import ScalarGenerator
from diffcv.logger import Logger
from diffcv.utils import inf_loop


def calculate_grad_norm(grads):
    flat_grads = jax.tree_util.tree_leaves(grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads))
    return grad_norm.item()


class EarlyStopping:
    def __init__(self, patience: int, strategy: str = "min"):
        self.patience = patience
        self.strategy = strategy
        self.steps_without_improvement = 0
        self.best_value = jnp.inf if strategy == "min" else -jnp.inf
        self.eps = 1e-6

    def need_to_stop(self, value):
        if self.strategy == "min":
            improvement_condition = value + self.eps < self.best_value
        else:
            improvement_condition = value > self.best_value + self.eps

        if improvement_condition:
            self.best_value = value
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement >= self.patience


class BaseTrainer:
    def __init__(
        self,
        logger: Logger,
        n_steps: int = 1000,
        log_every_n_steps: int = 100,
    ):
        self.logger = logger

        self.n_steps = n_steps
        self.log_every_n_steps = log_every_n_steps


class CVTrainer(BaseTrainer):
    def __init__(
        self,
        model: eqx.Module,
        fn: tp.Callable,
        dataloader: jdl.DataLoader,
        optimizer: optax.GradientTransformation,
        loss: tp.Callable,
        logger: Logger,
        n_steps: int = 1000,
        log_every_n_steps: int = 100,
        fn_mean: float | None = None,
        patience: int = 1000,
    ):
        super().__init__(
            logger=logger,
            n_steps=n_steps,
            log_every_n_steps=log_every_n_steps,
        )
        self.model = model
        self.fn = fn
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.fn_mean = fn_mean
        self.early_stopping = EarlyStopping(patience)

    def train(self, key: jax.random.PRNGKey):
        model = self.model
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        loss = eqx.filter_jit(eqx.filter_value_and_grad(self.loss))

        @eqx.filter_jit
        def step(model, data, opt_state, key):
            loss_score, grads = loss(model, data, key)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score, grads

        @eqx.filter_jit
        def step_with_fn(model, data, opt_state, key, fn_mean):
            loss_score, grads = loss(model, data, key, fn_mean)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score, grads

        pbar = tqdm(inf_loop(self.dataloader), total=self.n_steps)
        for batch_index, batch in enumerate(pbar):
            if batch_index >= self.n_steps:
                break
            batch = batch[0]  # dataloader returns tuple of size (1,)
            self.logger.set_step(batch_index)
            if self.fn_mean is not None:
                model, opt_state, loss_score, grads = step_with_fn(
                    model, batch, opt_state, key, self.fn_mean
                )
            else:
                model, opt_state, loss_score, grads = step(model, batch, opt_state, key)

            if batch_index % self.log_every_n_steps == 0:
                self.logger.add_scalar("grad_norm", calculate_grad_norm(grads))
                self.logger.add_scalar("loss", loss_score.item())
                self.logger.add_scalar(
                    "learning_rate", opt_state.hyperparams["learning_rate"].item()
                )
                pbar.set_description(f"loss: {loss_score.item(): .3f}")

            if self.early_stopping.need_to_stop(loss_score.item()):
                print(
                    f"Early stopping at step {batch_index} due to no improvement in loss over {self.early_stopping.patience} steps."
                )
                break

        return model


class CVALSTrainer(BaseTrainer):
    def __init__(
        self,
        model: eqx.Module,
        fn: tp.Callable,
        grad_log_prob: tp.Callable,
        dataloader: jdl.DataLoader,
        optimizer_diffusion: optax.GradientTransformation,
        optimizer_stein: optax.GradientTransformation,
        loss_diffusion: tp.Callable,
        loss_stein: tp.Callable,
        logger: Logger,
        switch_steps: int = 1000,
        n_steps_for_mean_recalculation: int = 1000,
        n_steps: int = 1000,
        log_every_n_steps: int = 100,
        patience: int = 1000,
    ):
        super().__init__(
            logger=logger,
            n_steps=n_steps,
            log_every_n_steps=log_every_n_steps,
        )
        self.model = model
        self.fn = fn
        self.grad_log_prob = grad_log_prob
        self.dataloader = dataloader
        self.optimizer_diffusion = optimizer_diffusion
        self.optimizer_stein = optimizer_stein
        self.loss_diffusion = loss_diffusion
        self.loss_stein = loss_stein
        self.switch_steps = switch_steps
        self.n_steps_for_mean_recalculation = n_steps_for_mean_recalculation
        self.early_stopping_diffusion = EarlyStopping(patience)
        self.early_stopping_stein = EarlyStopping(patience)

    def train(self, key: jax.random.PRNGKey):
        model = self.model
        opt_diffusion_state = self.optimizer_diffusion.init(
            eqx.filter(model, eqx.is_array)
        )
        opt_stein_state = self.optimizer_stein.init(eqx.filter(model, eqx.is_array))
        loss_diffusion = eqx.filter_jit(eqx.filter_value_and_grad(self.loss_diffusion))
        loss_stein = eqx.filter_jit(eqx.filter_value_and_grad(self.loss_stein))

        @eqx.filter_jit
        def step_diffusion(model, data, opt_state, key, fn_mean):
            loss_score, grads = loss_diffusion(model, data, key, fn_mean)
            updates, opt_state = self.optimizer_diffusion.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score, grads

        @eqx.filter_jit
        def step_stein(model, data, opt_state, key):
            loss_score, grads = loss_stein(model, data, key)
            updates, opt_state = self.optimizer_stein.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score, grads

        pbar = tqdm(inf_loop(self.dataloader), total=self.n_steps)
        diffusion_steps, stein_steps = 0, 0
        sample_mean_recalculated = False

        for batch_index, batch in enumerate(pbar):
            if batch_index >= self.n_steps:
                break
            batch = batch[0]  # dataloader returns tuple of size (1,)

            if (batch_index // self.switch_steps) % 2 == 0:
                self.logger.set_step(stein_steps)
                model, opt_stein_state, loss_score, grads = step_stein(
                    model, batch, opt_stein_state, key
                )

                if batch_index % self.log_every_n_steps == 0:
                    self.logger.add_scalar("loss_stein", loss_score.item())
                    self.logger.add_scalar("grad_norm", calculate_grad_norm(grads))
                    self.logger.add_scalar(
                        "learning_rate_stein",
                        opt_stein_state.hyperparams["learning_rate"].item(),
                    )
                    pbar.set_description(f"loss_stein: {loss_score.item(): .3f}")

                stein_steps += 1
                sample_mean_recalculated = False
            else:
                self.logger.set_step(diffusion_steps)

                if not sample_mean_recalculated:
                    generator = ScalarGenerator(self.grad_log_prob, model)
                    fn_mean = 0.0
                    for i, data in enumerate(inf_loop(self.dataloader)):
                        if i >= self.n_steps_for_mean_recalculation:
                            break
                        data = jax.lax.stop_gradient(data[0])
                        fn_mean += (
                            jax.vmap(self.fn)(data) + jax.vmap(generator)(data)
                        ).sum(axis=-1)
                    fn_mean /= batch.shape[0] * self.n_steps_for_mean_recalculation

                    sample_mean_recalculated = True
                    self.logger.add_scalar("fn_mean", fn_mean.item())
                model, opt_diffusion_state, loss_score, grads = step_diffusion(
                    model, batch, opt_diffusion_state, key, fn_mean
                )

                if batch_index % self.log_every_n_steps == 0:
                    self.logger.add_scalar("loss_diffusion", loss_score.item())
                    self.logger.add_scalar("grad_norm", calculate_grad_norm(grads))
                    self.logger.add_scalar(
                        "learning_rate_diffusion",
                        opt_diffusion_state.hyperparams["learning_rate"].item(),
                    )
                    pbar.set_description(f"loss_diffusion: {loss_score.item(): .3f}")

                diffusion_steps += 1

            if (batch_index // self.switch_steps) % 2 == 0:
                if self.early_stopping_stein.need_to_stop(loss_score.item()):
                    print(
                        f"Early stopping at step {batch_index} due to no improvement in loss over {self.early_stopping_stein.patience} steps."
                    )
                    break
            else:
                if self.early_stopping_diffusion.need_to_stop(loss_score.item()):
                    print(
                        f"Early stopping at step {batch_index} due to no improvement in loss over {self.early_stopping_diffusion.patience} steps."
                    )
                    break

        return model
