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


class CVTrainer:
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
    ):
        self.model = model
        self.fn = fn
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.logger = logger

        self.n_steps = n_steps
        self.log_every_n_steps = log_every_n_steps

        self.fn_mean = fn_mean

    def train(self, key: jax.random.PRNGKey):
        model = self.model
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        loss = eqx.filter_jit(eqx.filter_value_and_grad(self.loss))

        @eqx.filter_jit
        def step(model, data, opt_state, key):
            loss_score, grads = loss(model, data, key)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score

        @eqx.filter_jit
        def step_with_fn(model, data, opt_state, key, fn_mean):
            loss_score, grads = loss(model, data, key, fn_mean)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score

        pbar = tqdm(inf_loop(self.dataloader), total=self.n_steps)
        for batch_index, batch in enumerate(pbar):
            if batch_index >= self.n_steps:
                break
            batch = batch[0]  # dataloader returns tuple of size (1,)
            self.logger.set_step(batch_index)
            if self.fn_mean is not None:
                model, opt_state, loss_score = step_with_fn(
                    model, batch, opt_state, key, self.fn_mean
                )
            else:
                model, opt_state, loss_score = step(model, batch, opt_state, key)
            if batch_index % self.log_every_n_steps == 0:
                self.logger.add_scalar("loss", loss_score.item())
                self.logger.add_scalar(
                    "learning_rate", opt_state.hyperparams["learning_rate"].item()
                )
                pbar.set_description(f"loss: {loss_score.item(): .3f}")
        return model


class CVALSTrainer:
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
        n_steps: int = 1000,
        log_every_n_steps: int = 100,
        n_steps_for_mean_recalculation: int = 1000,
    ):
        self.model = model
        self.fn = fn
        self.grad_log_prob = grad_log_prob
        self.dataloader = dataloader
        self.optimizer_diffusion = optimizer_diffusion
        self.optimizer_stein = optimizer_stein
        self.loss_diffusion = loss_diffusion
        self.loss_stein = loss_stein
        self.logger = logger

        self.switch_steps = switch_steps
        self.n_steps = n_steps
        self.log_every_n_steps = log_every_n_steps
        self.n_steps_for_mean_recalculation = n_steps_for_mean_recalculation

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
            return model, opt_state, loss_score

        @eqx.filter_jit
        def step_stein(model, data, opt_state, key):
            loss_score, grads = loss_stein(model, data, key)
            updates, opt_state = self.optimizer_stein.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score

        pbar = tqdm(inf_loop(self.dataloader), total=self.n_steps)
        diffusion_steps, stein_steps = 0, 0
        sample_mean_recalculated = False

        for batch_index, batch in enumerate(pbar):
            if batch_index >= self.n_steps:
                break
            batch = batch[0]  # dataloader returns tuple of size (1,)

            if (batch_index // self.switch_steps) % 2 == 0:
                self.logger.set_step(stein_steps)
                model, opt_stein_state, loss_score = step_stein(
                    model, batch, opt_stein_state, key
                )

                if batch_index % self.log_every_n_steps == 0:
                    self.logger.add_scalar("loss_stein", loss_score.item())
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
                model, opt_diffusion_state, loss_score = step_diffusion(
                    model, batch, opt_diffusion_state, key, fn_mean
                )

                if batch_index % self.log_every_n_steps == 0:
                    self.logger.add_scalar("loss_diffusion", loss_score.item())
                    self.logger.add_scalar(
                        "learning_rate_diffusion",
                        opt_diffusion_state.hyperparams["learning_rate"].item(),
                    )
                    pbar.set_description(f"loss_diffusion: {loss_score.item(): .3f}")

                diffusion_steps += 1

        return model
