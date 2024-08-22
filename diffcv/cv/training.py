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
from diffcv.utils import MetricTracker


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
        fn: tp.Callable,
        grad_log_prob: tp.Callable,
        logger: Logger,
        n_steps: int = 1000,
        eval_every_n_steps: int = 1000,
        log_every_n_steps: int = 100,
        grad_clipping: int = -1,
        **kwargs,
    ):
        self.fn = fn
        self.grad_log_prob = grad_log_prob
        self.logger = logger
        self.n_steps = n_steps
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.train_metrics: MetricTracker = None
        self.evaluation_metrics: MetricTracker = None
        self.grad_clip_transform = (
            optax.clip_by_global_norm(grad_clipping)
            if grad_clipping > 0
            else optax.identity()
        )

    def _log_scalars(self, metric_tracker: MetricTracker):
        for metric_name in metric_tracker.keys():
            self.logger.add_scalar(metric_name, metric_tracker.avg(metric_name))

    def _calculate_fn_mean(self, model, dataloader, n_steps):
        generator = ScalarGenerator(self.grad_log_prob, model)
        fn_mean = 0.0
        for batch_index, batch in enumerate(inf_loop(dataloader)):
            if batch_index >= n_steps:
                break
            batch = jax.lax.stop_gradient(batch[0])
            fn_mean += (jax.vmap(self.fn)(batch) + jax.vmap(generator)(batch)).sum(
                axis=-1
            )
        fn_mean /= dataloader.dataloader.batch_size * n_steps
        return fn_mean

    def _evaluation(self, model, dataloader, n_steps):
        self.evaluation_metrics.reset()
        fn_mean = self._calculate_fn_mean(model, dataloader, n_steps)
        self.evaluation_metrics.update("fn_mean", fn_mean.item())
        self._log_scalars(self.evaluation_metrics)

    def _get_step_fn(self, loss, optimizer, with_fn_mean: bool = False):

        @eqx.filter_jit
        def opt_update(model, loss_score, grads, opt_state):
            grads, _ = self.grad_clip_transform.update(grads, None)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_score, grads

        @eqx.filter_jit
        def step(model, data, opt_state, key):
            loss_score, grads = loss(model=model, data=data, key=key)
            return opt_update(model, loss_score, grads, opt_state)

        @eqx.filter_jit
        def step_fn_mean(model, data, opt_state, key, fn_mean):
            loss_score, grads = loss(model=model, data=data, key=key, fn_mean=fn_mean)
            return opt_update(model, loss_score, grads, opt_state)

        if with_fn_mean:
            return step_fn_mean
        return step


class CVTrainer(BaseTrainer):
    def __init__(
        self,
        model: eqx.Module,
        fn: tp.Callable,
        grad_log_prob: tp.Callable,
        train_dataloader: jdl.DataLoader,
        eval_dataloader: jdl.DataLoader,
        optimizer: optax.GradientTransformation,
        loss: tp.Callable,
        logger: Logger,
        fn_mean: float | None = None,
        n_steps: int = 1000,
        eval_every_n_steps: int = 1000,
        log_every_n_steps: int = 100,
        patience: int = 1000,
        grad_clipping: int = -1,
        **kwargs,
    ):
        super().__init__(
            fn=fn,
            grad_log_prob=grad_log_prob,
            logger=logger,
            n_steps=n_steps,
            eval_every_n_steps=eval_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            grad_clipping=grad_clipping,
        )
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.fn_mean = fn_mean
        self.early_stopping = EarlyStopping(patience)
        self.train_metrics = MetricTracker("loss", "grad_norm")
        self.evaluation_metrics = MetricTracker("fn_mean")

    def train(self, key: jax.random.PRNGKey):
        model = self.model
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        loss = eqx.filter_jit(eqx.filter_value_and_grad(self.loss))
        if self.fn_mean is not None:
            step = self._get_step_fn(loss, self.optimizer, with_fn_mean=True)
        else:
            step = self._get_step_fn(loss, self.optimizer, with_fn_mean=False)

        pbar = tqdm(inf_loop(self.train_dataloader), total=self.n_steps)
        for batch_index, batch in enumerate(pbar):
            if batch_index >= self.n_steps:
                break
            batch = batch[0]  # dataloader returns tuple of size (1,)
            self.logger.set_step(batch_index)
            if self.fn_mean is not None:
                model, opt_state, loss_score, grads = step(
                    model, batch, opt_state, key, self.fn_mean
                )
            else:
                model, opt_state, loss_score, grads = step(model, batch, opt_state, key)

            self.train_metrics.update("grad_norm", calculate_grad_norm(grads))
            self.train_metrics.update("loss", loss_score.item())

            if batch_index % self.log_every_n_steps == 0:
                self._log_scalars(self.train_metrics)
                self.logger.add_scalar(
                    "learning_rate", opt_state.hyperparams["learning_rate"].item()
                )
                self.train_metrics.reset()
                pbar.set_description(f"loss: {loss_score.item(): .3f}")

            if batch_index % self.eval_every_n_steps == 0:
                self._evaluation(
                    model, self.eval_dataloader, len(self.eval_dataloader.dataloader)
                )
                pbar.set_postfix(
                    {"fn_mean": f'{self.evaluation_metrics.avg("fn_mean"): .3f}'}
                )

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
        train_dataloader: jdl.DataLoader,
        eval_dataloader: jdl.DataLoader,
        optimizer_diffusion: optax.GradientTransformation,
        optimizer_stein: optax.GradientTransformation,
        loss_diffusion: tp.Callable,
        loss_stein: tp.Callable,
        logger: Logger,
        switch_steps: int = 1000,
        n_steps: int = 1000,
        eval_every_n_steps: int = 1000,
        log_every_n_steps: int = 100,
        patience: int = 1000,
        grad_clipping: int = -1,
        **kwargs,
    ):
        super().__init__(
            fn=fn,
            grad_log_prob=grad_log_prob,
            logger=logger,
            n_steps=n_steps,
            eval_every_n_steps=eval_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            grad_clipping=grad_clipping,
        )
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer_diffusion = optimizer_diffusion
        self.optimizer_stein = optimizer_stein
        self.loss_diffusion = loss_diffusion
        self.loss_stein = loss_stein
        self.switch_steps = switch_steps
        self.early_stopping_diffusion = EarlyStopping(patience)
        self.early_stopping_stein = EarlyStopping(patience)
        self.train_metrics_diffusion = MetricTracker(
            "loss_diffusion", "grad_norm_diffusion"
        )
        self.train_metrics_stein = MetricTracker("loss_stein", "grad_norm_stein")
        self.evaluation_metrics = MetricTracker("fn_mean")

    def train(self, key: jax.random.PRNGKey):
        model = self.model
        opt_diffusion_state = self.optimizer_diffusion.init(
            eqx.filter(model, eqx.is_array)
        )
        opt_stein_state = self.optimizer_stein.init(eqx.filter(model, eqx.is_array))
        loss_diffusion = eqx.filter_jit(eqx.filter_value_and_grad(self.loss_diffusion))
        loss_stein = eqx.filter_jit(eqx.filter_value_and_grad(self.loss_stein))

        step_diffusion = self._get_step_fn(
            loss_diffusion, self.optimizer_diffusion, with_fn_mean=True
        )
        step_stein = self._get_step_fn(
            loss_stein, self.optimizer_stein, with_fn_mean=False
        )

        pbar = tqdm(inf_loop(self.train_dataloader), total=self.n_steps)
        diffusion_steps, stein_steps = 0, 0
        fn_mean_recalculated = False

        for batch_index, batch in enumerate(pbar):
            if batch_index >= self.n_steps:
                break
            batch = batch[0]  # dataloader returns tuple of size (1,)

            if (batch_index // self.switch_steps) % 2 == 0:
                self.logger.set_step(stein_steps)
                model, opt_stein_state, loss_score, grads = step_stein(
                    model, batch, opt_stein_state, key
                )

                self.train_metrics_stein.update("loss_stein", loss_score.item())
                self.train_metrics_stein.update(
                    "grad_norm_stein", calculate_grad_norm(grads)
                )
                stein_steps += 1
                fn_mean_recalculated = False
            else:
                self.logger.set_step(diffusion_steps)

                if not fn_mean_recalculated:
                    fn_mean = self._calculate_fn_mean(
                        model,
                        self.eval_dataloader,
                        len(
                            self.eval_dataloader.dataloader
                        ),  # decide whether to use less/more samples
                    )
                    print(f"train fn_mean: {fn_mean.item(): .3f}")
                    fn_mean_recalculated = True

                model, opt_diffusion_state, loss_score, grads = step_diffusion(
                    model, batch, opt_diffusion_state, key, fn_mean
                )

                self.train_metrics_diffusion.update("loss_diffusion", loss_score.item())
                self.train_metrics_diffusion.update(
                    "grad_norm_diffusion", calculate_grad_norm(grads)
                )
                diffusion_steps += 1

            if batch_index % self.log_every_n_steps == 0:
                if (batch_index // self.switch_steps) % 2 == 0:
                    self._log_scalars(self.train_metrics_stein)
                    self.train_metrics_stein.reset()
                    self.logger.add_scalar(
                        "learning_rate_stein",
                        opt_stein_state.hyperparams["learning_rate"].item(),
                    )
                    pbar.set_description(f"loss_stein: {loss_score.item(): .3f}")
                else:
                    self._log_scalars(self.train_metrics_diffusion)
                    self.train_metrics_diffusion.reset()
                    self.logger.add_scalar(
                        "learning_rate_diffusion",
                        opt_diffusion_state.hyperparams["learning_rate"].item(),
                    )
                    pbar.set_description(f"loss_diffusion: {loss_score.item(): .3f}")

            if batch_index % self.eval_every_n_steps == 0:
                self.logger.set_step(batch_index)
                self._evaluation(
                    model, self.eval_dataloader, len(self.eval_dataloader.dataloader)
                )
                pbar.set_postfix(
                    {"fn_mean": f'{self.evaluation_metrics.avg("fn_mean"): .3f}'}
                )

            if (batch_index // self.switch_steps) % 2 == 0:
                if self.early_stopping_stein.need_to_stop(loss_score.item()):
                    print(
                        f"Early stopping at step {batch_index} due to no improvement in stein loss over {self.early_stopping_stein.patience} steps."
                    )
                    break
            else:
                if self.early_stopping_diffusion.need_to_stop(loss_score.item()):
                    print(
                        f"Early stopping at step {batch_index} due to no improvement in diffusion loss over {self.early_stopping_diffusion.patience} steps."
                    )
                    break

        return model
