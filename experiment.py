import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
import jax.nn as nn
import equinox as eqx
import jax_dataloader as jdl

from tqdm import tqdm
import typing as tp
import pandas as pd
import math
from dataclasses import asdict

import sys

import optax
import numpyro.distributions as D

from diffcv.dist.gaussian_mixture import GaussianMixture
from diffcv.mcmc.base import Sampler
from diffcv.mcmc.langevin import ULASampler, MALASampler
from diffcv.mcmc.pyro import HMCSampler
from diffcv.cv.nn import (
    CVMLP,
    ModelWithConstant,
    init_linear,
    normal_init,
    he_uniform_init,
    xavier_uniform_init,
    zero_init,
)
from diffcv.cv.training import CVTrainer, CVALSTrainer
from diffcv.cv.loss import DiffusionLoss, DiffLoss, VarLoss
from diffcv.logger import Logger, plot_log_results
from diffcv.cv.data import get_data_from_sampler
from diffcv.cv.utils import recu, requ
from diffcv.cv.generator import ScalarGenerator
from diffcv.mcmc.estimator import Estimator
from diffcv.config import *

sns.set_style("darkgrid")


sampler_config = SamplerConfig(
    dim=1,
    gamma=8e-2,
    init_std=5.0,
)

sampling_config = SamplingConfig(
    steps=1_000,
    burnin_steps=1_000,
    skip_steps=2,
)

estimator_config = EstimatorConfig(
    sampling_config=sampling_config,
    total_samples=20_000,
    n_estimates=1_000,
)

rng = jax.random.PRNGKey(50)

# dist = D.MultivariateNormal(
#     loc=10 * jnp.ones((sampler_config.dim), dtype=float),
#     covariance_matrix=jnp.eye((sampler_config.dim), dtype=float),
# )
a = 5
dist = GaussianMixture(
    component_means=jnp.stack(
        [
            -a * jnp.ones((sampler_config.dim), dtype=float),
            a * jnp.ones((sampler_config.dim), dtype=float),
        ],
        axis=0,
    ),
    component_covs=jnp.stack(
        [
            jnp.ones((sampler_config.dim), dtype=float),
            jnp.ones((sampler_config.dim), dtype=float),
        ],
        axis=0,
    ),
)
log_prob = jax.jit(dist.log_prob)
grad_log_prob = jax.jit(jax.grad(dist.log_prob))

fn = jax.jit(lambda x: (x**2).sum(axis=-1))
true_pi = (dist.mean**2).sum() + dist.variance.sum()

print(f"true pi(f) = {true_pi: .3f}")

sampler = MALASampler(log_prob=log_prob, **asdict(sampler_config))


def estimate(
    key: jax.random.PRNGKey, fn: tp.Callable, estimator_config: EstimatorConfig
):
    estimates = Estimator(fn, sampler)(key, estimator_config)
    return {
        "estimates": estimates,
        "bias": Estimator.bias(true_pi, estimates),
        "std": Estimator.std(estimates),
    }


results = {}


def base_exp():
    global rng
    rng, key = jax.random.split(rng)
    results["base"] = estimate(key, fn, estimator_config)
    print(results["base"]["bias"], results["base"]["std"])


data_config = DataConfig(
    batch_size=1024,
    train_size=1024 * 5,
    eval_size=1024 * 20,
)

rng, key = jax.random.split(rng)
train_dataloader = get_data_from_sampler(
    key,
    data_config.batch_size,
    sampler,
    total_samples=data_config.train_size,
    sampling_config=sampling_config,
)
print(f"train_dataset length: {len(train_dataloader.dataloader.dataset)}")

rng, key = jax.random.split(rng)
eval_dataloader = get_data_from_sampler(
    key,
    data_config.batch_size,
    sampler,
    total_samples=data_config.eval_size,
    sampling_config=sampling_config,
)
print(f"eval_dataset length: {len(eval_dataloader.dataloader.dataset)}")


def get_model(key: jax.random.PRNGKey, in_size: int, model_config: ModelConfig):
    # key1, key2 = jax.random.split(key, 2)
    # model = CVMLP(in_size=in_size, **asdict(model_config), key=key1)
    # model = init_linear(model, key2, normal_init, zero_init)
    model = CVMLP(in_size=in_size, **asdict(model_config), key=key)
    return model


def get_scheduler(scheduler_name: str, **kwargs):
    if scheduler_name == "exponential":
        return optax.exponential_decay(**kwargs)
    else:
        raise NotImplementedError(f"unknown scheduler: {scheduler_name}")


def get_optimizer(scheduler: optax.Schedule, optimizer_name: str, **kwargs):
    if optimizer_name == "sgd":
        return optax.inject_hyperparams(optax.sgd)(learning_rate=scheduler, **kwargs)
    elif optimizer_name == "adam":
        return optax.inject_hyperparams(optax.adam)(learning_rate=scheduler, **kwargs)
    elif optimizer_name == "adamw":
        # weight_mask = lambda m: jax.tree_util.tree_map(lambda x: x.ndim > 1, m)
        return optax.inject_hyperparams(optax.adamw)(
            learning_rate=scheduler,
            **kwargs,
            # mask=weight_mask,
        )
    else:
        raise NotImplementedError(f"unknown optimizer: {optimizer_name}")


def diffusion_exp():
    global rng
    model_config = ModelConfig(
        depth=2,
        width_size=128,
        activation=requ,
    )

    trainer_config = TrainerConfig(
        grad_clipping=1,
        patience=10_000,
        eval_every_n_steps=2_000,
        n_steps=50_000,
    )

    logger = Logger()

    rng, key = jax.random.split(rng)
    model_diffusion = get_model(key, sampler_config.dim, model_config)
    loss = DiffusionLoss(fn=fn)

    scheduler = get_scheduler(
        "exponential",
        init_value=1e-4,
        transition_steps=20_000,
        decay_rate=0.9,
    )

    optimizer = get_optimizer(scheduler, "adamw", weight_decay=1e-1)

    trainer = CVTrainer(
        model_diffusion,
        fn,
        grad_log_prob,
        train_dataloader,
        eval_dataloader,
        optimizer,
        loss,
        logger,
        use_fn_mean=True,
        **asdict(trainer_config),
    )

    rng, key = jax.random.split(rng)
    model_diffusion = trainer.train(key)

    generator_diffusion = ScalarGenerator(grad_log_prob, model_diffusion)
    fn_with_cv = lambda x: fn(x) + generator_diffusion(x)

    rng, key = jax.random.split(rng)
    results["diffusion"] = estimate(key, fn_with_cv, estimator_config)
    print(results["diffusion"]["bias"], results["diffusion"]["std"])


def diffusion_als_exp():
    global rng
    model_config = ModelConfig(
        depth=2,
        width_size=128,
        activation=requ,
    )

    trainer_config = TrainerALSConfig(
        grad_clipping=1,
        patience=10_000,
        eval_every_n_steps=2_000,
        n_steps=50_000,
        switch_steps=10_000,
    )

    logger = Logger()

    rng, key = jax.random.split(rng)
    model_diffusion_als = get_model(key, sampler_config.dim, model_config)

    scheduler_diffusion = get_scheduler(
        "exponential",
        init_value=1e-4,
        transition_steps=20_000,
        decay_rate=0.9,
    )

    optimizer_diffusion = get_optimizer(scheduler_diffusion, "adamw", weight_decay=1e-1)

    scheduler_stein = get_scheduler(
        "exponential",
        init_value=1e-4,
        transition_steps=20_000,
        decay_rate=0.9,
    )

    optimizer_stein = get_optimizer(scheduler_stein, "adamw", weight_decay=1e-1)

    loss_diffusion = DiffusionLoss(fn=fn)
    loss_stein = DiffLoss(fn=fn, grad_log_prob=grad_log_prob, noise_std=1.0)

    trainer = CVALSTrainer(
        model_diffusion_als,
        fn,
        grad_log_prob,
        train_dataloader,
        eval_dataloader,
        optimizer_diffusion,
        optimizer_stein,
        loss_diffusion,
        loss_stein,
        logger=logger,
        **asdict(trainer_config),
    )

    rng, key = jax.random.split(rng)
    model_diffusion_als = trainer.train(key)

    generator_diffusion_als = ScalarGenerator(grad_log_prob, model_diffusion_als)
    fn_with_cv = lambda x: fn(x) + generator_diffusion_als(x)

    rng, key = jax.random.split(rng)
    results["diffusion_als"] = estimate(key, fn_with_cv, estimator_config)
    print(results["diffusion_als"]["bias"], results["diffusion_als"]["std"])


def diff_exp():
    global rng
    model_config = ModelConfig(
        depth=2,
        width_size=128,
        activation=requ,
    )

    trainer_config = TrainerConfig(
        grad_clipping=1,
        patience=10_000,
        eval_every_n_steps=2_000,
        n_steps=30_000,
    )

    logger = Logger()

    rng, key = jax.random.split(rng)
    model_diff = get_model(key, sampler_config.dim, model_config)
    loss = DiffLoss(fn=fn, grad_log_prob=grad_log_prob, noise_std=1.0)

    scheduler = get_scheduler(
        "exponential",
        init_value=1e-4,
        transition_steps=20_000,
        decay_rate=0.9,
    )

    optimizer = get_optimizer(scheduler, "adamw", weight_decay=1e-1)

    trainer = CVTrainer(
        model_diff,
        fn,
        grad_log_prob,
        train_dataloader,
        eval_dataloader,
        optimizer,
        loss,
        logger,
        **asdict(trainer_config),
    )
    rng, key = jax.random.split(rng)
    model_diff = trainer.train(key)

    generator_diff = ScalarGenerator(grad_log_prob, model_diff)
    fn_with_cv = lambda x: fn(x) + generator_diff(x)

    rng, key = jax.random.split(rng)
    results["diff"] = estimate(key, fn_with_cv, estimator_config)
    print(results["diff"]["bias"], results["diff"]["std"])


base_exp()
diffusion_exp()
diffusion_als_exp()
diff_exp()

print({key: f"{results[key]['bias']: .3f}" for key in results})
print({key: f"{results[key]['std']: .3f}" for key in results})
