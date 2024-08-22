import typing as tp
from dataclasses import dataclass


@dataclass
class SamplerConfig:
    dim: int
    n_samples: int
    gamma: float
    burnin_steps: int
    init_std: float
    skip_samples: int


@dataclass
class EstimatorConfig:
    n_chains: int
    n_estimates: int


@dataclass
class BaseTrainerConfig:
    batch_size: int
    train_size: int
    eval_size: int
    lr: float
    weight_decay: float
    grad_clipping: float
    patience: int
    eval_every_n_steps: int


@dataclass
class TrainerConfig(BaseTrainerConfig):
    n_steps: int


@dataclass
class ModelConfig:
    depth: int
    width_size: int
    activation: tp.Callable
