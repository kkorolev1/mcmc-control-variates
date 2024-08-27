import typing as tp
from dataclasses import dataclass


@dataclass
class SamplerConfig:
    dim: int
    gamma: float
    init_std: float


@dataclass
class SamplingConfig:
    steps: int
    burnin_steps: int
    skip_steps: int


@dataclass
class EstimatorConfig:
    sampling_config: SamplingConfig
    total_samples: int
    n_estimates: int


@dataclass
class DataConfig:
    batch_size: int
    train_size: int
    eval_size: int


@dataclass
class TrainerConfig:
    grad_clipping: float
    patience: int
    eval_every_n_steps: int
    n_steps: int


@dataclass
class TrainerALSConfig(TrainerConfig):
    switch_steps: int


@dataclass
class ModelConfig:
    depth: int
    width_size: int
    activation: tp.Callable
