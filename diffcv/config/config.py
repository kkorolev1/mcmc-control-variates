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
class SchedulerConfig:
    init_value: float
    transition_steps: int
    decay_rate: float


@dataclass
class OptimizerConfig:
    method: str
    weight_decay: float


@dataclass
class BaseTrainerConfig:
    batch_size: int
    train_size: int
    eval_size: int
    grad_clipping: float
    patience: int
    eval_every_n_steps: int
    sampling_config: SamplingConfig


@dataclass
class TrainerConfig(BaseTrainerConfig):
    scheduler_config: SchedulerConfig
    optimizer_config: OptimizerConfig
    n_steps: int


@dataclass
class TrainerALSConfig(BaseTrainerConfig):
    scheduler_config: SchedulerConfig
    optimizer_config: OptimizerConfig
    n_steps: int
    switch_steps: int


@dataclass
class ModelConfig:
    depth: int
    width_size: int
    activation: tp.Callable
