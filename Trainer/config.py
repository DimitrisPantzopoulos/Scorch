"""Immutable settings for learning, collection, and evaluation."""
from dataclasses import dataclass
from MCTX.mctx import SearchConfig

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate     : float = 3e-4
    weight_decay      : float = 1e-4
    max_grad_norm     : float = 1.0
    value_loss_weight : float = 1.0

    epochs            : int   = 256
    minibatch_size        : int   = 128

@dataclass(frozen=True)
class EvaluationConfig:
    num_games    : int = 32
    num_steps     : int = 256
    seed          : int = 42
    search_config : SearchConfig | None = None

    def __post_init__(self) -> None:
        if self.num_games < 2 or self.num_games % 2:
            raise ValueError("Evaluation num_games must be positive and even.")
        if self.num_steps < 1:
            raise ValueError("Evaluation num_steps must be positive.")


@dataclass(frozen=True)
class SelfPlayConfig:
    num_steps   : int   = 256
    temperature : float = 1.0

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("Self-play num_steps must be nonnegative.")
        if self.temperature < 0:
            raise ValueError("Self-play temperature must be nonnegative.")
