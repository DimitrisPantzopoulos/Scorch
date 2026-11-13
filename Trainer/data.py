"""Training examples, evaluation results, and shared array contracts."""
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, TypeAlias, Tuple

import chex
import jax
import pgx.core

Metrics      : TypeAlias = Dict[str, jax.Array | int | float]
Record       : TypeAlias = Dict[str, jax.Array]
Parameters   : TypeAlias = Dict[str, Any]
Carry        : TypeAlias = Tuple[pgx.core.State, jax.Array, jax.Array]
Observation  : TypeAlias = jax.Array
PolicyTarget : TypeAlias = jax.Array
ValueTarget  : TypeAlias = jax.Array
SideToMove   : TypeAlias = jax.Array
EpisodeId    : TypeAlias = jax.Array

@chex.dataclass
class SelfPlayExamples:
    observations   : Observation
    policy_targets : PolicyTarget
    value_targets  : ValueTarget
    side_to_moves  : SideToMove
    episode_ids    : EpisodeId

@dataclass(slots=True)
class Evaluation:
    wins       : int
    losses     : int
    draws      : int
    unfinished : int

    @property
    def completed_games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score_rate(self) -> float | None:
        if self.completed_games == 0:
            return None

        return (self.wins + 0.5 * self.draws) / (self.completed_games)

    @property
    def elo_delta(self) -> float | None:
        s : float | None = self.score_rate

        if s is None:
            return None
        if s == 0.0:
            return -math.inf
        if s == 1.0:
            return math.inf


        return 400 * math.log10(s / (1 - s))


@dataclass(frozen=True, slots=True)
class TrainingReport:
    iterations      : int
    promotions      : int
    optimizer_step  : int
    next_iteration  : int
    last_evaluation : Evaluation | None
    checkpoint_path : Path | None
