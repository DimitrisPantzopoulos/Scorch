"""Metric formatting and ownership of the TensorBoard event writer."""
from datetime import datetime
from pathlib import Path
import math
import jax
import jax.numpy as jnp
from tensorboardX import SummaryWriter
from .data import Evaluation, Metrics, SelfPlayExamples


class TensorBoardLogger:
    def __init__(
            self,
            log_dir  : str | Path | None = None,
            *,
            run_name : str | None        = None,
            base_dir : str | Path        = 'runs',
        ) -> None:
        self.log_dir : Path = (
            Path(log_dir) if log_dir is not None
            else Path(base_dir) / f"{run_name or 'scorch'}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        self.closed  : bool = False
        self._writer : SummaryWriter = SummaryWriter(log_dir=str(self.log_dir))

    def log_metrics(
            self,
            metrics  : Metrics,
            timestep : int,
            prefix   : str,
        ) -> None:
        if self.closed:
            raise RuntimeError("The TensorBoard logger is closed.")
        mean_metrics : Metrics = jax.tree_util.tree_map(lambda value : jnp.mean(value), metrics)
        for name, value in mean_metrics.items():
            self._writer.add_scalar(f"{prefix}/{name}", float(value), timestep)

    @staticmethod
    def self_play_metrics(
            examples   : SelfPlayExamples,
            time_delta : float,
        ) -> Metrics:
        return {
            "positions"      : int(examples.episode_ids.shape[0]),
            "games_finished" : int(jnp.unique(examples.episode_ids).shape[0]),
            "time_delta"     : float(time_delta),
        }

    def log_self_play(
            self,
            examples   : SelfPlayExamples,
            time_delta : float,
            iteration  : int,
        ) -> None:
        self.log_metrics(self.self_play_metrics(examples, time_delta), iteration, prefix="self_play")

    def log_evaluation(
            self,
            result    : Evaluation,
            iteration : int,
            promoted  : bool,
        ) -> None:
        metrics : Metrics = {
            "wins"            : result.wins,
            "losses"          : result.losses,
            "draws"           : result.draws,
            "unfinished"      : result.unfinished,
            "completed_games" : result.completed_games,
            "promoted"        : int(promoted),
        }
        if result.score_rate is not None:
            metrics["score_rate"] = result.score_rate
        elo : float | None = result.elo_delta
        if elo is not None and math.isfinite(elo):
            metrics["elo_delta"] = elo
        self.log_metrics(metrics, iteration, prefix="evaluation")

    def flush(self) -> None:
        if not self.closed:
            self._writer.flush()

    def close(self) -> None:
        if not self.closed:
            self._writer.close()
            self.closed = True
