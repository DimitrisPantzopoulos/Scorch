"""Factory and orchestration for self-play, learning, evaluation, and promotion."""
from pathlib import Path
from types import TracebackType
from typing import Self, Tuple
from tqdm import tqdm
import time

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.config import NetworkConfig
from Scorch.scorch import Scorch
from .checkpoint_manager import Checkpoint, CheckpointManager
from .config import EvaluationConfig, SelfPlayConfig, TrainingConfig
from .data import Evaluation, Metrics, SelfPlayExamples, TrainingReport
from .scorch_evaluator import ScorchEvaluator
from .scorch_learner import ScorchLearner
from .self_play_collector import SelfPlayCollector
from .tensorboard_logger import TensorBoardLogger


class ScorchTrainer:
    def __init__(
            self,
            scorch            : Scorch,
            env               : ScorchEnv,
            training_config   : TrainingConfig           = TrainingConfig(),
            *,
            self_play_config  : SelfPlayConfig           = SelfPlayConfig(),
            evaluation_config : EvaluationConfig | None  = EvaluationConfig(),
            seed              : int                      = 42,
            run_name          : str | None               = None,
            log_dir           : str | Path | None        = None,
            checkpoint_dir    : str | Path | None        = None,
            logger            : TensorBoardLogger | None = None,
        ) -> None:
        self.scorch            : Scorch                  = scorch
        self.env               : ScorchEnv               = env
        self.key               : jax.Array               = jax.random.PRNGKey(seed)
        self.learner           : ScorchLearner           = ScorchLearner(scorch, training_config)
        self.collector         : SelfPlayCollector       = SelfPlayCollector(scorch, env, self_play_config)
        self.evaluation_config : EvaluationConfig | None = evaluation_config
        self.evaluator         : ScorchEvaluator         = ScorchEvaluator(
            scorch,
            evaluation_config or EvaluationConfig(),
        )
        self.checkpoints       : CheckpointManager       = CheckpointManager()
        self.logger            : TensorBoardLogger       = (
            logger if logger is not None
            else TensorBoardLogger(log_dir, run_name=run_name)
        )
        self.checkpoint_dir    : Path | None              = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.next_episode_id   : int                      = 0
        self.next_iteration    : int                      = 0

    @classmethod
    def create(
            cls,
            *,
            num_envs          : int                     = 256,
            seed              : int                     = 42,
            network_config    : NetworkConfig           = NetworkConfig(),
            search_config     : SearchConfig            = SearchConfig(),
            training_config   : TrainingConfig          = TrainingConfig(),
            self_play_config  : SelfPlayConfig          = SelfPlayConfig(),
            evaluation_config : EvaluationConfig | None = EvaluationConfig(),
            run_name          : str | None              = None,
            log_dir           : str | Path | None       = None,
            checkpoint_dir    : str | Path | None       = None,
        ) -> Self:
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")

        key, env_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        env_seed   : int = int(jax.random.bits(env_key, dtype=jnp.uint32))
        model_seed : int = int(jax.random.bits(model_key, dtype=jnp.uint32))
        env        : ScorchEnv = ScorchEnv(batch_size=num_envs, seed=env_seed)
        scorch     : Scorch = Scorch(
            search_config  = search_config,
            env            = env,
            seed           = model_seed,
            network_config = network_config,
        )

        trainer : Self = cls(
            scorch,
            env,
            training_config,
            self_play_config  = self_play_config,
            evaluation_config = evaluation_config,
            seed              = seed,
            run_name          = run_name,
            log_dir           = log_dir,
            checkpoint_dir    = checkpoint_dir,
        )
        trainer.key = key
        return trainer

    @property
    def closed(self) -> bool:
        return self.logger.closed

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("The trainer is closed.")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
            self,
            exc_type  : type[BaseException] | None,
            exc_value : BaseException | None,
            traceback : TracebackType | None,
        ) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self.logger.close()

    def _snapshot(self) -> Checkpoint:
        return Checkpoint(
            state           = self.learner.state,
            next_episode_id = self.next_episode_id,
            trainer_key     = self.key,
            scorch_key      = self.scorch.prng_key,
            env_key         = self.env.key,
            next_iteration  = self.next_iteration,
        )

    def save_checkpoint(
            self,
            path : str | Path,
        ) -> None:
        self._ensure_open()
        self.checkpoints.save(path, self._snapshot())

    def load_checkpoint(
            self,
            path : str | Path,
        ) -> None:
        self._ensure_open()
        checkpoint : Checkpoint = self.checkpoints.load(
            path,
            self._snapshot(),
            num_envs = self.env.batch_size,
        )
        self.learner.restore(checkpoint.state)
        self.next_episode_id = int(checkpoint.next_episode_id)
        self.next_iteration  = int(checkpoint.next_iteration)
        self.key             = checkpoint.trainer_key
        self.scorch.prng_key = checkpoint.scorch_key
        self.env.key         = checkpoint.env_key

    def _collect_and_learn(
            self,
            key       : jax.Array,
            iteration : int,
        ) -> None:
        start_time : float = time.perf_counter()
        examples : SelfPlayExamples = jax.block_until_ready(
            self.collector.collect(start_episode_id=self.next_episode_id)
        )
        elapsed : float = time.perf_counter() - start_time
        self.logger.log_self_play(examples, elapsed, iteration)
        self.next_episode_id += self.env.batch_size

        if examples.episode_ids.shape[0] < self.learner.config.minibatch_size:
            return

        epochs : int = self.learner.config.epochs
        shuffle_key : jax.Array
        for epoch in range(iteration * epochs, (iteration + 1) * epochs):
            key, shuffle_key = jax.random.split(key)
            metrics : Metrics = self.learner.train_epoch(examples, shuffle_key)
            self.logger.log_metrics(metrics, epoch, prefix="train")

    def _run_iteration(self) -> Tuple[bool, Evaluation | None]:
        iteration : int        = self.next_iteration
        incumbent : TrainState = self.learner.state
        promoted  : bool       = False
        result    : Evaluation | None = None
        iter_key : jax.Array
        self.key, iter_key = jax.random.split(self.key)

        try:
            self._collect_and_learn(iter_key, iteration)
            if (self.evaluation_config is not None
                    and int(self.learner.state.step) > int(incumbent.step)):
                eval_key : jax.Array
                self.key, eval_key = jax.random.split(self.key)
                result = self.evaluator.evaluate(
                    candidate_params = self.learner.state.params,
                    opponent_params  = incumbent.params,
                    key              = eval_key,
                )
                score : float | None = result.score_rate
                decision : bool = (
                    result.unfinished == 0
                    and score is not None
                    and score > 0.5
                )
                self.logger.log_evaluation(result, iteration, decision)
                promoted = decision
        finally:
            if self.evaluation_config is not None and not promoted:
                self.learner.restore(incumbent)

        self.next_iteration += 1
        return promoted, result

    def train(
            self,
            iterations     : int,
            *,
            checkpoint_dir : str | Path | None = None,
        ) -> TrainingReport:
        self._ensure_open()
        if iterations < 0:
            raise ValueError("iterations must be nonnegative.")

        directory       : Path | None = (
            Path(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
        )
        promotions      : int = 0
        last_evaluation : Evaluation | None = None
        checkpoint_path : Path | None = None

        for _ in tqdm(range(iterations), desc='Training', unit='iteration'):
            promoted, result = self._run_iteration()
            promotions += int(promoted)
            if result is not None:
                last_evaluation = result

            if directory is not None:
                checkpoint_path = directory.resolve() / f"iteration_{self.next_iteration:06d}"
                self.save_checkpoint(checkpoint_path)
            self.logger.flush()

        return TrainingReport(
            iterations      = iterations,
            promotions      = promotions,
            optimizer_step  = int(self.learner.state.step),
            next_iteration  = self.next_iteration,
            last_evaluation = last_evaluation,
            checkpoint_path = checkpoint_path,
        )
