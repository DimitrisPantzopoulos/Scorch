import chex
import jax
import optax
import pgx
import pgx.core
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
from torch import float32

from MCTX.mctx     import MCTX, SearchConfig
from Scorch.scorch import Scorch
from Go.env        import ScorchEnv

import functools
import time
import math

from typing import Any, TypeAlias, Tuple, Dict, cast
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

Metrics : TypeAlias = Dict[str, jax.Array | int | float]
Record  : TypeAlias = Dict[str, jax.Array]

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

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate     : float = 3e-4
    weight_decay      : float = 1e-4
    max_grad_norm     : float = 1.0
    value_loss_weight : float = 1.0

    epochs            : int   = 256
    batch_size        : int   = 128

@dataclass(frozen=True)
class EvaluationConfig:
    batch_size    : int = 32
    num_steps     : int = 256
    seed          : int = 42
    search_config : SearchConfig | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("Evaluation batch_size must be positive and even.")
        if self.num_steps < 1:
            raise ValueError("Evaluation num_steps must be positive.")

@dataclass(slots=True)
class Checkpoint:
    state           : TrainState
    next_episode_id : int
    trainer_key     : jax.Array
    scorch_key      : jax.Array
    env_key         : jax.Array
    next_iteration  : int = 0

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

class SelfPlayCollector:
    @functools.partial(jax.jit, static_argnums=(0, 2, 6))
    def _play(
            self,
            params,
            mctx         : MCTX,
            state        : pgx.core.State,
            temperature  : float,
            search_key   : jax.Array,
            config       : SearchConfig,
            _
        ) -> Tuple[pgx.core.State, jax.Array]:
            moves, policy_targets = mctx._run_mctx(
                params, state, search_key, config, temperature
            )

            return mctx.step_fn(state, moves), policy_targets

    @functools.partial(jax.jit, static_argnums=(0,))
    def _skip(
            self,
            state : pgx.core.State,
            _ 
        ) -> Tuple[pgx.core.State, jax.Array]:

        return state, jnp.zeros(
            state.legal_action_mask.shape, dtype=jnp.float32
        )
    
    def _step(
            self, 
            carry,
            params,
            mctx        : MCTX,
            temperature : float,
            config      : SearchConfig,
            _
        ) -> Tuple[Tuple[pgx.core.State, jax.Array, jax.Array], Record]:
        state, key, final_rewards = carry

        active : jax.Array = ~(state.terminated | state.truncated)

        key, search_key = jax.random.split(key)

        next_state, policy_targets = cast(
            tuple[pgx.core.State, jax.Array],
            jax.lax.cond(
                active.any(),
                lambda operand : self._play(params, mctx, state, temperature, search_key, config, operand),
                lambda operand : self._skip(state, operand),
                operand = None,
            )
        )

        newly_finished : jax.Array = active & next_state.terminated

        final_rewards = jnp.where(
            newly_finished[:, None],
            next_state.rewards,
            final_rewards
        )

        record : Record = {
            "observations"   : state.observation,
            "policy_targets" : policy_targets,
            "side_to_move"   : state.current_player,
            "active"         : active,
        }

        return (next_state, key, final_rewards), record
    
    def _scan_step(
            self, 
            carry,
            params,
            mctx        : MCTX,
            config      : SearchConfig,
            temperature : float,
            scan_input  : None
        ) -> Tuple[Tuple[pgx.core.State, jax.Array, jax.Array], Record]:
        return self._step(carry, params, mctx, temperature, config, scan_input)
    
    @functools.partial(jax.jit, static_argnames=('self', 'mctx', 'config', 'num_steps'))
    def rollout(
            self,
            params,
            mctx        : MCTX,
            state       : pgx.core.State,
            rng_key     : jax.Array,
            config      : SearchConfig,
            temperature : float = 1.0,
            num_steps   : int   = 256,
        ) -> Tuple[Tuple[pgx.core.State, jax.Array, jax.Array], Record]:
            initial_carry : Tuple[pgx.core.State, jax.Array, jax.Array] = (state, rng_key, jnp.zeros_like(state.rewards))

            final_carry, history = jax.lax.scan(
                lambda carry, _ : self._scan_step(carry, params, mctx, config, temperature, None),
                initial_carry,
                xs     = None,
                length = num_steps
            )

            return final_carry, history

    def collect_self_play_games(
            self,
            scorch           : Scorch,
            env              : ScorchEnv,
            start_episode_id : int   = 0,
            temperature      : float = 1.0,
            num_steps        : int   = 256,
        ) -> SelfPlayExamples:
        env.reset()

        scorch.prng_key, rollout_key = jax.random.split(scorch.prng_key)

        final_carry, history = self.rollout(
            params      = scorch.params,
            mctx        = scorch.mctx,
            state       = env.state,
            rng_key     = rollout_key,
            config      = scorch.search_config,
            temperature = temperature,
            num_steps   = num_steps,
        )

        final_state, _, final_rewards = final_carry

        # Assign Value Targets
        n_steps, batch_size = history['side_to_move'].shape

        game_ids : jax.Array = jnp.arange(batch_size)[None, :] # (1, B)

        value_targets : jax.Array = final_rewards[
            game_ids, history['side_to_move']
        ]

        # Before reshape: (T, B); after reshape: (T * B,)
        valid : jax.Array = (history['active'] & final_state.terminated[None, :]).reshape(-1)

        observations   : jax.Array = history['observations']
        policy_targets : jax.Array = history['policy_targets']

        return SelfPlayExamples(
            observations   = observations.reshape(-1,   *observations.shape[2:])[valid].astype(jnp.float32),
            policy_targets = policy_targets.reshape(-1,  policy_targets.shape[-1])[valid],
            value_targets  = value_targets.reshape(-1)[valid],
            side_to_moves  = history['side_to_move'].reshape(-1)[valid],
            episode_ids    = jnp.broadcast_to(
                    start_episode_id + jnp.arange(batch_size),
                    (n_steps, batch_size),
                ).reshape(-1)[valid],
        )

class ScorchEvaluator:
    def _step(
        self,
        carry,
        candidate_params,
        opponent_params,
        mctx              : MCTX,
        search_config     : SearchConfig,
        temperature       : float,
        candidate_player  : int,
        _,
    ):
        state, key, final_rewards = carry

        key, step_key = jax.random.split(key)

        active : jax.Array = ~(state.terminated | state.truncated)

        def play(_):
            candidate_turn = state.current_player == candidate_player

            def moves(params):
                _, policy_targets = mctx._run_mctx(
                    params, state, step_key, search_config, temperature
                )
                return jnp.argmax(policy_targets, axis=1)

            def mixed_moves(_):
                candidate_moves = moves(candidate_params)
                opponent_moves = moves(opponent_params)
                return jnp.where(candidate_turn, candidate_moves, opponent_moves)

            # Player IDs are randomized independently by PGX. A batch can
            # therefore contain turns belonging to both networks at once.
            actions = jax.lax.cond(
                jnp.all(~active | candidate_turn),
                lambda _: moves(candidate_params),
                lambda _: jax.lax.cond(
                    jnp.all(~active | ~candidate_turn),
                    lambda _: moves(opponent_params), mixed_moves, operand=None,
                ),
                operand=None,
            )
            return mctx.step_fn(state, actions)

        def skip(_):
            return state

        next_state : pgx.core.State = jax.lax.cond(
            active.any(),
            play,
            skip,
            operand=None
        )

        newly_finished : jax.Array = active & next_state.terminated

        final_rewards = jnp.where(
            newly_finished[:, None],
            next_state.rewards,
            final_rewards
        )

        return (next_state, key, final_rewards), None

    @functools.partial(jax.jit, static_argnames=(
        'self', 'mctx', 'search_config', 'candidate_player', 'num_steps',
    ))
    def _evaluate_colour(
        self,
        candidate_params,
        opponent_params,
        mctx             : MCTX,
        state            : pgx.core.State,
        key              : jax.Array,
        search_config    : SearchConfig,
        temperature      : float,
        candidate_player : int,
        num_steps        : int
    ) -> Dict[str, jax.Array]:
        final_rewards : jax.Array = jnp.zeros_like(state.rewards)

        initial_carry = (state, key, final_rewards)

        final_carry, _ = jax.lax.scan(
            lambda carry, _ : self._step(carry, candidate_params, opponent_params, mctx, search_config, temperature, candidate_player, _),
            initial_carry,
            xs     = None,
            length = num_steps
        )

        final_state, _, final_rewards = final_carry

        candidate_rewards : jax.Array = final_rewards[:, candidate_player]

        wins : jax.Array = jnp.sum(
            final_state.terminated & (candidate_rewards > 0)
        )

        losses : jax.Array = jnp.sum(
            final_state.terminated & (candidate_rewards < 0)
        )

        draws : jax.Array = jnp.sum(
            final_state.terminated & (candidate_rewards == 0)
        )

        unfinished : jax.Array = jnp.sum(
            ~final_state.terminated
        )

        return {
            "wins"       : wins,
            "losses"     : losses,
            "draws"      : draws,
            "unfinished" : unfinished,
        }

    def evaluate(
            self,
            candidate_params,
            opponent_params,
            mctx             : MCTX,
            key              : jax.Array,
            search_config    : SearchConfig,
            temperature      : float,
            num_steps        : int,
            batch_size       : int,
            seed             : int
        ) -> Evaluation:
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("batch_size must be positive and even.")
        if num_steps < 0:
            raise ValueError("num_steps must be nonnegative.")

        scorch_env : ScorchEnv = ScorchEnv(batch_size // 2, seed)

        state_1 : pgx.core.State = scorch_env.state
        key, key_1 = jax.random.split(key)

        state_2 : pgx.core.State = scorch_env.state
        key, key_2 = jax.random.split(key)

        env_1_evaluation : Dict[str, jax.Array] = self._evaluate_colour(
            candidate_params,
            opponent_params,
            mctx,
            state_1,
            key_1,
            search_config,
            temperature,
            0,
            num_steps
        )

        env_2_evaluation : Dict[str, jax.Array] = self._evaluate_colour(
            candidate_params,
            opponent_params,
            mctx,
            state_2,
            key_2,
            search_config,
            temperature,
            1,
            num_steps
        )

        return Evaluation(
            wins       = env_1_evaluation['wins'].item() + env_2_evaluation['wins'].item(),
            losses     = env_1_evaluation['losses'].item() + env_2_evaluation['losses'].item(),
            draws      = env_1_evaluation['draws'].item() + env_2_evaluation['draws'].item(),
            unfinished = env_1_evaluation['unfinished'].item() + env_2_evaluation['unfinished'].item()
        )

class ScorchTrainer:
    def __init__(
        self,
        scorch          : Scorch,
        env             : ScorchEnv,
        training_config : TrainingConfig,
        training_run    : str | None = None,
    ) -> None:
        self.scorch : Scorch         = scorch
        self.env    : ScorchEnv      = env
        self.config : TrainingConfig = training_config

        optimizer = optax.chain(
            # Limit excessively large gradient updates.
            optax.clip_by_global_norm(self.config.max_grad_norm),

            # Adapt updates using gradient history, with weight decay.
            optax.adamw(
                learning_rate = self.config.learning_rate,
                weight_decay  = self.config.weight_decay,
            )
        )

        self.train_state : TrainState = TrainState.create(
            apply_fn = self.scorch.neural_network.apply,
            params   = self.scorch.params,
            tx       = optimizer
        )

        self.writer              : SummaryWriter     = SummaryWriter(f"runs/{training_run}_{datetime.now():%Y%m%d_%H%M%S_%f}")
        self.collector           : SelfPlayCollector = SelfPlayCollector()
        self.evaluator           : ScorchEvaluator   = ScorchEvaluator()
        self.next_episode_id     : int               = 0
        self.next_iteration      : int               = 0

    def _make_scorch_checkpoint(self, key : jax.Array) -> Checkpoint:
        return Checkpoint(
            state           = self.train_state,
            next_episode_id = self.next_episode_id,
            trainer_key     = key,
            scorch_key      = self.scorch.prng_key,
            env_key         = self.env.key,
            next_iteration  = self.next_iteration,
        )

    def _save_checkpoint(self, path : str, key : jax.Array) -> None:
        with ocp.StandardCheckpointer() as checkpointer:
            abs_path : Path = Path(path).resolve()

            scorch_checkpoint : Checkpoint = self._make_scorch_checkpoint(key)

            scorch_tree : Dict[str, Any] = {
                field.name : getattr(scorch_checkpoint, field.name)
                for field in fields(scorch_checkpoint)
            }

            checkpointer.save(abs_path, scorch_tree)
            checkpointer.wait_until_finished()

    def load_checkpoint(self, path : str) -> jax.Array:
        """Restore training state and return the trainer PRNG key."""
        with ocp.StandardCheckpointer() as checkpointer:
            # 1. Resolve the absolute path
            abs_path : Path = Path(path).resolve()

            # 2. Make Placeholder checkpoint which will be populate later
            template : Checkpoint = self._make_scorch_checkpoint(jax.random.PRNGKey(0))

            template_tree : Dict[str, Any] = {
                field.name : getattr(template, field.name)
                for field in fields(template)
            }

            # 3. Load the Checkpoint
            if not abs_path.exists():
                raise FileNotFoundError(abs_path)
            metadata = checkpointer.metadata(abs_path).item_metadata
            if metadata is None:
                raise ValueError("Checkpoint metadata is missing.")
            if "next_iteration" not in metadata.tree:
                # Previous checkpoints stored only episode IDs. Their fixed
                # batch collector allocated one batch of IDs per iteration.
                template_tree.pop("next_iteration")
            restored = checkpointer.restore(abs_path, target=template_tree)
            if "next_iteration" not in restored:
                restored["next_iteration"] = int(restored["next_episode_id"]) // self.env.batch_size
            loaded_checkpoint : Checkpoint = Checkpoint(**restored)

            # 4. Assign all the variables back
            self.train_state     = loaded_checkpoint.state
            self.next_episode_id = loaded_checkpoint.next_episode_id
            self.next_iteration  = loaded_checkpoint.next_iteration
            trainer_key          = loaded_checkpoint.trainer_key
            self.scorch.prng_key = loaded_checkpoint.scorch_key
            self.env.key         = loaded_checkpoint.env_key

            self.scorch.params = self.train_state.params

        return trainer_key

    def _log_metrics(self, metrics : Metrics, timestep : int, prefix : str) -> None:
        mean_metrics : Metrics = jax.tree_util.tree_map(
            lambda x : jnp.mean(x),
            metrics
        )

        for name, value in mean_metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', float(value), timestep)

    def _log_evaluation(self, result : Evaluation, iteration : int, promoted : bool) -> None:
        metrics : Metrics = {
            "wins": result.wins,
            "losses": result.losses,
            "draws": result.draws,
            "unfinished": result.unfinished,
            "completed_games": result.completed_games,
            "promoted": int(promoted),
        }
        if result.score_rate is not None:
            metrics["score_rate"] = result.score_rate
        elo = result.elo_delta
        if elo is not None and math.isfinite(elo):
            metrics["elo_delta"] = elo
        self._log_metrics(metrics, iteration, prefix="evaluation")

    def _calculate_self_play_metrics(self, examples : SelfPlayExamples, time_delta) -> Metrics:
        self_play_metrics : Metrics = {
            "positions"      : int(examples.episode_ids.shape[0]),
            "games_finished" : int(jnp.unique(examples.episode_ids).shape[0]),
            "time_delta"     : float(time_delta)
        }

        return self_play_metrics

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _train_step(
        self,
        train_state        : TrainState,
        self_play_examples : SelfPlayExamples,
        value_loss_weight  : float
    ) -> Tuple[TrainState, Metrics]:
        def _loss_fn(params) -> Tuple[jax.Array, Metrics]:
            policy_logits, predicted_values = cast(
                tuple[jax.Array, jax.Array],
                train_state.apply_fn(
                    {'params' : params}, self_play_examples.observations
                )
            )

            # Predict the outcome - Question: How accurate are our position evaluations?
            value_loss : jax.Array = jnp.mean(
                jnp.square(predicted_values - self_play_examples.value_targets)
            )

            # Imitatation learning from search - Question: How closely do we match MCTX's policy
            policy_loss : jax.Array = optax.softmax_cross_entropy(
                logits = policy_logits,
                labels = self_play_examples.policy_targets
            ).mean()

            total_loss : jax.Array = policy_loss + value_loss * value_loss_weight

            loss_metrics : Metrics = {
                "loss"        : total_loss,
                "policy_loss" : policy_loss,
                "value_loss"  : value_loss,
            }

            return total_loss, loss_metrics

        # Compute gradients of the loss with respect to the parameters.
        (_, loss_metrics), gradients =  jax.value_and_grad(
            _loss_fn,
            has_aux  = True
        )(train_state.params)

        updated_train_state : TrainState = train_state.apply_gradients(grads=gradients)

        return updated_train_state, loss_metrics

    def _shuffle_examples(self, examples : SelfPlayExamples, key : jax.Array) -> SelfPlayExamples:
        indices : jax.Array = jnp.arange(examples.episode_ids.shape[0])

        shuffled_permutation : jax.Array = jax.random.permutation(key, indices)

        return jax.tree_util.tree_map(
            lambda x : x[shuffled_permutation], examples
        )

    def _make_minibatches(self, examples : SelfPlayExamples, batch_size : int) -> SelfPlayExamples:
        # 1. Calculate how many full minibatches fit
        n_examples : int = examples.episode_ids.shape[0]

        n_minibatches : int = n_examples // batch_size

        # 2. Calculate the total number of examples used by full minibatches.
        n_used : int = batch_size * n_minibatches

        # 3. Trim every field to that length
        examples = jax.tree_util.tree_map(lambda x: x[:n_used], examples)

        # 4. Reshape every field to add a minibatch dimension
        examples = jax.tree_util.tree_map(lambda x: x.reshape(n_minibatches, batch_size, *x.shape[1:]), examples)
        
        return examples

    def _train_batch(
            self,
            self_play_examples : SelfPlayExamples,
        ) -> Metrics:
        self.train_state, metrics = self._train_step(
            self.train_state,
            self_play_examples,
            self.config.value_loss_weight
        )

        # Update the network params.
        self.scorch.params = self.train_state.params

        return metrics

    def _update_one_batch(self, carry, minibatch : SelfPlayExamples) -> Tuple[TrainState, Metrics]:
        return self._train_step(carry, minibatch, self.config.value_loss_weight)
    
    def _train_epoch(self, examples : SelfPlayExamples, key : jax.Array, batch_size : int) -> Metrics:
        # 1. Shuffle examples
        shuffled_examples : SelfPlayExamples = self._shuffle_examples(examples, key)

        # 2. Make them mini-batches
        minibatches : SelfPlayExamples = self._make_minibatches(shuffled_examples, batch_size)

        # 3. Loop over each, minibatch and call train_batch()
        self.train_state, metrics = jax.lax.scan(
            self._update_one_batch,
            self.train_state,
            xs=minibatches,
        )

        self.scorch.params = self.train_state.params

        return metrics

    def _train_iteration(
            self, 
            key         : jax.Array, 
            epochs      : int,
            batch_size  : int,
            iteration   : int   = 0,
            num_steps   : int   = 256,
            temperature : float = 1.0
        ) -> None:

        start_time : float = time.perf_counter()

        examples : SelfPlayExamples = jax.block_until_ready(self.collector.collect_self_play_games(
            scorch           = self.scorch,
            env              = self.env,
            start_episode_id = self.next_episode_id,
            temperature      = temperature,
            num_steps        = num_steps,
        ))

        end_time : float = time.perf_counter()

        # Calculate and log self play metrics
        self_play_metrics : Metrics = self._calculate_self_play_metrics(examples, end_time - start_time)

        self._log_metrics(self_play_metrics, iteration, prefix='self_play')

        self.next_episode_id += self.env.batch_size

        if examples.episode_ids.shape[0] < batch_size:
            return
    
        for i in range(iteration * epochs, iteration * epochs + epochs):
            key, shuffle_key = jax.random.split(key)

            loss_metrics : Metrics = self._train_epoch(examples, shuffle_key, batch_size)

            self._log_metrics(loss_metrics, i, prefix='train')

    def train(
            self,
            key               : jax.Array,
            iterations        : int,
            num_steps         : int                     = 256,
            temperature       : float                   = 1.0,
            *,
            evaluation_config : EvaluationConfig | None = EvaluationConfig(),
            checkpoint_dir    : str | Path | None        = None,
        ) -> jax.Array:
        if iterations < 0:
            raise ValueError("iterations must be nonnegative.")

        for _ in range(iterations):
            iteration : int        = int(self.next_iteration)
            incumbent : TrainState = self.train_state

            iter_key : jax.Array
            key, iter_key = jax.random.split(key)

            promoted : bool = False

            try:
                self._train_iteration(
                    iter_key,
                    self.config.epochs,
                    self.config.batch_size,
                    iteration,
                    num_steps,
                    temperature,
                )

                if (
                    evaluation_config is not None
                    and int(self.train_state.step) > int(incumbent.step)
                ):
                    eval_key : jax.Array
                    key, eval_key = jax.random.split(key)

                    search_config : SearchConfig | None = evaluation_config.search_config

                    if search_config is None:
                        search_config = replace(
                            self.scorch.search_config,
                            dirichlet_fraction = 0.0,
                        )

                    result : Evaluation = self.evaluator.evaluate(
                        candidate_params = self.train_state.params,
                        opponent_params  = incumbent.params,
                        mctx             = self.scorch.mctx,
                        key              = eval_key,
                        search_config    = search_config,
                        temperature      = 0.0,
                        num_steps        = evaluation_config.num_steps,
                        batch_size       = evaluation_config.batch_size,
                        seed             = evaluation_config.seed,
                    )

                    score    : float | None = result.score_rate
                    decision : bool         = (
                        result.unfinished == 0
                        and score is not None
                        and score > 0.5
                    )

                    self._log_evaluation(
                        result,
                        iteration,
                        decision,
                    )

                    promoted = decision

            finally:
                if evaluation_config is not None and not promoted:
                    self.train_state   = incumbent
                    self.scorch.params = incumbent.params

            self.next_iteration = iteration + 1

            if checkpoint_dir is not None:
                directory : Path = Path(checkpoint_dir).resolve()

                directory.mkdir(
                    parents  = True,
                    exist_ok = True,
                )

                self._save_checkpoint(
                    str(directory / f"iteration_{iteration + 1:06d}"),
                    key,
                )

            self.writer.flush()

        return key
        

