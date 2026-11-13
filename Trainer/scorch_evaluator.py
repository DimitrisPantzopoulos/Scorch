"""Balanced evaluation bound to an agent and evaluation settings."""
import functools
from dataclasses import replace
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import pgx.core
from Go.env import ScorchEnv
from MCTX.mctx import MCTX, SearchConfig
from Scorch.scorch import Scorch
from .config import EvaluationConfig
from .data import Carry, Evaluation, Parameters

class ScorchEvaluator:
    def __init__(
            self,
            scorch : Scorch,
            config : EvaluationConfig = EvaluationConfig(),
        ) -> None:
        self.scorch : Scorch           = scorch
        self.config : EvaluationConfig = config

    def _step(
            self,
            carry            : Carry,
            candidate_params : Parameters,
            opponent_params  : Parameters,
            mctx             : MCTX,
            search_config    : SearchConfig,
            temperature      : float,
            candidate_player : int,
            _                : None,
        ) -> Tuple[Carry, None]:
        state, key, final_rewards = carry

        key, step_key = jax.random.split(key)

        active : jax.Array = ~(state.terminated | state.truncated)

        def play(_ : None) -> pgx.core.State:
            candidate_turn : jax.Array = state.current_player == candidate_player

            def moves(params : Parameters) -> jax.Array:
                _, policy_targets = mctx._run_mctx(
                    params, state, step_key, search_config, temperature
                )
                return jnp.argmax(policy_targets, axis=1)

            def mixed_moves(_ : None) -> jax.Array:
                candidate_moves : jax.Array = moves(candidate_params)
                opponent_moves  : jax.Array = moves(opponent_params)
                return jnp.where(candidate_turn, candidate_moves, opponent_moves)

            # Player IDs are randomized independently by PGX. A batch can
            # therefore contain turns belonging to both networks at once.
            actions : jax.Array = jax.lax.cond(
                jnp.all(~active | candidate_turn),
                lambda _: moves(candidate_params),
                lambda _: jax.lax.cond(
                    jnp.all(~active | ~candidate_turn),
                    lambda _: moves(opponent_params), mixed_moves, operand=None,
                ),
                operand=None,
            )
            return mctx.step_fn(state, actions)

        def skip(_ : None) -> pgx.core.State:
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
            candidate_params : Parameters,
            opponent_params  : Parameters,
            mctx             : MCTX,
            state            : pgx.core.State,
            key              : jax.Array,
            search_config    : SearchConfig,
            temperature      : float,
            candidate_player : int,
            num_steps        : int,
        ) -> Dict[str, jax.Array]:
        final_rewards : jax.Array = jnp.zeros_like(state.rewards)

        initial_carry : Carry = (state, key, final_rewards)

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
            candidate_params : Parameters,
            opponent_params  : Parameters,
            key              : jax.Array,
            *,
            config           : EvaluationConfig | None = None,
        ) -> Evaluation:
        settings      : EvaluationConfig = self.config if config is None else config
        mctx          : MCTX              = self.scorch.mctx
        batch_size    : int               = settings.num_games
        num_steps     : int               = settings.num_steps
        seed          : int               = settings.seed
        temperature   : float             = 0.0
        search_config : SearchConfig      = settings.search_config or replace(
            self.scorch.search_config,
            dirichlet_fraction = 0.0,
        )

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
