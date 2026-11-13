"""Batched self-play collection bound to an agent and environment."""
import functools
from typing import Tuple, cast
import jax
import jax.numpy as jnp
import pgx.core
from Go.env import ScorchEnv
from MCTX.mctx import MCTX, SearchConfig
from Scorch.scorch import Scorch
from .config import SelfPlayConfig
from .data import Carry, Parameters, Record, SelfPlayExamples

class SelfPlayCollector:
    def __init__(
            self,
            scorch : Scorch,
            env    : ScorchEnv,
            config : SelfPlayConfig = SelfPlayConfig(),
        ) -> None:
        self.scorch : Scorch         = scorch
        self.env    : ScorchEnv      = env
        self.config : SelfPlayConfig = config

    @functools.partial(jax.jit, static_argnums=(0, 2, 6))
    def _play(
            self,
            params      : Parameters,
            mctx        : MCTX,
            state       : pgx.core.State,
            temperature : float,
            search_key  : jax.Array,
            config      : SearchConfig,
            _           : None,
        ) -> Tuple[pgx.core.State, jax.Array]:
        moves, policy_targets = mctx._run_mctx(
            params, state, search_key, config, temperature
        )

        return mctx.step_fn(state, moves), policy_targets

    @functools.partial(jax.jit, static_argnums=(0,))
    def _skip(
            self,
            state : pgx.core.State,
            _,
        ) -> Tuple[pgx.core.State, jax.Array]:
        return state, jnp.zeros(
            state.legal_action_mask.shape, dtype=jnp.float32
        )

    def _step(
            self,
            carry       : Carry,
            params      : Parameters,
            mctx        : MCTX,
            temperature : float,
            config      : SearchConfig,
            _           : None,
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
            carry       : Carry,
            params      : Parameters,
            mctx        : MCTX,
            config      : SearchConfig,
            temperature : float,
            scan_input  : None,
        ) -> Tuple[Tuple[pgx.core.State, jax.Array, jax.Array], Record]:
        return self._step(carry, params, mctx, temperature, config, scan_input)

    @functools.partial(jax.jit, static_argnames=('self', 'mctx', 'config', 'num_steps'))
    def rollout(
            self,
            params      : Parameters,
            mctx        : MCTX,
            state       : pgx.core.State,
            rng_key     : jax.Array,
            config      : SearchConfig,
            temperature : float          = 1.0,
            num_steps   : int            = 256,
        ) -> Tuple[Tuple[pgx.core.State, jax.Array, jax.Array], Record]:
        initial_carry : Tuple[pgx.core.State, jax.Array, jax.Array] = (state, rng_key, jnp.zeros_like(state.rewards))

        final_carry, history = jax.lax.scan(
            lambda carry, _ : self._scan_step(carry, params, mctx, config, temperature, None),
            initial_carry,
            xs     = None,
            length = num_steps
        )

        return final_carry, history

    def collect(
            self,
            start_episode_id : int = 0,
        ) -> SelfPlayExamples:
        num_steps   : int   = self.config.num_steps
        temperature : float = self.config.temperature

        self.env.reset()

        self.scorch.prng_key, rollout_key = jax.random.split(self.scorch.prng_key)

        final_carry, history = self.rollout(
            params      = self.scorch.params,
            mctx        = self.scorch.mctx,
            state       = self.env.state,
            rng_key     = rollout_key,
            config      = self.scorch.search_config,
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
