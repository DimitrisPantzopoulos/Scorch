import jax
import jax.numpy as jnp

import pgx
import pgx.core

from typing import ClassVar, Tuple

class ScorchEnv:
    ENV_ID : ClassVar[pgx.EnvId] = 'go_9x9'

    def __init__(
            self,
            batch_size  : int = 4096,
            seed        : int = 42
        ) -> None:
        # General Environment Information
        self.batch_size : int = batch_size
        self.seed       : int = seed

        # State Environment Information
        self.env : pgx.Env = pgx.make(self.ENV_ID)

        self.init_fn       = jax.jit(jax.vmap(self.env.init))
        self.step_fn       = jax.jit(jax.vmap(self.env.step))
        self.reset_if_done = jax.jit(jax.vmap(self._reset_if_done))

        self.key, init_keys = self._get_keys(jax.random.PRNGKey(self.seed))
        
        self.state = self.init_fn(init_keys)

    def _get_keys(self, key : jax.Array) -> Tuple[jax.Array, jax.Array]:
        key, other_key = jax.random.split(key)

        other_key = jax.random.split(other_key, self.batch_size)

        return key, other_key
    
    def _reset_if_done(self, state : pgx.core.State, key : jax.Array) -> pgx.core.State:
        done = state.terminated | state.truncated

        return jax.lax.cond(
            done,
            lambda k: self.env.init(k),
            lambda _: state,
            key
        )
    
    @property
    def legal_action_mask(self) -> jax.Array:
        return self.state.legal_action_mask

    @property
    def rewards(self) -> jax.Array:
        return self.state.rewards

    @property
    def terminated(self) -> jax.Array:
        return self.state.terminated

    @property
    def current_player(self) -> jax.Array:
        return self.state.current_player

    @property
    def done(self) -> jax.Array:
        return self.state.terminated | self.state.truncated
    
    def reset_finished(self) -> None:
        self.key, reset_key = self._get_keys(self.key)
        self.state = self.reset_if_done(self.state, reset_key)

    def reset(self) -> None:
        self.key, init_keys = self._get_keys(self.key)
        self.state = self.init_fn(init_keys)

    def step(self, actions : jax.Array) -> None:
        self.state = self.step_fn(self.state, actions)

    def observation(self) -> jax.Array:
        return self.state.observation.astype(jnp.float32)

    