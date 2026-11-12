import jax
import jax.numpy as jnp

import chex
import mctx

import pgx
import pgx.core

import functools

from dataclasses import dataclass
from Go.env import ScorchEnv
from Scorch.scorch_nn import ScorchNeuralNetwork
from typing import Tuple

@dataclass(frozen=True)
class SearchConfig:
    num_simulations    : int = 32
    max_depth          : int | None = None
    pb_c_init          : float = 1.25
    pb_c_base          : float = 19652.0
    dirichlet_fraction : float = 0.25
    dirichlet_alpha    : float = 0.3

class MCTX:
    def __init__(
            self,
            neural_network  : ScorchNeuralNetwork,
            env             : ScorchEnv,
        ) -> None:
        # Neural Network and Environment Information
        self.neural_network : ScorchNeuralNetwork    = neural_network
        self.env            : ScorchEnv = env

        # MCTX Functions
        self.step_fn = jax.jit(jax.vmap(self.env.env.step))

    def _root_fn(self, params, state : pgx.core.State) -> mctx.RootFnOutput:
        # Run the Neural Network to get the logits and value
        policy, value = self.neural_network.apply(
            {"params" : params}, state.observation.astype(jnp.float32)
        )

        return mctx.RootFnOutput(
            prior_logits = jnp.where(
                state.legal_action_mask,
                policy,
                -jnp.inf
            ),
            value     = value,
            embedding = state
        )

    def _recurrent_fn(
            self, 
            params,
            rng_key    : chex.PRNGKey,
            actions    : chex.Array, 
            embeddings : pgx.core.State
        ) -> Tuple[mctx.RecurrentFnOutput, pgx.core.State]:
        del rng_key

        state : pgx.core.State = embeddings

        players      : jax.Array = state.current_player
        already_done : jax.Array = (state.terminated | state.truncated)

        state = self.step_fn(state, actions)

        # Handle done
        done : jax.Array = (state.terminated | state.truncated)

        # Handle Reward
        reward : jax.Array = jnp.where(
            already_done,
            0.0,
            state.rewards[jnp.arange(actions.shape[0]), players]
        ).astype(jnp.float32)

        # Handle future discounted expected value
        discount : jax.Array = jnp.where(done, 0, -1).astype(jnp.float32)

        # Handle Policy logits and Value
        policy, value = self.neural_network.apply(
            {"params" : params}, state.observation.astype(jnp.float32)
        )

        # Handle Policy first
        policy_logits : jax.Array = jnp.where(state.legal_action_mask, policy, -jnp.inf)

        # Terminal States have no meaningful policy. keep their policy finite
        policy_logits = jnp.where(done[:, None], 0.0, policy_logits)

        # Handle Value
        value = jnp.where(done, 0, value).astype(jnp.float32)

        output : mctx.RecurrentFnOutput = mctx.RecurrentFnOutput(
            reward       = reward,
            discount     = discount,
            prior_logits = policy_logits,
            value        = value
        )

        return output, state

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def _run_mctx(
            self,
            params,
            state       : pgx.core.State,
            rng_key     : chex.PRNGKey,
            config      : SearchConfig,
            temperature : float
        ) -> Tuple[chex.Array, chex.Array]:
        root : mctx.RootFnOutput = self._root_fn(params, state)

        policy_output = mctx.muzero_policy(
            params             = params,
            rng_key            = rng_key,
            root               = root,
            recurrent_fn       = self._recurrent_fn,
            invalid_actions    = ~state.legal_action_mask,
            num_simulations    = config.num_simulations,
            max_depth          = config.max_depth,
            pb_c_init          = config.pb_c_init,
            pb_c_base          = config.pb_c_base,
            dirichlet_fraction = config.dirichlet_fraction,
            dirichlet_alpha    = config.dirichlet_alpha,
            temperature        = temperature,
        )

        return policy_output.action, policy_output.action_weights

    def search(
            self, 
            params,
            env         : ScorchEnv,
            config      : SearchConfig,
            rng_key     : chex.Array,
            temperature : float = 1.0
        ) -> Tuple[jax.Array, jax.Array]:

        return self._run_mctx(
            params, env.state, rng_key, config, temperature
        )