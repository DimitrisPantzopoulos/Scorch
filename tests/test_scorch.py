"""Behavioral tests for the Scorch agent with real network and MCTX search."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.scorch import Scorch


def assert_tree_equal(test, actual, expected):
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class ScorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=2, seed=21)
        cls.initial_state, cls.env_key = cls.env.state, cls.env.key
        cls.config = SearchConfig(
            num_simulations=4, max_depth=2, pb_c_init=2.0, pb_c_base=100.0,
            dirichlet_fraction=0.5, dirichlet_alpha=0.7,
        )
        cls.agent = Scorch(cls.config, cls.env, seed=123)
        cls.twin = Scorch(cls.config, cls.env, seed=123)
        cls.initial_key = cls.agent.prng_key
        cls.initial_params = cls.agent.params

    def setUp(self):
        self.env.state, self.env.key = self.initial_state, self.env_key
        for agent in (self.agent, self.twin):
            agent.prng_key = self.initial_key
            agent.params = self.initial_params
            agent.search_config = self.config

    def assert_valid_actions(self, actions, targets, env):
        self.assertEqual(actions.shape, (env.batch_size,))
        self.assertTrue(jnp.issubdtype(actions.dtype, jnp.integer))
        self.assertEqual(targets.shape, (env.batch_size, 82))
        self.assertTrue(np.isfinite(targets).all())
        self.assertTrue((np.asarray(targets) >= 0).all())
        np.testing.assert_allclose(targets.sum(axis=-1), 1, atol=1e-6)
        np.testing.assert_array_equal(
            env.legal_action_mask[jnp.arange(env.batch_size), actions], True
        )
        np.testing.assert_array_equal(
            np.asarray(targets)[~np.asarray(env.legal_action_mask)], 0
        )

    def test_seed_controls_parameter_initialization_and_predictions(self):
        # Construct fresh agents so setUp cannot mask initialization mistakes.
        same_seed = Scorch(self.config, self.env, seed=123)
        different_seed = Scorch(self.config, self.env, seed=124)
        assert_tree_equal(self, same_seed.params, self.initial_params)
        self.assertTrue(any(
            not np.array_equal(left, right)
            for left, right in zip(jax.tree_util.tree_leaves(same_seed.params),
                                   jax.tree_util.tree_leaves(different_seed.params))
        ))
        inputs = self.env.observation()
        same_predictions = same_seed.neural_network.apply(
            {"params": same_seed.params}, inputs
        )
        other_predictions = different_seed.neural_network.apply(
            {"params": different_seed.params}, inputs
        )
        self.assertTrue(any(not np.allclose(left, right)
                            for left, right in zip(same_predictions, other_predictions)))

    def test_act_returns_legal_actions_and_normalized_targets(self):
        self.env.step(jnp.array([0, 10]))
        actions, targets = self.agent.act(self.env)
        self.assert_valid_actions(actions, targets, self.env)
        # The selected actions can be applied to the actual games.
        players = self.env.current_player
        self.env.step(actions)
        np.testing.assert_array_equal(self.env.current_player, 1 - players)

    def test_act_forwards_current_parameters_config_temperature_and_split_key(self):
        self.env.step(jnp.array([0, 10]))
        # Replacing parameters must affect search; act must not cache init params.
        self.agent.params = jax.tree_util.tree_map(jnp.zeros_like, self.initial_params)
        next_key, search_key = jax.random.split(self.agent.prng_key)
        expected = self.agent.mctx.search(
            params=self.agent.params,
            env=self.env,
            config=self.config,
            rng_key=search_key,
            temperature=0.3,
        )
        actual = self.agent.act(self.env, temperature=0.3)
        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_array_equal(actual_array, expected_array)
        np.testing.assert_array_equal(self.agent.prng_key, next_key)

    def test_each_act_advances_and_splits_the_stored_random_key(self):
        previous = self.agent.prng_key
        for _ in range(2):
            expected_next, _ = jax.random.split(previous)
            self.agent.act(self.env)
            np.testing.assert_array_equal(self.agent.prng_key, expected_next)
            self.assertFalse(np.array_equal(self.agent.prng_key, previous))
            previous = expected_next

    def test_act_preserves_environment_state_environment_key_and_parameters(self):
        self.env.step(jnp.array([0, 10]))
        state_before, env_key_before = self.env.state, self.env.key
        params_before = jax.tree_util.tree_map(np.array, self.agent.params)
        self.agent.act(self.env)
        assert_tree_equal(self, self.env.state, state_before)
        np.testing.assert_array_equal(self.env.key, env_key_before)
        assert_tree_equal(self, self.agent.params, params_before)

    def test_equal_seeded_agents_reproduce_a_sequence_of_actions(self):
        for _ in range(2):
            first = self.agent.act(self.env)
            second = self.twin.act(self.env)
            for actual, expected in zip(first, second):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(self.agent.prng_key, self.twin.prng_key)
            self.env.step(first[0])

    def test_act_uses_supplied_environment_with_a_different_batch_size(self):
        supplied = ScorchEnv(batch_size=1, seed=7)
        supplied.step(jnp.array([0]))
        # Restrict the supplied root to passing to catch accidentally using
        # the constructor environment, which has two games and 82 legal moves.
        supplied.state = supplied.state.replace(
            legal_action_mask=jnp.zeros((1, 82), dtype=bool).at[:, 81].set(True)
        )
        before = supplied.state
        actions, targets = self.agent.act(supplied)
        self.assert_valid_actions(actions, targets, supplied)
        np.testing.assert_array_equal(actions, [81])
        np.testing.assert_array_equal(targets[:, :81], 0)
        np.testing.assert_array_equal(targets[:, 81], [1])
        assert_tree_equal(self, supplied.state, before)
        assert_tree_equal(self, self.env.state, self.initial_state)

    def test_act_uses_updated_search_configuration(self):
        self.agent.act(self.env)
        self.agent.search_config = SearchConfig(num_simulations=1, max_depth=1)
        actions, targets = self.agent.act(self.env)
        self.assert_valid_actions(actions, targets, self.env)
        np.testing.assert_array_equal(np.count_nonzero(targets, axis=-1), [1, 1])
        np.testing.assert_array_equal(targets[jnp.arange(2), actions], [1, 1])

    def test_zero_temperature_selects_a_most_visited_action(self):
        actions, targets = self.agent.act(self.env, temperature=0.0)
        self.assert_valid_actions(actions, targets, self.env)
        np.testing.assert_array_equal(targets[jnp.arange(2), actions],
                                      targets.max(axis=-1))


if __name__ == "__main__":
    unittest.main()
