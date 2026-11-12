"""Tests for the MCTX adapter using real PGX, ScorchNeuralNetwork, and MuZero search."""

import unittest

import jax
import jax.numpy as jnp
import mctx
import numpy as np

from Go.env import ScorchEnv
from MCTX.mctx import MCTX, SearchConfig
from Scorch.scorch_nn import ScorchNeuralNetwork


def assert_tree_equal(test, actual, expected):
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class MCTXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=2, seed=91)
        cls.initial_state, cls.initial_key = cls.env.state, cls.env.key
        cls.model = ScorchNeuralNetwork()
        cls.params = cls.model.init(
            jax.random.PRNGKey(123), cls.env.observation()
        )["params"]
        cls.searcher = MCTX(cls.model, cls.env)
        cls.config = SearchConfig(num_simulations=4, max_depth=2)

    def setUp(self):
        self.env.state, self.env.key = self.initial_state, self.initial_key
        self.key = jax.random.PRNGKey(456)

    def assert_valid_search(self, actions, weights, env):
        self.assertEqual(actions.shape, (env.batch_size,))
        self.assertTrue(jnp.issubdtype(actions.dtype, jnp.integer))
        self.assertEqual(weights.shape, (env.batch_size, 82))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((np.asarray(weights) >= 0).all())
        np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-6)
        np.testing.assert_array_equal(
            env.legal_action_mask[jnp.arange(env.batch_size), actions], True
        )
        np.testing.assert_array_equal(
            np.asarray(weights)[~np.asarray(env.legal_action_mask)], 0
        )

    def test_root_masks_illegal_moves_without_changing_legal_logits(self):
        self.env.step(jnp.array([0, 10]))
        logits, _ = self.model.apply({"params": self.params}, self.env.observation())
        root = self.searcher._root_fn(self.params, self.env.state)
        mask = np.asarray(self.env.legal_action_mask)
        np.testing.assert_allclose(np.asarray(root.prior_logits)[mask],
                                   np.asarray(logits)[mask], atol=1e-6)
        self.assertTrue(np.isneginf(np.asarray(root.prior_logits)[~mask]).all())

    def test_root_preserves_network_values_and_complete_state_embedding(self):
        self.env.step(jnp.array([0, 10]))
        before = self.env.state
        _, values = self.model.apply({"params": self.params}, self.env.observation())
        root = self.searcher._root_fn(self.params, before)
        np.testing.assert_allclose(root.value, values, atol=1e-6)
        assert_tree_equal(self, root.embedding, before)
        assert_tree_equal(self, self.env.state, before)

    def test_recurrent_live_transition_matches_pgx_and_network(self):
        actions = jnp.array([0, 10])
        before = self.env.state
        output, state = self.searcher._recurrent_fn(
            self.params, self.key, actions, before
        )
        for i in range(2):
            expected = self.env.env.step(
                jax.tree_util.tree_map(lambda leaf: leaf[i], before), actions[i]
            )
            assert_tree_equal(
                self, jax.tree_util.tree_map(lambda leaf: leaf[i], state), expected
            )
        logits, values = self.model.apply(
            {"params": self.params}, state.observation.astype(jnp.float32)
        )
        mask = np.asarray(state.legal_action_mask)
        np.testing.assert_allclose(np.asarray(output.prior_logits)[mask],
                                   np.asarray(logits)[mask], atol=1e-6)
        self.assertTrue(np.isneginf(np.asarray(output.prior_logits)[~mask]).all())
        np.testing.assert_allclose(output.value, values, atol=1e-6)
        np.testing.assert_array_equal(output.reward, [0, 0])
        # Alternating-player zero-sum search must flip future values' sign.
        np.testing.assert_array_equal(output.discount, [-1, -1])
        self.assertEqual(output.reward.dtype, jnp.float32)
        self.assertEqual(output.discount.dtype, jnp.float32)
        self.assertEqual(output.value.dtype, jnp.float32)
        assert_tree_equal(self, self.env.state, before)

    def test_terminal_reward_uses_the_player_who_took_the_action(self):
        self.env.step(jnp.array([81, 0]))
        output, state = self.searcher._recurrent_fn(
            self.params, self.key, jnp.array([81, 10]), self.env.state
        )
        np.testing.assert_array_equal(state.terminated, [True, False])
        # On an empty board, white (the second passer) wins by komi.
        # Reading rewards using the NEXT player would incorrectly yield -1.
        np.testing.assert_array_equal(output.reward, [1, 0])
        np.testing.assert_array_equal(output.discount, [0, -1])

    def test_new_terminal_state_has_zero_value_and_finite_neutral_policy(self):
        self.env.step(jnp.array([81, 0]))
        output, _ = self.searcher._recurrent_fn(
            self.params, self.key, jnp.array([81, 10]), self.env.state
        )
        np.testing.assert_array_equal(output.prior_logits[0], np.zeros(82))
        self.assertEqual(float(output.value[0]), 0)
        self.assertTrue(np.isfinite(output.prior_logits[0]).all())

    def test_finished_embeddings_do_not_repeat_rewards_or_bootstrap_values(self):
        self.env.step(jnp.array([81, 0]))
        self.env.step(jnp.array([81, 10]))
        finished = self.env.state
        for flag in ("terminated", "truncated"):
            with self.subTest(done_flag=flag):
                # Exercise both kinds of completed PGX State with the same
                # nonzero outcome, including truncation absent from normal Go.
                state = finished.replace(
                    terminated=jnp.array([flag == "terminated", False]),
                    truncated=jnp.array([flag == "truncated", False]),
                )
                output, next_state = self.searcher._recurrent_fn(
                    self.params, self.key, jnp.array([81, 20]), state
                )
                np.testing.assert_array_equal(output.reward, [0, 0])
                np.testing.assert_array_equal(output.discount, [0, -1])
                self.assertEqual(float(output.value[0]), 0)
                np.testing.assert_array_equal(output.prior_logits[0], 0)
                np.testing.assert_array_equal(next_state.observation[0],
                                              state.observation[0])

    def test_search_returns_legal_actions_and_normalized_visit_targets(self):
        self.env.step(jnp.array([0, 10]))
        actions, weights = self.searcher.search(self.params, self.env, self.config, self.key)
        self.assert_valid_search(actions, weights, self.env)

    def test_search_preserves_live_state_random_key_and_model_parameters(self):
        self.env.step(jnp.array([0, 10]))
        state_before, key_before = self.env.state, self.env.key
        params_before = jax.tree_util.tree_map(np.array, self.params)
        self.searcher.search(self.params, self.env, self.config, self.key)
        assert_tree_equal(self, self.env.state, state_before)
        np.testing.assert_array_equal(self.env.key, key_before)
        assert_tree_equal(self, self.params, params_before)

    def test_search_is_reproducible_with_the_same_rng_key(self):
        first = self.searcher.search(self.params, self.env, self.config, self.key)
        second = self.searcher.search(self.params, self.env, self.config, self.key)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_search_uses_the_supplied_environment_and_handles_one_legal_action(self):
        supplied = ScorchEnv(batch_size=2, seed=8)
        supplied.state = supplied.state.replace(
            legal_action_mask=jnp.zeros((2, 82), dtype=bool).at[:, 81].set(True)
        )
        before = supplied.state
        actions, weights = self.searcher.search(self.params, supplied, self.config, self.key)
        self.assert_valid_search(actions, weights, supplied)
        np.testing.assert_array_equal(actions, [81, 81])
        np.testing.assert_array_equal(weights[:, :81], 0)
        np.testing.assert_array_equal(weights[:, 81], [1, 1])
        assert_tree_equal(self, supplied.state, before)

    def test_search_supports_a_single_game_with_no_depth_limit(self):
        env = ScorchEnv(batch_size=1, seed=12)
        env.step(jnp.array([0]))
        searcher = MCTX(self.model, env)
        config = SearchConfig(num_simulations=2, max_depth=None)
        self.assert_valid_search(*searcher.search(self.params, env, config, self.key), env)

    def test_single_simulation_and_depth_one_produce_a_legal_one_hot_target(self):
        # The same searcher must accept different settings on successive calls.
        self.searcher.search(self.params, self.env, self.config, self.key)
        config = SearchConfig(num_simulations=1, max_depth=1)
        actions, weights = self.searcher.search(self.params, self.env, config, self.key)
        self.assert_valid_search(actions, weights, self.env)
        np.testing.assert_array_equal(np.count_nonzero(weights, axis=-1), [1, 1])
        np.testing.assert_array_equal(weights[jnp.arange(2), actions], [1, 1])

    def test_default_search_config_runs_successfully(self):
        self.assert_valid_search(
            *self.searcher.search(self.params, self.env, SearchConfig(), self.key), self.env
        )

    def test_zero_temperature_selects_a_most_visited_action(self):
        actions, weights = self.searcher.search(
            self.params, self.env, self.config, self.key, temperature=0.0
        )
        self.assert_valid_search(actions, weights, self.env)
        np.testing.assert_array_equal(
            weights[jnp.arange(2), actions], weights.max(axis=-1)
        )
        # Temperature changes action sampling, not the search visit targets.
        _, hot_weights = self.searcher.search(
            self.params, self.env, self.config, self.key, temperature=2.0
        )
        np.testing.assert_array_equal(hot_weights, weights)

    def test_custom_hyperparameters_match_direct_mctx_policy(self):
        config = SearchConfig(
            num_simulations=8, max_depth=3, pb_c_init=2.0, pb_c_base=100.0,
            dirichlet_fraction=0.5, dirichlet_alpha=0.7,
        )

        @jax.jit
        def reference_policy(params, state, key):
            # Independent, literal settings check the adapter's forwarding
            # against the real library, including temperature.
            output = mctx.muzero_policy(
                params=params,
                rng_key=key,
                root=self.searcher._root_fn(params, state),
                recurrent_fn=self.searcher._recurrent_fn,
                invalid_actions=~state.legal_action_mask,
                num_simulations=8,
                max_depth=3,
                pb_c_init=2.0,
                pb_c_base=100.0,
                dirichlet_fraction=0.5,
                dirichlet_alpha=0.7,
                temperature=0.3,
            )
            return output.action, output.action_weights

        self.env.step(jnp.array([0, 10]))
        actual = self.searcher.search(
            self.params, self.env, config, self.key, temperature=0.3
        )
        expected = reference_policy(self.params, self.env.state, self.key)
        self.assert_valid_search(*actual, self.env)
        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_array_equal(actual_array, expected_array)


if __name__ == "__main__":
    unittest.main()
