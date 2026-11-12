"""Behavioral tests for the real, batched PGX Go environment."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv


def assert_state_equal(test, actual, expected):
    """Compare every field, including PGX's internal board history."""
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class ScorchEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=3, seed=42)

    def setUp(self):
        self.env.reset()

    def test_initial_observations_and_actions_describe_empty_go_boards(self):
        observations = self.env.observation()
        self.assertEqual(observations.shape, (3, 9, 9, 17))
        self.assertEqual(observations.dtype, jnp.float32)
        np.testing.assert_array_equal(observations[..., :16], 0)
        np.testing.assert_array_equal(self.env.legal_action_mask,
                                      np.ones((3, 82), dtype=bool))
        np.testing.assert_array_equal(self.env.rewards, np.zeros((3, 2)))
        np.testing.assert_array_equal(self.env.done, [False, False, False])

    def test_step_matches_individual_pgx_games(self):
        actions = jnp.array([0, 10, 80], dtype=jnp.int32)
        before = self.env.state
        expected = [
            self.env.env.step(
                jax.tree_util.tree_map(lambda leaf: leaf[i], before), actions[i]
            )
            for i in range(3)
        ]
        self.env.step(actions)
        for i, state in enumerate(expected):
            with self.subTest(game=i):
                actual = jax.tree_util.tree_map(lambda leaf: leaf[i], self.env.state)
                assert_state_equal(self, actual, state)
        np.testing.assert_array_equal(
            self.env.legal_action_mask[jnp.arange(3), actions], False
        )
        np.testing.assert_array_equal(self.env.current_player, 1 - before.current_player)
        np.testing.assert_array_equal(self.env.rewards, 0)

    def test_two_passes_finish_only_the_game_that_passed(self):
        self.env.step(jnp.array([81, 0, 1]))
        np.testing.assert_array_equal(self.env.done, [False, False, False])
        self.env.step(jnp.array([81, 2, 3]))
        np.testing.assert_array_equal(self.env.terminated, [True, False, False])
        np.testing.assert_array_equal(self.env.done, [True, False, False])
        np.testing.assert_array_equal(np.sort(self.env.rewards[0]), [-1, 1])
        np.testing.assert_array_equal(self.env.rewards[1:], 0)

    def test_reset_finished_clears_completed_game_and_preserves_active_games(self):
        self.env.step(jnp.array([81, 0, 1]))
        self.env.step(jnp.array([81, 2, 3]))
        before = self.env.state
        self.env.reset_finished()
        np.testing.assert_array_equal(self.env.done, [False, False, False])
        np.testing.assert_array_equal(self.env.observation()[0, ..., :16], 0)
        np.testing.assert_array_equal(self.env.legal_action_mask[0], True)
        np.testing.assert_array_equal(self.env.rewards[0], 0)
        assert_state_equal(
            self,
            jax.tree_util.tree_map(lambda leaf: leaf[1:], self.env.state),
            jax.tree_util.tree_map(lambda leaf: leaf[1:], before),
        )

    def test_truncated_games_count_as_done_and_are_reset(self):
        self.env.step(jnp.array([0, 10, 80]))
        # PGX's normal Go end condition is termination. Inject a truncation
        # at the State boundary to exercise the wrapper's other reset path.
        self.env.state = self.env.state.replace(
            truncated=jnp.array([False, True, False])
        )
        before = self.env.state
        np.testing.assert_array_equal(self.env.terminated, False)
        np.testing.assert_array_equal(self.env.done, [False, True, False])
        self.env.reset_finished()
        np.testing.assert_array_equal(self.env.done, False)
        np.testing.assert_array_equal(self.env.observation()[1, ..., :16], 0)
        np.testing.assert_array_equal(self.env.legal_action_mask[1], True)
        for i in (0, 2):
            assert_state_equal(
                self,
                jax.tree_util.tree_map(lambda leaf: leaf[i], self.env.state),
                jax.tree_util.tree_map(lambda leaf: leaf[i], before),
            )

    def test_reset_finished_preserves_all_unfinished_states_and_advances_key(self):
        self.env.step(jnp.array([0, 10, 80]))
        before, old_key = self.env.state, self.env.key
        self.env.reset_finished()
        assert_state_equal(self, self.env.state, before)
        self.assertFalse(np.array_equal(self.env.key, old_key))

    def test_reset_clears_both_completed_and_active_games(self):
        self.env.step(jnp.array([81, 0, 1]))
        self.env.step(jnp.array([81, 2, 3]))
        old_key = self.env.key
        self.env.reset()
        np.testing.assert_array_equal(self.env.done, False)
        np.testing.assert_array_equal(self.env.rewards, 0)
        np.testing.assert_array_equal(self.env.observation()[..., :16], 0)
        np.testing.assert_array_equal(self.env.legal_action_mask, True)
        self.assertFalse(np.array_equal(self.env.key, old_key))

    def test_same_seed_reproduces_steps_and_resets(self):
        left = ScorchEnv(batch_size=2, seed=123)
        right = ScorchEnv(batch_size=2, seed=123)
        assert_state_equal(self, left.state, right.state)
        for actions in (jnp.array([81, 0]), jnp.array([81, 10])):
            left.step(actions)
            right.step(actions)
            assert_state_equal(self, left.state, right.state)
        for operation in ("reset_finished", "reset"):
            getattr(left, operation)()
            getattr(right, operation)()
            assert_state_equal(self, left.state, right.state)
            np.testing.assert_array_equal(left.key, right.key)

    def test_single_game_keeps_batch_dimension(self):
        env = ScorchEnv(batch_size=1, seed=7)
        env.step(jnp.array([0]))
        self.assertEqual(env.observation().shape, (1, 9, 9, 17))
        self.assertEqual(env.legal_action_mask.shape, (1, 82))
        self.assertEqual(env.rewards.shape, (1, 2))
        np.testing.assert_array_equal(env.legal_action_mask[:, 0], [False])


if __name__ == "__main__":
    unittest.main()
