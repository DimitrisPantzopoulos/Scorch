"""Evaluation scoring and correct per-game policy selection with real MCTX."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.scorch import Scorch
from Trainer.scorch_trainer import Evaluation, ScorchEvaluator


class EvaluationTests(unittest.TestCase):
    def test_completed_games_and_draws_contribute_to_score(self):
        result = Evaluation(wins=3, losses=1, draws=2, unfinished=4)
        self.assertEqual(result.completed_games, 6)
        self.assertAlmostEqual(result.score_rate, 2 / 3)
        self.assertAlmostEqual(result.elo_delta, 400 * math.log10(2))

    def test_no_completed_games_has_no_score_or_elo(self):
        result = Evaluation(0, 0, 0, 4)
        self.assertIsNone(result.score_rate)
        self.assertIsNone(result.elo_delta)

    def test_perfect_scores_have_unbounded_elo_and_ties_have_zero_elo(self):
        self.assertEqual(Evaluation(4, 0, 0, 0).elo_delta, math.inf)
        self.assertEqual(Evaluation(0, 4, 0, 0).elo_delta, -math.inf)
        self.assertEqual(Evaluation(1, 1, 2, 0).elo_delta, 0.0)


class ScorchEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=2, seed=91)
        cls.state, cls.env_key = cls.env.state, cls.env.key
        cls.config = SearchConfig(num_simulations=1, max_depth=1, dirichlet_fraction=0.0)
        cls.agent = Scorch(cls.config, cls.env, seed=123)
        cls.passing = jax.tree_util.tree_map(jnp.zeros_like, cls.agent.params)
        cls.passing["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        cls.passing["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        cls.placing = jax.tree_util.tree_map(jnp.zeros_like, cls.agent.params)
        cls.placing["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([3.0])
        cls.placing["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([-3.0])
        cls.evaluator = ScorchEvaluator()

    def test_each_game_uses_its_own_players_network(self):
        # Seed 91 gives different current player IDs in these two games.
        np.testing.assert_array_equal(self.state.current_player, [0, 1])
        for candidate_player in (0, 1):
            with self.subTest(candidate_player=candidate_player):
                (state, _, _), _ = self.evaluator._step(
                    (self.state, jax.random.PRNGKey(7), jnp.zeros((2, 2))),
                    self.passing, self.placing, self.agent.mctx,
                    self.config, 0.0, candidate_player, None,
                )
                for lane in (0, 1):
                    has_stones = bool(jnp.any(state.observation[lane, ..., :16]))
                    self.assertEqual(has_stones, lane != candidate_player)

    def test_paired_games_balance_candidate_colors(self):
        result = self.evaluator.evaluate(
            self.passing, self.passing, self.agent.mctx, jax.random.PRNGKey(7),
            self.config, 0.0, num_steps=2, batch_size=4, seed=91,
        )
        self.assertEqual(result, Evaluation(2, 2, 0, 0))

    def test_move_limit_reports_unfinished_games(self):
        result = self.evaluator.evaluate(
            self.passing, self.passing, self.agent.mctx, jax.random.PRNGKey(7),
            self.config, 0.0, num_steps=1, batch_size=4, seed=91,
        )
        self.assertEqual(result, Evaluation(0, 0, 0, 4))

    def test_evaluation_preserves_live_environment_and_agent(self):
        key = self.agent.prng_key
        self.evaluator.evaluate(
            self.passing, self.passing, self.agent.mctx, jax.random.PRNGKey(7),
            self.config, 0.0, num_steps=2, batch_size=4, seed=91,
        )
        for actual, expected in zip(jax.tree_util.tree_leaves(self.env.state),
                                    jax.tree_util.tree_leaves(self.state)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.env.key, self.env_key)
        np.testing.assert_array_equal(self.agent.prng_key, key)

    def test_invalid_evaluation_batch_sizes_are_rejected(self):
        for size in (0, 1, 3):
            with self.subTest(batch_size=size), self.assertRaises(ValueError):
                self.evaluator.evaluate(
                    self.passing, self.passing, self.agent.mctx, jax.random.PRNGKey(7),
                    self.config, 0.0, num_steps=2, batch_size=size, seed=91,
                )


if __name__ == "__main__":
    unittest.main()
