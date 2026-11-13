"""Candidate promotion, checkpointing, and real TensorBoard evaluation logs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.scorch import Scorch
from Trainer import (
    Evaluation, EvaluationConfig, ScorchEvaluator, ScorchTrainer, SelfPlayConfig,
    TensorBoardLogger, TrainingConfig,
)
from tests.tensorboard_utils import isolate_tensorboard_writers


def assert_tree_equal(test, actual, expected):
    test.assertEqual(jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected))
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(left, right)


def trees_differ(left, right):
    return any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)))


class FixedEvaluator:
    """Control match outcomes while training, files, and logging stay real."""
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


class FinalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = isolate_tensorboard_writers(cls)
        cls.env = ScorchEnv(batch_size=2, seed=91)
        cls.env_state, cls.env_key = cls.env.state, cls.env.key
        cls.search = SearchConfig(num_simulations=1, max_depth=1, dirichlet_fraction=0.0)
        cls.agent = Scorch(cls.search, cls.env, seed=123)
        cls.params = jax.tree_util.tree_map(jnp.zeros_like, cls.agent.params)
        cls.params["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        cls.params["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        cls.agent.params = cls.params
        cls.agent_key = cls.agent.prng_key
        cls.config = TrainingConfig(epochs=1, minibatch_size=2, weight_decay=0.0)
        cls.evaluation = EvaluationConfig(num_games=4, num_steps=2, seed=91, search_config=cls.search)
        cls.trainer = ScorchTrainer(
            cls.agent, cls.env, cls.config, run_name="final-training",
            evaluation_config=cls.evaluation,
            self_play_config=SelfPlayConfig(num_steps=4, temperature=0.0),
        )
        cls.initial_state = cls.trainer.learner.state

    def setUp(self):
        self.env.state, self.env.key = self.env_state, self.env_key
        self.agent.params, self.agent.prng_key = self.params, self.agent_key
        self.agent.search_config = self.search
        self.trainer.learner.restore(self.initial_state)
        self.trainer.next_episode_id = 0
        self.trainer.next_iteration = 0
        self.trainer.evaluation_config = self.evaluation
        self.trainer.collector.config = SelfPlayConfig(num_steps=4, temperature=0.0)
        self.trainer.logger.close()
        self.trainer.logger = TensorBoardLogger(self.logs / self._testMethodName)
        self.addCleanup(self.trainer.logger.close)
        self.fake = FixedEvaluator(Evaluation(3, 1, 0, 0))
        self.trainer.evaluator = self.fake
        temporary = tempfile.TemporaryDirectory(prefix="scorch-final-loop-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.key = jax.random.PRNGKey(7)
        self.trainer.key = self.key

    def train(self, iterations=1, **kwargs):
        return self.trainer.train(iterations=iterations, **kwargs)

    def events(self):
        self.trainer.logger.close()
        reader = EventAccumulator(str(self.trainer.logger.log_dir), size_guidance={"scalars": 0})
        reader.Reload()
        return {tag: [(event.step, event.value) for event in reader.Scalars(tag)]
                for tag in reader.Tags()["scalars"]}

    def assert_incumbent_restored(self):
        assert_tree_equal(self, self.trainer.learner.state, self.initial_state)
        assert_tree_equal(self, self.agent.params, self.params)

    def test_winning_candidate_is_promoted_and_compared_with_frozen_incumbent(self):
        self.train()
        self.assertEqual(int(self.trainer.learner.state.step), 2)
        self.assertTrue(trees_differ(self.agent.params, self.params))
        call = self.fake.calls[0]
        assert_tree_equal(self, call["opponent_params"], self.params)
        assert_tree_equal(self, call["candidate_params"], self.agent.params)
        self.assertEqual(set(call), {"candidate_params", "opponent_params", "key"})
        self.assertEqual(self.trainer.evaluation_config.num_games, 4)
        self.assertEqual(self.trainer.evaluation_config.num_steps, 2)
        self.assertEqual(self.events()["evaluation/promoted"], [(0, 1.0)])

    def test_losing_candidate_restores_parameters_and_optimizer_but_keeps_sampling_progress(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(1, 3, 0, 0))
        self.train()
        self.assert_incumbent_restored()
        self.assertEqual(self.trainer.next_episode_id, 2)
        self.assertFalse(np.array_equal(self.agent.prng_key, self.agent_key))
        self.assertFalse(np.array_equal(self.env.key, self.env_key))
        self.assertEqual(self.events()["evaluation/promoted"], [(0, 0.0)])

    def test_tied_evaluation_keeps_the_incumbent(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(1, 1, 2, 0))
        self.train()
        self.assert_incumbent_restored()
        events = self.events()
        self.assertEqual(events["evaluation/score_rate"], [(0, 0.5)])
        self.assertEqual(events["evaluation/elo_delta"], [(0, 0.0)])
        self.assertEqual(events["evaluation/draws"], [(0, 2.0)])

    def test_incomplete_evaluation_cannot_promote_a_candidate(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(3, 0, 0, 1))
        self.train()
        self.assert_incumbent_restored()
        events = self.events()
        self.assertEqual(events["evaluation/unfinished"], [(0, 1.0)])
        self.assertEqual(events["evaluation/promoted"], [(0, 0.0)])
        self.assertNotIn("evaluation/elo_delta", events)

    def test_no_completed_games_logs_counts_without_undefined_metrics(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(0, 0, 0, 4))
        self.train()
        self.assert_incumbent_restored()
        events = self.events()
        self.assertEqual(events["evaluation/completed_games"], [(0, 0.0)])
        self.assertNotIn("evaluation/score_rate", events)
        self.assertNotIn("evaluation/elo_delta", events)

    def test_all_wins_promote_without_writing_infinite_elo(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(4, 0, 0, 0))
        self.train()
        events = self.events()
        self.assertEqual(events["evaluation/score_rate"], [(0, 1.0)])
        self.assertEqual(events["evaluation/promoted"], [(0, 1.0)])
        self.assertNotIn("evaluation/elo_delta", events)
        for tag, entries in events.items():
            if tag.startswith("evaluation/"):
                self.assertTrue(np.isfinite([value for _, value in entries]).all())

    def test_next_candidate_is_compared_against_the_last_promoted_model(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(3, 1, 0, 0), Evaluation(3, 1, 0, 0))
        self.train(iterations=2)
        first, second = self.trainer.evaluator.calls
        assert_tree_equal(self, second["opponent_params"], first["candidate_params"])
        self.assertFalse(np.array_equal(first["key"], second["key"]))
        self.assertEqual(int(self.trainer.learner.state.step), 4)

    def test_repeated_train_calls_continue_tensorboard_iteration_numbers(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(3, 1, 0, 0), Evaluation(3, 1, 0, 0))
        self.train()
        self.trainer.train(iterations=1)
        events = self.events()
        self.assertEqual([step for step, _ in events["evaluation/promoted"]], [0, 1])
        self.assertEqual([step for step, _ in events["train/loss"]], [0, 1])

    def test_no_usable_examples_skip_evaluation_and_training(self):
        self.trainer.collector.config = replace(self.trainer.collector.config, num_steps=1)
        self.trainer.train(iterations=1)
        self.assert_incumbent_restored()
        self.assertEqual(self.fake.calls, [])
        self.assertFalse(any(tag.startswith("evaluation/") for tag in self.events()))

    def test_training_only_mode_keeps_updates_and_original_key_sequence(self):
        self.trainer.evaluation_config = None
        self.trainer.train(iterations=1)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(int(self.trainer.learner.state.step), 2)
        np.testing.assert_array_equal(self.trainer.key, jax.random.split(self.key)[0])

    def test_real_evaluator_logs_balanced_results_and_rejects_a_tie(self):
        evaluator = ScorchEvaluator(self.agent, self.evaluation)
        self.trainer.evaluator = evaluator
        with patch.object(evaluator, "_evaluate_colour", wraps=evaluator._evaluate_colour) as evaluate_colour:
            self.train()
        self.assertEqual(evaluate_colour.call_count, 2)
        for call in evaluate_colour.call_args_list:
            # Evaluation remains deterministic even when self-play is stochastic.
            temperature = call.kwargs.get("temperature", call.args[6] if len(call.args) > 6 else None)
            self.assertEqual(temperature, 0.0)
        self.assert_incumbent_restored()
        events = self.events()
        self.assertEqual(events["evaluation/wins"], [(0, 2.0)])
        self.assertEqual(events["evaluation/losses"], [(0, 2.0)])
        self.assertEqual(events["evaluation/completed_games"], [(0, 4.0)])
        self.assertEqual(events["evaluation/promoted"], [(0, 0.0)])

    def test_checkpoint_restores_the_promoted_state_and_owned_key(self):
        report = self.train(checkpoint_dir=self.directory)
        saved_key = self.trainer.key
        expected_state = self.trainer.learner.state
        path = self.directory / "iteration_000001"
        self.assertTrue(path.is_dir())
        self.trainer.learner.restore(self.initial_state)
        self.agent.params = self.params
        self.trainer.next_episode_id = 0
        self.trainer.next_iteration = 17
        self.trainer.key = jax.random.PRNGKey(999)
        self.trainer.load_checkpoint(str(path))
        assert_tree_equal(self, self.trainer.learner.state, expected_state)
        assert_tree_equal(self, self.agent.params, expected_state.params)
        np.testing.assert_array_equal(self.trainer.key, saved_key)
        self.assertEqual(Path(report.checkpoint_path), path)
        self.assertEqual(self.trainer.next_episode_id, 2)
        self.assertEqual(self.trainer.next_iteration, 1)

    def test_legacy_checkpoint_without_iteration_counter_can_be_loaded(self):
        path = self.directory / "legacy"
        # The previous schema had these five fields and no iteration count.
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(path, {
                "state": self.initial_state,
                "next_episode_id": 6,
                "trainer_key": self.key,
                "scorch_key": self.agent_key,
                "env_key": self.env_key,
            })
            checkpointer.wait_until_finished()
        self.trainer.load_checkpoint(str(path))
        self.assert_incumbent_restored()
        self.assertEqual(self.trainer.next_iteration, 3)
        np.testing.assert_array_equal(self.trainer.key, self.key)

    def test_rejected_candidate_checkpoint_contains_incumbent_optimizer_state(self):
        self.trainer.evaluator = FixedEvaluator(Evaluation(1, 3, 0, 0))
        self.train(checkpoint_dir=self.directory)
        self.trainer.learner.train_batch(self.trainer.collector.collect())
        self.trainer.load_checkpoint(str(self.directory / "iteration_000001"))
        self.assert_incumbent_restored()

    def test_evaluation_exception_restores_incumbent_before_propagating(self):
        class BrokenEvaluator:
            def evaluate(self, **kwargs):
                raise RuntimeError("evaluation failed")

        self.trainer.evaluator = BrokenEvaluator()
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            self.train()
        self.assert_incumbent_restored()

    def test_default_evaluation_disables_self_play_dirichlet_noise(self):
        self.agent.search_config = replace(self.search, dirichlet_fraction=0.25)
        self.trainer.evaluation_config = EvaluationConfig(num_games=4, num_steps=2)
        evaluator = ScorchEvaluator(self.agent, self.trainer.evaluation_config)
        self.trainer.evaluator = evaluator
        with patch.object(evaluator, "_evaluate_colour", wraps=evaluator._evaluate_colour) as evaluate_colour:
            self.trainer.train(iterations=1)
        self.assertEqual(evaluate_colour.call_count, 2)
        for call in evaluate_colour.call_args_list:
            search_config = call.kwargs.get("search_config", call.args[5] if len(call.args) > 5 else None)
            self.assertEqual(search_config.dirichlet_fraction, 0.0)

    def test_invalid_evaluation_configuration_is_rejected(self):
        for size in (0, 1, 3):
            with self.subTest(num_games=size), self.assertRaises(ValueError):
                EvaluationConfig(num_games=size)
        with self.assertRaises(ValueError):
            EvaluationConfig(num_steps=0)

    def test_negative_iterations_fail_before_training_or_evaluation(self):
        with self.assertRaises(ValueError):
            self.train(iterations=-1)
        self.assert_incumbent_restored()
        self.assertEqual(self.fake.calls, [])

    def test_zero_iterations_do_not_evaluate_or_create_checkpoints(self):
        path = self.directory / "unused"
        report = self.train(iterations=0, checkpoint_dir=path)
        self.assertEqual(report.iterations, 0)
        np.testing.assert_array_equal(self.trainer.key, self.key)
        self.assertEqual(self.fake.calls, [])
        self.assertFalse(path.exists())
        self.assert_incumbent_restored()


if __name__ == "__main__":
    unittest.main()
