"""Factory, owned RNG, reports, resource lifecycle, and resume API tests."""
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import Trainer
from MCTX.mctx import SearchConfig


class PublicTrainerAPITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="scorch-public-api-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def make_trainer(self, name="run", seed=42):
        trainer = Trainer.ScorchTrainer.create(
            num_envs=2,
            seed=seed,
            search_config=SearchConfig(num_simulations=1, max_depth=1, dirichlet_fraction=0.0),
            training_config=Trainer.TrainingConfig(epochs=1, minibatch_size=2, weight_decay=0.0),
            self_play_config=Trainer.SelfPlayConfig(num_steps=4, temperature=0.0),
            evaluation_config=None,
            log_dir=self.directory / name,
        )
        self.addCleanup(trainer.close)
        return trainer

    def make_passing(self, trainer):
        params = jax.tree_util.tree_map(jnp.zeros_like, trainer.learner.state.params)
        params["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        params["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        trainer.learner.restore(trainer.learner.state.replace(params=params))

    def test_package_exposes_the_public_training_components(self):
        for name in ("ScorchTrainer", "ScorchLearner", "SelfPlayCollector", "ScorchEvaluator",
                     "TrainingConfig", "EvaluationConfig", "SelfPlayConfig", "SelfPlayExamples",
                     "TrainingReport", "Evaluation", "CheckpointManager", "TensorBoardLogger"):
            with self.subTest(public_name=name):
                self.assertTrue(hasattr(Trainer, name), f"Trainer.{name} is not exported")

    def test_factory_constructs_bound_components_and_requested_environment_count(self):
        trainer = self.make_trainer()
        self.assertEqual(trainer.env.batch_size, 2)
        self.assertIs(trainer.collector.scorch, trainer.scorch)
        self.assertIs(trainer.collector.env, trainer.env)
        self.assertIs(trainer.learner.scorch, trainer.scorch)
        self.assertEqual(trainer.learner.config.minibatch_size, 2)
        self.assertEqual(trainer.logger.log_dir, self.directory / "run")

    def test_factory_seed_reproduces_models_and_independent_random_streams(self):
        first, second = self.make_trainer("first"), self.make_trainer("second")
        for a, b in zip(jax.tree_util.tree_leaves(first.learner.state.params),
                         jax.tree_util.tree_leaves(second.learner.state.params)):
            np.testing.assert_array_equal(a, b)
        for name in ("key",):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        np.testing.assert_array_equal(first.env.key, second.env.key)
        np.testing.assert_array_equal(first.scorch.prng_key, second.scorch.prng_key)
        keys = (first.key, first.env.key, first.scorch.prng_key)
        for i in range(3):
            for j in range(i):
                self.assertFalse(np.array_equal(keys[i], keys[j]))

    def test_context_manager_closes_trainer_and_logging_resources(self):
        trainer = self.make_trainer()
        with trainer as entered:
            self.assertIs(entered, trainer)
            self.assertFalse(trainer.closed)
        self.assertTrue(trainer.closed)
        self.assertTrue(trainer.logger.closed)

    def test_context_manager_closes_resources_without_swallowing_exceptions(self):
        trainer = self.make_trainer()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with trainer:
                raise RuntimeError("intentional")
        self.assertTrue(trainer.closed)
        self.assertTrue(trainer.logger.closed)

    def test_close_is_idempotent_and_training_after_close_is_rejected(self):
        trainer = self.make_trainer()
        trainer.close()
        trainer.close()
        with self.assertRaises(RuntimeError):
            trainer.train(iterations=0)

    def test_training_owns_key_and_returns_a_structured_report(self):
        trainer = self.make_trainer()
        self.make_passing(trainer)
        key = trainer.key
        report = trainer.train(iterations=1)
        self.assertIsInstance(report, Trainer.TrainingReport)
        self.assertEqual(report.iterations, 1)
        self.assertEqual(report.promotions, 0)
        self.assertEqual(report.optimizer_step, 2)
        self.assertEqual(report.next_iteration, 1)
        self.assertIsNone(report.last_evaluation)
        self.assertIsNone(report.checkpoint_path)
        np.testing.assert_array_equal(trainer.key, jax.random.split(key)[0])

    def test_checkpoint_restores_owned_key_and_resumes_the_next_update(self):
        first = self.make_trainer("first")
        self.make_passing(first)
        first.train(iterations=1)
        path = self.directory / "checkpoint"
        key = first.key
        first.save_checkpoint(path)
        first.train(iterations=1)
        expected = first.learner.state

        resumed = self.make_trainer("resumed", seed=999)
        self.assertIsNone(resumed.load_checkpoint(path))
        np.testing.assert_array_equal(resumed.key, key)
        report = resumed.train(iterations=1)
        self.assertEqual(report.next_iteration, 2)
        for actual, wanted in zip(jax.tree_util.tree_leaves(resumed.learner.state),
                                   jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(resumed.key, first.key)

    def test_factory_rejects_an_empty_environment_batch(self):
        with self.assertRaises(ValueError):
            Trainer.ScorchTrainer.create(num_envs=0, log_dir=self.directory / "invalid")

    def test_learner_rejects_empty_examples_before_changing_optimizer_state(self):
        trainer = self.make_trainer()
        state = trainer.learner.state
        examples = Trainer.SelfPlayExamples(
            observations=jnp.zeros((0, 9, 9, 17)),
            policy_targets=jnp.zeros((0, 82)),
            value_targets=jnp.zeros(0),
            side_to_moves=jnp.zeros(0, dtype=jnp.int32),
            episode_ids=jnp.zeros(0, dtype=jnp.int32),
        )
        with self.assertRaises(ValueError):
            trainer.learner.train_batch(examples)
        self.assertIs(trainer.learner.state, state)

    def test_closed_logger_rejects_metrics_instead_of_reopening_a_writer(self):
        logger = Trainer.TensorBoardLogger(self.directory / "closed-logger")
        logger.close()
        with self.assertRaises(RuntimeError):
            logger.log_metrics({"loss": 1.0}, timestep=0, prefix="train")


if __name__ == "__main__":
    unittest.main()
