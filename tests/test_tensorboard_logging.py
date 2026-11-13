"""Read real TensorBoard event files to verify trainer logging end to end."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.scorch import Scorch
from Trainer import scorch_trainer as trainer_module
from Trainer import ScorchTrainer, SelfPlayConfig, SelfPlayExamples, TensorBoardLogger, TrainingConfig
from tests.tensorboard_utils import isolate_tensorboard_writers


class TensorBoardLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_root = isolate_tensorboard_writers(cls)
        cls.env = ScorchEnv(batch_size=2, seed=91)
        cls.env_state, cls.env_key = cls.env.state, cls.env.key
        cls.agent = Scorch(
            SearchConfig(num_simulations=1, max_depth=1, dirichlet_fraction=0.0),
            cls.env, seed=123,
        )
        cls.params = jax.tree_util.tree_map(jnp.zeros_like, cls.agent.params)
        cls.params["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        cls.params["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        cls.agent.params = cls.params
        cls.agent_key = cls.agent.prng_key
        cls.config = TrainingConfig(epochs=2, minibatch_size=2, weight_decay=0.0,
                                    value_loss_weight=0.5)
        cls.trainer = ScorchTrainer(cls.agent, cls.env, cls.config,
                                    run_name="tensorboard-tests", evaluation_config=None,
                                    self_play_config=SelfPlayConfig(num_steps=4, temperature=0.0))
        cls.initial_train_state = cls.trainer.learner.state

    def setUp(self):
        self.env.state, self.env.key = self.env_state, self.env_key
        self.agent.params, self.agent.prng_key = self.params, self.agent_key
        self.trainer.learner.restore(self.initial_train_state)
        self.trainer.next_episode_id = 0
        self.trainer.next_iteration = 0
        self.trainer.key = jax.random.PRNGKey(7)
        self.trainer.learner.config = self.config
        self.trainer.collector.config = SelfPlayConfig(num_steps=4, temperature=0.0)
        self.trainer.logger.close()
        self.trainer.logger = TensorBoardLogger(self.log_root / self._testMethodName)
        self.addCleanup(self.trainer.logger.close)

    def events(self, logger=None):
        logger = self.trainer.logger if logger is None else logger
        # tensorboardX flushes the file without waiting for its background
        # queue. Closing joins that worker and makes readback deterministic.
        logger.close()
        accumulator = EventAccumulator(str(logger.log_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        return {
            tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
            for tag in accumulator.Tags()["scalars"]
        }

    @staticmethod
    def examples(episode_ids):
        count = len(episode_ids)
        return SelfPlayExamples(
            observations=jnp.zeros((count, 9, 9, 17)),
            policy_targets=jax.nn.one_hot(jnp.full((count,), 81), 82),
            value_targets=jnp.zeros(count),
            side_to_moves=jnp.zeros(count, dtype=jnp.int32),
            episode_ids=jnp.array(episode_ids, dtype=jnp.int32),
        )

    def clock(self, *values):
        # Replace only the trainer's clock reference, leaving JAX and writer
        # internals free to use the process-wide time module normally.
        return patch.object(trainer_module, "time", SimpleNamespace(
            perf_counter=Mock(side_effect=values)
        ))

    def test_constructor_creates_a_named_writable_run_directory(self):
        trainer = ScorchTrainer(self.agent, self.env, self.config, run_name="named-run")
        self.addCleanup(trainer.close)
        path = trainer.logger.log_dir
        self.assertTrue(path.is_dir())
        self.assertTrue(path.name.startswith("named-run_"))
        trainer.logger.log_metrics({"loss": 2.0}, timestep=7, prefix="train")
        self.assertEqual(self.events(trainer.logger), {"train/loss": [(7, 2.0)]})

    def test_scalar_metrics_write_python_and_jax_values_at_requested_step(self):
        self.trainer.logger.log_metrics(
            {"loss": jnp.array(1.5), "positions": 4, "time_delta": 2.25},
            timestep=7, prefix="train",
        )
        self.assertEqual(self.events(), {
            "train/loss": [(7, 1.5)],
            "train/positions": [(7, 4.0)],
            "train/time_delta": [(7, 2.25)],
        })

    def test_array_metrics_are_averaged_over_all_minibatch_dimensions(self):
        self.trainer.logger.log_metrics(
            {"loss": jnp.array([1.0, 3.0]),
             "value_loss": jnp.array([[2.0, 4.0], [6.0, 8.0]])},
            timestep=5, prefix="train",
        )
        self.assertEqual(self.events(), {
            "train/loss": [(5, 2.0)], "train/value_loss": [(5, 5.0)],
        })

    def test_prefixes_and_repeated_steps_keep_separate_scalar_histories(self):
        self.trainer.logger.log_metrics({"loss": 1.0}, 2, "train")
        self.trainer.logger.log_metrics({"loss": 3.0}, 5, "train")
        self.trainer.logger.log_metrics({"loss": 4.0}, 5, "validation")
        self.assertEqual(self.events(), {
            "train/loss": [(2, 1.0), (5, 3.0)],
            "validation/loss": [(5, 4.0)],
        })

    def test_empty_metric_dictionary_writes_no_scalar_events(self):
        self.trainer.logger.log_metrics({}, timestep=0, prefix="train")
        self.assertEqual(self.events(), {})

    def test_self_play_counts_positions_and_distinct_finished_episode_ids(self):
        metrics = TensorBoardLogger.self_play_metrics(
            self.examples([100, 100, 104, 104, 999]), time_delta=2.75
        )
        self.assertEqual(metrics, {"positions": 5, "games_finished": 3, "time_delta": 2.75})

    def test_empty_self_play_metrics_report_zero_positions_and_finished_games(self):
        metrics = TensorBoardLogger.self_play_metrics(self.examples([]), time_delta=0.25)
        self.assertEqual(metrics, {"positions": 0, "games_finished": 0, "time_delta": 0.25})

    def test_real_training_logs_each_iteration_and_epoch_with_correct_steps(self):
        with self.clock(10.0, 12.5, 20.0, 23.75):
            self.trainer.train(iterations=2)
        events = self.events()
        self.assertEqual(set(events), {
            "self_play/positions", "self_play/games_finished", "self_play/time_delta",
            "train/loss", "train/policy_loss", "train/value_loss",
        })
        self.assertEqual(events["self_play/positions"], [(0, 4.0), (1, 4.0)])
        self.assertEqual(events["self_play/games_finished"], [(0, 2.0), (1, 2.0)])
        self.assertEqual(events["self_play/time_delta"], [(0, 2.5), (1, 3.75)])
        for tag in ("train/loss", "train/policy_loss", "train/value_loss"):
            self.assertEqual([step for step, _ in events[tag]], [0, 1, 2, 3])
            self.assertTrue(np.isfinite([value for _, value in events[tag]]).all())
        for (_, loss), (_, policy), (_, value) in zip(
            events["train/loss"], events["train/policy_loss"], events["train/value_loss"]
        ):
            self.assertAlmostEqual(loss, policy + 0.5 * value, places=5)
        self.assertEqual(int(self.trainer.learner.state.step), 8)

    def test_insufficient_samples_log_self_play_without_training_loss(self):
        with self.clock(1.0, 1.75):
            self.trainer.learner.config = replace(self.config, minibatch_size=5)
            self.trainer.next_iteration = 3
            self.trainer.train(iterations=1)
        self.assertEqual(self.events(), {
            "self_play/positions": [(3, 4.0)],
            "self_play/games_finished": [(3, 2.0)],
            "self_play/time_delta": [(3, 0.75)],
        })
        self.assertEqual(int(self.trainer.learner.state.step), 0)

    def test_unfinished_games_log_zero_counts_without_training_loss(self):
        with self.clock(1.0, 1.5):
            self.trainer.collector.config = replace(self.trainer.collector.config, num_steps=1)
            self.trainer.train(iterations=1)
        self.assertEqual(self.events(), {
            "self_play/positions": [(0, 0.0)],
            "self_play/games_finished": [(0, 0.0)],
            "self_play/time_delta": [(0, 0.5)],
        })
        self.assertEqual(int(self.trainer.learner.state.step), 0)

    def test_iteration_offset_is_applied_to_epoch_logging_steps(self):
        with self.clock(5.0, 6.0):
            self.trainer.next_iteration = 4
            self.trainer.train(iterations=1)
        events = self.events()
        self.assertEqual(events["self_play/positions"], [(4, 4.0)])
        self.assertEqual([step for step, _ in events["train/loss"]], [8, 9])


if __name__ == "__main__":
    unittest.main()
