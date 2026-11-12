"""Real Orbax save/restore and deterministic training-resumption tests."""

from contextlib import chdir
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.scorch import Scorch
from Trainer.scorch_trainer import ScorchTrainer, SelfPlayExamples, TrainingConfig
from tests.tensorboard_utils import isolate_tensorboard_writers


def assert_tree_equal(test, actual, expected):
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual),
                                          jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def assert_train_state_equal(test, actual, expected):
    # apply_fn and tx are reconstructed from the receiving trainer. Compare
    # the persisted payload rather than identities of Python callables.
    np.testing.assert_array_equal(actual.step, expected.step)
    assert_tree_equal(test, actual.params, expected.params)
    assert_tree_equal(test, actual.opt_state, expected.opt_state)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_root = isolate_tensorboard_writers(cls)
        cls.env = ScorchEnv(batch_size=2, seed=91)
        cls.env_state = cls.env.state
        cls.search_config = SearchConfig(num_simulations=1, max_depth=1)
        cls.agent = Scorch(cls.search_config, cls.env, seed=123)
        cls.config = TrainingConfig(weight_decay=0.0, epochs=1, batch_size=2)
        cls.trainer = ScorchTrainer(cls.agent, cls.env, cls.config,
                                    training_run="checkpoint-tests")
        cls.untrained_state = cls.trainer.train_state
        cls.examples = SelfPlayExamples(
            observations=cls.env.observation(),
            policy_targets=jax.nn.one_hot(jnp.array([0, 81]), 82),
            value_targets=jnp.array([0.75, -0.5]),
            side_to_moves=cls.env.current_player,
            episode_ids=jnp.array([40, 41]),
        )
        # Nonzero Adam moments and step count make missing optimizer state
        # observable when comparing resumed and uninterrupted training.
        cls.trainer._train_batch(cls.examples)
        cls.trained_state = cls.trainer.train_state

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="scorch-checkpoint-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "checkpoint"
        self.trainer.train_state = self.trained_state
        self.trainer.next_episode_id = 42
        self.agent.params = self.trained_state.params
        self.agent.prng_key = jax.random.PRNGKey(1234)
        self.env.key = jax.random.PRNGKey(5678)
        self.env.state = self.env_state
        self.key = jax.random.PRNGKey(9012)

    def snapshot(self):
        return (self.trainer.train_state, self.trainer.next_episode_id,
                self.agent.prng_key, self.env.key)

    def assert_restored(self, trainer, expected):
        state, episode_id, agent_key, env_key = expected
        assert_train_state_equal(self, trainer.train_state, state)
        assert_tree_equal(self, trainer.scorch.params, state.params)
        self.assertEqual(trainer.next_episode_id, episode_id)
        np.testing.assert_array_equal(trainer.scorch.prng_key, agent_key)
        np.testing.assert_array_equal(trainer.env.key, env_key)

    def test_round_trip_restores_trained_state_optimizer_counters_and_random_keys(self):
        expected = self.snapshot()
        self.trainer._save_checkpoint(str(self.path), self.key)
        self.assertTrue(self.path.is_dir())
        self.trainer._train_batch(self.examples)
        self.trainer.next_episode_id = 99
        self.agent.prng_key = jax.random.PRNGKey(222)
        self.env.key = jax.random.PRNGKey(333)
        restored_key = self.trainer.load_checkpoint(str(self.path))
        self.assert_restored(self.trainer, expected)
        np.testing.assert_array_equal(restored_key, self.key)

    def test_fresh_trainer_restores_parameters_and_network_predictions(self):
        expected = self.snapshot()
        predictions = self.agent.neural_network.apply(
            {"params": self.agent.params}, self.examples.observations
        )
        self.trainer._save_checkpoint(str(self.path), self.key)
        env = ScorchEnv(batch_size=2, seed=222)
        agent = Scorch(self.search_config, env, seed=456)
        fresh = ScorchTrainer(agent, env, self.config, training_run="fresh-restore")
        self.assertEqual(int(fresh.train_state.step), 0)
        restored_key = fresh.load_checkpoint(str(self.path))
        self.assert_restored(fresh, expected)
        np.testing.assert_array_equal(restored_key, self.key)
        restored_predictions = agent.neural_network.apply(
            {"params": agent.params}, self.examples.observations
        )
        for actual, wanted in zip(restored_predictions, predictions):
            np.testing.assert_array_equal(actual, wanted)

    def test_next_optimizer_update_matches_uninterrupted_training(self):
        self.trainer._save_checkpoint(str(self.path), self.key)
        expected_metrics = self.trainer._train_batch(self.examples)
        expected_state = self.trainer.train_state
        self.assertEqual(int(expected_state.step), 2)
        self.trainer.load_checkpoint(str(self.path))
        actual_metrics = self.trainer._train_batch(self.examples)
        assert_train_state_equal(self, self.trainer.train_state, expected_state)
        assert_tree_equal(self, actual_metrics, expected_metrics)
        assert_tree_equal(self, self.agent.params, expected_state.params)

    def test_restored_keys_reproduce_environment_reset_and_next_agent_action(self):
        self.trainer._save_checkpoint(str(self.path), self.key)
        self.env.reset()
        expected_state = self.env.state
        expected_action = self.agent.act(self.env)
        expected_agent_key, expected_env_key = self.agent.prng_key, self.env.key
        returned_key = self.trainer.load_checkpoint(str(self.path))
        # The checkpoint stores the environment RNG, not a live board. Reset
        # explicitly to reproduce the next self-play game's initial state.
        self.env.reset()
        assert_tree_equal(self, self.env.state, expected_state)
        actual_action = self.agent.act(self.env)
        for actual, wanted in zip(actual_action, expected_action):
            np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(self.agent.prng_key, expected_agent_key)
        np.testing.assert_array_equal(self.env.key, expected_env_key)
        np.testing.assert_array_equal(jax.random.split(returned_key),
                                      jax.random.split(self.key))

    def test_relative_paths_are_resolved_for_save_and_load(self):
        expected = self.snapshot()
        with chdir(self.root):
            self.trainer._save_checkpoint("relative-checkpoint", self.key)
            self.assertTrue((self.root / "relative-checkpoint").is_dir())
            self.trainer.next_episode_id = 999
            restored_key = self.trainer.load_checkpoint("relative-checkpoint")
        self.assert_restored(self.trainer, expected)
        np.testing.assert_array_equal(restored_key, self.key)

    def test_saving_preserves_live_training_environment_and_sampling_state(self):
        expected = self.snapshot()
        state = self.env.state
        self.trainer._save_checkpoint(str(self.path), self.key)
        self.assert_restored(self.trainer, expected)
        assert_tree_equal(self, self.env.state, state)
        np.testing.assert_array_equal(self.key, jax.random.PRNGKey(9012))

    def test_untrained_checkpoint_restores_zero_step_counter_and_keys(self):
        self.trainer.train_state = self.untrained_state
        self.agent.params = self.untrained_state.params
        self.trainer.next_episode_id = 0
        self.agent.prng_key = jax.random.PRNGKey(0)
        self.env.key = jax.random.PRNGKey(0)
        expected = self.snapshot()
        self.trainer._save_checkpoint(str(self.path), jax.random.PRNGKey(0))
        self.trainer.train_state = self.trained_state
        self.agent.params = self.trained_state.params
        self.trainer.next_episode_id = 7
        restored_key = self.trainer.load_checkpoint(str(self.path))
        self.assert_restored(self.trainer, expected)
        np.testing.assert_array_equal(restored_key, jax.random.PRNGKey(0))

    def test_missing_checkpoint_raises_without_changing_live_state(self):
        expected = self.snapshot()
        state = self.env.state
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_checkpoint(str(self.root / "missing"))
        self.assert_restored(self.trainer, expected)
        assert_tree_equal(self, self.env.state, state)

    def test_existing_checkpoint_is_not_silently_overwritten(self):
        expected = self.snapshot()
        self.trainer._save_checkpoint(str(self.path), self.key)
        self.trainer._train_batch(self.examples)
        self.trainer.next_episode_id = 99
        with self.assertRaises(ValueError):
            self.trainer._save_checkpoint(str(self.path), jax.random.PRNGKey(222))
        restored_key = self.trainer.load_checkpoint(str(self.path))
        self.assert_restored(self.trainer, expected)
        np.testing.assert_array_equal(restored_key, self.key)

    def test_separate_checkpoint_paths_keep_independent_training_snapshots(self):
        first = self.snapshot()
        first_path, second_path = self.root / "first", self.root / "second"
        self.trainer._save_checkpoint(str(first_path), self.key)
        self.trainer._train_batch(self.examples)
        self.trainer.next_episode_id = 100
        self.agent.prng_key = jax.random.PRNGKey(444)
        self.env.key = jax.random.PRNGKey(555)
        second = self.snapshot()
        second_key = jax.random.PRNGKey(666)
        self.trainer._save_checkpoint(str(second_path), second_key)
        restored_first_key = self.trainer.load_checkpoint(str(first_path))
        self.assert_restored(self.trainer, first)
        np.testing.assert_array_equal(restored_first_key, self.key)
        restored_second_key = self.trainer.load_checkpoint(str(second_path))
        self.assert_restored(self.trainer, second)
        np.testing.assert_array_equal(restored_second_key, second_key)


if __name__ == "__main__":
    unittest.main()
