"""Training and collection tests using real Flax, Optax, PGX, and MCTX."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch.scorch import Scorch
from Trainer.scorch_trainer import (
    ScorchTrainer, SelfPlayCollector, SelfPlayExamples, TrainingConfig,
)


def assert_tree_equal(test, actual, expected):
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for left, right in zip(jax.tree_util.tree_leaves(actual),
                           jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(left, right)


def trees_differ(left, right):
    return any(not np.array_equal(a, b)
               for a, b in zip(jax.tree_util.tree_leaves(left),
                               jax.tree_util.tree_leaves(right)))


class ScorchTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=2, seed=21)
        cls.agent = Scorch(SearchConfig(num_simulations=1), cls.env, seed=123)
        cls.params = cls.agent.params
        cls.agent_key = cls.agent.prng_key
        cls.config = TrainingConfig(
            learning_rate=3e-4, weight_decay=0.0,
            max_grad_norm=1.0, value_loss_weight=2.5,
        )
        cls.trainer = ScorchTrainer(cls.agent, cls.env, cls.config)
        cls.initial_train_state = cls.trainer.train_state
        cls.examples = SelfPlayExamples(
            observations=cls.env.observation(),
            policy_targets=jax.nn.one_hot(jnp.array([0, 81]), 82),
            value_targets=jnp.array([0.75, -0.5]),
            side_to_moves=cls.env.current_player,
            episode_ids=jnp.array([10, 11]),
        )

    def setUp(self):
        self.agent.params = self.params
        self.agent.prng_key = self.agent_key
        self.trainer.train_state = self.initial_train_state

    def test_metrics_match_uniform_policy_loss_and_weighted_value_error(self):
        zero_params = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        state = self.initial_train_state.replace(params=zero_params)
        examples = self.examples.replace(value_targets=jnp.array([1.0, -1.0]))
        _, metrics = self.trainer._train_step(state, examples, 2.5)
        self.assertAlmostEqual(float(metrics["policy_loss"]), math.log(82), places=5)
        self.assertAlmostEqual(float(metrics["value_loss"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["loss"]), math.log(82) + 2.5, places=5)

    def test_train_batch_uses_configured_value_loss_weight(self):
        self.trainer.train_state = self.initial_train_state.replace(
            params=jax.tree_util.tree_map(jnp.zeros_like, self.params)
        )
        examples = self.examples.replace(value_targets=jnp.array([1.0, -1.0]))
        metrics = self.trainer._train_batch(examples)
        self.assertAlmostEqual(float(metrics["loss"]), math.log(82) + 2.5, places=5)

    def test_scan_callback_carries_optimizer_state_and_configured_loss_weight(self):
        state = self.initial_train_state.replace(
            params=jax.tree_util.tree_map(jnp.zeros_like, self.params)
        )
        examples = self.examples.replace(value_targets=jnp.array([1.0, -1.0]))
        updated, metrics = self.trainer._update_one_batch(state, examples)
        self.assertEqual(int(updated.step), 1)
        self.assertEqual(int(state.step), 0)
        self.assertAlmostEqual(float(metrics["loss"]), math.log(82) + 2.5, places=5)

    def test_policy_loss_uses_soft_search_targets(self):
        controlled = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        controlled["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([math.log(81)])
        # Pass has probability 1/2; each placement has probability 1/162.
        targets = jnp.zeros((2, 82)).at[:, 0].set(0.25).at[:, 81].set(0.75)
        examples = self.examples.replace(
            policy_targets=targets, value_targets=jnp.zeros(2),
        )
        state = self.initial_train_state.replace(params=controlled)
        _, metrics = self.trainer._train_step(state, examples, 1.0)
        expected = 0.25 * math.log(162) + 0.75 * math.log(2)
        self.assertAlmostEqual(float(metrics["policy_loss"]), expected, places=5)
        self.assertAlmostEqual(float(metrics["loss"]), expected, places=5)

    def test_training_updates_agent_parameters_and_predictions(self):
        before = self.agent.neural_network.apply(
            {"params": self.params}, self.examples.observations
        )
        metrics = self.trainer._train_batch(self.examples)
        self.assertEqual(int(self.trainer.train_state.step), 1)
        self.assertTrue(trees_differ(self.agent.params, self.params))
        assert_tree_equal(self, self.agent.params, self.trainer.train_state.params)
        after = self.agent.neural_network.apply(
            {"params": self.agent.params}, self.examples.observations
        )
        self.assertTrue(trees_differ(after, before))
        self.assertTrue(all(np.isfinite(value).all() for value in metrics.values()))

    def test_repeated_updates_keep_optimizer_state_and_reduce_loss(self):
        losses = []
        for expected_step in range(1, 5):
            metrics = self.trainer._train_batch(self.examples)
            self.assertEqual(int(self.trainer.train_state.step), expected_step)
            self.assertTrue(all(np.isfinite(value).all() for value in metrics.values()))
            losses.append(float(metrics["loss"]))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(trees_differ(self.trainer.train_state.opt_state,
                                    self.initial_train_state.opt_state))

    def test_training_preserves_environment_sampling_key_and_input_examples(self):
        state, env_key = self.env.state, self.env.key
        examples_before = jax.tree_util.tree_map(np.array, self.examples)
        self.trainer._train_batch(self.examples)
        assert_tree_equal(self, self.env.state, state)
        np.testing.assert_array_equal(self.env.key, env_key)
        np.testing.assert_array_equal(self.agent.prng_key, self.agent_key)
        assert_tree_equal(self, self.examples, examples_before)

    def test_zero_learning_rate_preserves_parameters(self):
        trainer = ScorchTrainer(
            self.agent, self.env,
            TrainingConfig(learning_rate=0.0, weight_decay=0.0),
        )
        trainer._train_batch(self.examples)
        assert_tree_equal(self, self.agent.params, self.params)
        self.assertEqual(int(trainer.train_state.step), 1)

    def test_weight_decay_changes_a_parameter_with_zero_loss_gradient(self):
        # Zero LayerNorm scales disconnect these embeddings from both heads,
        # so only AdamW weight decay can change this nonzero parameter.
        controlled = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        controlled["Tokenizer"]["row_embeddings"] = jnp.ones((9, 128))
        self.agent.params = controlled
        trainer = ScorchTrainer(
            self.agent, self.env,
            TrainingConfig(learning_rate=0.001, weight_decay=0.1),
        )
        trainer._train_batch(self.examples)
        np.testing.assert_allclose(
            self.agent.params["Tokenizer"]["row_embeddings"], 0.9999,
            atol=1e-7, rtol=0,
        )


class SelfPlayCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=2, seed=91)
        cls.initial_state, cls.env_key = cls.env.state, cls.env.key
        cls.config = SearchConfig(
            num_simulations=1, max_depth=1, dirichlet_fraction=0.0,
        )
        cls.agent = Scorch(cls.config, cls.env, seed=123)
        # A controlled real network makes passing the unique best move.
        # Games end naturally after two passes instead of requiring long Go
        # matches or replacing the collector/search with mocks.
        cls.params = jax.tree_util.tree_map(jnp.zeros_like, cls.agent.params)
        cls.params["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        cls.params["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        cls.agent_key = cls.agent.prng_key
        cls.collector = SelfPlayCollector()

    def setUp(self):
        self.env.state, self.env.key = self.initial_state, self.env_key
        self.agent.params = self.params
        self.agent.prng_key = self.agent_key

    def rollout(self, state=None, num_steps=4):
        return self.collector._rollout(
            params=self.params, mctx=self.agent.mctx,
            state=self.env.state if state is None else state,
            rng_key=jax.random.PRNGKey(456), config=self.config,
            temperature=0.0, num_steps=num_steps,
        )

    def test_skip_preserves_state_and_returns_zero_policy_targets(self):
        state, targets = self.collector._skip(self.env.state, None)
        assert_tree_equal(self, state, self.env.state)
        np.testing.assert_array_equal(targets, np.zeros((2, 82)))
        self.assertEqual(targets.dtype, jnp.float32)

    def test_step_records_position_before_play_and_advances_random_key(self):
        key = jax.random.PRNGKey(456)
        carry = (self.env.state, key, jnp.zeros((2, 2)))
        (state, next_key, rewards), record = self.collector._step(
            carry, self.params, self.agent.mctx, 0.0, self.config, None
        )
        np.testing.assert_array_equal(record["observations"], self.env.state.observation)
        np.testing.assert_array_equal(record["side_to_move"], self.env.current_player)
        np.testing.assert_array_equal(record["active"], [True, True])
        np.testing.assert_array_equal(record["policy_targets"][:, 81], [1, 1])
        np.testing.assert_array_equal(state.current_player, 1 - self.env.current_player)
        np.testing.assert_array_equal(next_key, jax.random.split(key)[0])
        np.testing.assert_array_equal(rewards, 0)

    def test_rollout_ends_games_then_skips_remaining_steps(self):
        (state, _, rewards), history = self.rollout()
        np.testing.assert_array_equal(state.terminated, [True, True])
        np.testing.assert_array_equal(history["active"],
                                      [[True, True], [True, True], [False, False], [False, False]])
        self.assertEqual(history["observations"].shape, (4, 2, 9, 9, 17))
        np.testing.assert_array_equal(history["policy_targets"][:2, :, 81], 1)
        np.testing.assert_array_equal(history["policy_targets"][2:], 0)
        # Empty-board black loses and white wins by komi, regardless of the
        # randomized mapping between colors and PGX player IDs.
        ids = jnp.arange(2)
        np.testing.assert_array_equal(rewards[ids, self.initial_state.current_player], [-1, -1])
        np.testing.assert_array_equal(rewards[ids, 1 - self.initial_state.current_player], [1, 1])

    def test_mixed_length_games_preserve_rewards_of_earlier_finishers(self):
        start = self.agent.mctx.step_fn(self.initial_state, jnp.array([81, 0]))
        (state, _, rewards), history = self.rollout(state=start)
        np.testing.assert_array_equal(history["active"],
                                      [[True, True], [False, True], [False, False], [False, False]])
        np.testing.assert_array_equal(state.terminated, [True, True])
        np.testing.assert_array_equal(rewards[0, self.initial_state.current_player[0]], -1)
        np.testing.assert_array_equal(np.sort(rewards, axis=-1), [[-1, 1], [-1, 1]])

    def test_truncated_positions_are_inactive_and_not_played(self):
        start = self.initial_state.replace(truncated=jnp.array([True, True]))
        (state, _, rewards), history = self.rollout(state=start)
        assert_tree_equal(self, state, start)
        np.testing.assert_array_equal(history["active"], False)
        np.testing.assert_array_equal(history["policy_targets"], 0)
        np.testing.assert_array_equal(rewards, 0)

    def test_collection_filters_finished_padding_and_labels_player_outcomes(self):
        examples = self.collector.collect_self_play_games(
            self.agent, self.env, start_episode_id=100, temperature=0.0, num_steps=4,
        )
        self.assertEqual(examples.observations.shape, (4, 9, 9, 17))
        self.assertEqual(examples.observations.dtype, jnp.float32)
        self.assertEqual(examples.policy_targets.shape, (4, 82))
        np.testing.assert_array_equal(examples.policy_targets[:, 81], [1, 1, 1, 1])
        np.testing.assert_array_equal(examples.policy_targets[:, :81], 0)
        np.testing.assert_array_equal(examples.value_targets, [-1, -1, 1, 1])
        np.testing.assert_array_equal(examples.episode_ids, [100, 101, 100, 101])
        np.testing.assert_array_equal(examples.side_to_moves[2:], 1 - examples.side_to_moves[:2])
        np.testing.assert_array_equal(examples.observations[:2, ..., :16], 0)

    def test_unfinished_games_do_not_produce_training_examples(self):
        examples = self.collector.collect_self_play_games(
            self.agent, self.env, temperature=0.0, num_steps=1,
        )
        self.assertEqual(examples.observations.shape, (0, 9, 9, 17))
        self.assertEqual(examples.policy_targets.shape, (0, 82))
        for values in (examples.value_targets, examples.side_to_moves, examples.episode_ids):
            self.assertEqual(values.shape, (0,))

    def test_zero_step_collection_produces_correctly_shaped_empty_arrays(self):
        examples = self.collector.collect_self_play_games(self.agent, self.env, num_steps=0)
        self.assertEqual(examples.observations.shape, (0, 9, 9, 17))
        self.assertEqual(examples.policy_targets.shape, (0, 82))
        self.assertEqual(examples.value_targets.shape, (0,))
        self.assertEqual(examples.episode_ids.shape, (0,))

    def test_collection_resets_live_games_and_advances_agent_key_once(self):
        self.env.step(jnp.array([0, 10]))
        before_key = self.agent.prng_key
        before_env_key = self.env.key
        examples = self.collector.collect_self_play_games(
            self.agent, self.env, temperature=0.0, num_steps=4,
        )
        np.testing.assert_array_equal(self.agent.prng_key, jax.random.split(before_key)[0])
        self.assertFalse(np.array_equal(self.env.key, before_env_key))
        np.testing.assert_array_equal(self.env.observation()[..., :16], 0)
        np.testing.assert_array_equal(self.env.done, False)
        np.testing.assert_array_equal(examples.observations[:2, ..., :16], 0)
        assert_tree_equal(self, self.agent.params, self.params)

    def test_collected_examples_can_train_the_real_scorch_network(self):
        examples = self.collector.collect_self_play_games(
            self.agent, self.env, temperature=0.0, num_steps=4,
        )
        trainer = ScorchTrainer(self.agent, self.env, TrainingConfig(weight_decay=0.0))
        metrics = trainer._train_batch(examples)
        self.assertTrue(all(np.isfinite(value).all() for value in metrics.values()))
        self.assertEqual(int(trainer.train_state.step), 1)
        self.assertTrue(trees_differ(self.agent.params, self.params))
        assert_tree_equal(self, self.agent.params, trainer.train_state.params)


class TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.config = TrainingConfig(
            learning_rate=3e-4, weight_decay=0.0, value_loss_weight=0.5,
            epochs=2, batch_size=2,
        )
        cls.trainer = ScorchTrainer(cls.agent, cls.env, cls.config)
        cls.initial_train_state = cls.trainer.train_state

    def setUp(self):
        self.reset_training()

    def reset_training(self):
        self.env.state, self.env.key = self.env_state, self.env_key
        self.agent.params, self.agent.prng_key = self.params, self.agent_key
        self.trainer.train_state = self.initial_train_state
        self.trainer.next_episode_id = 0

    @staticmethod
    def indexed_examples(count):
        indices = jnp.arange(count)
        return SelfPlayExamples(
            observations=jnp.broadcast_to(indices[:, None, None, None],
                                           (count, 9, 9, 17)).astype(jnp.float32),
            policy_targets=jax.nn.one_hot(indices, 82),
            value_targets=indices.astype(jnp.float32) / 10,
            side_to_moves=indices % 2,
            episode_ids=100 + indices,
        )

    def assert_aligned_examples(self, examples):
        # Every field encodes a sample's identity independently of its order.
        ids = np.asarray(examples.episode_ids).reshape(-1) - 100
        observations = np.asarray(examples.observations).reshape((-1, 9, 9, 17))
        policies = np.asarray(examples.policy_targets).reshape((-1, 82))
        for position, identity in enumerate(ids):
            np.testing.assert_array_equal(observations[position], identity)
            np.testing.assert_array_equal(policies[position], np.eye(82)[identity])
        np.testing.assert_allclose(np.asarray(examples.value_targets).reshape(-1), ids / 10)
        np.testing.assert_array_equal(np.asarray(examples.side_to_moves).reshape(-1), ids % 2)

    def test_shuffle_preserves_all_samples_field_alignment_and_original_data(self):
        examples = self.indexed_examples(7)
        before = jax.tree_util.tree_map(np.array, examples)
        shuffled = self.trainer._shuffle_examples(examples, jax.random.PRNGKey(7))
        np.testing.assert_array_equal(np.sort(shuffled.episode_ids),
                                      [100, 101, 102, 103, 104, 105, 106])
        self.assertFalse(np.array_equal(shuffled.episode_ids, examples.episode_ids))
        self.assert_aligned_examples(shuffled)
        assert_tree_equal(self, examples, before)

    def test_shuffle_is_reproducible_for_the_same_key(self):
        examples = self.indexed_examples(7)
        first = self.trainer._shuffle_examples(examples, jax.random.PRNGKey(7))
        repeated = self.trainer._shuffle_examples(examples, jax.random.PRNGKey(7))
        different = self.trainer._shuffle_examples(examples, jax.random.PRNGKey(8))
        assert_tree_equal(self, first, repeated)
        self.assertFalse(np.array_equal(first.episode_ids, different.episode_ids))

    def test_minibatches_preserve_fields_and_drop_only_the_incomplete_tail(self):
        for count in (6, 7):
            with self.subTest(example_count=count):
                examples = self.indexed_examples(count)
                batches = self.trainer._make_minibatches(examples, 3)
                self.assertEqual(batches.observations.shape, (2, 3, 9, 9, 17))
                self.assertEqual(batches.policy_targets.shape, (2, 3, 82))
                self.assertEqual(batches.value_targets.shape, (2, 3))
                np.testing.assert_array_equal(batches.episode_ids,
                                              [[100, 101, 102], [103, 104, 105]])
                self.assert_aligned_examples(batches)
                self.assertEqual(examples.episode_ids.shape, (count,))

    def test_fewer_examples_than_batch_size_produce_no_full_minibatches(self):
        batches = self.trainer._make_minibatches(self.indexed_examples(1), 2)
        self.assertEqual(batches.observations.shape, (0, 2, 9, 9, 17))
        self.assertEqual(batches.policy_targets.shape, (0, 2, 82))
        for values in (batches.value_targets, batches.side_to_moves, batches.episode_ids):
            self.assertEqual(values.shape, (0, 2))

    def test_empty_examples_produce_correctly_shaped_minibatches(self):
        batches = self.trainer._make_minibatches(self.indexed_examples(0), 2)
        self.assertEqual(batches.observations.shape, (0, 2, 9, 9, 17))
        self.assertEqual(batches.policy_targets.shape, (0, 2, 82))
        self.assertEqual(batches.episode_ids.shape, (0, 2))

    def test_epoch_matches_sequential_updates_on_shuffled_full_batches(self):
        examples = self.indexed_examples(5)
        key = jax.random.PRNGKey(7)
        order = jax.random.permutation(key, jnp.arange(5))
        expected = self.initial_train_state
        # The five examples produce exactly two full batches; one is dropped.
        for indices in (order[:2], order[2:4]):
            batch = jax.tree_util.tree_map(lambda values: values[indices], examples)
            expected, _ = self.trainer._train_step(
                expected, batch, self.config.value_loss_weight
            )
        self.trainer._train_epoch(examples, key, batch_size=2)
        self.assertEqual(int(self.trainer.train_state.step), 2)
        self.assertEqual(jax.tree_util.tree_structure(self.trainer.train_state),
                         jax.tree_util.tree_structure(expected))
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(self.trainer.train_state),
            jax.tree_util.tree_leaves(expected),
        ):
            np.testing.assert_allclose(actual_leaf, expected_leaf, atol=1e-6, rtol=1e-6)
        assert_tree_equal(self, self.agent.params, self.trainer.train_state.params)

    def test_epoch_without_full_batches_preserves_optimizer_and_parameters(self):
        self.trainer._train_epoch(self.indexed_examples(1), jax.random.PRNGKey(7), 2)
        assert_tree_equal(self, self.trainer.train_state, self.initial_train_state)
        assert_tree_equal(self, self.agent.params, self.params)

    def test_iteration_runs_the_requested_epochs_and_allocates_episode_ids(self):
        self.trainer._train_iteration(
            jax.random.PRNGKey(7), epochs=3, batch_size=2, num_steps=4, temperature=0.0,
        )
        # Two games x two moves = four samples, two batches per epoch.
        self.assertEqual(int(self.trainer.train_state.step), 6)
        self.assertEqual(self.trainer.next_episode_id, 2)
        self.assertTrue(trees_differ(self.agent.params, self.params))
        assert_tree_equal(self, self.agent.params, self.trainer.train_state.params)

    def test_iteration_with_too_few_samples_skips_training_but_advances_episodes(self):
        self.trainer._train_iteration(
            jax.random.PRNGKey(7), epochs=3, batch_size=5, num_steps=4, temperature=0.0,
        )
        assert_tree_equal(self, self.trainer.train_state, self.initial_train_state)
        assert_tree_equal(self, self.agent.params, self.params)
        self.assertEqual(self.trainer.next_episode_id, 2)
        np.testing.assert_array_equal(self.agent.prng_key, jax.random.split(self.agent_key)[0])

    def test_zero_epoch_iteration_collects_games_without_optimizer_updates(self):
        self.trainer._train_iteration(
            jax.random.PRNGKey(7), epochs=0, batch_size=2, num_steps=4, temperature=0.0,
        )
        assert_tree_equal(self, self.trainer.train_state, self.initial_train_state)
        self.assertEqual(self.trainer.next_episode_id, 2)

    def test_zero_iterations_preserve_keys_state_and_episode_counter(self):
        key = jax.random.PRNGKey(7)
        returned = self.trainer.train(key, iterations=0, num_steps=4, temperature=0.0)
        np.testing.assert_array_equal(returned, key)
        np.testing.assert_array_equal(self.agent.prng_key, self.agent_key)
        np.testing.assert_array_equal(self.env.key, self.env_key)
        assert_tree_equal(self, self.env.state, self.env_state)
        assert_tree_equal(self, self.trainer.train_state, self.initial_train_state)
        self.assertEqual(self.trainer.next_episode_id, 0)

    def test_train_uses_configured_epochs_batch_size_and_fresh_iteration_keys(self):
        key = jax.random.PRNGKey(7)
        returned = self.trainer.train(key, iterations=2, num_steps=4, temperature=0.0)
        self.assertEqual(int(self.trainer.train_state.step), 8)
        self.assertEqual(self.trainer.next_episode_id, 4)
        expected_key, _ = jax.random.split(key)
        expected_key, _ = jax.random.split(expected_key)
        np.testing.assert_array_equal(returned, expected_key)
        expected_agent_key, _ = jax.random.split(self.agent_key)
        expected_agent_key, _ = jax.random.split(expected_agent_key)
        np.testing.assert_array_equal(self.agent.prng_key, expected_agent_key)
        assert_tree_equal(self, self.agent.params, self.trainer.train_state.params)
        self.assertTrue(trees_differ(self.agent.params, self.params))

    def test_unfinished_self_play_skips_training_without_nan_updates(self):
        key = jax.random.PRNGKey(7)
        returned = self.trainer.train(key, iterations=1, num_steps=1, temperature=0.0)
        assert_tree_equal(self, self.trainer.train_state, self.initial_train_state)
        assert_tree_equal(self, self.agent.params, self.params)
        self.assertEqual(self.trainer.next_episode_id, 2)
        np.testing.assert_array_equal(returned, jax.random.split(key)[0])

    def test_returned_key_allows_training_to_resume_reproducibly(self):
        key = jax.random.PRNGKey(7)
        next_key = self.trainer.train(key, iterations=1, num_steps=4, temperature=0.0)
        resumed_key = self.trainer.train(next_key, iterations=1, num_steps=4, temperature=0.0)
        resumed_state = self.trainer.train_state
        resumed_agent_key = self.agent.prng_key
        self.reset_training()
        combined_key = self.trainer.train(key, iterations=2, num_steps=4, temperature=0.0)
        np.testing.assert_array_equal(resumed_key, combined_key)
        np.testing.assert_array_equal(resumed_agent_key, self.agent.prng_key)
        assert_tree_equal(self, self.trainer.train_state, resumed_state)
        self.assertEqual(self.trainer.next_episode_id, 4)


if __name__ == "__main__":
    unittest.main()
