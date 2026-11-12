"""Deterministic component, forward, and training checks for ScorchNeuralNetwork."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv
from Scorch.scorch_nn import (
    BoardTokenizer, MultiHeadAttention, PolicyHead, ScorchNeuralNetwork,
    TransformerBlock, ValueHead,
)


class ModelComponentTests(unittest.TestCase):
    def test_tokenizer_preserves_board_order_positions_and_global_token(self):
        tokenizer = BoardTokenizer()
        observations = jnp.zeros((2, 9, 9, 17)).at[..., 0].set(1)
        params = {
            "Dense_0": {
                "kernel": jnp.zeros((17, 128)).at[0, 0].set(1),
                "bias": jnp.zeros(128),
            },
            "row_embeddings": jnp.zeros((9, 128)).at[:, 0].set(jnp.arange(9) * 100),
            "col_embeddings": jnp.zeros((9, 128)).at[:, 0].set(jnp.arange(9) * 10),
            "global_token": jnp.full((1, 1, 128), 999.0),
        }
        tokens = tokenizer.apply({"params": params}, observations)
        self.assertEqual(tokens.shape, (2, 82, 128))
        np.testing.assert_array_equal(tokens[:, [0, 1, 9, 80], 0],
                                      [[1, 11, 101, 881], [1, 11, 101, 881]])
        np.testing.assert_array_equal(tokens[:, :81, 1:], 0)
        np.testing.assert_array_equal(tokens[:, 81], 999)

    def test_attention_uses_scaled_scores_and_normalizes_over_source_tokens(self):
        attention = MultiHeadAttention()
        # Each head sees [1, 0, ...] and [0, 0, ...]. Identity projections
        # give first-query scores [1/sqrt(32), 0], second-query scores [0, 0].
        x = jnp.zeros((1, 2, 128)).at[0, 0, jnp.array([0, 32, 64, 96])].set(1)
        params = {
            name: {"kernel": jnp.eye(128), "bias": jnp.zeros(128)}
            for name in ("query", "key", "value", "output")
        }
        output = attention.apply({"params": params}, x)
        expected = np.zeros((1, 2, 128), dtype=np.float32)
        # Hand-calculated softmax([1/sqrt(32), 0])[0].
        expected[0, 0, [0, 32, 64, 96]] = 0.544079444
        expected[0, 1, [0, 32, 64, 96]] = 0.5
        np.testing.assert_allclose(output, expected, atol=1e-6, rtol=1e-6)

    def test_transformer_preserves_input_when_residual_branches_are_zero(self):
        block = TransformerBlock()
        x = jax.random.normal(jax.random.PRNGKey(9), (1, 3, 128))
        variables = block.init(jax.random.PRNGKey(10), x)
        zero_variables = jax.tree_util.tree_map(jnp.zeros_like, variables)
        np.testing.assert_array_equal(block.apply(zero_variables, x), x)

    def test_policy_head_scores_placements_and_pass_from_their_own_tokens(self):
        head = PolicyHead()
        x = jnp.zeros((1, 82, 128)).at[0, :, 0].set(jnp.arange(82))
        params = {
            "policy_placement_logits_mlp": {
                "kernel": jnp.zeros((128, 1)).at[0, 0].set(2),
                "bias": jnp.array([1.0]),
            },
            "policy_passing_logits_mlp": {
                "kernel": jnp.zeros((128, 1)).at[0, 0].set(3),
                "bias": jnp.array([-1.0]),
            },
        }
        logits = head.apply({"params": params}, x)
        self.assertEqual(logits.shape, (1, 82))
        np.testing.assert_array_equal(logits[0, [0, 1, 80, 81]], [1, 3, 161, 242])
        changed = x.at[0, 81, 0].set(10)
        changed_logits = head.apply({"params": params}, changed)
        np.testing.assert_array_equal(changed_logits[:, :81], logits[:, :81])
        self.assertEqual(float(changed_logits[0, 81]), 29)

    def test_value_head_reads_only_global_token_and_bounds_extreme_scores(self):
        head = ValueHead()
        params = {
            "value_policy_mlp_1": {
                "kernel": jnp.zeros((128, 1)).at[0, 0].set(1),
                "bias": jnp.zeros(1),
            },
            "value_policy_mlp_2": {
                "kernel": jnp.array([[1.0]]), "bias": jnp.zeros(1),
            },
        }
        x = jnp.full((2, 82, 128), 100.0).at[:, -1, :].set(0)
        np.testing.assert_array_equal(head.apply({"params": params}, x), [0, 0])
        x = x.at[:, -1, 0].set(100)
        np.testing.assert_array_equal(head.apply({"params": params}, x), [1, 1])
        params["value_policy_mlp_2"]["kernel"] = jnp.array([[-1.0]])
        np.testing.assert_array_equal(head.apply({"params": params}, x), [-1, -1])


class ScorchNeuralNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ScorchEnv(batch_size=2, seed=21)
        cls.env.step(jnp.array([0, 10]))
        cls.observations = cls.env.observation()
        cls.model = ScorchNeuralNetwork()
        cls.variables = cls.model.init(jax.random.PRNGKey(123), cls.observations)
        cls.forward = staticmethod(jax.jit(cls.model.apply))

    def test_forward_returns_finite_action_logits_and_bounded_values(self):
        logits, values = self.forward(self.variables, self.observations)
        self.assertEqual(logits.shape, (2, 82))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((np.abs(values) <= 1).all())

    def test_jit_matches_eager_forward(self):
        eager = self.model.apply(self.variables, self.observations)
        compiled = self.forward(self.variables, self.observations)
        for actual, expected in zip(compiled, eager):
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_batch_items_are_independent_and_single_game_is_supported(self):
        batched = self.forward(self.variables, self.observations)
        for i in range(2):
            single = self.forward(self.variables, self.observations[i:i + 1])
            self.assertEqual(single[0].shape, (1, 82))
            self.assertEqual(single[1].shape, (1,))
            for actual, expected in zip(single, batched):
                np.testing.assert_allclose(actual[0], expected[i], atol=2e-5, rtol=2e-5)

    def test_same_initialization_seed_reproduces_predictions(self):
        repeated = self.model.init(jax.random.PRNGKey(123), self.observations)
        for actual, expected in zip(
            self.forward(repeated, self.observations),
            self.forward(self.variables, self.observations),
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_legal_mask_removes_occupied_moves_and_greedy_actions_can_step(self):
        logits, _ = self.forward(self.variables, self.observations)
        mask = self.env.legal_action_mask
        probabilities = jax.nn.softmax(jnp.where(mask, logits, -jnp.inf), axis=-1)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=-1), [1, 1], atol=1e-6)
        np.testing.assert_array_equal(np.asarray(probabilities)[~np.asarray(mask)], 0)
        actions = jnp.argmax(probabilities, axis=-1)
        np.testing.assert_array_equal(mask[jnp.arange(2), actions], True)
        # Use a separate state so this integration check cannot affect other tests.
        next_state = self.env.step_fn(self.env.state, actions)
        np.testing.assert_array_equal(next_state.current_player, 1 - self.env.current_player)
        np.testing.assert_array_equal(next_state.terminated, False)

    def test_training_gradients_reach_tokenizer_all_blocks_and_both_heads(self):
        def loss(params):
            logits, values = self.model.apply({"params": params}, self.observations)
            log_probabilities = jax.nn.log_softmax(logits, axis=-1)
            # Different, legal placement targets and nonzero value targets.
            policy_loss = -jnp.mean(log_probabilities[jnp.arange(2), jnp.array([1, 2])])
            value_loss = jnp.mean((values - jnp.array([0.5, -0.5])) ** 2)
            return policy_loss + value_loss

        initial_loss, gradients = jax.jit(jax.value_and_grad(loss))(self.variables["params"])
        self.assertTrue(np.isfinite(initial_loss))
        for name, subtree in gradients.items():
            with self.subTest(parameter_group=name):
                leaves = jax.tree_util.tree_leaves(subtree)
                self.assertTrue(all(np.isfinite(leaf).all() for leaf in leaves))
                self.assertGreater(sum(float(jnp.sum(leaf ** 2)) for leaf in leaves), 0)
        updated = jax.tree_util.tree_map(
            lambda param, grad: param - 1e-4 * grad,
            self.variables["params"], gradients,
        )
        self.assertLess(float(loss(updated)), float(initial_loss))


if __name__ == "__main__":
    unittest.main()
