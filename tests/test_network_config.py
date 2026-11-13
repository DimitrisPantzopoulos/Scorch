"""Configurable architecture, agent search, and trainer factory integration."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from Go.env import ScorchEnv
from MCTX.mctx import SearchConfig
from Scorch import NetworkConfig, Scorch, ScorchNeuralNetwork
from Scorch.scorch_nn import BoardTokenizer, MultiHeadAttention
from Trainer import EvaluationConfig, ScorchTrainer, SelfPlayConfig, SelfPlayExamples, TrainingConfig


def assert_tree_equal(test : unittest.TestCase, actual, expected) -> None:
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual),
                                          jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class NetworkConfigTests(unittest.TestCase):
    def test_defaults_preserve_the_original_architecture(self) -> None:
        config : NetworkConfig = NetworkConfig()
        self.assertEqual((config.d_model, config.num_heads, config.num_blocks,
                          config.mlp_ratio, config.value_hidden_dim,
                          config.embedding_stddev), (128, 4, 4, 4, 1, 0.02))

    def test_config_is_immutable_hashable_and_can_be_replaced_for_sweeps(self) -> None:
        original : NetworkConfig = NetworkConfig()
        with self.assertRaises(FrozenInstanceError):
            original.d_model = 64
        smaller : NetworkConfig = replace(original, d_model=32, num_blocks=1)
        self.assertEqual(original.d_model, 128)
        self.assertEqual(smaller.d_model, 32)
        self.assertEqual({original: "default", smaller: "small"}[smaller], "small")

    def test_invalid_dimensions_are_rejected_before_initialization(self) -> None:
        for field in ("d_model", "num_heads", "mlp_ratio", "value_hidden_dim"):
            for value in (0, -1, 1.5, True, "4"):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        NetworkConfig(**{field: value})
        for value in (-1, 1.5, True, "4"):
            with self.subTest(num_blocks=value):
                with self.assertRaisesRegex(ValueError, "num_blocks"):
                    NetworkConfig(num_blocks=value)

    def test_attention_width_must_be_divisible_by_head_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            NetworkConfig(d_model=25, num_heads=4)

    def test_embedding_standard_deviation_must_be_finite_and_nonnegative(self) -> None:
        for value in (-0.1, float("nan"), float("inf"), True, "0.02"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "embedding_stddev"):
                    NetworkConfig(embedding_stddev=value)


class ConfiguredNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = NetworkConfig(d_model=24, num_heads=3, num_blocks=2,
                                   mlp_ratio=2, value_hidden_dim=8)
        cls.observations = jax.random.normal(jax.random.PRNGKey(7), (2, 9, 9, 17))
        cls.model = ScorchNeuralNetwork(config=cls.config)
        cls.variables = cls.model.init(jax.random.PRNGKey(123), cls.observations)

    def test_custom_dimensions_reach_every_parameter_group(self) -> None:
        params = self.variables["params"]
        self.assertEqual(params["Tokenizer"]["Dense_0"]["kernel"].shape, (17, 24))
        self.assertEqual(params["Tokenizer"]["row_embeddings"].shape, (9, 24))
        self.assertEqual(params["Tokenizer"]["global_token"].shape, (1, 1, 24))
        blocks = sorted(name for name in params if name.startswith("Attention_Block_"))
        self.assertEqual(blocks, ["Attention_Block_0", "Attention_Block_1"])
        for name in blocks:
            self.assertEqual(params[name]["attention"]["query"]["kernel"].shape, (24, 24))
            self.assertEqual(params[name]["transformer_attention_mlp_layer_1"]["kernel"].shape, (24, 48))
            self.assertEqual(params[name]["transformer_attention_mlp_layer_2"]["kernel"].shape, (48, 24))
        self.assertEqual(params["value_head"]["value_policy_mlp_1"]["kernel"].shape, (24, 8))
        self.assertEqual(params["value_head"]["value_policy_mlp_2"]["kernel"].shape, (8, 1))
        self.assertEqual(params["policy_head"]["policy_placement_logits_mlp"]["kernel"].shape, (24, 1))

    def test_custom_forward_is_finite_bounded_and_jit_compatible(self) -> None:
        eager = self.model.apply(self.variables, self.observations)
        compiled = jax.jit(self.model.apply)(self.variables, self.observations)
        self.assertEqual(compiled[0].shape, (2, 82))
        self.assertEqual(compiled[1].shape, (2,))
        for actual, expected in zip(compiled, eager):
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertTrue((np.abs(compiled[1]) <= 1).all())

    def test_tokenizer_derives_board_dimensions_and_features_from_observations(self) -> None:
        model : ScorchNeuralNetwork = ScorchNeuralNetwork(config=self.config)
        observations : jax.Array = jnp.zeros((2, 3, 5, 7))
        variables = model.init(jax.random.PRNGKey(17), observations)
        logits, values = model.apply(variables, observations)
        self.assertEqual(logits.shape, (2, 16))
        self.assertEqual(values.shape, (2,))
        params = variables["params"]["Tokenizer"]
        self.assertEqual(params["Dense_0"]["kernel"].shape, (7, 24))
        self.assertEqual(params["row_embeddings"].shape, (3, 24))
        self.assertEqual(params["col_embeddings"].shape, (5, 24))

    def test_tokenizer_rejects_invalid_observation_shapes_clearly(self) -> None:
        tokenizer : BoardTokenizer = BoardTokenizer(config=self.config)
        for shape in ((2, 81, 17), (2, 0, 9, 17), (2, 9, 9, 0)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "observations"):
                    tokenizer.init(jax.random.PRNGKey(0), jnp.zeros(shape))

    def test_embedding_stddev_controls_only_learned_embedding_initialization(self) -> None:
        variables = []
        for stddev in (0.0, 0.1, 0.2):
            tokenizer = BoardTokenizer(config=replace(self.config, embedding_stddev=stddev))
            variables.append(tokenizer.init(jax.random.PRNGKey(5), self.observations)["params"])
        for name in ("row_embeddings", "col_embeddings", "global_token"):
            np.testing.assert_array_equal(variables[0][name], 0)
            np.testing.assert_allclose(variables[2][name], 2 * variables[1][name], atol=1e-7)
            self.assertGreater(float(jnp.linalg.norm(variables[1][name])), 0)
        assert_tree_equal(self, variables[0]["Dense_0"], variables[2]["Dense_0"])

    def test_attention_uses_configured_heads_and_head_dimension_for_scaling(self) -> None:
        config : NetworkConfig = NetworkConfig(d_model=8, num_heads=2)
        attention : MultiHeadAttention = MultiHeadAttention(config=config)
        x : jax.Array = jnp.zeros((1, 2, 8)).at[0, 0, jnp.array([0, 4])].set(1)
        params = {name: {"kernel": jnp.eye(8), "bias": jnp.zeros(8)}
                  for name in ("query", "key", "value", "output")}
        output : jax.Array = attention.apply({"params": params}, x)
        expected = np.zeros((1, 2, 8), dtype=np.float32)
        # softmax([1/sqrt(4), 0])[0], independently calculated.
        expected[0, 0, [0, 4]] = 0.622459331
        expected[0, 1, [0, 4]] = 0.5
        np.testing.assert_allclose(output, expected, atol=1e-6, rtol=1e-6)

    def test_zero_blocks_supports_a_tokenizer_and_heads_only_ablation(self) -> None:
        model : ScorchNeuralNetwork = ScorchNeuralNetwork(config=replace(self.config, num_blocks=0))
        variables = model.init(jax.random.PRNGKey(3), self.observations)
        self.assertFalse(any(name.startswith("Attention_Block_") for name in variables["params"]))
        logits, values = model.apply(variables, self.observations)
        self.assertEqual(logits.shape, (2, 82))
        self.assertTrue(np.isfinite(values).all())

    def test_gradients_reach_custom_tokenizer_blocks_and_both_heads(self) -> None:
        def loss(params) -> jax.Array:
            logits, values = self.model.apply({"params": params}, self.observations)
            policy = -jax.nn.log_softmax(logits)[jnp.arange(2), jnp.array([1, 2])].mean()
            return policy + jnp.mean((values - jnp.array([0.4, -0.8])) ** 2)

        value, gradients = jax.jit(jax.value_and_grad(loss))(self.variables["params"])
        self.assertTrue(np.isfinite(value))
        for name, subtree in gradients.items():
            with self.subTest(group=name):
                leaves = jax.tree_util.tree_leaves(subtree)
                self.assertTrue(all(np.isfinite(leaf).all() for leaf in leaves))
                self.assertGreater(sum(float(jnp.sum(leaf ** 2)) for leaf in leaves), 0)

    def test_different_architectures_do_not_share_or_mutate_settings(self) -> None:
        config : NetworkConfig = replace(self.config, d_model=12, num_blocks=1)
        model : ScorchNeuralNetwork = ScorchNeuralNetwork(config=config)
        variables = model.init(jax.random.PRNGKey(0), self.observations)
        self.assertEqual(variables["params"]["Tokenizer"]["Dense_0"]["kernel"].shape, (17, 12))
        self.assertEqual(self.variables["params"]["Tokenizer"]["Dense_0"]["kernel"].shape, (17, 24))
        self.assertEqual(self.model.config, self.config)


class ConfiguredAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = NetworkConfig(d_model=16, num_heads=2, num_blocks=1,
                                   mlp_ratio=2, value_hidden_dim=4)
        cls.search = SearchConfig(num_simulations=2, max_depth=1, dirichlet_fraction=0.0)
        cls.env = ScorchEnv(batch_size=2, seed=21)

    def test_agent_uses_the_configured_network_for_real_legal_search(self) -> None:
        agent : Scorch = Scorch(self.search, self.env, seed=123, network_config=self.config)
        before = self.env.state
        self.assertEqual(agent.network_config, self.config)
        self.assertEqual(agent.neural_network.config, self.config)
        self.assertIs(agent.mctx.neural_network, agent.neural_network)
        self.assertEqual(agent.params["Tokenizer"]["Dense_0"]["kernel"].shape, (17, 16))
        actions, targets = agent.act(self.env, temperature=0.0)
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(targets.shape, (2, 82))
        self.assertTrue(bool(self.env.legal_action_mask[jnp.arange(2), actions].all()))
        np.testing.assert_allclose(targets.sum(-1), 1, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(targets)[~np.asarray(self.env.legal_action_mask)], 0)
        assert_tree_equal(self, self.env.state, before)

    def test_configured_agents_reproduce_initialization_and_search_with_same_seed(self) -> None:
        first : Scorch = Scorch(self.search, self.env, seed=123, network_config=self.config)
        second : Scorch = Scorch(self.search, self.env, seed=123, network_config=self.config)
        assert_tree_equal(self, first.params, second.params)
        assert_tree_equal(self, first.act(self.env), second.act(self.env))
        np.testing.assert_array_equal(first.prng_key, second.prng_key)

    def test_public_package_imports_work_in_fresh_search_agent_and_trainer_orders(self) -> None:
        for imports in (
                "import MCTX.mctx; from Scorch import NetworkConfig, Scorch, ScorchNeuralNetwork; import Trainer",
                "import Scorch; import Trainer; import MCTX.mctx",
                "import Trainer; import Scorch; import MCTX.mctx",
            ):
            with self.subTest(imports=imports):
                result = subprocess.run([sys.executable, "-c", imports], capture_output=True,
                                        text=True, cwd=Path(__file__).resolve().parents[1], timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)


class ConfiguredTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="scorch-network-config-")
        self.addCleanup(directory.cleanup)
        self.directory : Path = Path(directory.name)
        self.config : NetworkConfig = NetworkConfig(d_model=16, num_heads=2, num_blocks=1,
                                                    mlp_ratio=2, value_hidden_dim=4)

    def make_trainer(self, name : str, seed : int = 42) -> ScorchTrainer:
        trainer : ScorchTrainer = ScorchTrainer.create(
            num_envs          = 2,
            seed              = seed,
            network_config    = self.config,
            search_config     = SearchConfig(num_simulations=1, max_depth=1, dirichlet_fraction=0.0),
            training_config   = TrainingConfig(epochs=1, minibatch_size=2, weight_decay=0.0),
            self_play_config  = SelfPlayConfig(num_steps=4, temperature=0.0),
            evaluation_config = None,
            log_dir           = self.directory / name,
        )
        self.addCleanup(trainer.close)
        return trainer

    def examples(self, trainer : ScorchTrainer) -> SelfPlayExamples:
        return SelfPlayExamples(
            observations   = trainer.env.observation(),
            policy_targets = jax.nn.one_hot(jnp.array([0, 81]), 82),
            value_targets  = jnp.array([0.75, -0.5]),
            side_to_moves  = trainer.env.current_player,
            episode_ids    = jnp.array([40, 41]),
        )

    def test_factory_uses_custom_model_in_collection_learning_and_evaluation(self) -> None:
        trainer : ScorchTrainer = self.make_trainer("factory")
        self.assertEqual(trainer.scorch.neural_network.config, self.config)
        self.assertIs(trainer.collector.scorch, trainer.scorch)
        self.assertIs(trainer.evaluator.scorch, trainer.scorch)
        before = trainer.learner.state.params
        metrics = trainer.learner.train_batch(self.examples(trainer))
        self.assertTrue(all(np.isfinite(value).all() for value in metrics.values()))
        self.assertEqual(int(trainer.learner.state.step), 1)
        self.assertIs(trainer.scorch.params, trainer.learner.state.params)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(trainer.scorch.params))))

    def test_custom_model_checkpoint_resumes_the_next_update_exactly(self) -> None:
        first : ScorchTrainer = self.make_trainer("first")
        examples : SelfPlayExamples = self.examples(first)
        first.learner.train_batch(examples)
        first.save_checkpoint(self.directory / "checkpoint")
        expected_metrics = first.learner.train_batch(examples)
        resumed : ScorchTrainer = self.make_trainer("resumed", seed=999)
        resumed.load_checkpoint(self.directory / "checkpoint")
        actual_metrics = resumed.learner.train_batch(examples)
        assert_tree_equal(self, actual_metrics, expected_metrics)
        # TrainState static callables differ between independently built trainers.
        assert_tree_equal(self, resumed.learner.state.params, first.learner.state.params)
        assert_tree_equal(self, resumed.learner.state.opt_state, first.learner.state.opt_state)
        self.assertEqual(int(resumed.learner.state.step), int(first.learner.state.step))

    def test_training_loop_collects_and_trains_with_a_custom_architecture(self) -> None:
        trainer : ScorchTrainer = self.make_trainer("loop")
        params = jax.tree_util.tree_map(jnp.zeros_like, trainer.scorch.params)
        params["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        params["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        trainer.learner.restore(trainer.learner.state.replace(params=params))
        report = trainer.train(iterations=1)
        self.assertEqual(report.optimizer_step, 2)
        self.assertEqual(report.next_iteration, 1)
        self.assertEqual(trainer.scorch.params["Tokenizer"]["Dense_0"]["kernel"].shape, (17, 16))

    def test_evaluation_runs_a_completed_match_with_the_custom_architecture(self) -> None:
        trainer : ScorchTrainer = self.make_trainer("evaluation")
        params = jax.tree_util.tree_map(jnp.zeros_like, trainer.scorch.params)
        params["policy_head"]["policy_placement_logits_mlp"]["bias"] = jnp.array([-3.0])
        params["policy_head"]["policy_passing_logits_mlp"]["bias"] = jnp.array([3.0])
        before = trainer.scorch.params
        result = trainer.evaluator.evaluate(
            candidate_params = params,
            opponent_params  = params,
            key              = jax.random.PRNGKey(23),
            config           = EvaluationConfig(num_games=2, num_steps=4),
        )
        self.assertEqual(result.unfinished, 0)
        self.assertEqual(result.wins + result.losses + result.draws, 2)
        self.assertEqual(result.score_rate, 0.5)
        assert_tree_equal(self, trainer.scorch.params, before)


if __name__ == "__main__":
    unittest.main()
