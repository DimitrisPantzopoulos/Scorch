"""Optimizer state, losses, shuffling, and minibatch learning."""
import functools
from typing import Tuple, cast
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from Scorch.scorch import Scorch
from .config import TrainingConfig
from .data import Metrics, Parameters, SelfPlayExamples

class ScorchLearner:
    def __init__(
            self,
            scorch : Scorch,
            config : TrainingConfig = TrainingConfig(),
        ) -> None:
        if config.minibatch_size < 1 or config.epochs < 0:
            raise ValueError("minibatch_size must be positive and epochs nonnegative.")
        self.scorch : Scorch         = scorch
        self.config : TrainingConfig = config

        optimizer : optax.GradientTransformation = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adamw(
                learning_rate = self.config.learning_rate,
                weight_decay  = self.config.weight_decay,
            ),
        )
        self.state : TrainState = TrainState.create(
            apply_fn = self.scorch.neural_network.apply,
            params   = self.scorch.params,
            tx       = optimizer,
        )

    def restore(
            self,
            state : TrainState,
        ) -> None:
        self.state         = state
        self.scorch.params = state.params

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _train_step(
            self,
            train_state        : TrainState,
            self_play_examples : SelfPlayExamples,
            value_loss_weight  : float,
        ) -> Tuple[TrainState, Metrics]:
        def _loss_fn(params : Parameters) -> Tuple[jax.Array, Metrics]:
            policy_logits, predicted_values = cast(
                tuple[jax.Array, jax.Array],
                train_state.apply_fn(
                    {'params' : params}, self_play_examples.observations
                )
            )

            # Predict the outcome - Question: How accurate are our position evaluations?
            value_loss : jax.Array = jnp.mean(
                jnp.square(predicted_values - self_play_examples.value_targets)
            )

            # Imitatation learning from search - Question: How closely do we match MCTX's policy
            policy_loss : jax.Array = optax.softmax_cross_entropy(
                logits = policy_logits,
                labels = self_play_examples.policy_targets
            ).mean()

            total_loss : jax.Array = policy_loss + value_loss * value_loss_weight

            loss_metrics : Metrics = {
                "loss"        : total_loss,
                "policy_loss" : policy_loss,
                "value_loss"  : value_loss,
            }

            return total_loss, loss_metrics

        # Compute gradients of the loss with respect to the parameters.
        (_, loss_metrics), gradients =  jax.value_and_grad(
            _loss_fn,
            has_aux  = True
        )(train_state.params)

        updated_train_state : TrainState = train_state.apply_gradients(grads=gradients)

        return updated_train_state, loss_metrics

    def shuffle(
            self,
            examples : SelfPlayExamples,
            key      : jax.Array,
        ) -> SelfPlayExamples:
        indices : jax.Array = jnp.arange(examples.episode_ids.shape[0])

        shuffled_permutation : jax.Array = jax.random.permutation(key, indices)

        return jax.tree_util.tree_map(
            lambda x : x[shuffled_permutation], examples
        )

    def minibatches(
            self,
            examples   : SelfPlayExamples,
            batch_size : int | None       = None,
        ) -> SelfPlayExamples:
        batch_size = self.config.minibatch_size if batch_size is None else batch_size
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        # 1. Calculate how many full minibatches fit
        n_examples : int = examples.episode_ids.shape[0]

        n_minibatches : int = n_examples // batch_size

        # 2. Calculate the total number of examples used by full minibatches.
        n_used : int = batch_size * n_minibatches

        # 3. Trim every field to that length
        examples = jax.tree_util.tree_map(lambda x: x[:n_used], examples)

        # 4. Reshape every field to add a minibatch dimension
        examples = jax.tree_util.tree_map(lambda x: x.reshape(n_minibatches, batch_size, *x.shape[1:]), examples)

        return examples

    def train_batch(
            self,
            self_play_examples : SelfPlayExamples,
        ) -> Metrics:
        if self_play_examples.observations.shape[0] == 0:
            raise ValueError("Cannot train on an empty batch of examples.")

        self.state, metrics = self._train_step(
            self.state,
            self_play_examples,
            self.config.value_loss_weight
        )

        # Update the network params.
        self.scorch.params = self.state.params

        return metrics

    def _update_one_batch(
            self,
            carry     : TrainState,
            minibatch : SelfPlayExamples,
        ) -> Tuple[TrainState, Metrics]:
        return self._train_step(carry, minibatch, self.config.value_loss_weight)

    def train_epoch(
            self,
            examples   : SelfPlayExamples,
            key        : jax.Array,
            batch_size : int | None       = None,
        ) -> Metrics:
        shuffled_examples : SelfPlayExamples = self.shuffle(examples, key)

        # 2. Make them mini-batches
        minibatches : SelfPlayExamples = self.minibatches(shuffled_examples, batch_size)

        # 3. Loop over each, minibatch and call train_batch()
        self.state, metrics = jax.lax.scan(
            self._update_one_batch,
            self.state,
            xs=minibatches,
        )

        self.scorch.params = self.state.params

        return metrics
