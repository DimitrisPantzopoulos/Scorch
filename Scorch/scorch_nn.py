import jax
import jax.numpy as jnp

from typing import Tuple
from flax   import linen as nn

from .config import NetworkConfig

class BoardTokenizer(nn.Module):
    config : NetworkConfig = NetworkConfig()

    @nn.compact
    def __call__(
            self,
            observations : jax.Array,
        ) -> jax.Array:
        if observations.ndim != 4 or any(size < 1 for size in observations.shape[1:]):
            raise ValueError(
                "observations must have shape (batch, rows, columns, features) "
                "with positive board dimensions and features."
            )

        batch_size : int = observations.shape[0]
        n_rows     : int = observations.shape[1]
        n_cols     : int = observations.shape[2]
        n_features : int = observations.shape[3]
        n_tokens   : int = n_rows * n_cols
        d_model    : int = self.config.d_model

        # One Token Per Intersection
        x : jax.Array = observations.reshape(batch_size, n_tokens, n_features)

        # Apply Learned Projection to every intersection - Question : What happened here?
        x = nn.Dense(features=d_model)(x)

        # Apply Positional Embeddings - Question : Where did it happen?
        row_embeddings : jax.Array = self.param(
            "row_embeddings",
            nn.initializers.normal(stddev=self.config.embedding_stddev),
            (n_rows, d_model)
        )

        col_embeddings : jax.Array = self.param(
            "col_embeddings",
            nn.initializers.normal(stddev=self.config.embedding_stddev),
            (n_cols, d_model)
        )

        positions : jax.Array = (
            row_embeddings[:, None, :] + col_embeddings[None, :, :]
        ).reshape(n_tokens, d_model)

        x = x + positions[None, :, :]

        global_token : jax.Array = self.param(
            "global_token",
            nn.initializers.normal(stddev=self.config.embedding_stddev),
            (1, 1, d_model)
        )

        # Dedicated vector to collect information about the whole board - Question: What should I know about this board.
        global_tokens : jax.Array = jnp.broadcast_to(
            global_token,
            (batch_size, 1, d_model)
        )

        return jnp.concatenate(
            (x, global_tokens), axis=1
        )

class MultiHeadAttention(nn.Module):
    config : NetworkConfig = NetworkConfig()

    @nn.compact
    def __call__(
            self,
            x : jax.Array,
        ) -> jax.Array:
        batch_size, n_tokens, _ = x.shape

        d_model  : int = self.config.d_model
        head_dim : int = d_model // self.config.num_heads

        split_heads_reshape : Tuple[int, ...] = (batch_size, n_tokens, self.config.num_heads, head_dim)
        token_axis          : Tuple[int, ...] = (0, 2, 1, 3)

        # Split features into heads, then place the head axis before tokens.
        query : jax.Array = nn.Dense(
            features=d_model, name='query'
        )(x).reshape(*split_heads_reshape).transpose(*token_axis)

        key   : jax.Array = nn.Dense(
            features=d_model, name='key'
        )(x).reshape(*split_heads_reshape).transpose(*token_axis)

        value : jax.Array = nn.Dense(
            features=d_model, name='value'
        )(x).reshape(*split_heads_reshape).transpose(*token_axis)

        scores : jax.Array = query @ key.swapaxes(-1, -2) / jnp.sqrt(head_dim)

        # Apply the self attention to the tokens. - Question: Which tokens should each token gather information from?? 
        attention : jax.Array = (
            nn.softmax(scores, axis=-1) @ value
        ).transpose(*token_axis).reshape(batch_size, n_tokens, d_model)

        # Mix information from the different attention heads. - Question: How should their gathered information be mixed?
        return nn.Dense(features=d_model, name='output')(attention)

class TransformerBlock(nn.Module):
    config : NetworkConfig = NetworkConfig()

    @nn.compact
    def __call__(
            self,
            x : jax.Array,
        ) -> jax.Array:
        # Normalise Input features - Question: Can we apply attention a consistent input scale?
        normalised_features : jax.Array = nn.LayerNorm(name='feature_attention_norm')(x)

        # Exchange Information using MultiHeadAttention - Question: What information should each token gather from the others?
        attention : jax.Array = MultiHeadAttention(
            config = self.config,
            name   = 'attention',
        )(normalised_features)

        # Residual connection - Question: What information should we preserve and add to the existing tokens.
        x = x + attention

        normalised_mlp_features : jax.Array = nn.LayerNorm(name='mlp_features_norm')(x)

        hidden : jax.Array = nn.gelu(nn.Dense(
            features = self.config.d_model * self.config.mlp_ratio,
            name     = 'transformer_attention_mlp_layer_1',
        )(normalised_mlp_features))

        hidden = nn.Dense(
            features = self.config.d_model,
            name     = 'transformer_attention_mlp_layer_2'
        )(hidden)

        return x + hidden

class PolicyHead(nn.Module):
    @nn.compact
    def __call__(
            self,
            x : jax.Array,
        ) -> jax.Array:
        board_token  : jax.Array = x[:, :-1, :]  # (B, 81, D_MODEL)
        global_token : jax.Array = x[:, -1,  :]  # (B, D_MODEL)

        # Logit distribution for each intersection - Question: How promising is placing here?
        policy_logits : jax.Array = nn.Dense(
            features = 1,
            name     = "policy_placement_logits_mlp",
        )(board_token).squeeze(-1) # (B, 81)

        # Score Passing 
        pass_logits : jax.Array = nn.Dense(
            features = 1,
            name     = "policy_passing_logits_mlp"
        )(global_token) # (B, 1)

        return jnp.concatenate(
            (policy_logits, pass_logits), axis=-1
        ) # (B, 82)

class ValueHead(nn.Module):
    config : NetworkConfig = NetworkConfig()

    @nn.compact
    def __call__(
            self,
            x : jax.Array,
        ) -> jax.Array:
        global_token : jax.Array = x[:, -1, :] # (B, D_MODEL)

        # Evaluate the position - Question: How favourable is this position?
        hidden : jax.Array = nn.gelu(nn.Dense(
            features = self.config.value_hidden_dim,
            name     = 'value_policy_mlp_1'
        )(global_token))

        value : jax.Array = nn.Dense(
            features = 1,
            name     = 'value_policy_mlp_2'
        )(hidden)

        return jnp.tanh(value).squeeze(-1) # (B, )
    
class ScorchNeuralNetwork(nn.Module):
    config : NetworkConfig = NetworkConfig()

    @nn.compact
    def __call__(
            self,
            observations : jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
        # Encode the Board
        x : jax.Array = BoardTokenizer(
            config = self.config,
            name   = 'Tokenizer',
        )(observations)

        # Refine what relationships within the board.
        for i in range(self.config.num_blocks):
            x = TransformerBlock(
                config = self.config,
                name   = f'Attention_Block_{i}',
            )(x)

        x = nn.LayerNorm(name='post_attention_norm')(x)

        policy_logits : jax.Array = PolicyHead(name='policy_head')(x)
        value         : jax.Array = ValueHead(
            config = self.config,
            name   = 'value_head',
        )(x)

        return policy_logits, value
