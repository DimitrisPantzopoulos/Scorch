import jax
import jax.numpy as jnp

from typing import ClassVar, Tuple
from flax   import linen as nn

D_MODEL : int   = 128
STDDEV  : float = 0.02

class BoardTokenizer(nn.Module):
    TOKENS     : ClassVar[int] = 81
    N_FEATURES : ClassVar[int] = 17

    @nn.compact
    def __call__(self, observations : jax.Array) -> jax.Array:
        batch_size : int = observations.shape[0]
        board_size : int = observations.shape[1]

        
        # One Token Per Intersection
        x : jax.Array = observations.reshape(batch_size, self.TOKENS, self.N_FEATURES)

        # Apply Learned Projection to every intersection - Question : What happened here?
        x = nn.Dense(features=D_MODEL)(x)

        # Apply Positional Embeddings - Question : Where did it happen?
        row_embeddings : jax.Array = self.param(
            "row_embeddings",
            nn.initializers.normal(stddev=STDDEV),
            (board_size, D_MODEL)
        )

        col_embeddings : jax.Array = self.param(
            "col_embeddings",
            nn.initializers.normal(stddev=STDDEV),
            (board_size, D_MODEL)
        )

        positions : jax.Array = (row_embeddings[:, None, :] + col_embeddings[None, :, :]).reshape(self.TOKENS, D_MODEL)

        x = x + positions[None, :, :]

        global_token : jax.Array = self.param(
            "global_token",
            nn.initializers.normal(stddev=STDDEV),
            (1, 1, D_MODEL)
        )

        # Dedicated vector to collect information about the whole board - Question: What should I know about this board.
        global_tokens : jax.Array = jnp.broadcast_to(
            global_token,
            (batch_size, 1, D_MODEL)
        )

        return jnp.concatenate(
            (x, global_tokens), axis=1
        )

class MultiHeadAttention(nn.Module):
    HEADS : ClassVar[int] = 4

    @nn.compact
    def __call__(self, x : jax.Array) -> jax.Array:
        batch_size, n_tokens, _ = x.shape

        head_dim : int = D_MODEL // self.HEADS

        split_heads_reshape : Tuple[int, ...] = (batch_size, n_tokens, self.HEADS, head_dim)
        token_axis          : Tuple[int, ...] = (0, 2, 1, 3)

        # (B, 82, 128)   -> Reshape   -> (B, 82, 4, 32). 
        # (B, 82, 4, 32) -> Transpose -> (B, 4, 82, 32). 
        query : jax.Array = nn.Dense(
            features=D_MODEL, name='query'
        )(x).reshape(*split_heads_reshape).transpose(*token_axis)

        key   : jax.Array = nn.Dense(
            features=D_MODEL, name='key'
        )(x).reshape(*split_heads_reshape).transpose(*token_axis)

        value : jax.Array = nn.Dense(
            features=D_MODEL, name='value'
        )(x).reshape(*split_heads_reshape).transpose(*token_axis)

        scores : jax.Array = query @ key.swapaxes(-1, -2) / jnp.sqrt(head_dim)

        # Apply the self attention to the tokens. - Question: Which tokens should each token gather information from?? 
        attention : jax.Array = (
            nn.softmax(scores, axis=-1) @ value
        ).transpose(*token_axis).reshape(batch_size, n_tokens, D_MODEL)

        # Mix information from the different attention heads. - Question: How should their gathered information be mixed?
        return nn.Dense(features=D_MODEL, name='output')(attention)

class TransformerBlock(nn.Module):
    MLP_RATIO : ClassVar[int] = 4

    @nn.compact
    def __call__(self, x : jax.Array) -> jax.Array:
        # Normalise Input features - Question: Can we apply attention a consistent input scale?
        normalised_features : jax.Array = nn.LayerNorm(name='feature_attention_norm')(x)

        # Exchange Information using MultiHeadAttention - Question: What information should each token gather from the others?
        attention : jax.Array = MultiHeadAttention(name='attention')(normalised_features)

        # Residual connection - Question: What information should we preserve and add to the existing tokens.
        x = x + attention

        normalised_mlp_features : jax.Array = nn.LayerNorm(name='mlp_features_norm')(x)

        hidden : jax.Array = nn.gelu(nn.Dense(
            features = D_MODEL * self.MLP_RATIO,
            name     = 'transformer_attention_mlp_layer_1' 
        )(normalised_mlp_features))

        hidden = nn.Dense(
            features = D_MODEL,
            name     = 'transformer_attention_mlp_layer_2'
        )(hidden)

        return x + hidden

class PolicyHead(nn.Module):
    @nn.compact
    def __call__(self, x : jax.Array) -> jax.Array:
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
    @nn.compact
    def __call__(self, x : jax.Array) -> jax.Array:
        global_token : jax.Array = x[:, -1, :] # (B, D_MODEL)

        # Evaluate the position - Question: How favourable is this position?
        hidden : jax.Array = nn.gelu(nn.Dense(
            features = 1,
            name     = 'value_policy_mlp_1'
        )(global_token))

        value : jax.Array = nn.Dense(
            features = 1,
            name     = 'value_policy_mlp_2'
        )(hidden)

        return jnp.tanh(value).squeeze(-1) # (B, )
    
class ScorchNeuralNetwork(nn.Module):
    N_MHA_BLOCKS : ClassVar[int] = 4

    @nn.compact
    def __call__(self, observations : jax.Array) -> Tuple[jax.Array, jax.Array]:
        # Encode the Board
        x = BoardTokenizer(name='Tokenizer')(observations)

        # Refine what relationships within the board.
        for i in range(self.N_MHA_BLOCKS):
            x = TransformerBlock(name=f'Attention_Block_{i}')(x)

        x = nn.LayerNorm(name='post_attention_norm')(x)

        policy_logits : jax.Array = PolicyHead(name='policy_head')(x)
        value         : jax.Array = ValueHead(name='value_head')(x)

        return policy_logits, value

        









