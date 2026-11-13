# Scorch network and agent

Use an immutable `NetworkConfig` to define an architecture for each experiment.
Import the public classes from `Scorch`:

```python
from Scorch import NetworkConfig, Scorch, ScorchNeuralNetwork

network_config : NetworkConfig = NetworkConfig(
    d_model          = 256,
    num_heads        = 8,
    num_blocks       = 6,
    mlp_ratio        = 4,
    value_hidden_dim = 64,
    embedding_stddev = 0.02,
)
```

| Setting | Default | Meaning |
|---|---|---|
| `d_model` | `128` | Token width and attention projection width |
| `num_heads` | `4` | Attention heads; must divide `d_model` |
| `num_blocks` | `4` | Transformer blocks; zero allows a tokenizer/heads ablation |
| `mlp_ratio` | `4` | Transformer hidden width as a multiple of `d_model` |
| `value_hidden_dim` | `1` | Hidden width of the value head |
| `embedding_stddev` | `0.02` | Initialization standard deviation of row, column, and global embeddings |

Widths, heads, and MLP ratio must be positive integers. The number of blocks
must be a nonnegative integer, and embedding standard deviation must be finite
and nonnegative. Invalid settings raise `ValueError` at config construction.
The default value head retains its original hidden width of one; increase
`value_hidden_dim` to experiment with a wider value head.

## Neural network alone

```python
import jax
from Go.env import ScorchEnv

env   : ScorchEnv          = ScorchEnv(batch_size=2)
model : ScorchNeuralNetwork = ScorchNeuralNetwork(config=network_config)

variables = model.init(jax.random.PRNGKey(123), env.observation())
policy_logits, values = model.apply(variables, env.observation())
```

Policy logits have shape `(batch_size, 82)` and values have shape
`(batch_size,)` for the 9x9 Go environment. Values are bounded by `tanh`.
The tokenizer infers rows, columns, and feature count from observations with
shape `(batch, rows, columns, features)`. A parameter set is initialized for
that input shape; this does not change the environment's 9x9 Go rules.

## Scorch agent with search

```python
from MCTX.mctx import SearchConfig

agent : Scorch = Scorch(
    search_config  = SearchConfig(num_simulations=64),
    env            = env,
    seed           = 123,
    network_config = network_config,
)

actions, policy_targets = agent.act(env, temperature=1.0)
env.step(actions)
```

`agent.network_config` reports the network's architecture. `agent.search_config`
controls MCTX. `act()` owns and advances the agent's random key, and uses the
current `agent.params`. The existing constructor form
`Scorch(search_config, env, seed=123)` still uses the original default network.

## Training and parameter sweeps

```python
from dataclasses import replace
from Trainer import ScorchTrainer, TrainingConfig

for num_blocks in (2, 4, 6):
    experiment : NetworkConfig = replace(network_config, num_blocks=num_blocks)

    with ScorchTrainer.create(
            num_envs        = 256,
            seed            = 42,
            network_config  = experiment,
            search_config   = SearchConfig(num_simulations=64),
            training_config = TrainingConfig(
                                  learning_rate  = 1e-4,
                                  epochs         = 4,
                                  minibatch_size = 128,
                              ),
            run_name        = f"blocks_{num_blocks}",
        ) as trainer:

        report = trainer.train(iterations=100)
```

Create a new network or trainer for each architecture. Replacing a config does
not resize existing parameters or change a compiled model. Different shapes
can require fresh JAX compilation. Search settings remain separate and can be
replaced on the agent for search experiments.

Resume checkpoints with the same architecture and training/search/collection
settings. Configs are supplied when constructing the trainer; they are not
stored or reconstructed by checkpoints. The defaults retain the original
parameter names, shapes, initialization, and predictions, so existing default
checkpoints remain compatible. See `Trainer/README.md` for the complete training
and checkpoint API.
