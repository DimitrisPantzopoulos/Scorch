# Tests

From the project root, using a Python interpreter with `jax`, `flax`, `pgx`, `mctx`, `chex`, `optax`,
and `numpy` installed:

```powershell
python -m unittest discover -s tests -v
```

The suite uses Python's standard-library `unittest`; no test runner dependency
is required. Small CPU batches exercise real PGX games, Flax modules, JIT
compilation, and gradients. The existing `test.py` remains a standalone smoke
test and can be run separately with `python test.py`.

Environment tests cover independent batched moves, rewards after two passes,
termination and truncation resets, preservation of active games, reproducible
random keys, and a batch of one. Model tests cover token positions, scaled
attention, residuals, placement/pass routing, value bounds, eager/JIT agreement,
batch independence, legal-action integration, and a decreasing training loss.

MCTX adapter tests additionally check root masking, recurrent PGX transitions,
rewards from the acting player's perspective, alternating-player discounts,
terminal/truncated states, legal search targets, depth and simulation limits,
reproducibility, and preservation of live state, random keys, and parameters.
They use the real `ScorchNeuralNetwork` and the installed mctx library.

Run just the MCTX class tests from the project root:

```powershell
python -m unittest tests.test_mctx -v
```

Scorch agent tests exercise seeded parameter initialization, legal actions and
training targets, parameter/configuration/temperature forwarding, random-key
advancement, reproducible action sequences, different environment batch sizes,
and preservation of environment state and network parameters. They use the real
neural network, PGX environment, and MCTX search.

Run just the Scorch agent tests from the project root:

```powershell
python -m unittest tests.test_scorch -v
```

Trainer and self-play collector tests cover weighted value loss, soft policy
targets, optimizer updates, learning rate and weight decay, agent parameter
synchronization, finished-game filtering, outcome labels, episode IDs, padding,
truncation, empty rollouts, and collection-to-training integration. A controlled
real network prefers passing so real Go games finish in two moves.

Training-loop tests also check aligned shuffling, full minibatches and discarded
tails, empty datasets, epoch updates against sequential batches, configured
iteration/epoch counts, episode allocation, early exits, and reproducible
resumption using the returned random key.
Run these tests from the project root:

```powershell
python -m unittest tests.test_training -v
```
