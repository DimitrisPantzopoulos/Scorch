# Tests

From the project root, using a Python interpreter with `jax`, `flax`, `pgx`, `mctx`, `chex`, `optax`, `orbax-checkpoint`, `tensorboardX`, `tensorboard`,
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

TensorBoard logging tests read real event files to verify scalar names, values,
minibatch averaging, iteration/epoch steps, self-play counts, timings, and
logging when there are too few completed examples to train. Writers are closed
before readback to drain asynchronous writes. Training fixtures redirect real
writers to temporary folders and clean up their files automatically.

Run the logging tests from the project root:

```powershell
python -m unittest tests.test_tensorboard_logging -v
```

Checkpoint tests save and restore real Orbax checkpoints in temporary folders.
They verify model parameters, Adam optimizer state, training steps, episode
counters, trainer/agent/environment random keys, fresh-trainer predictions,
deterministic optimizer and action resumption, relative paths, zero-state
checkpoints, missing paths, overwrite protection, and independent snapshots.
The environment RNG is restored; reset the environment to reproduce the next
game, since live board state is not part of the checkpoint payload.

Run the checkpoint tests from the project root:

```powershell
python -m unittest tests.test_checkpoints -v
```
