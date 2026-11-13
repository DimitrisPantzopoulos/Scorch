# Tests

Use a Python interpreter with `jax`, `flax`, `pgx`, `mctx`, `chex`, `optax`,
`orbax-checkpoint`, `tensorboardX`, `tensorboard`, and `numpy` installed.
The runner is Python's standard-library `unittest`.

From the project root:

```powershell
python -m unittest discover -s tests -v
```

The `-s tests` option runs the automated suite without executing the standalone
`test.py` TensorBoard example at the project root.

Run individual areas with:

```powershell
python -m unittest tests.test_env tests.test_model -v
python -m unittest tests.test_mctx tests.test_scorch -v
python -m unittest tests.test_training -v
python -m unittest tests.test_evaluator tests.test_final_training -v
python -m unittest tests.test_checkpoints tests.test_tensorboard_logging -v
python -m unittest tests.test_public_api -v
```

## Coverage

- Environment: batched moves, rewards, termination/truncation resets, random
  reproducibility, and preservation of unfinished games.
- Neural network: token ordering, attention, residuals, heads, JIT agreement,
  batch independence, legal-action masking, and useful gradients.
- Agent/search: legal moves, normalized targets, reward perspective,
  terminal handling, configuration forwarding, RNG advancement, and state
  preservation.
- Learner: weighted losses, soft policy targets, parameter synchronization,
  optimizer state, learning rate/decay, aligned shuffling, and full minibatches.
- Collection/training: completed-game filtering, labels and episode IDs,
  empty/unfinished rollouts, epoch and iteration counts, and reproducibility.
- Evaluation/promotion: balanced colours, mixed player IDs, comparison against
  the accepted model, complete-match requirements, optimizer rollback, and
  errors without accidentally accepting a candidate.
- Checkpoints: real Orbax round trips, full optimizer state and keys, independent
  snapshots, missing paths, overwrite protection, and legacy payloads.
- TensorBoard: real event-file readback of values, means, tags, timings,
  iteration/epoch steps, evaluation results, and skipped-training cases.
- Public API: factory setup and seed streams, reports, owned RNG keys,
  context-manager cleanup, closed-resource behavior, and checkpoint resumption.

Tests use real PGX games, the Flax network, Optax, MCTX, Orbax files, and
TensorBoard events. Controlled passing networks finish small Go games quickly.
Fixed evaluation results are used only to force promotion/error branches;
separate evaluator tests and a real end-to-end match cover actual evaluation.
Logs/checkpoints live in temporary folders and are cleaned up after use.

## Refactored API

Import public components from `Trainer`. See `Trainer/README.md` for factory,
explicit construction, component APIs, checkpoint behavior, and complete usage.

| Previous API | Current API |
|---|---|
| `TrainingConfig(batch_size=...)` | `TrainingConfig(minibatch_size=...)` |
| `EvaluationConfig(batch_size=...)` | `EvaluationConfig(num_games=...)` |
| Collector settings on every call | `SelfPlayCollector(scorch, env, config).collect()` |
| Repeated evaluator dependencies | `ScorchEvaluator(scorch, config).evaluate(candidate, opponent, key)` |
| `trainer.train_state` | `trainer.learner.state` |
| Trainer's private loss/batch helpers | Learner's update and batching methods |
| `train(key, iterations=...)` returns key | `train(iterations=...)` returns `TrainingReport` |
| `_save_checkpoint(path, key)` | `save_checkpoint(path)` |
| `load_checkpoint(path)` returns key | Restores the internally owned `trainer.key` |
| `trainer.writer.close()` | Context manager or `trainer.close()` |

The state payload and promotion rules are preserved. Callers supply matching
training/search/collection settings when restoring checkpoints. Older payloads
without an iteration field infer it from episode IDs and the current number of
environments. Reset games after restoration to use the restored environment RNG.
