# Scorch training API

Import public training components from `Trainer`. The factory constructs the
environment and Scorch agent, initializes independent seeded random streams,
and binds collection, learning, evaluation, logging, and checkpoints.

```python
from MCTX.mctx import SearchConfig
from Trainer import (
    EvaluationConfig,
    ScorchTrainer,
    SelfPlayConfig,
    TrainingConfig,
    TrainingReport,
)

with ScorchTrainer.create(
        num_envs          = 256,
        seed              = 42,
        search_config     = SearchConfig(num_simulations=32),
        training_config   = TrainingConfig(epochs=4, minibatch_size=128),
        self_play_config  = SelfPlayConfig(num_steps=256, temperature=1.0),
        evaluation_config = EvaluationConfig(num_games=32, num_steps=256),
        run_name          = "go_run_01",
        checkpoint_dir    = "checkpoints/go_run_01",
    ) as trainer:
    report : TrainingReport = trainer.train(iterations=100)

    print(report.promotions)
    print(report.last_evaluation)
    print(report.checkpoint_path)
```

`num_envs` counts parallel self-play games, `minibatch_size` controls optimizer
batches, and `num_games` counts balanced evaluation games. Configure these
components when constructing the trainer; `train()` only needs an iteration
count. Passing `evaluation_config=None` enables training without evaluation or
promotion gating.

Training reports contain iterations completed in that call, promotions,
the accepted optimizer step, the next iteration number, the latest evaluation
performed in the call, and the latest automatic checkpoint path.

The trainer owns its RNG key. Save and restore checkpoints without passing or
receiving keys:

```python
trainer.save_checkpoint("checkpoints/manual_snapshot")
trainer.load_checkpoint("checkpoints/manual_snapshot")
report = trainer.train(iterations=10)
```

Restoration includes model parameters, optimizer state, counters, and trainer,
agent, and environment RNG keys. Use the same model, training, search, and
collection configuration when resuming. Existing checkpoint payloads remain
compatible; older checkpoints without an iteration counter infer it from the
episode counter and current environment batch size. Live boards are not saved;
collection resets games using the restored environment RNG.

Automatic checkpoints use unique `iteration_000001` folders. Existing checkpoint
destinations are never overwritten. The context manager closes TensorBoard
resources even if training raises; explicit users can call `trainer.close()`.
Training a closed trainer raises `RuntimeError`.

## Components

| Module | Responsibility |
|---|---|
| `config.py` | Immutable collection, learning, and evaluation settings |
| `data.py` | Training examples, evaluation results, reports, array aliases |
| `self_play_collector.py` | Bound game collection and pure JAX rollout helpers |
| `scorch_evaluator.py` | Balanced, per-game candidate/opponent evaluation |
| `scorch_learner.py` | Optimizer state, losses, minibatches, and epochs |
| `checkpoint_manager.py` | Orbax persistence and checkpoint compatibility |
| `tensorboard_logger.py` | Metrics, event files, and logger lifecycle |
| `scorch_trainer.py` | Factory and collect/learn/evaluate/promote orchestration |

Explicit dependency construction remains available:

```python
with ScorchTrainer(
        scorch,
        env,
        training_config,
        self_play_config  = SelfPlayConfig(num_steps=256),
        evaluation_config = EvaluationConfig(num_games=32),
        seed              = 42,
        run_name          = "custom_run",
    ) as trainer:
    report = trainer.train(iterations=10)
```

Advanced users can use `SelfPlayCollector(scorch, env, config).collect()`,
`ScorchLearner(scorch, config).train_batch(examples)`, and
`ScorchEvaluator(scorch, config).evaluate(candidate_params, opponent_params, key)`
independently. Learners reject empty direct training batches. Low-level JAX
helpers keep state and RNG arguments explicit; filesystem I/O stays outside JIT.

Promotion rules are unchanged: all evaluation games must finish and the score
must exceed 50%, counting draws as half a win. Rejection restores both model
and optimizer state while retaining consumed sampling randomness and episode
IDs. TensorBoard continues to use `self_play/`, `train/`, and `evaluation/` tags.
