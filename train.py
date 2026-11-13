import jax

from MCTX.mctx import SearchConfig
from Scorch import NetworkConfig
from Trainer import (
    EvaluationConfig,
    ScorchTrainer,
    SelfPlayConfig,
    TrainingConfig,
    TrainingReport,
)


def main() -> None:
    print("Initializing JAX...", flush=True)
    print("Devices:", jax.devices(), flush=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("This training run requires an NVIDIA GPU. Install requirements-gpu.txt on the VM first.")

    run_name : str = "scorch_standard_transformer_run_01"

    with ScorchTrainer.create(
            num_envs          = 1024,
            seed              = 42,
            network_config    = NetworkConfig(
                                    d_model          = 128,
                                    num_heads        = 4,
                                    num_blocks       = 4,
                                    mlp_ratio        = 4,
                                    value_hidden_dim = 1,
                                ),
            search_config     = SearchConfig(num_simulations=32),
            training_config   = TrainingConfig(epochs=4, minibatch_size=128),
            self_play_config  = SelfPlayConfig(num_steps=256, temperature=1.0),
            evaluation_config = EvaluationConfig(num_games=32, num_steps=256),
            run_name          = run_name,
            log_dir           = f"runs/{run_name}",
            checkpoint_dir    = f"checkpoints/{run_name}",
        ) as trainer:

        report : TrainingReport = trainer.train(iterations=1000)

        print("Promotions:", report.promotions)
        print("Last evaluation:", report.last_evaluation)
        print("Latest checkpoint:", report.checkpoint_path)


if __name__ == "__main__":
    main()
