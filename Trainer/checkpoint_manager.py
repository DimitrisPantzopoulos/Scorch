"""Checkpoint persistence, including the original checkpoint schema."""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import jax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState


@dataclass(slots=True)
class Checkpoint:
    state           : TrainState
    next_episode_id : int
    trainer_key     : jax.Array
    scorch_key      : jax.Array
    env_key         : jax.Array
    next_iteration  : int = 0


class CheckpointManager:
    def save(
            self,
            path       : str | Path,
            checkpoint : Checkpoint,
        ) -> None:
        directory : Path = Path(path).resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        payload : Dict[str, Any] = {
            field.name : getattr(checkpoint, field.name)
            for field in fields(checkpoint)
        }
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(directory, payload)
            checkpointer.wait_until_finished()

    def load(
            self,
            path     : str | Path,
            template : Checkpoint,
            *,
            num_envs : int,
        ) -> Checkpoint:
        directory : Path = Path(path).resolve()
        if not directory.exists():
            raise FileNotFoundError(directory)
        target : Dict[str, Any] = {
            field.name : getattr(template, field.name)
            for field in fields(template)
        }
        with ocp.StandardCheckpointer() as checkpointer:
            metadata = checkpointer.metadata(directory).item_metadata
            if metadata is None:
                raise ValueError("Checkpoint metadata is missing.")
            if "next_iteration" not in metadata.tree:
                target.pop("next_iteration")
            restored : Dict[str, Any] = checkpointer.restore(directory, target=target)
        if "next_iteration" not in restored:
            restored["next_iteration"] = int(restored["next_episode_id"]) // num_envs
        return Checkpoint(**restored)
