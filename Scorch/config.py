"""Immutable architecture settings shared by the network and agent."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NetworkConfig:
    d_model          : int   = 128
    num_heads        : int   = 4
    num_blocks       : int   = 4
    mlp_ratio        : int   = 4
    value_hidden_dim : int   = 1
    embedding_stddev : float = 0.02

    def __post_init__(self) -> None:
        for name in ("d_model", "num_heads", "mlp_ratio", "value_hidden_dim"):
            value : int = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if (isinstance(self.num_blocks, bool)
                or not isinstance(self.num_blocks, int)
                or self.num_blocks < 0):
            raise ValueError("num_blocks must be a nonnegative integer.")

        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")

        if (isinstance(self.embedding_stddev, bool)
                or not isinstance(self.embedding_stddev, (int, float))
                or not math.isfinite(self.embedding_stddev)
                or self.embedding_stddev < 0):
            raise ValueError("embedding_stddev must be finite and nonnegative.")
