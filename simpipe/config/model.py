from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "gpt"
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    seq_len: int = 4096
    vocab_size: int = 50257
    micro_batch_size: int = 1
    intermediate_size: int | None = None
    use_moe: bool = False
    num_experts: int = 8
    top_k: int = 2
    expert_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
