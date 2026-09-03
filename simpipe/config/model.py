from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


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
    hf_config_path: str | None = None
    # JSON with {"pattern": "E...L", "forward_ms": {...}, "backward_ms": {...},
    # "weight_ms": {...}} (solver fit output); overrides registry presets.
    profile_times_path: str | None = None
    flash_attention: bool = True
    # Full activation recompute: backward re-runs forward first, so each
    # checkpointed body layer costs f+b in the backward pass (emb/head excluded).
    recompute: bool = False

    def __post_init__(self) -> None:
        if self.hf_config_path:
            self._apply_hf_config(self.hf_config_path)
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def _apply_hf_config(self, path: str) -> None:
        cfg_path = Path(path).expanduser()
        with cfg_path.open() as f:
            data = json.load(f)
        if self.name == "gpt":
            self.name = data.get("model_type", self.name)
        self.hidden_size = int(data.get("hidden_size", self.hidden_size))
        self.num_layers = int(data.get("num_hidden_layers", self.num_layers))
        self.num_attention_heads = int(data.get("num_attention_heads", self.num_attention_heads))
        self.vocab_size = int(data.get("vocab_size", self.vocab_size))
        if data.get("intermediate_size") is not None:
            self.intermediate_size = int(data["intermediate_size"])
        elif data.get("moe_intermediate_size") is not None:
            self.intermediate_size = int(data["moe_intermediate_size"])
        if data.get("n_routed_experts") is not None:
            self.use_moe = True
            self.num_experts = int(data["n_routed_experts"])
        if data.get("num_experts_per_tok") is not None:
            self.top_k = int(data["num_experts_per_tok"])
