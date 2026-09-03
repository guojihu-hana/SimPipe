from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from simpipe.config.sim_config import SimConfig
from simpipe.models.pattern import stack_layer_count, stack_layer_symbols
from simpipe.models.profile_times import ProfileTimes, profile_times_from_preset

# Fitted per-model layer times (pattern + per-symbol f/b/w ms) live in
# profiles/<name>.json at the repository root; PRESETS below only carries
# model metadata and synthetic test fixtures.
PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

PRESETS: dict[str, dict] = {
    "nemotron-h-4B": {
        "model": {
            "name": "nemotron-h-4B",
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "seq_len": 4096,
            "vocab_size": 131072,
        },
    },
    "nemotron-nano-v2-9B": {
        "model": {
            "name": "nemotron-nano-v2-9B",
            "hidden_size": 4480,
            "num_attention_heads": 40,
            "seq_len": 4096,
            "vocab_size": 131072,
        },
    },
    "nemotron-h-47B": {
        "model": {
            "name": "nemotron-h-47B",
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "seq_len": 4096,
            "vocab_size": 131072,
        },
    },
    "test_model": {
        "model": {
            "name": "test_model",
            "hidden_size": 1024,
            "num_layers": 48,
            "num_attention_heads": 16,
            "seq_len": 4096,
            "vocab_size": 131072,
        },
        "layer_f_times": [10, 12, 11, 15, 9, 11, 14, 8] * 6,
        "layer_b_times": [15, 18, 17, 22, 12, 14, 17, 11] * 6,
        "layer_w_times": [5, 6, 5, 8, 3, 4, 4, 6] * 6,
        "embedding_f_times": 5.0,
        "embedding_b_times": 4.0,
        "embedding_w_times": 3.0,
        "head_f_times": 30.0,
        "head_b_times": 35.0,
        "head_w_times": 10.0,
    },
    "gpt-13B": {
        "model": {
            "name": "gpt-13B",
            "hidden_size": 5120,
            "num_layers": 40,
            "num_attention_heads": 40,
            "seq_len": 2048,
        },
    },
    "deepseek-16B": {
        "model": {
            "name": "deepseek-16B",
            "hidden_size": 2048,
            "num_layers": 28,
            "use_moe": True,
            "num_experts": 64,
            "top_k": 6,
        },
    },
}


@lru_cache(maxsize=None)
def profile_data(name: str) -> dict | None:
    """Fitted timing data for a model (pattern + per-symbol f/b/w ms), if any."""
    path = PROFILES_DIR / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def profiled_model_names() -> list[str]:
    """Names of all models with a profiles/<name>.json timing file."""
    return sorted(path.stem for path in PROFILES_DIR.glob("*.json"))


def _timing_data(name: str) -> dict:
    """Preset metadata merged with the profiles/ JSON timing data (JSON wins)."""
    data = dict(PRESETS.get(name, {}))
    profile = profile_data(name)
    if profile:
        data.update(profile)
    return data


def _model_data_from_preset(data: dict) -> dict:
    model_data = dict(data.get("model", {}))
    if "pattern" in data and "num_layers" not in model_data:
        model_data["num_layers"] = stack_layer_count(data["pattern"])
    return model_data


def preset_model_data(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"Unknown model preset: {name}")
    return _model_data_from_preset(_timing_data(name))


def stack_layer_symbols_for_model(name: str, num_layers: int) -> list[str] | None:
    pattern = _timing_data(name).get("pattern")
    if not pattern:
        return None
    return stack_layer_symbols(pattern)[:num_layers]


def get_preset(name: str) -> SimConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown model preset: {name}")
    data = _timing_data(name)
    return SimConfig.from_dict(
        {
            "model": _model_data_from_preset(data),
            "schedule": data.get("schedule", "1f1b"),
        }
    )


def get_profile_times(name: str) -> ProfileTimes:
    return profile_times_from_preset(_timing_data(name))


def get_layer_times(name: str) -> tuple[list[float], list[float], list[float]]:
    profile = get_profile_times(name)
    return profile.layer_f, profile.layer_b, profile.layer_w
