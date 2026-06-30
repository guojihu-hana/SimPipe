from __future__ import annotations

from simpipe.config.sim_config import SimConfig
from simpipe.models.pattern import stack_layer_count, stack_layer_symbols
from simpipe.models.profile_times import ProfileTimes, profile_times_from_preset

_NEMOTRONH_FORWARD_MS = {
    "E": 2.564743408403899,
    "M": 2.734447820687338,  # mamba
    "-": 1.6631777588932555,  # mlp
    "*": 2.342341311290737,  # attn
    "L": 12.715130125848873,
}
_NEMOTRONH_BACKWARD_MS = {
    "E": 2.695218854201464,
    "M": 6.950846335235035,  # mamba
    "-": 2.2246038983742937,  # mlp
    "*": 1.8680110522756872,  # attn
    "L": 11.092865020350409,
}
_NEMOTRONH_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

_NEMOTRON_NANO_V2_F_MS = {
    "E": 0.7589,
    "M": 2.6768,  # mamba
    "-": 1.8477,  # mlp
    "*": 1.9254,  # attn
    "L": 14.2155,
}
_NEMOTRON_NANO_V2_B_MS = {
    "E": 2.8654,
    "M": 4.2984,  # mamba
    "-": 1.9090,  # mlp
    "*": 2.4534,  # attn
    "L": 15.7056,
}
_NEMOTRON_NANO_V2_W_MS = {
    "E": 0,
    "M": 1.7599,  # mamba
    "-": 1.6866,  # mlp
    "*": 0.6547,  # attn
    "L": 0.0595,
}

_NEMOTRON_NANO_V2_PATTERN = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"

PRESETS: dict[str, dict] = {
    "nemotronh-4B": {
        "model": {
            "name": "nemotronh-4B",
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "seq_len": 4096,
            "vocab_size": 131072,
        },
        "pattern": _NEMOTRONH_PATTERN,
        "forward_ms": _NEMOTRONH_FORWARD_MS,
        "backward_ms": _NEMOTRONH_BACKWARD_MS,
    },
    "nemotronh-nano-v2-9B": {
        "model": {
            "name": "nemotronh-nano-v2-9B",
            "hidden_size": 4480,
            "num_attention_heads": 40,
            "seq_len": 4096,
            "vocab_size": 131072,
        },
        "pattern": _NEMOTRON_NANO_V2_PATTERN,
        "forward_ms": _NEMOTRON_NANO_V2_F_MS,
        "backward_ms": _NEMOTRON_NANO_V2_B_MS,
        "weight_ms": _NEMOTRON_NANO_V2_W_MS,
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


def _model_data_from_preset(data: dict) -> dict:
    model_data = dict(data.get("model", {}))
    if "pattern" in data and "num_layers" not in model_data:
        model_data["num_layers"] = stack_layer_count(data["pattern"])
    return model_data


def preset_model_data(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"Unknown model preset: {name}")
    return _model_data_from_preset(PRESETS[name])


def stack_layer_symbols_for_model(name: str, num_layers: int) -> list[str] | None:
    preset = PRESETS.get(name)
    if not preset or "pattern" not in preset:
        return None
    return stack_layer_symbols(preset["pattern"])[:num_layers]


def get_preset(name: str) -> SimConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown model preset: {name}")
    data = PRESETS[name]
    return SimConfig.from_dict(
        {
            "model": _model_data_from_preset(data),
            "schedule": data.get("schedule", "1f1b"),
        }
    )


def get_profile_times(name: str) -> ProfileTimes:
    preset = PRESETS.get(name, {})
    return profile_times_from_preset(preset)


def get_layer_times(name: str) -> tuple[list[float], list[float], list[float]]:
    profile = get_profile_times(name)
    return profile.layer_f, profile.layer_b, profile.layer_w
