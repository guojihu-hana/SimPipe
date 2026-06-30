from simpipe.models.pattern import (
    ATTN,
    MAMBA,
    MLP,
    encode_layer_pattern,
    layer_count,
    normalize_timing_value,
    stack_layer_count,
    stack_layer_symbols,
    tokenize_pattern,
)
from simpipe.models.registry import _NEMOTRONH_PATTERN, get_preset, get_profile_times


def test_each_pattern_character_is_one_layer():
    assert tokenize_pattern("M-M*") == [MAMBA, MLP, MAMBA, ATTN]
    assert stack_layer_symbols("M-M*") == [MAMBA, MLP, MAMBA, ATTN]
    assert stack_layer_symbols(_NEMOTRONH_PATTERN) == tokenize_pattern(_NEMOTRONH_PATTERN)


def test_stage_layer_pattern_strings():
    from simpipe.models.pattern import stage_layer_pattern_strings

    patterns = stage_layer_pattern_strings([1, 2, 1], ["M", "-", "*"])
    assert patterns == ["EM", "-*", "L"]


def test_encode_layer_pattern_concatenates_symbols():
    assert encode_layer_pattern([MAMBA, MLP, ATTN]) == "M-*"


def test_normalize_timing_value_scales_fractional_ms():
    assert normalize_timing_value(10) == 10
    assert normalize_timing_value(10.0) == 10.0
    assert normalize_timing_value(2.564743408403899) == 256
    assert normalize_timing_value(12.715130125848873) == 1272


def test_nemotronh_pattern_layer_count():
    assert stack_layer_count(_NEMOTRONH_PATTERN) == 52
    assert layer_count(_NEMOTRONH_PATTERN) == 54


def test_nemotronh_stack_layer_mix():
    stack = stack_layer_symbols(_NEMOTRONH_PATTERN)
    assert stack.count(MAMBA) == 24
    assert stack.count(MLP) == 24
    assert stack.count(ATTN) == 4


def test_nemotronh_profile_times_match_pattern():
    profile = get_profile_times("nemotronh-4B")

    assert len(profile.layer_f) == 52
    assert profile.layer_f[0] == 273
    assert profile.layer_f[1] == 166
    assert profile.layer_f[7] == 234
    assert profile.embedding_f == 256
    assert profile.head_f == 1272
    assert profile.embedding_w == 270


def test_get_preset_derives_num_layers_from_pattern():
    cfg = get_preset("nemotronh-4B")
    assert cfg.model.num_layers == 52
    assert layer_count(_NEMOTRONH_PATTERN) == 54


def test_nemotron_nano_v2_profile_times_use_weight_ms():
    from simpipe.models.registry import (
        _NEMOTRON_NANO_V2_PATTERN,
        get_preset,
        get_profile_times,
    )

    profile = get_profile_times("nemotronh-nano-v2-9B")

    assert stack_layer_count(_NEMOTRON_NANO_V2_PATTERN) == 56
    assert len(profile.layer_w) == 56
    assert profile.layer_w[0] == 176
    assert profile.layer_w[1] == 169
    assert profile.embedding_w == 0.0
    assert profile.head_w == 6.0
    assert profile.embedding_w != 287

    cfg = get_preset("nemotronh-nano-v2-9B")
    assert cfg.model.num_layers == 56


def test_test_model_profile_times_use_weight_ms():
    profile = get_profile_times("test_model")

    assert len(profile.layer_f) == 48
    assert profile.layer_f[:3] == [10.0, 12.0, 11.0]
    assert profile.layer_w[:3] == [5.0, 6.0, 5.0]
    assert profile.embedding_f == 5.0
    assert profile.embedding_w == 3.0
    assert profile.head_f == 30.0
    assert profile.head_w == 10.0
