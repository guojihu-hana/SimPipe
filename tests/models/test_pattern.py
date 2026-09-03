from simpipe.models.pattern import (
    ATTN,
    MOE,
    MAMBA,
    MLP,
    TRANSFORMER,
    encode_layer_pattern,
    layer_count,
    layer_kinds,
    normalize_timing_keys,
    normalize_timing_value,
    pattern_tokens,
    stack_layer_count,
    stack_layer_symbols,
    tokenize_pattern,
)
from simpipe.models.registry import get_preset, get_profile_times, profile_data

NEMOTRON_H_4B_PATTERN = profile_data("nemotron-h-4B")["pattern"]
NEMOTRON_NANO_V2_PATTERN = profile_data("nemotron-nano-v2-9B")["pattern"]


def test_each_pattern_character_is_one_layer():
    assert tokenize_pattern("M-M*") == [MAMBA, MLP, MAMBA, ATTN]
    assert stack_layer_symbols("M-M*") == [MAMBA, MLP, MAMBA, ATTN]
    # The nano pattern spells out no E/L, so stack symbols == all tokens.
    assert stack_layer_symbols(NEMOTRON_NANO_V2_PATTERN) == tokenize_pattern(
        NEMOTRON_NANO_V2_PATTERN
    )


def test_pattern_supports_transformer_and_moe_layers():
    assert tokenize_pattern("T#") == [TRANSFORMER, MOE]
    assert stack_layer_symbols("ET#L") == [TRANSFORMER, MOE]
    assert pattern_tokens("ET#L") == ["E", TRANSFORMER, MOE, "L"]
    assert layer_kinds("T#") == ["embedding", "transformer", "moe", "head"]
    assert normalize_timing_keys({"transformer": 1, "moe": 2}) == {
        TRANSFORMER: 1.0,
        MOE: 2.0,
    }


def test_stage_layer_pattern_strings():
    from simpipe.models.pattern import stage_layer_pattern_strings

    patterns = stage_layer_pattern_strings([1, 2, 2], ["M", "-", "*", "T", "#"])
    assert patterns == ["EM", "-*", "T#L"]


def test_encode_layer_pattern_concatenates_symbols():
    assert encode_layer_pattern([MAMBA, MLP, ATTN, TRANSFORMER, MOE]) == "M-*T#"


def test_normalize_timing_value_scales_fractional_ms():
    assert normalize_timing_value(10) == 10
    assert normalize_timing_value(10.0) == 10.0
    assert normalize_timing_value(2.564743408403899) == 256
    assert normalize_timing_value(12.715130125848873) == 1272


def test_nemotronh_pattern_layer_count():
    assert stack_layer_count(NEMOTRON_H_4B_PATTERN) == 52
    assert layer_count(NEMOTRON_H_4B_PATTERN) == 54


def test_nemotronh_stack_layer_mix():
    stack = stack_layer_symbols(NEMOTRON_H_4B_PATTERN)
    assert stack.count(MAMBA) == 24
    assert stack.count(MLP) == 24
    assert stack.count(ATTN) == 4


def test_nemotronh_profile_times_match_pattern():
    profile = get_profile_times("nemotron-h-4B")

    assert len(profile.layer_f) == 52
    assert profile.layer_f[0] == 109
    assert profile.layer_f[1] == 67
    assert profile.layer_f[7] == 51
    assert profile.embedding_f == 278
    assert profile.head_f == 964
    assert profile.embedding_w == 0.0


def test_get_preset_derives_num_layers_from_pattern():
    cfg = get_preset("nemotron-h-4B")
    assert cfg.model.num_layers == 52
    assert layer_count(NEMOTRON_H_4B_PATTERN) == 54


def test_nemotron_nano_v2_profile_times_use_weight_ms():
    profile = get_profile_times("nemotron-nano-v2-9B")

    assert stack_layer_count(NEMOTRON_NANO_V2_PATTERN) == 56
    assert len(profile.layer_w) == 56
    assert profile.layer_w[0] == 176
    assert profile.layer_w[1] == 169
    assert profile.embedding_w == 0.0
    assert profile.head_w == 6.0
    assert profile.embedding_w != 287

    cfg = get_preset("nemotron-nano-v2-9B")
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
