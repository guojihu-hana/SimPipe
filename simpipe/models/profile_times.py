from __future__ import annotations

from dataclasses import dataclass
import warnings

from simpipe.models.pattern import (
    ATTN,
    HEAD,
    MAMBA,
    MLP,
    STACK_LAYER_SYMBOLS,
    layer_kind,
    normalize_timing_keys,
    pattern_tokens,
    tokenize_pattern,
)


@dataclass(frozen=True)
class ProfileTimes:
    layer_f: list[float]
    layer_b: list[float]
    layer_w: list[float]
    embedding_f: float | None = None
    embedding_b: float | None = None
    embedding_w: float | None = None
    head_f: float | None = None
    head_b: float | None = None
    head_w: float | None = None

    def with_full_recompute(self) -> "ProfileTimes":
        """Model --recompute-granularity full: b ops re-run the layer forward."""
        return ProfileTimes(
            layer_f=self.layer_f,
            layer_b=[f + b for f, b in zip(self.layer_f, self.layer_b)],
            layer_w=self.layer_w,
            embedding_f=self.embedding_f,
            embedding_b=self.embedding_b,
            embedding_w=self.embedding_w,
            head_f=self.head_f,
            head_b=self.head_b,
            head_w=self.head_w,
        )

    def slice_layers(self, num_layers: int) -> ProfileTimes:
        return ProfileTimes(
            layer_f=self.layer_f[:num_layers],
            layer_b=self.layer_b[:num_layers],
            layer_w=self.layer_w[:num_layers],
            embedding_f=self.embedding_f,
            embedding_b=self.embedding_b,
            embedding_w=self.embedding_w,
            head_f=self.head_f,
            head_b=self.head_b,
            head_w=self.head_w,
        )


def _scalar_time(data: dict, key: str) -> float | None:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return float(value)


def _lookup_ms(table: dict[str, float], token: str) -> float:
    if token not in table:
        raise KeyError(f"Missing timing for pattern token {token!r}")
    return float(table[token])


def _weight_ms(
    forward_ms: dict[str, float],
    backward_ms: dict[str, float],
    token: str,
    weight_ms: dict[str, float] | None,
) -> float:
    if weight_ms is not None and token in weight_ms:
        return float(weight_ms[token])
    return _lookup_ms(backward_ms, token)


def profile_times_from_pattern(
    *,
    pattern: str,
    forward_ms: dict[str, float],
    backward_ms: dict[str, float],
    weight_ms: dict[str, float] | None = None,
) -> ProfileTimes:
    forward_ms = normalize_timing_keys(forward_ms)
    backward_ms = normalize_timing_keys(backward_ms)
    weight_ms = normalize_timing_keys(weight_ms) if weight_ms else None

    tokens = pattern_tokens(pattern)

    layer_f: list[float] = []
    layer_b: list[float] = []
    layer_w: list[float] = []
    embedding_f = embedding_b = embedding_w = None
    head_f = head_b = head_w = None

    for token in tokens:
        if token == "E":
            embedding_f = _lookup_ms(forward_ms, "E")
            embedding_b = _lookup_ms(backward_ms, "E")
            embedding_w = _weight_ms(forward_ms, backward_ms, "E", weight_ms)
        elif token == "L":
            head_f = _lookup_ms(forward_ms, "L")
            head_b = _lookup_ms(backward_ms, "L")
            head_w = _weight_ms(forward_ms, backward_ms, "L", weight_ms)
        elif token in STACK_LAYER_SYMBOLS:
            layer_f.append(_lookup_ms(forward_ms, token))
            layer_b.append(_lookup_ms(backward_ms, token))
            layer_w.append(_weight_ms(forward_ms, backward_ms, token, weight_ms))
        else:
            raise ValueError(f"Unsupported pattern token {token!r}")

    return ProfileTimes(
        layer_f=layer_f,
        layer_b=layer_b,
        layer_w=layer_w,
        embedding_f=embedding_f,
        embedding_b=embedding_b,
        embedding_w=embedding_w,
        head_f=head_f,
        head_b=head_b,
        head_w=head_w,
    )


def profile_times_from_preset(preset: dict) -> ProfileTimes:
    if "pattern" in preset:
        return profile_times_from_pattern(
            pattern=preset["pattern"],
            forward_ms=preset["forward_ms"],
            backward_ms=preset["backward_ms"],
            weight_ms=preset.get("weight_ms"),
        )

    n = preset.get("model", {}).get("num_layers", 32)
    if "layer_f_times" in preset:
        f = preset["layer_f_times"]
        b = preset.get("layer_b_times", f)
        w = preset.get("layer_w_times", b)
    else:
        model_name = preset.get("model", {}).get("name", "<unknown>")
        warnings.warn(
            f"No profiled layer times or pattern found for {model_name}; "
            "using uniform fallback timings fwd=1.0, bwd=2.0, weight=0.0 per transformer layer "
            "and embedding/head=0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        f = [1.0] * n
        b = [2.0] * n
        w = [0.0] * n
        return ProfileTimes(
            layer_f=f,
            layer_b=b,
            layer_w=w,
            embedding_f=0.0,
            embedding_b=0.0,
            embedding_w=0.0,
            head_f=0.0,
            head_b=0.0,
            head_w=0.0,
        )
    return ProfileTimes(
        layer_f=list(f),
        layer_b=list(b),
        layer_w=list(w),
        embedding_f=_scalar_time(preset, "embedding_f_times"),
        embedding_b=_scalar_time(preset, "embedding_b_times"),
        embedding_w=_scalar_time(preset, "embedding_w_times"),
        head_f=_scalar_time(preset, "head_f_times"),
        head_b=_scalar_time(preset, "head_b_times"),
        head_w=_scalar_time(preset, "head_w_times"),
    )
