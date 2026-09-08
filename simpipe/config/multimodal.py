from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cached_property

from simpipe.config.model import ModelConfig

# Placement aliases understood by AuxModuleConfig.placement (an explicit
# device-id list is also accepted, giving a device-mesh style spec).
FIRST_STAGE = "first_stage"
LAST_STAGE = "last_stage"
REPLICATED = "replicated"
DEDICATED = "dedicated"
_PLACEMENT_ALIASES = (FIRST_STAGE, LAST_STAGE, REPLICATED, DEDICATED)

_MODEL_FIELDS = {f.name for f in fields(ModelConfig)}


def _default_aux_model() -> ModelConfig:
    return ModelConfig(name="mock_model", num_layers=1, pattern="T", forward_ms={"T": 1.0})


@dataclass
class AuxModuleConfig:
    """One encoder or decoder module attached to the backbone pipeline.

    An aux module *is* a model: ``model`` holds a full ModelConfig and its
    timings flow through the same pipeline as the backbone's
    (profile_times_for_model -- inline mock times, a profile_times_path
    YAML, or the registry profile looked up by model.name).  The module
    runs as one atomic workload per pass whose F/B/W durations are the
    sums over the model's layers (embedding/head included), and
    model.recompute folds the forward re-run into B exactly like the
    backbone.  ``name`` is the module label (Gantt/legend); model.name is
    which model it runs.

    Flat ``forward_ms/backward_ms/weight_ms`` scalars (module totals per
    microbatch in ms) are still accepted as a shorthand and are normalized
    into a single-layer mock model at parse time; a top-level ``recompute``
    key maps onto model.recompute.

    Encoders run before the backbone: every microbatch must finish all its
    encoders before backbone stage 0 can start its F, and each encoder's B
    waits for backbone stage 0's B (activation gradients flow back through
    the same P2P link).  Decoders mirror this after the last backbone stage:
    their F waits for the last stage's F, and the last stage's B waits for
    the decoder's B (the loss lives behind the decoders).  Multiple encoders
    (or decoders) are independent of each other unless the placement makes
    them share a device.

    placement decides where the module's compute runs:
      - "first_stage" / "last_stage": on the device holding backbone stage 0
        / the last stage (no extra P2P when the dependency edge is local).
      - "replicated": one weight copy per backbone device, microbatches
        sharded round-robin over the copies -- each microbatch runs its
        encoder F/B exactly once, on one rank (data parallelism over the
        module, up to pp_size microbatches in flight).  Pays the weight
        memory on every device but splits the compute.
      - "dedicated": one extra device appended after the backbone devices;
        data crosses to/from the backbone via P2P.
      - [device ids]: explicit mesh-style list; same round-robin microbatch
        sharding as replicated, restricted to the listed devices.

    params_gb is the parameter-tensor size in GB used for the memory
    estimate (model state is derived from it); act_gb is the activation
    footprint of one microbatch, resident from the module's F until its B
    completes, so the memory estimate charges it by the peak number of
    in-flight microbatches under the simulated schedule.  act_gb is not
    derived from recompute automatically -- with recompute on, set it to
    the checkpointed footprint yourself.
    """

    name: str = "encoder"
    model: ModelConfig = field(default_factory=_default_aux_model)
    placement: str | list[int] = FIRST_STAGE
    params_gb: float = 0.0
    act_gb: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.placement, str):
            key = self.placement.lower()
            if key not in _PLACEMENT_ALIASES:
                raise ValueError(
                    f"aux module {self.name!r}: unknown placement {self.placement!r} "
                    f"(expected one of {', '.join(_PLACEMENT_ALIASES)} or a device-id list)"
                )
            self.placement = key
        else:
            devices = list(self.placement)
            if not devices or any((not isinstance(d, int)) or d < 0 for d in devices):
                raise ValueError(
                    f"aux module {self.name!r}: placement list must hold non-negative device ids"
                )
            self.placement = devices

    @cached_property
    def _profile(self):
        """ProfileTimes of the module's model, via the shared backbone path.

        Deferred import: simpipe.models.registry imports the config package
        back (circular at module level).
        """
        from simpipe.models.registry import profile_times_for_model

        try:
            return profile_times_for_model(self.model)
        except Exception as exc:  # surface which module is broken
            raise ValueError(
                f"aux module {self.name!r} (model {self.model.name!r}): {exc}"
            ) from exc

    # Module totals for one pass in 0.01 ms ticks: layer sums plus
    # embedding/head, each rounded per layer exactly like the backbone.
    @property
    def f_ticks(self) -> float:
        pt = self._profile
        return float(sum(pt.layer_f) + (pt.embedding_f or 0.0) + (pt.head_f or 0.0))

    @property
    def b_ticks(self) -> float:
        pt = self._profile
        return float(sum(pt.layer_b) + (pt.embedding_b or 0.0) + (pt.head_b or 0.0))

    @property
    def w_ticks(self) -> float:
        pt = self._profile
        return float(sum(pt.layer_w) + (pt.embedding_w or 0.0) + (pt.head_w or 0.0))

    @classmethod
    def from_dict(cls, data: dict, *, default_name: str) -> AuxModuleConfig:
        name = data.get("name", default_name)
        model = _parse_aux_model(data, name)
        return cls(
            name=name,
            model=model,
            placement=data.get("placement", FIRST_STAGE),
            params_gb=float(data.get("params_gb", 0.0)),
            act_gb=float(data.get("act_gb", 0.0)),
        )


def _parse_aux_model(data: dict, name: str) -> ModelConfig:
    """ModelConfig from an aux entry: a model: mapping, or flat scalars.

    Flat forward_ms/backward_ms/weight_ms (module totals in ms) become a
    single-layer mock model -- an aux module is a model either way.  A
    top-level recompute maps onto model.recompute.
    """
    md = data.get("model")
    if md is not None:
        if not isinstance(md, dict):
            raise ValueError(f"aux module {name!r}: model must be a mapping")
        unknown = sorted(set(md) - _MODEL_FIELDS)
        if unknown:
            raise ValueError(
                f"aux module {name!r}: unknown model field(s) {', '.join(unknown)}"
            )
        md = dict(md)
        model_name = md.get("name", "mock_model")
        md["name"] = model_name
        if model_name == "mock_model":
            if not md.get("pattern") and md.get("layer_time") is None:
                raise ValueError(
                    f"aux module {name!r}: mock model needs a 'pattern' with "
                    f"forward_ms times (like the backbone's mock model)"
                )
            if md.get("pattern") and md.get("num_layers") is None:
                # deferred import (circular via the registry at module level)
                from simpipe.models.pattern import (expand_pattern,
                                                    stack_layer_symbols)

                md["num_layers"] = len(stack_layer_symbols(expand_pattern(str(md["pattern"]))))
        elif md.get("num_layers") is None or md.get("pattern") is None:
            # profiled model: fill num_layers/pattern etc. from the registry
            # profile, exactly like the backbone config loader does
            from simpipe.models.registry import timing_model_data

            merged = timing_model_data(model_name)
            if merged is None:
                raise ValueError(
                    f"aux module {name!r}: unknown model {model_name!r} "
                    f"(no registry preset or profiles/ timing data)"
                )
            md = {**{k: v for k, v in merged.items() if k in _MODEL_FIELDS}, **md}
    else:
        fwd = float(data.get("forward_ms", 1.0))
        bwd = data.get("backward_ms")
        wgt = data.get("weight_ms")
        tables: dict = {"forward_ms": {"T": fwd}}
        if bwd is not None:
            tables["backward_ms"] = {"T": float(bwd)}
        if wgt is not None:
            tables["weight_ms"] = {"T": float(wgt)}
        md = {"name": "mock_model", "num_layers": 1, "pattern": "T", **tables}
    if data.get("recompute"):
        md["recompute"] = True
    return ModelConfig(**md)


def aux_modules_from_config(
    items: list | None, *, role: str
) -> list[AuxModuleConfig]:
    """Parse the encoders: / decoders: YAML list (each entry a mapping)."""
    if not items:
        return []
    out: list[AuxModuleConfig] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ValueError(f"{role}s[{i}] must be a mapping, got {type(entry).__name__}")
        default = f"{role}{i}" if len(items) > 1 else role
        out.append(AuxModuleConfig.from_dict(entry, default_name=default))
    return out
