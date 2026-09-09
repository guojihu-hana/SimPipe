from __future__ import annotations

from dataclasses import dataclass, field

from simpipe.config.multimodal import (DEDICATED, FIRST_STAGE, LAST_STAGE,
                                       REPLICATED, AuxModuleConfig)

ENCODER = "encoder"
DECODER = "decoder"


@dataclass(frozen=True)
class AuxInstance:
    """One physical copy of an encoder/decoder on one device.

    Stage ids continue after the backbone's (aux_sid >= backbone stage_num),
    so aux workloads reuse the plain Workload/constraint machinery without
    colliding with backbone stages.  Multi-copy placements (replicated /
    explicit device list) shard microbatches round-robin over the copies:
    each microbatch runs on exactly one copy (data parallelism over the
    module), and mids holds this copy's shard.
    """

    aux_sid: int
    name: str
    role: str  # ENCODER | DECODER
    module_index: int  # index into config.encoders / config.decoders
    device_id: int  # may be >= backbone device_num (dedicated devices)
    mids: tuple[int, ...]  # local microbatch ids (0..nmb-1) this copy runs
    f_ticks: float
    b_ticks: float
    w_ticks: float
    params_gb: float
    act_gb: float = 0.0


@dataclass
class AuxPlan:
    instances: list[AuxInstance] = field(default_factory=list)
    # dedicated devices appended after the backbone's (device ids
    # device_num .. device_num + extra_device_num - 1)
    extra_device_num: int = 0
    # Memory-driven throttle: an encoder copy may keep at most this many
    # microbatch activations alive (F started, B not finished).  None = no
    # limit (encoders run as early as idle slots allow).  build_simulation
    # searches the smallest value that keeps the makespan unchanged, so the
    # schedule holds encoder activations low at zero bubble cost.
    encoder_inflight_limit: int | None = None

    @property
    def encoders(self) -> list[AuxInstance]:
        return [i for i in self.instances if i.role == ENCODER]

    @property
    def decoders(self) -> list[AuxInstance]:
        return [i for i in self.instances if i.role == DECODER]

    def instances_on_device(self, did: int) -> list[AuxInstance]:
        return [i for i in self.instances if i.device_id == did]


def _did_of_stage(device_stages: list[list[int]], sid: int) -> int:
    for did, sids in enumerate(device_stages):
        if sid in sids:
            return did
    raise ValueError(f"stage {sid} not found in placement {device_stages}")


def build_aux_plan(
    encoders: list[AuxModuleConfig],
    decoders: list[AuxModuleConfig],
    device_stages: list[list[int]],
    stage_num: int,
    micro_batch_num: int,
) -> AuxPlan | None:
    """Expand encoder/decoder configs into per-device instances.

    Placement resolution (see AuxModuleConfig): first_stage/last_stage pin
    the copy to the boundary stage's device, dedicated appends a fresh
    device, and replicated / an explicit device-id list put one weight copy
    per device and shard microbatches round-robin over the copies (each
    microbatch runs on exactly one copy -- data parallelism over the module;
    replicated is shorthand for "every backbone device").
    """
    if not encoders and not decoders:
        return None
    device_num = len(device_stages)
    plan = AuxPlan()
    next_sid = stage_num
    next_extra_did = device_num
    all_mids = tuple(range(micro_batch_num))

    def add(cfg: AuxModuleConfig, role: str, module_index: int) -> None:
        nonlocal next_sid, next_extra_did

        def instance(did: int, mids: tuple[int, ...], tag: str = "") -> None:
            nonlocal next_sid
            plan.instances.append(
                AuxInstance(
                    aux_sid=next_sid,
                    name=cfg.name + tag,
                    role=role,
                    module_index=module_index,
                    device_id=did,
                    mids=mids,
                    f_ticks=cfg.f_ticks,
                    b_ticks=cfg.b_ticks,
                    w_ticks=cfg.w_ticks,
                    params_gb=cfg.params_gb,
                    act_gb=cfg.act_gb,
                )
            )
            next_sid += 1

        def shard(devices: list[int]) -> None:
            for slot, did in enumerate(devices):
                if did >= device_num:
                    raise ValueError(
                        f"aux module {cfg.name!r}: device {did} out of range "
                        f"(backbone has {device_num} devices; use 'dedicated' "
                        f"for extra devices)"
                    )
                mids = tuple(m for m in all_mids if m % len(devices) == slot)
                if mids:
                    instance(did, mids, tag=f"@d{did}" if len(devices) > 1 else "")

        placement = cfg.placement
        if placement == FIRST_STAGE:
            instance(_did_of_stage(device_stages, 0), all_mids)
        elif placement == LAST_STAGE:
            instance(_did_of_stage(device_stages, stage_num - 1), all_mids)
        elif placement == REPLICATED:
            shard(list(range(device_num)))
        elif placement == DEDICATED:
            instance(next_extra_did, all_mids)
            next_extra_did += 1
        else:  # explicit device-id list
            shard(list(placement))

    for idx, cfg in enumerate(encoders):
        add(cfg, ENCODER, idx)
    for idx, cfg in enumerate(decoders):
        add(cfg, DECODER, idx)
    plan.extra_device_num = next_extra_did - device_num
    return plan
