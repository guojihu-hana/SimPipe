from __future__ import annotations

from dataclasses import dataclass, field

PACK = "pack"
PAD = "pad"


@dataclass
class BatchConfig:
    """Variable-length microbatch specification.

    Two mutually exclusive ways to describe the microbatches:

    1. ``microbatches``: each entry lists the raw sequence lengths (tokens)
       grouped into one microbatch; ``mode`` selects how they share it:

       - ``pack``: sequences are concatenated without padding (varlen
         kernels, block-diagonal attention).  Linear cost ~ sum(len),
         attention quadratic cost ~ sum(len^2).
       - ``pad``: sequences are padded to the longest one in the microbatch.
         Linear cost ~ n * max_len, attention quadratic cost ~ n * max_len^2.

    2. ``time_scales``: direct per-microbatch compute-time multipliers vs the
       profiled reference microbatch (1.0 = reference).  Values in arbitrary
       units (e.g. measured times [2000, 3000, 4000]) can be normalized by
       setting ``time_ref`` (scale = value / time_ref).  The same factor
       weights activation memory (token-linear approximation), and no
       linear/quadratic split is applied.
    """

    mode: str = PACK
    microbatches: list[list[int]] = field(default_factory=list)
    time_scales: list[float] | None = None
    time_ref: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in (PACK, PAD):
            raise ValueError(f"batch.mode must be 'pack' or 'pad', got {self.mode!r}")
        if bool(self.microbatches) == bool(self.time_scales):
            raise ValueError(
                "batch requires exactly one of 'microbatches' or 'time_scales'"
            )
        if self.time_scales is not None:
            if self.time_ref <= 0:
                raise ValueError(f"batch.time_ref must be positive, got {self.time_ref}")
            for i, value in enumerate(self.time_scales):
                if float(value) <= 0:
                    raise ValueError(
                        f"batch.time_scales[{i}] must be positive, got {value}"
                    )
            return
        for i, seqs in enumerate(self.microbatches):
            if not seqs:
                raise ValueError(f"batch.microbatches[{i}] is empty")
            for length in seqs:
                if int(length) <= 0:
                    raise ValueError(
                        f"batch.microbatches[{i}] has non-positive sequence length {length}"
                    )

    @classmethod
    def from_dict(cls, data: dict) -> BatchConfig:
        time_scales = data.get("time_scales")
        return cls(
            mode=str(data.get("mode", PACK)),
            microbatches=[[int(x) for x in seqs] for seqs in data.get("microbatches", [])],
            time_scales=[float(x) for x in time_scales] if time_scales else None,
            time_ref=float(data.get("time_ref", 1.0)),
        )

    @property
    def num_microbatches(self) -> int:
        if self.time_scales is not None:
            return len(self.time_scales)
        return len(self.microbatches)

    def token_counts(self) -> list[tuple[int, int]]:
        """Per-microbatch (linear_tokens, quadratic_tokens).

        linear_tokens drives token-proportional cost (GEMMs, mamba, norms,
        activation bytes); quadratic_tokens drives attention-score cost.
        """
        if self.time_scales is not None:
            raise ValueError("token_counts requires 'microbatches' (got 'time_scales')")
        counts: list[tuple[int, int]] = []
        for seqs in self.microbatches:
            if self.mode == PACK: # pack
                lin = sum(seqs)
                quad = sum(length * length for length in seqs)
            else:  # pad
                pad_len = max(seqs)
                lin = len(seqs) * pad_len
                quad = len(seqs) * pad_len * pad_len
            counts.append((lin, quad))
        return counts

    def scales(self, ref_rows: int, ref_seq_len: int) -> list[tuple[float, float]]:
        """Per-microbatch (linear_scale, quadratic_scale) vs the profiled shape.

        The profile reference microbatch is ``ref_rows`` sequences of
        ``ref_seq_len`` tokens (model.micro_batch_size x model.seq_len), so
        its linear size is ref_rows*ref_seq_len and its quadratic size is
        ref_rows*ref_seq_len^2.

        With ``time_scales`` both components carry the same factor, so every
        workload duration (and the activation-memory weight) becomes exactly
        base_time * scale.
        """
        if self.time_scales is not None:
            return [
                (float(v) / self.time_ref, float(v) / self.time_ref)
                for v in self.time_scales
            ]
        rows = max(1, int(ref_rows))
        base = max(1, int(ref_seq_len))
        ref_lin = float(rows * base)
        ref_quad = float(rows * base * base)
        return [(lin / ref_lin, quad / ref_quad) for lin, quad in self.token_counts()]
