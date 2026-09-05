from __future__ import annotations


def _layer_comp(
    layer_f: list[float],
    layer_b: list[float],
    layer_w: list[float] | None,
    start: int,
    end: int,
) -> float:
    total = 0.0
    for i in range(start, end):
        w = layer_w[i] if layer_w is not None and i < len(layer_w) else 0.0
        total += layer_f[i] + layer_b[i] + w
    return total


def partition_variance(stage_times: list[float]) -> float:
    if not stage_times:
        return 0.0
    mean = sum(stage_times) / len(stage_times)
    return sum((t - mean) ** 2 for t in stage_times) / len(stage_times)


def stage_times_for_partition(
    layer_f: list[float],
    layer_b: list[float],
    layer_w: list[float] | None,
    partition_layers: list[int],
    *,
    embedding_f_time: float = 0.0,
    embedding_b_time: float = 0.0,
    embedding_w_time: float = 0.0,
    head_f_time: float = 0.0,
    head_b_time: float = 0.0,
    head_w_time: float = 0.0,
) -> list[float]:
    stage_times: list[float] = []
    cursor = 0
    for idx, count in enumerate(partition_layers):
        comp = _layer_comp(layer_f, layer_b, layer_w, cursor, cursor + count)
        if idx == 0:
            comp += embedding_f_time + embedding_b_time + embedding_w_time
        if idx == len(partition_layers) - 1:
            comp += head_f_time + head_b_time + head_w_time
        stage_times.append(comp)
        cursor += count
    return stage_times


def legal_chunk_values(num_layers: int, device_num: int, fixed_chunk: int | None = None) -> list[int]:
    if fixed_chunk is not None:
        stage_num = device_num * fixed_chunk
        if stage_num <= num_layers and stage_num >= device_num:
            return [fixed_chunk]
        return []
    max_chunk = num_layers // device_num
    return [c for c in range(max_chunk, 0, -1) if device_num * c <= num_layers]


def legal_stage_nums(
    num_layers: int,
    device_num: int,
    fixed_chunk: int | None = None,
    *,
    include_irregular: bool = False,
    max_stage_num: int | None = None,
    min_stage_num: int | None = None,
) -> list[int]:
    """Candidate total stage counts.

    Regular mode keeps the historical behaviour (multiples of device_num).
    Irregular mode enumerates every stage_num in [device_num, limit], so the
    per-device stage count no longer has to be uniform (cf. the nemotron-nano
    9B config with 24 stages on 4 devices placed as 7/6/6/5).
    """
    if fixed_chunk is not None:
        stage_num = device_num * fixed_chunk
        return [stage_num] if device_num <= stage_num <= num_layers else []
    limit = num_layers if max_stage_num is None else min(num_layers, max_stage_num)
    floor = device_num if min_stage_num is None else max(device_num, min_stage_num)
    if include_irregular:
        return list(range(limit, floor - 1, -1))
    max_chunk = limit // device_num
    return [device_num * c for c in range(max_chunk, 0, -1) if device_num * c >= floor]


def top_partitions_by_stage_variance(
    layer_f: list[float],
    layer_b: list[float],
    layer_w: list[float] | None,
    num_layers: int,
    stage_num: int,
    top_k: int,
    *,
    embedding_f_time: float = 0.0,
    embedding_b_time: float = 0.0,
    embedding_w_time: float = 0.0,
    head_f_time: float = 0.0,
    head_b_time: float = 0.0,
    head_w_time: float = 0.0,
) -> list[tuple[float, list[int]]]:
    """Return up to top_k contiguous layer partitions minimizing stage comp variance."""
    if stage_num <= 0 or stage_num > num_layers:
        return []

    # Prefix sums make the per-stage comp an O(1) range query.  Beam entries
    # are parent-pointer chains (count, stage_comp, parent) carrying running
    # (sum, sum-of-squares), so extending a candidate is O(1): no per-step
    # counts/times list copies, and the trim ranking uses the O(1) identity
    # var = s2/n - (s/n)^2.  Only the trim survivors ever get materialized.
    prefix = [0.0]
    for i in range(num_layers):
        w = layer_w[i] if layer_w is not None and i < len(layer_w) else 0.0
        prefix.append(prefix[-1] + layer_f[i] + layer_b[i] + w)

    # entry = (sum, sumsq, count, stage_comp, parent_entry | None)
    Entry = tuple  # recursive tuple chain
    partial: dict[tuple[int, int], list[Entry]] = {(0, 0): [(0.0, 0.0, 0, 0.0, None)]}

    emb_time = embedding_f_time + embedding_b_time + embedding_w_time
    head_time = head_f_time + head_b_time + head_w_time

    for stages_used in range(1, stage_num + 1):
        next_partial: dict[tuple[int, int], list[Entry]] = {}
        for (pos, prev_stages), entries in partial.items():
            if prev_stages != stages_used - 1:
                continue
            remaining_layers = num_layers - pos
            remaining_stages = stage_num - stages_used
            max_count = remaining_layers - remaining_stages
            base = prefix[pos]
            extra = (emb_time if stages_used == 1 else 0.0) + (
                head_time if stages_used == stage_num else 0.0
            )
            for count in range(1, max_count + 1):
                stage_comp = prefix[pos + count] - base + extra
                sq = stage_comp * stage_comp
                bucket = next_partial.setdefault((pos + count, stages_used), [])
                for entry in entries:
                    bucket.append(
                        (entry[0] + stage_comp, entry[1] + sq, count, stage_comp, entry)
                    )

        trimmed: dict[tuple[int, int], list[Entry]] = {}
        keep = max(top_k * 4, top_k)
        n = float(stages_used)
        for key, entries in next_partial.items():
            ranked = sorted(entries, key=lambda e: e[1] / n - (e[0] / n) ** 2)
            trimmed[key] = ranked[:keep]
        partial = trimmed

    def _materialize(entry: Entry) -> tuple[list[int], list[float]]:
        counts: list[int] = []
        times: list[float] = []
        while entry is not None and entry[2]:
            counts.append(entry[2])
            times.append(entry[3])
            entry = entry[4]
        counts.reverse()
        times.reverse()
        return counts, times

    finished: list[tuple[float, list[int]]] = []
    for (pos, stages_used), entries in partial.items():
        if pos != num_layers or stages_used != stage_num:
            continue
        for entry in entries:
            counts, times = _materialize(entry)
            finished.append((partition_variance(times), counts))

    finished.sort(key=lambda x: x[0])
    unique: list[tuple[float, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for variance, counts in finished:
        key = tuple(counts)
        if key in seen:
            continue
        seen.add(key)
        unique.append((variance, counts))
        if len(unique) >= top_k:
            break
    return unique


def best_partition_by_stage_variance(
    layer_f: list[float],
    layer_b: list[float],
    layer_w: list[float] | None,
    num_layers: int,
    stage_num: int,
    *,
    embedding_f_time: float = 0.0,
    embedding_b_time: float = 0.0,
    embedding_w_time: float = 0.0,
    head_f_time: float = 0.0,
    head_b_time: float = 0.0,
    head_w_time: float = 0.0,
) -> list[int]:
    """Return the contiguous layer partition with minimum stage FBW variance."""
    results = top_partitions_by_stage_variance(
        layer_f,
        layer_b,
        layer_w,
        num_layers,
        stage_num,
        top_k=1,
        embedding_f_time=embedding_f_time,
        embedding_b_time=embedding_b_time,
        embedding_w_time=embedding_w_time,
        head_f_time=head_f_time,
        head_b_time=head_b_time,
        head_w_time=head_w_time,
    )
    if results:
        return results[0][1]
    base, rem = divmod(num_layers, stage_num)
    return [base + (1 if i < rem else 0) for i in range(stage_num)]
