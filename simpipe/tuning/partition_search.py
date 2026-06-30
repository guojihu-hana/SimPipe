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

    partial: dict[tuple[int, int], list[tuple[list[int], list[float]]]] = {
        (0, 0): [([], [])]
    }

    for stages_used in range(1, stage_num + 1):
        next_partial: dict[tuple[int, int], list[tuple[list[int], list[float]]]] = {}
        for (pos, prev_stages), entries in partial.items():
            if prev_stages != stages_used - 1:
                continue
            remaining_layers = num_layers - pos
            remaining_stages = stage_num - stages_used
            for counts, times in entries:
                max_count = remaining_layers - remaining_stages
                for count in range(1, max_count + 1):
                    start = pos
                    end = pos + count
                    stage_comp = _layer_comp(layer_f, layer_b, layer_w, start, end)
                    if stages_used == 1:
                        stage_comp += embedding_f_time + embedding_b_time + embedding_w_time
                    if stages_used == stage_num:
                        stage_comp += head_f_time + head_b_time + head_w_time
                    key = (end, stages_used)
                    next_partial.setdefault(key, []).append(
                        (counts + [count], times + [stage_comp])
                    )

        trimmed: dict[tuple[int, int], list[tuple[list[int], list[float]]]] = {}
        keep = max(top_k * 4, top_k)
        for key, entries in next_partial.items():
            ranked = sorted(entries, key=lambda item: partition_variance(item[1]))
            trimmed[key] = ranked[:keep]
        partial = trimmed

    finished: list[tuple[float, list[int]]] = []
    for (pos, stages_used), entries in partial.items():
        if pos != num_layers or stages_used != stage_num:
            continue
        for counts, times in entries:
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
