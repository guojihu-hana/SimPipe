from __future__ import annotations

import heapq


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

    # entry = (sumsq, count, stage_comp, parent_entry | None).  Every chain
    # reaching layer position `pos` covers exactly layers [0, pos), so the
    # running sum equals prefix[pos] plus the emb/head extras for every entry
    # in a bucket: ranking by variance within a bucket is ranking by the sum
    # of squares alone, and the running sum need not be carried at all.
    Entry = tuple  # recursive tuple chain
    prev: dict[int, list[Entry]] = {0: [(0.0, 0, 0.0, None)]}

    emb_time = embedding_f_time + embedding_b_time + embedding_w_time
    head_time = head_f_time + head_b_time + head_w_time

    keep = max(top_k * 4, top_k)
    for stages_used in range(1, stage_num + 1):
        # One bounded max-heap per new position.  Iterating count upward
        # makes the last stage's comp (hence sq) non-decreasing while parent
        # buckets stay ascending, so three prunes apply: stop the position
        # once sq alone exceeds the kept worst (later counts only get worse),
        # skip a parent bucket whose best entry loses, and stop within a
        # bucket at the first losing entry.
        #
        # Historically candidates were generated ascending by pos (== count
        # descending here) and trimmed with a stable sort on variance, and
        # which candidate survives a tie is visible in the final result.  The
        # heap key (-sumsq, count, -k) reproduces that order exactly, which
        # is why the prunes above must use strict comparisons: a tie on
        # sumsq is settled by the (count, -k) part, not dropped early.
        nxt: dict[int, list[Entry]] = {}
        remaining_stages = stage_num - stages_used
        extra = (emb_time if stages_used == 1 else 0.0) + (
            head_time if stages_used == stage_num else 0.0
        )
        for new_pos in range(stages_used, num_layers - remaining_stages + 1):
            bucket: list = []
            full = False
            for count in range(1, new_pos - stages_used + 2):
                entries = prev.get(new_pos - count)
                if entries is None:
                    continue
                stage_comp = prefix[new_pos] - prefix[new_pos - count] + extra
                sq = stage_comp * stage_comp
                if full:
                    worst_kept = -bucket[0][0]
                    if sq > worst_kept:
                        break
                    if sq + entries[0][0] > worst_kept:
                        continue
                for k, entry in enumerate(entries):
                    total_sq = entry[0] + sq
                    item = (-total_sq, count, -k, (total_sq, count, stage_comp, entry))
                    if full:
                        if item > bucket[0]:
                            heapq.heapreplace(bucket, item)
                        else:
                            break  # entries ascend: the rest only get worse
                    else:
                        heapq.heappush(bucket, item)
                        full = len(bucket) >= keep
            if bucket:
                # descending heap items == the legacy stable-sorted order
                # (sumsq ascending, generation order breaking ties); the next
                # round relies on entries[0] being each bucket's best
                nxt[new_pos] = [item[3] for item in sorted(bucket, reverse=True)]
        prev = nxt

    def _materialize(entry: Entry) -> tuple[list[int], list[float]]:
        counts: list[int] = []
        times: list[float] = []
        while entry is not None and entry[1]:
            counts.append(entry[1])
            times.append(entry[2])
            entry = entry[3]
        counts.reverse()
        times.reverse()
        return counts, times

    finished: list[tuple[float, list[int]]] = []
    for entry in prev.get(num_layers, []):
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
