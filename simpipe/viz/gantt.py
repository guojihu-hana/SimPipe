from __future__ import annotations

from pathlib import Path

from simpipe.metrics.comp_bubble import analyze_pipeline_comp_bubble


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Workload fill colors (darkened from base palette)
_COLOR_F = "#E8C66A"  # was rgb(255, 242, 204)
_COLOR_B = "#94B8E8"  # was rgb(218, 232, 252)
_COLOR_W = "#8FBD8C"  # was rgb(213, 232, 212)
_COLOR_BUBBLE = "#D49A96"
_STROKE = "#000000"


def _color(wtype: str) -> str:
    w = wtype.lower()
    if w == "f":
        return _COLOR_F
    if w == "b":
        return _COLOR_B
    if w == "w":
        return _COLOR_W
    if w == "r":
        return "#F8CECC"
    return "#cccccc"


_TABLE_FONT_SIZE = 11.0
_TABLE_CHAR_W = 6.0
_TABLE_CHAR_W_BOLD = 6.3
_TABLE_SPACING_SCALE = 1.1
_TABLE_CELL_PAD_L = 6.0 * _TABLE_SPACING_SCALE
_TABLE_COL_GAP = 6.0 * _TABLE_SPACING_SCALE


def _estimate_text_width(text: str, *, bold: bool = False) -> float:
    factor = _TABLE_CHAR_W_BOLD if bold else _TABLE_CHAR_W
    return len(text) * factor


def _fit_column_widths(headers: list[str], rows: list[list[str]]) -> list[float]:
    """Column widths from content, with minimal horizontal padding."""
    all_rows = [headers, *rows]
    widths: list[float] = []
    for col_idx in range(len(headers)):
        max_w = max(
            _estimate_text_width(row[col_idx], bold=(row_idx == 0))
            for row_idx, row in enumerate(all_rows)
        )
        pad_r = _TABLE_COL_GAP if col_idx < len(headers) - 1 else 0.0
        widths.append(_TABLE_CELL_PAD_L + max_w + pad_r)
    return widths


def _append_three_line_table(
    parts: list[str],
    x: float,
    y: float,
    headers: list[str],
    rows: list[list[str]],
    *,
    row_h: float = 18.0,
) -> tuple[float, float]:
    """Draw an academic-style three-line table; return (height, width)."""
    col_widths = _fit_column_widths(headers, rows)
    table_w = sum(col_widths)
    body_rows = [headers, *rows]
    table_h = row_h * len(body_rows)

    parts.append(
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + table_w:.1f}" y2="{y:.1f}" '
        f'stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{x:.1f}" y1="{y + row_h:.1f}" x2="{x + table_w:.1f}" y2="{y + row_h:.1f}" '
        f'stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{x:.1f}" y1="{y + table_h:.1f}" x2="{x + table_w:.1f}" y2="{y + table_h:.1f}" '
        f'stroke="#333" stroke-width="1"/>'
    )

    col_x = x
    for row_idx, row in enumerate(body_rows):
        baseline = y + row_h * (row_idx + 0.72)
        is_header = row_idx == 0
        col_x = x
        for col_idx, cell in enumerate(row):
            col_w = col_widths[col_idx]
            if col_idx == 0:
                cell_x = col_x + _TABLE_CELL_PAD_L
                anchor = "start"
            else:
                cell_x = col_x + col_w / 2.0
                anchor = "middle"
            weight = ' font-weight="600"' if is_header else ""
            parts.append(
                f'<text x="{cell_x:.1f}" y="{baseline:.1f}" text-anchor="{anchor}"{weight}>'
                f"{_esc(cell)}</text>"
            )
            col_x += col_w

    return table_h, table_w


def _device_stats_table_rows(stats: list[dict]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Device",
        "Computation",
        "Bubble",
        "Warmup",
        "Residual",
        "Cooldown",
        "Total",
    ]
    rows: list[list[str]] = []
    for row in stats:
        rows.append(
            [
                f"D{row['did']}",
                f"{row['comp']:.0f} ({row['comp_ratio']:.1%})",
                f"{row['bubble']:.0f} ({row['bubble_ratio']:.1%})",
                f"{row['warmup_bubble']:.0f}",
                f"{row['residual_bubble']:.0f}",
                f"{row['cooldown_bubble']:.0f}",
                f"{row['total']:.0f} ({row['total_ratio']:.0%})",
            ]
        )
    return headers, rows


def _format_placement_lines(placement: list[list[int]]) -> list[str]:
    if not placement:
        return ["placement = []"]
    if len(placement) == 1:
        return [f"placement = [ {placement[0]}]"]
    lines = [f"placement = [ {placement[0]},"]
    for device_stages in placement[1:-1]:
        lines.append(f" {device_stages},")
    lines.append(f" {placement[-1]}]")
    return lines


def _partition_placement_lines(
    partition_layers: list[int],
    placement: list[list[int]],
) -> list[str]:
    return [f"partition = {partition_layers}", *_format_placement_lines(placement)]


def _append_text_block(
    parts: list[str],
    x: float,
    y: float,
    lines: list[str],
    *,
    row_h: float = 18.0,
) -> tuple[float, float]:
    """Draw plain left-aligned text lines; return (height, width)."""
    block_w = max(_estimate_text_width(line) for line in lines) if lines else 0.0
    for idx, line in enumerate(lines):
        baseline = y + row_h * (idx + 0.72)
        parts.append(
            f'<text x="{x:.1f}" y="{baseline:.1f}" text-anchor="start">'
            f"{_esc(line)}</text>"
        )
    return row_h * len(lines), block_w


def write_gantt_svg(
    records: list[dict],
    output: Path,
    title: str = "Pipeline Schedule",
    *,
    partition_layers: list[int] | None = None,
    placement: list[list[int]] | None = None,
    width: float = 1200.0,
    min_workload_px: float = 30.0,
    row_h: float = 28.0,
    label_gap: float = 4.0,
    margin_l: float = 36.0,
    margin_t: float = 40.0,
    margin_b: float = 32.0,
) -> None:
    """Render pipeline Gantt: one row per device, blocks at start/end time."""
    if not records:
        output.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>No data</text></svg>')
        return

    blocks: list[tuple[str, int, int, int, float, float]] = []
    for r in records:
        wtype = str(r.get("wtype", "F"))
        mid = int(r["mid"])
        sid = int(r.get("sid", 0))
        did = int(r["did"])
        start = float(r.get("start") or 0)
        end = float(r.get("end") or start + (r.get("duration") or 1))
        if end < start:
            start, end = end, start
        blocks.append((wtype, mid, sid, did, start, end))

    max_t = max(b[5] for b in blocks)
    min_t = 0.0
    span = max(max_t - min_t, 1.0)
    n_dev = max(b[3] for b in blocks) + 1
    blocks_by_device: dict[int, list[tuple[str, int, int, int, float, float]]] = {}
    for block in blocks:
        blocks_by_device.setdefault(block[3], []).append(block)
    for device_blocks in blocks_by_device.values():
        device_blocks.sort(key=lambda b: (b[4], b[5]))
    pad = 0.0
    base_plot_w = max(width - margin_l - 16.0, 1.0)
    positive_durations = [end - start for *_rest, start, end in blocks if end > start]
    min_duration = min(positive_durations, default=1.0)
    min_scale = (min_workload_px + 2 * pad) / min_duration
    scale = max(base_plot_w / span, min_scale)
    plot_w = span * scale
    stats = analyze_pipeline_comp_bubble(
        [
            {"did": did, "mid": mid, "sid": sid, "wtype": wtype, "start": start, "end": end}
            for wtype, mid, sid, did, start, end in blocks
        ]
    ).to_dict()["per_device"]
    headers, table_rows = _device_stats_table_rows(stats)
    stats_table_w = sum(_fit_column_widths(headers, table_rows))
    stats_row_h = 18.0
    panel_gap = 24.0 * _TABLE_SPACING_SCALE
    config_lines: list[str] = []
    config_block_w = 0.0
    config_block_h = 0.0
    if partition_layers is not None and placement is not None:
        config_lines = _partition_placement_lines(partition_layers, placement)
        config_block_w = max(_estimate_text_width(line) for line in config_lines)
        config_block_h = stats_row_h * len(config_lines)
    bottom_w = stats_table_w + (panel_gap + config_block_w if config_lines else 0.0)
    width = max(width, margin_l + plot_w + 16.0, margin_l + bottom_w + 16.0)
    table_title_h = 18.0
    stats_body_h = stats_row_h * (len(stats) + 1)
    table_body_h = max(stats_body_h, config_block_h)
    table_h = table_title_h + table_body_h + 8.0
    height = margin_t + n_dev * row_h + margin_b + table_h

    def x_pos(t: float) -> float:
        return margin_l + (t - min_t) * scale

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">',
        '<style>text{font-family:system-ui,Segoe UI,sans-serif;font-size:12px;}</style>',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        f'<text x="{margin_l:.0f}" y="22">{_esc(title)}</text>',
        f'<text x="{margin_l:.0f}" y="{height - 10:.0f}">time: ({min_t:.0f}, {max_t:.0f})</text>',
    ]

    # One lane per device
    for d in range(n_dev):
        y0 = margin_t + d * row_h
        parts.append(
            f'<text x="{margin_l - label_gap:.1f}" y="{y0 + row_h * 0.72:.1f}" '
            f'text-anchor="end">D{d}</text>',
        )
        parts.append(
            f'<line x1="{margin_l:.1f}" y1="{y0 + row_h:.1f}" x2="{width - 8:.1f}" '
            f'y2="{y0 + row_h:.1f}" stroke="#ddd"/>',
        )
        prev_end = 0.0
        for _wtype, _mid, _sid, _did, start, end in blocks_by_device.get(d, []):
            if start > prev_end:
                bx = x_pos(prev_end) + pad
                bw = max((start - prev_end) * scale - 2 * pad, 1.0)
                parts.append(
                    f'<rect x="{bx:.2f}" y="{y0 + pad:.2f}" width="{bw:.2f}" '
                    f'height="{row_h - 2 * pad:.2f}" fill="{_COLOR_BUBBLE}" '
                    f'stroke="{_STROKE}" stroke-width="1" rx="2"/>'
                )
            prev_end = end
        if max_t > prev_end:
            bx = x_pos(prev_end) + pad
            bw = max((max_t - prev_end) * scale - 2 * pad, 1.0)
            parts.append(
                f'<rect x="{bx:.2f}" y="{y0 + pad:.2f}" width="{bw:.2f}" '
                f'height="{row_h - 2 * pad:.2f}" fill="{_COLOR_BUBBLE}" '
                f'stroke="{_STROKE}" stroke-width="1" rx="2"/>'
            )

    # Blocks on device rows
    for wtype, mid, sid, did, start, end in blocks:
        y0 = margin_t + did * row_h
        x0 = x_pos(start) + pad
        wpx = max((end - start) * scale - 2 * pad, 1.0)
        col = _color(wtype)
        parts.append(
            f'<rect x="{x0:.2f}" y="{y0 + pad:.2f}" width="{wpx:.2f}" '
            f'height="{row_h - 2 * pad:.2f}" fill="{col}" stroke="{_STROKE}" '
            f'stroke-width="1" rx="2"/>',
        )
        # label = f"{wtype[0].lower()}{mid}_{sid}"
        label = f"{mid},{sid}"
        parts.append(
            f'<text x="{x0 + wpx / 2:.1f}" y="{y0 + row_h * 0.65:.1f}" '
            f'text-anchor="middle" fill="#111">{label}</text>',
        )

    # Legend: 1 row × 3 columns
    legend_y = 22
    col_w = 92
    legend_items = [
        ("F forward", _COLOR_F, ""),
        ("B backward", _COLOR_B, ""),
        ("W weight", _COLOR_W, ""),
        ("bubble", _COLOR_BUBBLE, ""),
    ]
    legend_x = width - col_w * len(legend_items) - 8
    for i, (label, col, extra) in enumerate(legend_items):
        x = legend_x + i * col_w
        parts.append(
            f'<rect x="{x:.0f}" y="{legend_y - 10:.0f}" width="12" height="12" '
            f'fill="{col}" stroke="{_STROKE}" stroke-width="1"{extra}/>'
        )
        parts.append(f'<text x="{x + 16:.0f}" y="{legend_y:.0f}" font-size="12">{label}</text>')

    table_y = margin_t + n_dev * row_h + 20.0
    table_top = table_y + 8.0
    parts.append(
        f'<text x="{margin_l:.0f}" y="{table_y:.0f}" font-weight="600">Device Time Statistics</text>'
    )
    _append_three_line_table(
        parts,
        margin_l,
        table_top,
        headers,
        table_rows,
        row_h=stats_row_h,
    )
    if config_lines:
        config_x = margin_l + stats_table_w + panel_gap
        _append_text_block(
            parts,
            config_x,
            table_top,
            config_lines,
            row_h=stats_row_h,
        )

    parts.append("</svg>")
    output.write_text("\n".join(parts))


