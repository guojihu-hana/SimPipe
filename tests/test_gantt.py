import re

import pytest

from simpipe.viz.gantt import _fit_column_widths, format_gantt_detailed_info, write_gantt_svg


def test_gantt_one_row_per_device(tmp_path):
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 10, "duration": 10},
        {"did": 0, "mid": 0, "sid": 0, "wtype": "B", "start": 10, "end": 20, "duration": 10},
        {"did": 1, "mid": 0, "sid": 1, "wtype": "F", "start": 5, "end": 15, "duration": 10},
    ]
    out = tmp_path / "gantt.svg"
    write_gantt_svg(records, out, title="test")
    svg = out.read_text()
    # 2 devices -> compact height (~100px), not 3 rows * 20px each for 3 records
    assert 'height="1' in svg
    assert svg.count('text-anchor="end">D0</text>') == 1
    assert svg.count('text-anchor="end">D1</text>') == 1
    assert "0,0" in svg
    assert "time:" not in svg
    assert "Device Time Statistics" not in svg


def test_gantt_expands_width_and_labels_every_workload(tmp_path):
    records = [
        {"did": 0, "mid": i, "sid": 0, "wtype": "F", "start": i, "end": i + 1, "duration": 1}
        for i in range(80)
    ]
    out = tmp_path / "wide_gantt.svg"
    write_gantt_svg(records, out, title="wide", min_workload_px=50)
    svg = out.read_text()

    assert 'width="4052"' in svg
    assert 'width="50.00"' in svg
    for i in range(80):
        assert f">{i},0</text>" in svg


def test_gantt_min_workload_px_is_minimum_visible_block_width(tmp_path):
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 1, "duration": 1},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 1, "end": 3, "duration": 2},
        {"did": 0, "mid": 2, "sid": 0, "wtype": "W", "start": 3, "end": 6, "duration": 3},
    ]
    out = tmp_path / "scaled_gantt.svg"

    write_gantt_svg(records, out, title="scaled", width=100, min_workload_px=20)

    widths = [
        float(w)
        for w in re.findall(
            # rx="2" selects workload blocks; the legend swatches use rx="1".
            r'<rect x="[0-9.]+" y="[0-9.]+" width="([0-9.]+)"[^>]*fill="#(?:E8C66A|94B8E8|8FBD8C)"[^>]*rx="2"',
            out.read_text(),
        )[:3]
    ]
    assert widths == [20.0, 40.0, 60.0]


def test_stats_table_uses_content_fitted_column_widths(tmp_path):
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 5, "end": 7, "duration": 2},
    ]
    out = tmp_path / "pipeline.svg"
    write_gantt_svg(records, out, title="pipeline", detailed=True)
    svg = out.read_text()

    widths = _fit_column_widths(
        ["Device", "Computation", "Bubble", "Warmup", "Residual", "Cooldown", "Total"],
        [["D0", "5 (50.0%)", "5 (50.0%)", "2", "0", "3", "10 (100%)"]],
    )
    assert sum(widths) < 750
    assert 'text-anchor="start">D0</text>' in svg
    assert 'text-anchor="middle" font-weight="600">Computation</text>' in svg


def test_gantt_shows_partition_and_placement_panel(tmp_path):
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
        {"did": 1, "mid": 0, "sid": 1, "wtype": "F", "start": 1, "end": 4, "duration": 3},
    ]
    out = tmp_path / "pipeline.svg"
    write_gantt_svg(
        records,
        out,
        title="pipeline",
        partition_layers=[2, 2],
        placement=[[0], [1]],
        detailed=True,
    )
    svg = out.read_text()
    assert "partition = [2, 2]" in svg
    assert "placement = [ [0]," in svg
    assert " [1]]" in svg


def test_pipeline_gantt_marks_bubble_and_device_stats(tmp_path):
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 5, "end": 7, "duration": 2},
    ]
    out = tmp_path / "pipeline.svg"

    write_gantt_svg(records, out, title="pipeline", detailed=True)

    svg = out.read_text()
    assert "#D49A96" in svg
    assert "bubbleShadow" not in svg
    assert "Device Time Statistics" in svg
    assert ">Computation</text>" in svg
    assert ">Warmup</text>" in svg
    assert ">Cooldown</text>" in svg
    assert ">D0</text>" in svg
    assert svg.count('stroke="#333" stroke-width="1"/>') >= 3


def test_gantt_detailed_info_is_markdown_table():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 5, "end": 7, "duration": 2},
    ]

    text = format_gantt_detailed_info(
        records,
        partition_layers=[2],
        placement=[[0]],
    )

    assert "## Device Time Statistics" in text
    assert "| Device | Computation | Bubble | Warmup | Residual | Cooldown | Total |" in text
    assert "| D0 | 5 (71.4%) | 2 (28.6%) | 2 | 0 | 0 | 7 (100%) |" in text
    assert "## Pipeline Layout" in text
    assert "| Partition | [2] |" in text
    assert "| Placement | [[0]] |" in text
