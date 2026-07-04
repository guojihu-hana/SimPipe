from click.testing import CliRunner

from simpipe.cli import main


def test_run_writes_detailed_info_and_hides_gantt_details_by_default(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  num_layers: 2
  hidden_size: 64
  num_attention_heads: 4
  seq_len: 32
parallel:
  pp_size: 2
  micro_batch_num: 2
schedule: 1f1b
hardware:
  gpu_hbm_gb: 1
  comm_alpha_us: 0
"""
    )
    out = tmp_path / "results"

    result = CliRunner().invoke(main, ["run", "--config", str(cfg), "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "detailed_info.md").exists()
    assert "## Device Time Statistics" in (out / "detailed_info.md").read_text()
    assert "Device Time Statistics" not in (out / "pipeline_gantt.svg").read_text()


def test_run_detailed_gantt_renders_gantt_details(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  num_layers: 2
  hidden_size: 64
  num_attention_heads: 4
  seq_len: 32
parallel:
  pp_size: 2
  micro_batch_num: 2
schedule: 1f1b
hardware:
  gpu_hbm_gb: 1
  comm_alpha_us: 0
"""
    )
    out = tmp_path / "results"

    result = CliRunner().invoke(
        main,
        ["run", "--config", str(cfg), "--output", str(out), "--detailed-gantt"],
    )

    assert result.exit_code == 0, result.output
    assert "Device Time Statistics" in (out / "pipeline_gantt.svg").read_text()
    assert "## Device Time Statistics" in (out / "detailed_info.md").read_text()
