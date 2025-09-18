from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configuration import build_config


def test_build_config_defaults(tmp_path):
    cli_args = {
        "input": str(tmp_path / "input.mp4"),
        "output": str(tmp_path / "output.mp4"),
    }
    config = build_config(cli_args)
    assert config.input_path == (tmp_path / "input.mp4")
    assert config.output_path == (tmp_path / "output.mp4")
    assert config.flow_algo == "DIS"
    assert config.mode == "full"
    assert config.rd_steps == 5
    assert config.effects_profile is not None


def test_preview_flag_adjusts_defaults(tmp_path):
    cli_args = {
        "input": str(tmp_path / "input.mp4"),
        "output": str(tmp_path / "output.mp4"),
        "preview": True,
    }
    config = build_config(cli_args)
    assert config.mode == "preview"
    assert config.width == 640
    assert config.rd_steps == 1
    assert config.effects_profile is not None
