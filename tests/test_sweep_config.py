from __future__ import annotations

from pathlib import Path

import pytest

from multicam_bench.bench.sweep_config import load_sweep_config

REPO_ROOT = Path(__file__).parents[1]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(text, encoding="utf-8")
    return path


VALID_TEXT = """\
n_values: [1, 2, 4, 8, 12, 16]
codecs: [libx264, libx265]
backends: [ffmpeg-cpu, d3d11va, qsv, cuda]
repetitions: 3
cooldown_s: 60
resolution: 360p
fps: 30
duration_s: 60
rtsp_base_url: rtsp://127.0.0.1:8554
mediamtx_config: tools/mediamtx.yml
"""


def test_load_sweep_config(tmp_path: Path) -> None:
    config = load_sweep_config(_write(tmp_path, VALID_TEXT))

    assert config.n_values == [1, 2, 4, 8, 12, 16]
    assert config.codecs == ["libx264", "libx265"]
    assert config.backends == ["ffmpeg-cpu", "d3d11va", "qsv", "cuda"]
    assert config.repetitions == 3
    assert config.cooldown_s == 60
    assert config.resolution == "360p"
    assert config.fps == 30
    assert config.duration_s == 60
    assert config.rtsp_base_url == "rtsp://127.0.0.1:8554"
    assert config.mediamtx_config == Path("tools/mediamtx.yml")


def test_load_sweep_config_rejects_fewer_than_three_repetitions(tmp_path: Path) -> None:
    # THREATS-TO-VALIDITY.md T3: minimum 3 repetitions for a variance estimate.
    text = VALID_TEXT.replace("repetitions: 3", "repetitions: 2")
    with pytest.raises(ValueError, match="repetitions"):
        load_sweep_config(_write(tmp_path, text))


def test_load_sweep_config_rejects_empty_n_values(tmp_path: Path) -> None:
    text = VALID_TEXT.replace("n_values: [1, 2, 4, 8, 12, 16]", "n_values: []")
    with pytest.raises(ValueError, match="n_values"):
        load_sweep_config(_write(tmp_path, text))


def test_load_sweep_config_rejects_empty_codecs(tmp_path: Path) -> None:
    text = VALID_TEXT.replace("codecs: [libx264, libx265]", "codecs: []")
    with pytest.raises(ValueError, match="codecs"):
        load_sweep_config(_write(tmp_path, text))


def test_load_sweep_config_rejects_empty_backends(tmp_path: Path) -> None:
    text = VALID_TEXT.replace(
        "backends: [ffmpeg-cpu, d3d11va, qsv, cuda]", "backends: []"
    )
    with pytest.raises(ValueError, match="backends"):
        load_sweep_config(_write(tmp_path, text))


def test_repo_sweep_config_is_valid_and_meets_minimum_repetitions() -> None:
    config = load_sweep_config(REPO_ROOT / "configs" / "sweep.yaml")
    assert config.repetitions >= 3
    assert config.n_values == [1, 2, 4, 8, 12, 16]
    assert config.codecs == ["libx264", "libx265"]
    assert config.backends == ["ffmpeg-cpu", "d3d11va", "qsv", "cuda"]
