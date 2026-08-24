from __future__ import annotations

from pathlib import Path

import pytest

from multicam_bench.config import load_thresholds

REPO_ROOT = Path(__file__).parents[1]


def test_load_thresholds(tmp_path: Path) -> None:
    text = """\
saturation:
  lag_p95_ms: 200
  drop_rate_pct: 1.0
publisher_drift_reject_ms: 50
warmup_s: 20
measure_s: 90
"""
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")

    thresholds = load_thresholds(path)

    assert thresholds.lag_p95_ms == 200
    assert thresholds.drop_rate_pct == 1.0
    assert thresholds.publisher_drift_reject_ms == 50
    assert thresholds.warmup_s == 20
    assert thresholds.measure_s == 90


def test_repo_thresholds_match_pre_registered_values() -> None:
    # Guards THREATS-TO-VALIDITY.md §1.1: these are fixed before data collection
    # and must not silently change. configs/thresholds.yaml is never written by code.
    thresholds = load_thresholds(REPO_ROOT / "configs" / "thresholds.yaml")

    assert thresholds.lag_p95_ms == 200
    assert thresholds.drop_rate_pct == 1.0
    assert thresholds.publisher_drift_reject_ms == 50
    assert thresholds.warmup_s == 20
    assert thresholds.measure_s == 90


def test_repo_thresholds_file_has_no_byte_order_mark() -> None:
    # Regression test: configs/thresholds.yaml was committed with a UTF-8 BOM
    # (b"\xef\xbb\xbf") from the very first commit. yaml.safe_load happened to
    # tolerate it on the environment that wrote test_repo_thresholds_match_
    # pre_registered_values above (PyYAML strips an embedded U+FEFF from a
    # decoded string), so that test passed and the bug went uncaught for two
    # milestones. This test reads the real committed file as raw bytes — the one
    # thing the value-based test above could never catch — so a BOM regression
    # fails loudly here even if some future YAML parser is BOM-tolerant too.
    raw = (REPO_ROOT / "configs" / "thresholds.yaml").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), (
        "configs/thresholds.yaml has a UTF-8 byte-order mark — strip it "
        "(the threshold values are unaffected, only the leading 3 bytes)"
    )


def test_load_thresholds_tolerates_a_byte_order_mark(tmp_path: Path) -> None:
    # load_thresholds must not *rely* on the YAML parser's incidental BOM
    # tolerance — it strips the BOM itself (utf-8-sig) before parsing.
    text = (
        "saturation:\n"
        "  lag_p95_ms: 200\n"
        "  drop_rate_pct: 1.0\n"
        "publisher_drift_reject_ms: 50\n"
        "warmup_s: 20\n"
        "measure_s: 90\n"
    )
    path = tmp_path / "thresholds_bom.yaml"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))

    thresholds = load_thresholds(path)

    assert thresholds.lag_p95_ms == 200
    assert thresholds.warmup_s == 20


def test_load_thresholds_missing_saturation_key_fails_loudly(tmp_path: Path) -> None:
    text = """\
publisher_drift_reject_ms: 50
warmup_s: 20
measure_s: 90
"""
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="saturation"):
        load_thresholds(path)


def test_load_thresholds_missing_nested_key_fails_loudly(tmp_path: Path) -> None:
    text = """\
saturation:
  lag_p95_ms: 200
publisher_drift_reject_ms: 50
warmup_s: 20
measure_s: 90
"""
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="drop_rate_pct"):
        load_thresholds(path)


def test_load_thresholds_missing_top_level_key_fails_loudly(tmp_path: Path) -> None:
    text = """\
saturation:
  lag_p95_ms: 200
  drop_rate_pct: 1.0
warmup_s: 20
measure_s: 90
"""
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="publisher_drift_reject_ms"):
        load_thresholds(path)


def test_load_thresholds_saturation_wrong_type_fails_loudly(tmp_path: Path) -> None:
    text = """\
saturation: not_a_mapping
publisher_drift_reject_ms: 50
warmup_s: 20
measure_s: 90
"""
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_thresholds(path)


def test_load_thresholds_rejects_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_thresholds(path)
