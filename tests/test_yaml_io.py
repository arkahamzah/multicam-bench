from __future__ import annotations

from pathlib import Path

import pytest

from multicam_bench.yaml_io import load_yaml_mapping, require


def test_load_yaml_mapping_strips_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"\xef\xbb\xbfkey: value\n")

    data = load_yaml_mapping(path)

    assert data == {"key": "value"}


def test_load_yaml_mapping_without_bom_still_works(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("key: value\n", encoding="utf-8")

    assert load_yaml_mapping(path) == {"key": "value"}


def test_load_yaml_mapping_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_yaml_mapping(path)


def test_load_yaml_mapping_rejects_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_yaml_mapping(path)


def test_require_returns_value_when_present() -> None:
    assert require({"a": 1}, "a", "ctx") == 1


def test_require_raises_with_context_and_key_when_missing() -> None:
    with pytest.raises(ValueError, match="ctx.*missing.*'a'"):
        require({}, "a", "ctx")
