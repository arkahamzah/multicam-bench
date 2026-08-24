"""Shared, BOM-safe YAML mapping loader for every `configs/*.yaml` file in this
repo.

Reads with `utf-8-sig` so a leading UTF-8 byte-order mark is stripped before
parsing — relying on a YAML parser's own BOM tolerance is not something to
depend on: PyYAML's scanner happens to strip an embedded U+FEFF character from a
decoded string on this machine's installed version, but that is incidental
parser behaviour, not a documented guarantee, and nothing here should depend on
it holding on every platform/version this project runs on (see
tests/test_config.py's regression test, which reads the real, committed
`configs/thresholds.yaml` byte-for-byte). `require()` then makes every missing
required key a loud, specific `ValueError` — a config loader must never
silently substitute a default for an absent value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read `path` as YAML and require the top-level document to be a mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}"
        )
    return raw


def require(data: dict[str, Any], key: str, context: str) -> Any:
    """`data[key]`, or a `ValueError` naming `context` and the missing key —
    never a bare `KeyError` and never a silently-substituted default.
    """
    if key not in data:
        raise ValueError(f"{context}: missing required key {key!r}")
    return data[key]
