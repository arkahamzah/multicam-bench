from __future__ import annotations

import getpass
import json
import socket
from pathlib import Path

import pytest

from multicam_bench.bench.env import _cpu_model, collect_env


def test_collect_env_uses_the_given_machine_label() -> None:
    env = collect_env(machine_label="machine-a")
    assert env["machine_label"] == "machine-a"


def test_collect_env_default_label_is_generic_not_auto_detected() -> None:
    env = collect_env()
    assert env["machine_label"] == "unlabelled-machine"


def test_collect_env_never_leaks_real_hostname_or_username() -> None:
    # CLAUDE.md hard rule 2 / this repo is published: no hostname, no username,
    # anywhere in the payload — check against this machine's actual values so a
    # regression (e.g. someone adding platform.node() to the payload) fails here.
    hostname = socket.gethostname()
    username = getpass.getuser()
    payload = json.dumps(collect_env(machine_label="machine-a"))

    if hostname:
        assert hostname not in payload
    if username:
        assert username not in payload


def test_collect_env_has_no_absolute_path_looking_strings() -> None:
    env = collect_env(machine_label="machine-a")
    for key, value in env.items():
        if isinstance(value, str):
            assert not value.startswith("/home/"), f"{key} looks like an absolute path"
            assert not value.startswith("C:\\Users"), f"{key} looks like an absolute path"
            assert "Users\\" not in value or key == "os", f"{key} looks like a user path"


def test_collect_env_reports_expected_hardware_fields() -> None:
    env = collect_env(machine_label="machine-a")
    assert isinstance(env["cpu_cores_physical"], int) or env["cpu_cores_physical"] is None
    assert isinstance(env["cpu_cores_logical"], int)
    assert isinstance(env["ram_total_bytes"], int)
    assert isinstance(env["gpus"], list)


def test_cpu_model_prefers_platform_processor_when_nonempty() -> None:
    # On this dev machine platform.processor() is non-empty, so the /proc/cpuinfo
    # fallback path must never be consulted.
    result = _cpu_model(cpuinfo_path=Path("/does/not/exist"))
    assert result  # non-empty string from platform.processor()


def test_cpu_model_falls_back_to_proc_cpuinfo_when_processor_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("multicam_bench.bench.env.platform.processor", lambda: "")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: Test CPU Model X9\nflags\t: fpu\n",
        encoding="utf-8",
    )

    assert _cpu_model(cpuinfo_path=cpuinfo) == "Test CPU Model X9"


def test_cpu_model_returns_none_when_processor_empty_and_no_cpuinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("multicam_bench.bench.env.platform.processor", lambda: "")
    assert _cpu_model(cpuinfo_path=Path("/does/not/exist")) is None
