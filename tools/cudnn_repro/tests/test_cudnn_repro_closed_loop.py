import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd, env):
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def _last_payload(log_path):
    import cudnn_repro as repro

    lines = log_path.read_text().splitlines()
    entries = list(repro._iter_context_entries(lines))
    assert entries, f"No context entries found in {log_path}"
    return entries[-1]


def _target_tests():
    raw = os.environ.get("CUDNN_REPRO_TARGETS")
    if raw is None:
        raw = ",".join(
            [
                *(f"test/python/test_mhas_v2.py::test_sdpa_random_fwd_L0[test{i}]" for i in range(1, 11)),
                *(f"test/python/test_mhas_v2.py::test_sdpa_random_fwd_ragged_L0[test{i}]" for i in range(1, 11)),
                *(f"test/python/test_mhas_v2.py::test_sdpa_random_bwd_L0[test{i}]" for i in range(1, 6)),
            ]
        )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _target_test_id(target):
    return target.split("::", 1)[-1]


def _assert_reproducer_json_matches_target(tmp_path, target):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to generate SDPA logs")

    env_base = os.environ.copy()
    env_base["CUDNN_FRONTEND_LOG_INFO"] = "1"
    log_a = tmp_path / "initial.log"
    log_b = tmp_path / "repro.log"

    cmd_test = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        "-s",
        "-rA",
        target,
    ]
    env_first = env_base.copy()
    env_first["CUDNN_FRONTEND_LOG_FILE"] = str(log_a)
    _run(cmd_test, env_first)

    import cudnn_repro as repro

    raw_line, payload = _last_payload(log_a)
    cfg = repro._build_cfg(raw_line, payload)

    repro_cmd = shlex.split(repro._build_command(cfg, payload))
    env_second = env_base.copy()
    env_second["CUDNN_FRONTEND_LOG_FILE"] = str(log_b)
    _run(repro_cmd, env_second)

    _, repro_payload = _last_payload(log_b)
    assert payload == repro_payload, f"Payload mismatch for target {target}"


@pytest.mark.parametrize("target", _target_tests(), ids=_target_test_id)
def test_reproducer_json_matches(tmp_path, target):
    _assert_reproducer_json_matches_target(tmp_path, target)
