import json
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


def _normalize_tensor_entry(entry):
    normalized = {}
    for key in (
        "data_type",
        "dim",
        "stride",
        "is_virtual",
        "is_pass_by_value",
        "pass_by_value",
        "reordering_type",
        "ragged_offset_uid",
    ):
        if key in entry:
            normalized[key] = entry[key]
    return normalized


def _normalize_payload(payload):
    tensors = payload.get("tensors", {})

    def resolve(uid):
        return _normalize_tensor_entry(tensors[str(uid)])

    normalized = {
        "context": payload.get("context"),
        "nodes": [],
        "tensors": sorted(
            json.dumps(_normalize_tensor_entry(entry), sort_keys=True) for entry in tensors.values()
        ),
    }
    for node in payload.get("nodes", []):
        normalized_node = {}
        for key, value in node.items():
            if key == "inputs":
                normalized_node[key] = {label: resolve(uid) for label, uid in value.items()}
            elif key == "outputs":
                normalized_node[key] = {label: resolve(uid) for label, uid in value.items()}
            else:
                normalized_node[key] = value
        normalized["nodes"].append(normalized_node)
    return normalized


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
    import cudnn_repro.routing as routing

    raw_line, payload = _last_payload(log_a)
    stage1, stage2 = routing.select_stage_modules(payload)
    stage1_json = stage1.extract_and_annotate(raw_line, payload, log_a.read_text())
    seed = stage1_json.get("repro_metadata", {}).get("rng_data_seed")
    cfg = stage1.build_cfg(raw_line, stage1_json, seed)
    repro_cmd = shlex.split(stage2.build_command(cfg))
    env_second = env_base.copy()
    env_second["CUDNN_FRONTEND_LOG_FILE"] = str(log_b)
    _run(repro_cmd, env_second)

    _, repro_payload = _last_payload(log_b)
    assert _normalize_payload(payload) == _normalize_payload(repro_payload), f"Payload mismatch for target {target}"


@pytest.mark.parametrize("target", _target_tests(), ids=_target_test_id)
def test_reproducer_json_matches(tmp_path, target):
    _assert_reproducer_json_matches_target(tmp_path, target)
