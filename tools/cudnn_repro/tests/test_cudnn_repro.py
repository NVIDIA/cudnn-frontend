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


def _target_tests(request):
    raw = os.environ.get("CUDNN_REPRO_TARGETS")
    if raw is None:
        raw = ",".join(f"test/python/test_mhas_v2.py::test_sdpa_random_fwd_L0[test{i}]" for i in range(1, 11))
    return [item.strip() for item in raw.split(",") if item.strip()]


def test_reproducer_json_matches(tmp_path, request):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to generate SDPA logs")

    env_base = os.environ.copy()
    env_base["CUDNN_FRONTEND_LOG_INFO"] = "1"

    for idx, target in enumerate(_target_tests(request)):
        log_a = tmp_path / f"initial_{idx}.log"
        log_b = tmp_path / f"repro_{idx}.log"

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

        repro_cmd = shlex.split(repro._build_command(cfg))
        env_second = env_base.copy()
        env_second["CUDNN_FRONTEND_LOG_FILE"] = str(log_b)
        _run(repro_cmd, env_second)

        _, repro_payload = _last_payload(log_b)
        assert payload == repro_payload, f"Payload mismatch for target {target}"


def test_build_cfg_maps_causal_without_explicit_right_bound():
    import cudnn_repro as repro

    payload = {
        "context": {"io_data_type": "FLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_FWD",
                "name": "sdpa_fwd",
                "inputs": {"Q": 1, "K": 2, "V": 3},
                "outputs": {"O": 4},
                "diagonal_alignment": "TOP_LEFT",
                "causal_mask": True,
                "left_bound": None,
                "right_bound": None,
            }
        ],
        "tensors": {
            "1": {"uid": 1, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            "2": {"uid": 2, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            "3": {"uid": 3, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            "4": {"uid": 4, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
        },
    }

    cfg = repro._build_cfg("{}", payload)
    assert cfg["left_bound"] is None
    assert cfg["right_bound"] == 0
    assert cfg["diag_align"] == 0


def test_build_cfg_preserves_logged_tensor_layout():
    import cudnn_repro as repro

    payload = {
        "context": {"io_data_type": "FLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_FWD",
                "name": "sdpa_fwd",
                "inputs": {"Q": 1, "K": 2, "V": 3},
                "outputs": {"O": 4},
                "diagonal_alignment": "TOP_LEFT",
                "left_bound": None,
                "right_bound": None,
            }
        ],
        # BSHD shape/stride: (b, s, h, d)
        "tensors": {
            "1": {"uid": 1, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            "2": {"uid": 2, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            "3": {"uid": 3, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            "4": {"uid": 4, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
        },
    }

    cfg = repro._build_cfg("{}", payload)
    assert cfg["shape_q"] == (2, 128, 4, 64)
    assert cfg["stride_q"] == (32768, 64, 8192, 1)
    assert cfg["h_q"] == 128
    assert cfg["s_q"] == 4
    assert cfg["left_bound"] is None
    assert cfg["right_bound"] is None


def test_build_command_normalizes_enum_fields():
    import cudnn_repro as repro

    cfg = {
        "data_type": "torch.float16",
        "rng_data_seed": 123,
        "batches": 1,
        "h_q": 2,
        "h_k": 2,
        "h_v": 2,
        "s_q": 16,
        "s_kv": 16,
        "d_qk": 64,
        "d_v": 64,
        "shape_q": (1, 2, 16, 64),
        "stride_q": (2048, 1024, 64, 1),
        "shape_k": (1, 2, 16, 64),
        "stride_k": (2048, 1024, 64, 1),
        "shape_v": (1, 2, 16, 64),
        "stride_v": (2048, 1024, 64, 1),
        "shape_o": (1, 2, 16, 64),
        "stride_o": (2048, 1024, 64, 1),
        "seq_len_q": [],
        "seq_len_kv": [],
        "left_bound": None,
        "right_bound": 0,
        "diag_align": 0,
        "implementation": "AUTO",
    }

    command = repro._build_command(cfg)
    assert "cudnn.diagonal_alignment.TOP_LEFT" in command
    assert "cudnn.attention_implementation.AUTO" in command
