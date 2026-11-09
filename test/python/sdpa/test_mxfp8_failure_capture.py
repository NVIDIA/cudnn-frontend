# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure evidence contracts; synthetic failures do not qualify SDPA numerics."""

import ast
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0
_ROOT = Path(__file__).resolve().parents[1]


def _capture_module():
    assert importlib.util.find_spec("sdpa.mxfp8_failure_capture") is not None, "MXFP8 failure capture must ship with the harness"
    return importlib.import_module("sdpa.mxfp8_failure_capture")


class _Graph:
    _plan_index = 2

    def get_plan_name_at_index(self, index):
        assert index == 2
        return "synthetic-backward-plan"

    def get_engine_and_knobs_at_index(self, index):
        return (7, {"tile": 64})

    def serialize(self, **kwargs):
        return b"synthetic-plan-serialization"


def exec_sdpa_mxfp8(device="cpu"):
    # Match the harness frame/locals, with genuine tensor data and fake plans.
    cfg = SimpleNamespace(serialize=lambda: {"dO_type": "torch.float8_e5m2", "rng_data_seed": 955})
    graph_fwd, graph_bwd = _Graph(), _Graph()
    dO_f32 = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    dV_ref = dO_f32.double()
    variant_pack_bwd = {
        uid: torch.arange(256, dtype=torch.uint8, device=device).reshape(16, 16).view(dtype)
        for uid, dtype in enumerate((torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu), 1)
    }
    variant_pack_bwd[4] = dO_f32.to(torch.bfloat16)[:, ::2, :]
    raise AssertionError("synthetic dV accuracy failure")


def _failure(device="cpu"):
    try:
        exec_sdpa_mxfp8(device)
    except AssertionError as error:
        return error
    raise AssertionError("the synthetic detector must fail")


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_capture_preserves_raw_codes_strides_references_and_both_plans(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA payload control requires a GPU")
    capture = _capture_module()
    monkeypatch.setenv("MXFP8_CAPTURE_DIR", str(tmp_path))
    error = _failure(device)
    capture.capture_mxfp8_failure("first-node", error)
    capture.capture_mxfp8_failure("second-node", _failure(device))
    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1, "retain only the first failed case per process"
    bundle = torch.load(paths[0], map_location="cpu", weights_only=True)
    assert bundle["nodeid"] == "first-node"
    assert bundle["failure"] == str(error)
    assert bundle["cfg"]["dO_type"] == "torch.float8_e5m2"
    for uid, dtype in enumerate((torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu), 1):
        item = bundle["variant_pack_bwd"][uid]
        assert item["dtype"] == str(dtype)
        assert item["shape"] == [16, 16] and item["stride"] == [16, 1]
        assert torch.equal(item["bytes"].reshape(-1), torch.arange(256, dtype=torch.uint8))
    padded = bundle["variant_pack_bwd"][4]
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(torch.bfloat16)[:, ::2, :]
    assert padded["shape"] == [2, 2, 4] and padded["stride"] == [12, 8, 1]
    assert torch.equal(padded["bytes"], expected.contiguous().view(torch.uint8))
    assert torch.equal(bundle["reference_tensors"]["dV_ref"]["bytes"], torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).view(torch.uint8))
    for graph in ("graph_fwd", "graph_bwd"):
        assert bundle["graphs"][graph]["index"] == 2
        assert bundle["graphs"][graph]["plan"] == "synthetic-backward-plan"
        assert bundle["graphs"][graph]["serialized"] == b"synthetic-plan-serialization"
    assert not list(tmp_path.glob("*.tmp"))


def test_disabled_capture_does_not_inspect_a_failure(monkeypatch):
    capture = _capture_module()
    monkeypatch.delenv("MXFP8_CAPTURE_DIR", raising=False)

    class NoInspection:
        @property
        def __traceback__(self):
            pytest.fail("disabled capture inspected the failure")

    capture.capture_mxfp8_failure("disabled", NoInspection())


def test_unrelated_failure_creates_no_bundle(tmp_path, monkeypatch):
    capture = _capture_module()
    monkeypatch.setenv("MXFP8_CAPTURE_DIR", str(tmp_path / "unused"))
    try:
        raise AssertionError("not an MXFP8 harness failure")
    except AssertionError as error:
        capture.capture_mxfp8_failure("other-node", error)
    assert not (tmp_path / "unused").exists()


def test_failed_write_leaves_no_partial_bundle_and_can_retry(tmp_path, monkeypatch, capsys):
    capture = _capture_module()
    monkeypatch.setenv("MXFP8_CAPTURE_DIR", str(tmp_path))
    save = torch.save

    def broken_save(bundle, path):
        save({"partial": True}, path)
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(torch, "save", broken_save)
    capture.capture_mxfp8_failure("write-failure", _failure())
    assert "original test failure retained" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(torch, "save", save)
    capture.capture_mxfp8_failure("retry", _failure())
    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    assert torch.load(paths[0], weights_only=True)["nodeid"] == "retry"


def test_cleanup_error_cannot_replace_the_original_failure(tmp_path, monkeypatch, capsys):
    capture = _capture_module()
    monkeypatch.setenv("MXFP8_CAPTURE_DIR", str(tmp_path))

    def broken_save(*args, **kwargs):
        raise OSError("synthetic save failure")

    def broken_unlink(*args, **kwargs):
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(torch, "save", broken_save)
    monkeypatch.setattr(Path, "unlink", broken_unlink)
    capture.capture_mxfp8_failure("cleanup-failure", _failure())
    assert "original test failure retained" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ("success", "save_error", "cleanup_error"))
def test_broken_stderr_cannot_replace_the_original_failure(tmp_path, monkeypatch, mode):
    capture = _capture_module()
    monkeypatch.setenv("MXFP8_CAPTURE_DIR", str(tmp_path))

    class BrokenStderr:
        def write(self, text):
            raise OSError("synthetic stderr failure")

    def broken_io(*args, **kwargs):
        raise OSError("synthetic capture IO failure")

    error = _failure()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", BrokenStderr())
        if mode != "success":
            patch.setattr(torch, "save", broken_io)
        if mode == "cleanup_error":
            patch.setattr(Path, "unlink", broken_io)
        capture.capture_mxfp8_failure("broken-stderr", error)
    assert str(error) == "synthetic dV accuracy failure"


def test_missing_plan_metadata_does_not_lose_the_tensor_evidence(tmp_path, monkeypatch):
    capture = _capture_module()
    monkeypatch.setenv("MXFP8_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delattr(_Graph, "_plan_index")
    capture.capture_mxfp8_failure("missing-plan-metadata", _failure())
    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    bundle = torch.load(paths[0], weights_only=True)
    assert "unavailable" in bundle["graphs"]["graph_bwd"]["index"]
    assert torch.equal(bundle["variant_pack_bwd"][1]["bytes"].reshape(-1), torch.arange(256, dtype=torch.uint8))


def _child(command, cwd, env):
    try:
        return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"capture child timed out\nstdout: {error.stdout}\nstderr: {error.stderr}")


def test_import_keeps_torch_and_cudnn_lazy(tmp_path):
    _capture_module()
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    result = _child(
        [sys.executable, "-c", "import sys; import sdpa.mxfp8_failure_capture; assert 'torch' not in sys.modules; assert 'cudnn' not in sys.modules"],
        tmp_path,
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ("failure", "pass", "disabled", "write_error"))
def test_report_hook_keeps_original_pytest_result(tmp_path, mode):
    # Use the actual conftest hook without importing its CUDA session setup.
    tree = ast.parse((_ROOT / "conftest.py").read_text())
    hook = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pytest_runtest_makereport")
    assert "capture_mxfp8_failure" in ast.unparse(hook), "ordinary pytest runs must wire in opt-in capture"
    (tmp_path / "conftest.py").write_text("import os, pytest\n" + ast.unparse(hook) + "\n")
    source = Path(__file__).read_text()
    synthetic = ast.parse(source)
    definitions = [node for node in synthetic.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("_Graph", "exec_sdpa_mxfp8")]
    (tmp_path / "test_synthetic.py").write_text(
        "import faulthandler; faulthandler.dump_traceback_later(45, exit=True)\n"
        "import os, torch\nfrom types import SimpleNamespace\n"
        + "\n".join(ast.unparse(node) for node in definitions)
        + "\ndef test_synthetic():\n    if os.environ['CAPTURE_MODE'] != 'pass':\n        exec_sdpa_mxfp8()\n"
    )
    capture_dir = tmp_path / "bundles"
    if mode == "write_error":
        capture_dir.write_text("not a directory")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTEST_XDIST_") and key not in ("CUDNN_TEST_RUN_ID", "MXFP8_CAPTURE_DIR", "PYTEST_ADDOPTS")
    }
    env.update(PYTHONPATH=str(_ROOT), CAPTURE_MODE=mode)
    if mode != "disabled":
        env["MXFP8_CAPTURE_DIR"] = str(capture_dir)
    result = _child([sys.executable, "-m", "pytest", "-q", "-c", "/dev/null", str(tmp_path / "test_synthetic.py")], tmp_path, env)
    assert result.returncode == (0 if mode == "pass" else 1), result.stdout + result.stderr
    if mode != "pass":
        assert "synthetic dV accuracy failure" in result.stdout
    if mode == "failure":
        assert len(list(capture_dir.glob("*.pt"))) == 1
    elif mode == "write_error":
        assert "original test failure retained" in result.stdout + result.stderr
    else:
        assert not capture_dir.exists()
