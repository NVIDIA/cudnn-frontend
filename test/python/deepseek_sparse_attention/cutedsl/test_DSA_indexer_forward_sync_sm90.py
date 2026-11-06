# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check warp-role shared-memory publication in the SM90 indexer forward."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("tool", ["racecheck", "synccheck"])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("test_DSA_indexer_forward_wrapper_qh16_causal_block_boundary", id="bf16"),
        pytest.param("test_DSA_indexer_forward_wrapper_fp8_sm90_thd_matches_dequant_reference", id="fp8-thd"),
        pytest.param("test_indexer_forward_sm90_fp8_k_scale_reuse[32]", id="fp8-reuse-h32"),
        pytest.param("test_indexer_forward_sm90_fp8_k_scale_reuse[64]", id="fp8-reuse-h64"),
    ],
)
def test_indexer_forward_sm90_sanitizer(case, tool):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    toolkit_sanitizer = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "compute-sanitizer"
    sanitizer = str(toolkit_sanitizer) if toolkit_sanitizer.is_file() and os.access(toolkit_sanitizer, os.X_OK) else shutil.which("compute-sanitizer")
    if sanitizer is None:
        sanitizer = next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/compute-sanitizer"))), None)
    if sanitizer is None:
        pytest.skip("compute-sanitizer is not installed")
    test_timeout = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
    timeout = min(300.0, 0.6 * test_timeout) if test_timeout > 0 else 300.0
    child = f"import faulthandler,sys; faulthandler.dump_traceback_later({timeout * 0.9!r}); import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    target = (
        Path(__file__).resolve()
        if case.startswith("test_indexer_forward_sm90_fp8_k_scale_reuse")
        else Path(__file__).with_name("test_DSA_indexer_forward.py").resolve()
    )
    cmd = [
        sanitizer,
        "--tool",
        tool,
        "--report-api-errors",
        "no",
        "--error-exitcode",
        "9",
        "--print-limit",
        "10",
        sys.executable,
        "-u",
        "-c",
        child,
        f"{target}::{case}",
        "-n0",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        rest, _ = proc.communicate()
        prior = exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else (exc.output or "")
        raise AssertionError(f"{tool} timed out after {timeout}s\n{prior}{rest}") from None
    assert proc.returncode == 0, out[-8000:]
    assert "1 passed" in out, out[-8000:]
    summary = "ERROR SUMMARY: 0 errors" if tool == "synccheck" else "0 errors, 0 warnings"
    assert summary in out, out[-8000:]


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("heads", [32, 64])
def test_indexer_forward_sm90_fp8_k_scale_reuse(heads):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from cudnn import DSA
    from deepseek_sparse_attention.cutedsl.dsa_reference import ref_indexer_forward
    from deepseek_sparse_attention.cutedsl.dsa_utils import quantize_fp8_1x128

    torch.manual_seed(111 + heads)
    batch, seq_q, seq_k, dim, ratio = 1, 8, 259, 128, 4
    q_ref = torch.randn(batch, seq_q, heads, dim, dtype=torch.bfloat16, device="cuda")
    k_ref = torch.randn(batch, seq_k, 1, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(batch, seq_q, heads, dtype=torch.bfloat16, device="cuda") * 0.1
    q, q_scale = quantize_fp8_1x128(q_ref)
    k, k_scale = quantize_fp8_1x128(k_ref)
    offsets = torch.tensor([seq_k * ratio - seq_q], device="cuda", dtype=torch.int32)
    result = DSA.indexer_forward_wrapper(
        q,
        k,
        weights,
        ratio=ratio,
        qhead_per_kv_head=heads,
        q_causal_offsets=offsets,
        precision="fp8",
        q_scale=q_scale,
        k_scale=k_scale,
        return_lse=True,
    )
    expected = ref_indexer_forward(
        q.float() * q_scale.unsqueeze(-1),
        k.float() * k_scale.unsqueeze(-1),
        weights,
        ratio,
        q_causal_offsets=offsets,
    )
    torch.testing.assert_close(result["scores"], expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(result["lse"], torch.logsumexp(expected, dim=-1), atol=5e-3, rtol=5e-3)
