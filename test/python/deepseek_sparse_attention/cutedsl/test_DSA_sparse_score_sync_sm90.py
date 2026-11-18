# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 sparse score warp-role barrier and reduction synchronization."""

import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_sparse_score_recompute
from test_utils import torch_fork_set_rng


def _require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")


@pytest.mark.L1
@torch_fork_set_rng(seed=1234)
@pytest.mark.parametrize("score_type", ["attention", "indexer"])
def test_sm90_sparse_score_sync_reference(score_type):
    _require_sm90()
    from cudnn import DSA

    heads, dim = (64, 512) if score_type == "attention" else (32, 128)
    b, sq, sk, topk = 1, 4, 256, 256
    q = torch.randn(b, sq, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, sk, dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, 1, topk).expand(b, sq, topk).clone()
    lengths = torch.tensor([[129, 193, 255, 256]], device="cuda", dtype=torch.int32)
    indices.masked_fill_(torch.arange(topk, device="cuda").view(1, 1, topk) >= lengths.unsqueeze(-1), -1)
    if score_type == "attention":
        lse = torch.randn(b, sq, heads, device="cuda", dtype=torch.float32)
        scale = 1.0 / math.sqrt(dim)
        out = DSA.sparse_attn_score_recompute_wrapper(q, k, lse, indices, scale, qhead_per_kv_head=heads, topk_length=lengths)["target"]
        torch.cuda.synchronize()
        check_ref_sparse_score_recompute("attention", q, lse, indices, out, aux=k, softmax_scale=scale, topk_length=lengths)
    else:
        weights = torch.randn(b, sq, heads, device="cuda", dtype=torch.bfloat16)
        out = DSA.sparse_indexer_score_recompute_wrapper(q, k, weights, indices, qhead_per_kv_head=heads, topk_length=lengths)["predict"]
        torch.cuda.synchronize()
        check_ref_sparse_score_recompute("indexer", q, k, indices, out, aux=weights, topk_length=lengths)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("score_type", ["attention", "indexer"])
def test_sm90_sparse_score_synccheck(score_type):
    _require_sm90()
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        sanitizer = next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/compute-sanitizer"))), None)
    if sanitizer is None:
        pytest.skip("compute-sanitizer is not installed")
    test_timeout = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
    timeout = min(300.0, 0.6 * test_timeout) if test_timeout > 0 else 300.0
    child = f"import faulthandler,sys; faulthandler.dump_traceback_later({timeout * 0.9!r}); import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    cmd = [
        sanitizer,
        "--tool",
        "synccheck",
        "--report-api-errors",
        "no",
        "--error-exitcode",
        "9",
        sys.executable,
        "-u",
        "-c",
        child,
        f"{Path(__file__).resolve()}::test_sm90_sparse_score_sync_reference[{score_type}]",
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
        raise AssertionError(f"synccheck timed out after {timeout}s\n{prior}{rest}") from None
    assert proc.returncode == 0, out[-6000:]
    assert "ERROR SUMMARY: 0 errors" in out, out[-6000:]
    assert "1 passed" in out, out[-6000:]
