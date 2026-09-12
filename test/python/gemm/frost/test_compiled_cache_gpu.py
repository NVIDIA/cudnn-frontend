# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A frost GEMM plan compiled in one process is reloaded, not recompiled, in the
next -- and computes the same thing. Two child processes share a fresh cache
directory; the first must miss and export, the second must hit."""

import json
import os
import subprocess
import sys

import pytest

from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]

_CHILD = r"""
import hashlib, json, sys, torch, cudnn
from cudnn.frost import compiled_cache as cc
M, N, K = 256, 512, 256
torch.manual_seed(0)
a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
handle = cudnn.create_handle()
g = cudnn.pygraph(handle=handle, io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
A, B = g.tensor_like(a), g.tensor_like(b)
C = g.matmul(A=A, B=B); C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
idx = next(i for i in range(g.get_execution_plan_count()) if g.get_plan_name_at_index(i).startswith("frost_gemm"))
g.select_plan(idx); g.check_support(); g.build_plans()
c = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
g.execute({A: a, B: b, C: c}, ws, handle=handle); torch.cuda.synchronize()
launch = g._compiled_plans[idx]._compiled._launchable
print(json.dumps({"stats": cc.stats(), "digest": hashlib.sha256(c.view(torch.int16).cpu().numpy().tobytes()).hexdigest(), "reloaded": hasattr(launch, "_compiled_cache_entry"), "root": str(cc.get_cache_dir())}))
"""


def _run(cache_dir):
    env = dict(os.environ, CUDNN_FRONTEND_ENABLE_FROST_ENGINES="1", CUDNN_FRONTEND_COMPILED_CACHE=str(cache_dir))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    out = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_second_process_reloads_the_exported_kernel(tmp_path):
    first = _run(tmp_path)
    assert first["stats"]["misses"] >= 1 and first["stats"]["hits"] == 0, first
    assert first["reloaded"], "the miss path hands back the reloaded artifact so hit and miss run the same thing"
    entries = list((tmp_path / "v1").glob("*/*/entry.json"))
    assert entries, "no entry committed"
    record = json.loads(entries[0].read_text())
    assert record["schema"] == "v1" and record["symbol"] == "frost_gemm" and (entries[0].parent / "kernel.o").stat().st_size > 0
    assert json.loads((entries[0].parent.parent / "manifest.json").read_text())["schema"] == "v1"

    second = _run(tmp_path)
    assert second["stats"]["misses"] == 0 and second["stats"]["hits"] >= 1, second
    assert second["digest"] == first["digest"], "the reloaded kernel must compute exactly what the compiled one did"
