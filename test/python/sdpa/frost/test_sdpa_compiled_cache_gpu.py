# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A frost SDPA forward plan compiled in one process is reloaded, not recompiled,
in the next -- and writes the same O. The template's key is its file + params
digest joined with the compile() call's arguments (compiled_cache.template_key)."""

import json
import os
import subprocess
import sys

import pytest

from cudnn.frost import compiled_cache as cc

from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_blackwell, requires_dsl]

_CHILD = r"""
import hashlib, json, math, sys, torch, cudnn, cudnn.sdpa
from cudnn.frost import compiled_cache as cc
sys.path.insert(0, %(tests)r)
from frost_test_utils import select_engine, _SM
from cudnn.sdpa.fwd.engines import engine_name
_ARCH = "sm107" if _SM == 107 else "sm100"
b, h, s, d = 2, 8, 256, 128
torch.manual_seed(0)
mk = lambda: torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
q_gpu, k_gpu, v_gpu = mk(), mk(), mk()
o_gpu = torch.empty(b, s, h, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
q, k, v = g.tensor_like(q_gpu), g.tensor_like(k_gpu), g.tensor_like(v_gpu)
o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=1.0 / math.sqrt(d), use_causal_mask=True)
o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
select_engine(g, engine_name(arch=_ARCH)); g.check_support(); g.build_plans()
ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws); torch.cuda.synchronize()
print(json.dumps({"stats": cc.stats(), "digest": hashlib.sha256(o_gpu.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()}))
"""


def _run(cache_dir):
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(cache_dir))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    child = _CHILD % {"tests": os.path.dirname(os.path.abspath(__file__))}
    out = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_second_process_reloads_the_sdpa_forward_kernel(tmp_path):
    first = _run(tmp_path)
    assert first["stats"]["misses"] >= 1 and first["stats"]["hits"] == 0, first
    entries = list((tmp_path / cc._SCHEMA).glob("*/*/entry.json"))
    assert entries, "no entry committed"
    record = json.loads(entries[0].read_text())
    assert record["symbol"] == "frost_sdpa_fwd" and "|compile|" in record["key"], record
    assert "('b', 2)" in record["key"] and "('qh', 8)" in record["key"], "the compile() arguments are part of the key"

    second = _run(tmp_path)
    assert second["stats"]["misses"] == 0 and second["stats"]["hits"] >= 1, second
    assert second["digest"] == first["digest"], "the reloaded kernel must write exactly what the compiled one did"
