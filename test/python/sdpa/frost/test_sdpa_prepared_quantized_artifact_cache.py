# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The gate and hybrid pointer hosts retain their exported calling contract."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl, pytest.mark.L0]


@pytest.mark.parametrize("route", ["pv128", "pv192", "gate_fp8", "gate_mxfp8", "paged1", "paged4"])
def test_quantized_variant_artifact_reloads_in_fresh_process(route, tmp_path):
    cc = torch.cuda.get_device_capability()
    if (route.startswith(("pv", "paged")) and cc not in ((10, 0), (10, 3))) or (route.startswith("gate") and cc != (10, 7)):
        pytest.skip("variant needs its native architecture")
    child = r"""
import hashlib, json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
tests, package, route, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
sys.path[:0] = [tests, str(Path(tests).parents[1])]
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh-process quantized variant invoked JIT")
    cute.compile = forbidden
if route.startswith("paged"):
    import test_sdpa_prepared_fp8_paged as helper
    g, vp, ws, bufs, ts = helper.paged._run_graph_fp8(
        2, 4, 2, 128, 16, 16, [256, 256], False, out_dt=torch.bfloat16,
        s_q=16, explicit_split=int(route[5:]), return_case=True,
        v_table_layout="batch_inner", override=True,
    )
    owner = g._compiled_plans[g._plan_index]._prepared.spec.owner
    bufs["v_table"].copy_(bufs["v_table"].flip(2))
    run = lambda: g.execute(vp, ws)
    check = lambda: helper._check(bufs)
    outputs = [bufs[n] for n in ("o", "lse", "amax_o")]
elif route.startswith("pv"):
    import test_sdpa_prepared_pv_bf16 as helper
    api, bufs, ws, scales = helper._case(d=int(route[2:]))
    owner = api._dense_spec.owner
    run = lambda: api.execute(**bufs, workspace=ws)
    check = lambda: helper._check(bufs, scales)
    outputs = [bufs[n] for n in ("o_tensor", "lse_tensor", "amax_o")]
else:
    from test_sdpa_prepared_quantized_gate import _helper
    helper = _helper(route == "gate_mxfp8")
    gate = torch.full((2, 128, 4, 256), 0.7, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    g, vp, ws, bufs, ts = helper._case(d=256, arch="sm107", gate=gate)
    owner = g._compiled_plans[g._plan_index]._prepared.spec.owner
    run = lambda: g.execute(vp, ws)
    check = lambda: helper._check(bufs, thd=False)
    outputs = [bufs[n] for n in ("o", "lse", "amax_o")]
if reload == "1":
    assert hasattr(owner, "_compiled_cache_raw")
run()
check()
def digest():
    return [hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for t in outputs]
expected = digest()
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        run()
    for t in outputs:
        t.fill_(float("nan"))
    graph.replay()
    check()
    assert digest() == expected
finally:
    graph.reset()
print(json.dumps(dict(digest=expected, stats=compiled_cache.stats())))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, route, str(reload)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["stats"]["misses"] > 0 and first["stats"]["hits"] == 0, first
    assert second["stats"]["misses"] == 0 and second["stats"]["hits"] > 0, second
    assert first["digest"] == second["digest"]
