# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native second-process checks for prepared backward artifact export."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cudnn


def check_backward_artifact_reload(arch, route, dtype, cache_dir):
    child = r"""
import json, sys
from pathlib import Path
from unittest.mock import patch
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
tests, package, arch, route, dtype, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
sys.path[:0] = [tests, str(Path(tests).parents[1])]
if reload == "1":
    def forbidden(*a, **kw):
        raise AssertionError("prepared backward plan invoked JIT in the second process")
    cute.compile = forbidden
dtype = getattr(torch, dtype)
if arch == "sm120" and route.startswith("aux_"):
    import test_sdpa_bwd_dsl_sm120 as helper
    torch.manual_seed(903)
    b, h, hk, sq, d = 2, 4, 2, 128, 64
    q, k, v, do = [helper._bhsd(b, heads, sq, d, dtype) for heads in (h, hk, hk, h)]
    bias = torch.randn(1, h, sq, sq, device="cuda", dtype=torch.float32)
    sink = torch.randn(1, h, 1, 1, device="cuda", dtype=torch.float32)
    o, stats, dq, dk, dv, aux = helper._ref_bwd(q, k, v, do, scale=d**-0.5, is_causal=True, bias=bias, sink_token=sink)
    o = helper._bhsd(b, h, sq, d, dtype, empty=True).copy_(o)
    dbias = torch.empty_like(bias, dtype=torch.float32 if route == "aux_fp32" else dtype)
    calls = []
    original = cudnn.pygraph.execute
    def record(g, *args, **kwargs):
        calls.append((g, args))
        return original(g, *args, **kwargs)
    with patch.object(cudnn.pygraph, "execute", record):
        grads = helper._run_bwd_graph(q, k, v, o, do, stats, scale=d**-0.5, is_causal=True, bias_gpu=bias, dbias_gpu=dbias, sink_gpu=sink)
    assert grads[-1] == "sdpa_bwd_sm120"
    g, (vp, ws) = calls[-1]
    outputs = [*grads[:4], dbias]
    expected = [dq, dk, dv, aux.dsink, aux.dbias]
    def check():
        for actual, want in zip(outputs, expected):
            torch.testing.assert_close(actual.float(), want.float(), **helper._tolerances(dtype))
elif arch == "sm120":
    from test_sdpa_bwd_dsl_sm120 import _prepared_bwd_case, _check_prepared_bwd
    case = _prepared_bwd_case(dtype=dtype, route=route)
    g, vp, ws = case.graph, case.pack, case.workspace
    outputs = [case.tensors[n] for n in ("dq", "dk", "dv")]
    check = lambda: _check_prepared_bwd(case)
elif route == "dense":
    from test_sdpa_bwd_dsl_sm100 import _prepared_case, _check_prepared
    case = _prepared_case(dtype=dtype, chunks=True)
    g, vp, ws = case.graph, case.pack, case.workspace
    outputs = [case.tensors[n] for n in ("dq", "dk", "dv")]
    check = lambda: _check_prepared(case)
else:
    from test_sdpa_bwd_thd_sm100 import _run_graph, _check
    calls = []
    original = cudnn.pygraph.execute
    def record(g, *args, **kwargs):
        calls.append((g, args))
        return original(g, *args, **kwargs)
    with patch.object(cudnn.pygraph, "execute", record):
        case, *outputs = _run_graph((97, 63), (83, 79), h=4, hkv=2, dtype=dtype, stats_layout="token_major", use_causal_mask=True)
    g, (vp, ws) = calls[-1]
    check = lambda: _check(case, *outputs, hkv=2)
spec = g._compiled_plans[g._plan_index]._prepared.spec
if reload == "1":
    assert hasattr(spec.artifact.entry, "_compiled_cache_raw")
g.execute(vp, ws)
check()
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        g.execute(vp, ws)
    for t in outputs:
        t.fill_(float("nan"))
    ws.fill_(0xBD)
    graph.replay()
    check()
finally:
    graph.reset()
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(cache_dir))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, arch, route, dtype, str(reload)],
            env=env,
            cwd=cache_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["misses"] > 0 and first["hits"] == 0, first
    assert second["misses"] == 0 and second["hits"] > 0, second
