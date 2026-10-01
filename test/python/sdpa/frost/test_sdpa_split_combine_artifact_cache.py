# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Normally imported split helpers must persist every calling convention."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cudnn
import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("route", ["dense", "ragged32", "ragged64", "quantized", "packed"])
def test_shared_combine_artifact_reloads_in_fresh_process(route, tmp_path):
    child = r"""
import hashlib, json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
tests, package, route, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
sys.path[:0] = [tests, str(Path(tests).parents[1])]
import test_sdpa_split_combine_sm100 as helper
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh-process shared combine invoked JIT")
    cute.compile = forbidden
packed = route == "packed"
b, h, sq, d, splits = 1 if packed else 2, 3, 5, 160, 3
op, lp, ref_o, ref_lse = helper._partials(b, h, sq, d, splits)
ragged = route.startswith("ragged")
if ragged:
    ref_o = torch.cat((ref_o[0, :3], ref_o[1, :2]), 0).unsqueeze(0)
    ref_lse = torch.cat((ref_lse[0, :, :3], ref_lse[1, :, :2]), 1).unsqueeze(0)
out_b = 1 if ragged else b
ostride, lstride = helper._strides(out_b, h, sq, d, "strided")
o, os_, ou = helper._output((out_b, sq, h, d), ostride, torch.float16)
lse, ls, lu = helper._output((out_b, h, sq), lstride, torch.float32)
quantized = route == "quantized"
amax = torch.zeros(1, device="cuda") if quantized else None
scale = torch.tensor([1.75], device="cuda") if quantized else None
owner = comb.compile_ptr(dtype_o="f16", has_lse=True, ragged=ragged, ragged_i64=route == "ragged64", quantized=quantized,
                         has_amax=quantized, has_scale_o=quantized, has_scale_o_input=quantized, packed=packed)
fn = compiled_cache.positional_entry(owner)
extra = ()
if ragged:
    offsets = torch.tensor([0,3,5], device="cuda", dtype=torch.int64 if route == "ragged64" else torch.int32)
    extra = (offsets.data_ptr(), offsets.data_ptr(), offsets.data_ptr(), (1,1,1), (sq,sq))
elif quantized:
    extra = (amax.data_ptr(), scale.data_ptr())
elif packed:
    total = torch.tensor([sq], device="cuda", dtype=torch.int32)
    extra = (total.data_ptr(),)
def run():
    if amax is not None: amax.zero_()
    fn(op.data_ptr(),lp.data_ptr(),o.data_ptr(),lse.data_ptr(),(b,h,sq,d),splits,ostride,lstride,*extra,torch.cuda.current_stream().cuda_stream)
if reload == "1":
    assert hasattr(owner, "_compiled_cache_raw")
def check():
    helper._check(o,lse,os_,ou,ls,lu,ref_o * (1.75 if quantized else 1.0),ref_lse,False)
    if quantized: torch.testing.assert_close(amax.cpu(), ref_o.abs().amax().float().view(1), atol=2e-6, rtol=2e-6)
run();check()
graph=torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):run()
    op.mul_(.5);ref_o.mul_(.5)
    o.fill_(float("nan"));lse.fill_(float("nan"))
    graph.replay();check()
finally:graph.reset()
digest=[hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for x in (o,lse)]
print(json.dumps(dict(digest=digest,stats=compiled_cache.stats())))
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
