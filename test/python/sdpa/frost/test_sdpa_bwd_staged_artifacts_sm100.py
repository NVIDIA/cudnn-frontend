# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent artifact ownership for the large-head dense conversion chain."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cudnn
import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_staged_artifact_fresh_process(dtype, tmp_path):
    child = r"""
import json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.bwd import prepared_sm100
tests, package, dtype, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
sys.path[:0] = [tests, str(Path(tests).parents[1])]
if reload == "1":
    def forbidden(*a, **kw):
        raise AssertionError("staged backward invoked JIT while reloading its artifact")
    cute.compile = forbidden
owners = []
original = prepared_sm100.compile_staged
def record(*args):
    result = original(*args)
    owners.append(result.spec.artifact.entry)
    owners.extend(entry[0] for entry in result.launches if entry is not None)
    owners.extend(single[0] for entry in result.launches if entry is not None for single in entry[3])
    return result
prepared_sm100.compile_staged = record
from test_sdpa_bwd_staged_sm100 import _case, _check
case = _case(dtype=getattr(torch, dtype))
assert owners
if reload == "1":
    assert all(hasattr(entry, "_compiled_cache_raw") for entry in owners)
case.graph.execute(case.pack, case.workspace)
_check(case)
capture = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(capture):
        case.graph.execute(case.pack, case.workspace)
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    case.workspace.fill_(0xBD)
    capture.replay()
    _check(case)
finally:
    capture.reset()
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, dtype, str(reload)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["misses"] > 0 and first["hits"] == 0, first
    assert second["misses"] == 0 and second["hits"] > 0, second
