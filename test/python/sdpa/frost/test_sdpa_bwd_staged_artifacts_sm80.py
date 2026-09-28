# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staged SM80 backward artifacts retain their ABI across interpreters."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl

pytestmark = [
    requires_dsl,
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80"),
]


@pytest.mark.parametrize("route", ["dense", "packed", "aux", "rope"])
def test_staged_backward_artifact_fresh_process(route, tmp_path):
    child = r"""
import json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.bwd import staged_sm80
tests, package, route, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
sys.path[:0] = [tests, str(Path(tests).parents[1])]
import test_sdpa_bwd_staged_sm80 as helper
artifacts = []
original = staged_sm80.compile_staged
def record(api, *args, **kwargs):
    spec = original(api, *args, **kwargs)
    artifacts.append(spec.artifact)
    if api._staged_copies is not None:
        copies, auxiliary = api._staged_copies
        artifacts.extend(entry[0] for entry in copies if entry is not None)
        artifacts.extend(single[0] for entry in copies if entry is not None for single in entry[3])
        artifacts.extend(entry[1] for _, _, entries in auxiliary for entry in entries)
    return spec
staged_sm80.compile_staged = record
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh-process staged backward invoked JIT")
    cute.compile = forbidden
if route == "dense":
    helper.test_dense_staged_rebind_and_replay(torch.bfloat16, 160, 112, 1, False, True)
elif route == "aux":
    helper.test_dense_staged_rebind_and_replay(torch.bfloat16, 96, 80, 2, True, False)
elif route == "packed":
    helper.test_packed_staged_preserves_output_tail(torch.bfloat16, 160, 112, 2, "token_major")
else:
    helper.test_rope_staged_replay(torch.bfloat16, 128)
assert artifacts
if reload == "1":
    assert all(hasattr(a, "_compiled_cache_raw") for a in artifacts)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path), CUDNN_FRONTEND_ENABLE_FROST_ENGINES="1")
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
    assert first["misses"] > 0 and first["hits"] == 0, first
    assert second["misses"] == 0 and second["hits"] > 0, second


@pytest.mark.parametrize("route", ["dense", "packed", "aux", "rope"])
@pytest.mark.parametrize("cache_mode", ["disabled", "unknown_manifest"])
def test_staged_backward_replan_without_disk_cache(route, cache_mode, tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache
    from cudnn.sdpa.bwd.kernels.sm80 import prepared_host
    from sdpa.frost import test_sdpa_bwd_staged_sm80 as helper

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    if cache_mode == "disabled":
        monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    else:
        monkeypatch.delenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", raising=False)
        manifest = dict(compiled_cache.environment_manifest(), test_unknown="unknown")
        monkeypatch.setattr(compiled_cache, "environment_manifest", lambda: manifest)
    from cudnn.sdpa.fwd.kernels.sm80.staged_copy import compile_gather
    from cudnn.sdpa.fwd.kernels.staged_copy import compile_copy
    from cudnn.sdpa.bwd.kernels.sm80.staged_copy import compile_cast

    for cached in (compile_gather, compile_copy, compile_cast):
        cached.cache_clear()
    prepared_host._compile_thd_artifact.cache_clear()
    memo = getattr(prepared_host, "_compile_staged_artifact", None)
    if memo is not None:
        memo.cache_clear()

    def run():
        if route == "dense":
            helper.test_dense_staged_rebind_and_replay(torch.bfloat16, 160, 112, 1, False, True)
        elif route == "aux":
            helper.test_dense_staged_rebind_and_replay(torch.bfloat16, 96, 80, 2, True, False)
        elif route == "packed":
            helper.test_packed_staged_preserves_output_tail(torch.bfloat16, 160, 112, 2, "token_major")
        else:
            helper.test_rope_staged_replay(torch.bfloat16, 128)

    run()

    def forbidden(*args, **kwargs):
        pytest.fail("staged backward replan invoked JIT without disk persistence")

    monkeypatch.setattr(cute, "compile", forbidden)
    run()
