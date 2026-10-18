# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process FROST forward artifacts must run without invoking the compiler."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state

pytestmark = pytest.mark.L0

_CHILD = r"""
import json, sys, time
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache as cc
from linear_attention.test_la import Backend, FWD_TOL, STATE_TOL, make_case, random_state, reference, pinned_op, op_args, assert_rms_close
variant, phase, result_path, test_root = sys.argv[1:]
assert Path(cudnn.__file__).resolve().is_relative_to(Path(test_root).resolve() / 'python')
dtype = torch.bfloat16
calls = []
compile_fn = cute.compile
def count_compile(*args, **kwargs):
    calls.append(getattr(args[0], '__name__', str(type(args[0]))))
    if phase == 'warm':
        raise AssertionError('a fresh process recompiled a cached forward: ' + calls[-1])
    return compile_fn(*args, **kwargs)
cute.compile = count_compile
cc.reset_stats()
started = time.perf_counter()
for length in (17, 128, 512):
    case = make_case(variant, dtype, seq_lens=[length, length // 2], H=4, seed=173 + length)
    initial = random_state(case)
    kwargs = dict(initial_state=initial, output_final_state=True)
    op = pinned_op(Backend('frost'), variant)
    args = op_args(case)
    expected = reference(case, initial_state=initial)
    output, state = op(*args, **kwargs)
    assert_rms_close('eager output', output, expected[0], FWD_TOL[dtype])
    assert_rms_close('eager state', state, expected[1], STATE_TOL[dtype])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured_output, captured_state = op(*args, **kwargs)
        captured_output.fill_(float('nan')); captured_state.fill_(float('nan'))
        graph.replay(); torch.cuda.synchronize()
        assert_rms_close('replay output', captured_output, expected[0], FWD_TOL[dtype])
        assert_rms_close('replay state', captured_state, expected[1], STATE_TOL[dtype])
    finally:
        graph.reset()
Path(result_path).write_text(json.dumps({'stats': cc.stats(), 'compile_calls': calls, 'source': cudnn.__file__, 'case_seconds': time.perf_counter() - started}))
"""


@pytest.mark.parametrize("variant", ["gdn", "gdn2", "gdp", "kda"])
def test_frost_forward_reloads_in_a_fresh_process(tmp_path, variant):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FROST linear-attention forward cache qualification requires SM100/SM103")
    if not cutedsl_state()[0]:
        pytest.skip("CuTe DSL is not installed")
    reason = cutedsl_requirement_error("FROST forward disk cache")
    if reason:
        pytest.skip(reason)
    source = Path(__file__).resolve().parents[4]
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path / "cache"), CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO="0")
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    env["PYTHONPATH"] = os.pathsep.join([str(source / "python"), str(source / "test/python"), env.get("PYTHONPATH", "")])
    results = []
    for phase in ("cold", "warm"):
        path = tmp_path / f"{phase}.json"
        process = subprocess.run(
            [sys.executable, "-c", _CHILD, variant, phase, str(path), str(source)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=600
        )
        assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-6000:]
        results.append(json.loads(path.read_text()))
    print(json.dumps({"variant": variant, "cold": results[0], "warm": results[1]}))
    assert results[0]["compile_calls"] and results[0]["stats"]["misses"] > 0, results
    assert results[1]["compile_calls"] == [] and results[1]["stats"]["hits"] > 0, results
