# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper attention artifact reload across processes and runtime extents."""

import os
import subprocess
import sys

import pytest
import torch

from QSA.cutedsl.test_qsa import supported_gpu


@pytest.mark.L0
def test_attention_artifacts_reload(supported_gpu, tmp_path):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper attention test")
    script = f"""
import sys
import torch
import cutlass.cute as cute
from cudnn.frost.compiled_cache import set_cache_dir
from QSA.cutedsl.test_qsa import check_case, make_inputs
from QSA.cutedsl.test_qsa_indexed_training import check_case as check_indexed, indices_for
set_cache_dir({str(tmp_path)!r})
if sys.argv[1] == 'reload':
    def forbidden(*args, **kwargs):
        raise AssertionError('attention artifact was recompiled')
    cute.compile = forbidden
lengths = (3, 9) if sys.argv[1] == 'build' else (5, 14)
for heads in ((6, 1), (12, 1), (24, 2)):
    inputs = make_inputs(lengths, *heads, empty=True)
    check_case(inputs)
    dense_lengths = (257, 263) if sys.argv[1] == 'build' else (259, 269)
    check_case(make_inputs(dense_lengths, *heads, empty=True))
    if heads[1] == 1:
        groups = indices_for(inputs)
        indices = torch.full((sum(lengths), 32), -1, dtype=torch.int32, device="cuda")
        indices[:, :groups.shape[1]] = groups
        check_indexed(inputs, indices)
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path for path in sys.path if path))
    for mode in ("build", "reload"):
        run = subprocess.run([sys.executable, "-c", script, mode], capture_output=True, text=True, timeout=240, env=env)
        assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.L0
def test_dense_replay_rechecks_value_range(supported_gpu):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper attention test")
    from cudnn.ops import qsa_forward
    from QSA.cutedsl.test_qsa import assert_close, make_inputs, reference

    inputs = make_inputs((513,), 12, 1, all_selected=True)
    inputs["q"].zero_()
    inputs["k"].zero_()
    inputs["v"].fill_(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qsa_forward(**inputs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            result = qsa_forward(**inputs)
        for magnitude in (1e34, 1.0):
            inputs["v"].fill_(magnitude)
            for tensor in result:
                tensor.fill_(torch.nan)
            graph.replay()
            for actual, expected in zip(result, reference(inputs)):
                assert_close(actual, expected)
    finally:
        graph.reset()
