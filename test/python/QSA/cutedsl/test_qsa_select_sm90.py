# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper selector: head groups, native strides and changed-metadata replay."""

import os
import subprocess
import sys

import pytest
import torch

from cudnn.ops import qsa_select_forward
from cudnn.frost.buffers import cutedsl_requirement_error
from gated_attention_block.cutedsl.test_qsa_select import _assert_selection

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper selector test")
    reason = cutedsl_requirement_error("QSA selector")
    if reason:
        pytest.skip(reason)


def _reference(q, k, cu, pos, weights, scale):
    result = torch.full((q.shape[0], k.shape[1]), -torch.inf, dtype=torch.float64, device=q.device)
    prefix, origins = cu.tolist(), pos.tolist()
    for batch, (first, last) in enumerate(zip(prefix, prefix[1:])):
        if first == last:
            continue
        score = torch.einsum("thd,nd->thn", q[first:last].double(), k[batch].double()).relu()
        if weights is not None:
            score *= weights[first:last, :, None].double()
        score = score.sum(1) * scale
        limit = (origins[batch] + torch.arange(last - first, device=q.device) + 1) // 4
        result[first:last] = score.masked_fill(
            torch.arange(k.shape[1], device=q.device)[None] >= limit[:, None],
            -torch.inf,
        )
    return result


@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_weighted_heads_and_strided_outputs(heads, dtype):
    torch.manual_seed(47)
    q = torch.randn(42, heads, 131, device="cuda").to(dtype)[..., :128]
    k = torch.randn(3, 17, 128, device="cuda").to(dtype)
    cu = torch.tensor([0, 5, 5, 42], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 0, 19], dtype=torch.int32, device="cuda")
    weights = torch.randn(
        42,
        heads,
        device="cuda",
        dtype=torch.float32 if dtype == torch.bfloat16 else torch.bfloat16,
    )
    ids = torch.full((42, 10), -9, dtype=torch.int32, device="cuda")[:, :7]
    scores = torch.full((42, 10), torch.nan, device="cuda")[:, :7]
    result = qsa_select_forward(
        q,
        k,
        cu,
        top_k=7,
        q_pos0=pos,
        w=weights,
        scale=-0.125,
        block_ids_out=ids,
        scores_out=scores,
    )
    assert result["block_ids"].data_ptr() == ids.data_ptr()
    assert result["scores"].data_ptr() == scores.data_ptr()
    _assert_selection(ids, scores, _reference(q, k, cu, pos, weights, -0.125), 7)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_selector_replay_uses_current_metadata_without_allocation(dtype, monkeypatch):
    import cutlass.cute as cute

    torch.manual_seed(51)
    q = torch.randn(22, 4, 128, device="cuda").to(dtype)
    k = torch.randn(2, 17, 128, device="cuda").to(dtype)
    cu = torch.tensor([0, 5, 22], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 11], dtype=torch.int32, device="cuda")
    # A slab per sequence permits new lengths and origins under the same capture.
    offsets = torch.tensor([0, 22 * 17, 2 * 22 * 17], dtype=torch.int64, device="cuda")
    cand = torch.empty(2 * 22 * 17, device="cuda")
    ids = torch.empty((22, 7), dtype=torch.int32, device="cuda")
    scores = torch.empty((22, 7), device="cuda")
    kwargs = dict(
        top_k=7,
        q_pos0=pos,
        max_seqlen_q=22,
        block_ids_out=ids,
        scores_out=scores,
        cand_buffer=cand,
        cand_batch_offsets=offsets,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qsa_select_forward(q, k, cu, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            old_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                qsa_select_forward(q, k, cu, **kwargs)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            finally:
                torch.cuda.set_sync_debug_mode(old_mode)
        for boundary, origin in ((9, 7), (5, 11)):
            cu[1] = boundary
            pos[1] = origin
            ids.fill_(-9)
            scores.fill_(torch.nan)
            graph.replay()
            _assert_selection(ids, scores, _reference(q, k, cu, pos, None, 128**-0.5), 7)
        monkeypatch.setattr(
            cute,
            "compile",
            lambda *a, **kw: pytest.fail("runtime extents caused compilation"),
        )
        shorter_cu = torch.tensor([0, 9, 18], dtype=torch.int32, device="cuda")
        result = qsa_select_forward(
            q[:18],
            k,
            shorter_cu,
            top_k=7,
            q_pos0=pos,
        )
        _assert_selection(result["block_ids"], result["scores"], _reference(q[:18], k, shorter_cu, pos, None, 128**-0.5), 7)
    finally:
        graph.reset()


def test_selector_artifacts_reload_in_a_fresh_process(tmp_path):
    script = f"""
import sys
import torch
import cutlass.cute as cute
from cudnn.frost.compiled_cache import set_cache_dir
from QSA.cutedsl.test_qsa_select_sm90 import test_weighted_heads_and_strided_outputs
set_cache_dir({str(tmp_path)!r})
if sys.argv[1] == 'reload':
    def forbidden(*args, **kwargs):
        raise AssertionError('selector artifact was recompiled')
    cute.compile = forbidden
for dtype in (torch.bfloat16, torch.float8_e4m3fn):
    test_weighted_heads_and_strided_outputs(4, dtype)
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path for path in sys.path if path))
    for mode in ("build", "reload"):
        run = subprocess.run([sys.executable, "-c", script, mode], capture_output=True, text=True, timeout=180, env=env)
        assert run.returncode == 0, run.stdout + run.stderr
