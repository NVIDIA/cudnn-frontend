# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical THD origins are independent of live-length prefixes on SM80."""

import math

import pytest
import torch
import cutlass.cute as cute
import cudnn
from frost_test_utils import requires_dsl
from test_sdpa_fwd_thd_sm80 import _make_graph, _plan

pytestmark = [
    pytest.mark.L0,
    requires_dsl,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80"),
]


@pytest.mark.parametrize("d,dv", [(48, 32), (96, 80), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_major", [False, True])
def test_independent_origins_rebind_replay(d, dv, dtype, token_major, cudnn_handle, monkeypatch, dense_o=False, single_q=False):
    torch.manual_seed(812 + d)
    h, hk, capacity = 4, 2, 160
    graph, vp, nodes, _ = _make_graph(
        (1, 0, 1) if single_q else (33, 0, 17),
        (41, 3, 23),
        h=h,
        hkv=hk,
        d=d,
        dv=dv,
        dtype=dtype,
        cu_lens=(d == 256),
        stats_layout="token_major" if token_major else "head_major",
    )
    q, k, v, o, stats = nodes
    ports = (q, k, v, o, stats)
    multipliers = (1, 2, 4, 8, 1)
    geometries = []
    for i, node in enumerate(ports):
        if node is stats:
            strides = (0, 1, h, 1) if token_major else (0, capacity, 1, 1)
            shape = (capacity, h) if token_major else (h, capacity)
        else:
            heads, width = node.get_dim()[1], node.get_dim()[3]
            head_stride = width + 8 * (i + 1)
            token_stride = heads * head_stride + 8 * (i + 2)
            strides = (7, head_stride, token_stride, 1)
            shape = (capacity, heads, width)
        node.set_stride(strides).set_ragged_offset_multiplier(multipliers[i])
        offset = node.ragged_offset
        offset.set_stride((i + 1, 1, 1, 1))
        offset.set_data_type(cudnn.data_type.INT32 if i % 2 else cudnn.data_type.INT64)
        if dense_o and i == 3:
            del vp[offset]
            node.set_ragged_offset(None)
            strides = (48 * token_stride, head_stride, token_stride, 1)
            node.set_stride(strides)
            offset = None
        geometries.append((shape, strides, offset))
    graph_lifetimes = []
    workspace = _plan(graph)

    def allocate():
        storage, views = [], []
        for i, (node, (shape, strides, offset)) in enumerate(zip(ports, geometries)):
            if node is stats:
                view_strides = (h, 1) if token_major else (capacity, 1)
            else:
                view_strides = (strides[2], strides[1], 1)
            span = 1 + sum((n - 1) * st for n, st in zip(shape, view_strides))
            raw = torch.full((span + 16,), 77.0, dtype=torch.float32 if node is stats else dtype, device="cuda")
            view = raw.as_strided(shape, view_strides, 8)
            # A graph carrier is raw storage; the declaration owns addressing.
            vp[node] = raw[8:-8]
            if offset is not None:
                vp[offset] = torch.empty(4 * (i + 1), dtype=torch.int32 if i % 2 else torch.int64, device="cuda")[:: i + 1]
            storage.append(raw)
            views.append(view)
        return storage, views

    def forbidden(*args, **kwargs):
        raise AssertionError("execute must not allocate, copy or compile")

    def launch():
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with monkeypatch.context() as patch:
                for obj, names in ((cute, ("compile",)), (torch, ("empty", "zeros")), (torch.Tensor, ("clone", "contiguous"))):
                    for name in names:
                        patch.setattr(obj, name, forbidden)
                graph.execute(vp, workspace, handle=cudnn_handle)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    for binding in range(2):
        storage, views = allocate()
        capture = torch.cuda.CUDAGraph()
        graph_lifetimes.append(capture)
        try:
            phases = (((1, 0, 1), (41, 3, 23)), ((0, 1, 1), (26, 0, 41))) if single_q else (((33, 0, 17), (41, 3, 23)), ((0, 25, 25), (26, 0, 41)))
            for phase, (ql, kl) in enumerate(phases):
                starts = [(tuple((3 + 2 * i, 48 + i, 85 - i)) if phase == 0 else tuple((6 + i, 50 - i, 91 + i))) for i in range(5)]
                if dense_o:
                    starts[3] = (0, 48, 96)
                for i, (node, raw, view, (_, strides, offsets)) in enumerate(zip(ports, storage, views, geometries)):
                    raw.fill_(77 if i >= 3 else float("nan"))
                    lengths = kl if i in (1, 2) else ql
                    if i < 3:
                        for start, count in zip(starts[i], lengths):
                            view[start : start + count].normal_()
                    physical = [start * strides[2] // multipliers[i] for start in starts[i]] + [capacity * strides[2] // multipliers[i]]
                    if offsets is not None:
                        vp[offsets].copy_(torch.tensor(physical, dtype=vp[offsets].dtype, device="cuda"))
                for node in vp:
                    name = node.get_name()
                    if name in ("seq_len_q", "seq_len_kv", "cu_seq_len_q", "cu_seq_len_kv"):
                        lengths = ql if name.endswith("_q") else kl
                        values = [0, lengths[0], sum(lengths[:2]), sum(lengths)] if name.startswith("cu_") else lengths
                        vp[node].copy_(torch.tensor(values, dtype=torch.int32, device="cuda").view_as(vp[node]))
                workspace.fill_(0xA5)
                if phase == 0:
                    # Invalid current metadata rejects before any output or workspace write.
                    offset = geometries[0][2]
                    good = vp[offset]
                    for bad in (good[:1], good.to(torch.float32)):
                        vp[offset] = bad
                        with pytest.raises(ValueError, match="ragged_q"):
                            launch()
                        assert torch.all(storage[3] == 77) and torch.all(storage[4] == 77) and torch.all(workspace == 0xA5)
                    vp[offset] = good
                    launch()
                else:
                    capture.replay()
                torch.cuda.synchronize()
                o_mask = torch.ones_like(storage[3], dtype=torch.bool)
                l_mask = torch.ones_like(storage[4], dtype=torch.bool)
                om = o_mask.as_strided(views[3].shape, views[3].stride(), 8)
                lm = l_mask.as_strided(views[4].shape, views[4].stride(), 8)
                for batch, (nq, nk) in enumerate(zip(ql, kl)):
                    qq, kk, vv = [views[i][starts[i][batch] : starts[i][batch] + (nk if i else nq)].transpose(0, 1).double() for i in range(3)]
                    scores = qq @ kk.repeat_interleave(h // hk, 0).transpose(-1, -2) / math.sqrt(d)
                    expected_o = torch.nan_to_num(scores.softmax(-1), nan=0) @ vv.repeat_interleave(h // hk, 0)
                    expected_l = scores.logsumexp(-1)
                    so, sl = starts[3][batch], starts[4][batch]
                    actual_o = views[3][so : so + nq].transpose(0, 1)
                    actual_l = views[4][sl : sl + nq].T if token_major else views[4][:, sl : sl + nq]
                    torch.testing.assert_close(actual_o.double(), expected_o, rtol=1e-2, atol=4e-3)
                    torch.testing.assert_close(actual_l.double(), expected_l, rtol=1e-3, atol=2e-3)
                    om[so : so + nq] = False
                    if token_major:
                        lm[sl : sl + nq] = False
                    else:
                        lm[:, sl : sl + nq] = False
                assert torch.all(storage[3][o_mask] == 77) and torch.all(storage[4][l_mask] == 77)
                if phase == 0:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    previous_stream = cudnn.get_stream(cudnn_handle)
                    try:
                        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
                        with torch.cuda.graph(capture, stream=stream):
                            launch()
                    finally:
                        cudnn.set_stream(cudnn_handle, previous_stream)
        finally:
            capture.reset()


@pytest.mark.parametrize("token_major", [False, True])
def test_dense_o_keeps_batch_pitch_with_ragged_inputs(token_major, cudnn_handle, monkeypatch):
    test_independent_origins_rebind_replay(96, 80, torch.bfloat16, token_major, cudnn_handle, monkeypatch, dense_o=True)
