# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed SDPA metadata follows the actual batch, including loaded plans.

The no-launch controls exercise the native validator with owned CPU backing.
The execution tests use real native descriptors and analytic attention outputs.
"""

import math

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import is_backend_engine

pytestmark = pytest.mark.L0


def _metadata_pack(bits=32, extra=1, stride=1, rules=None, enabled=True):
    native = cudnn._pybind_module
    layout = native.DeclaredLayout(1)
    layout.set_backend_bound(0, 7, [4096 + extra, 1, 1, 1], [stride, 1, 1, 1], bits, enabled, rules or [(1, 4096, extra)])
    backing = torch.empty(65536, dtype=torch.uint8)
    pack = native.VariantPackNative(1)
    pack.describe_from(layout, [])

    def bind(size):
        pack.set_operand(0, backing.data_ptr(), [size], [1], 1, 8, observed_bytes=size)

    return pack, bind


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("extra", [0, 1])
@pytest.mark.parametrize("stride", [1, 8])
@pytest.mark.parametrize("override_stride", [[], [1, 1, 1, 1], [100, 1, 1, 1]])
def test_metadata_capacity_covers_actual_batch_even_if_override_understates_rows(bits, extra, stride, override_stride):
    pack, bind = _metadata_pack(bits, extra, stride)
    # Metadata claims one row, while the SDPA query requires three batches.
    pack._set_backend_overrides([1, 7], [[3, 4, 128, 64], [1, 1, 1, 1]], [[], override_stride])
    required = ((3 + extra - 1) * max(stride, override_stride[0] if override_stride else 1) + 1) * (bits // 8)
    bind(required)
    pack._validate_backend_capacity()
    bind(required - 1)
    with pytest.raises(ValueError, match=f"requires a {required}-byte buffer but received {required - 1} bytes"):
        pack._validate_backend_capacity()


@pytest.mark.parametrize("missing", ["query", "metadata", "enabled"])
def test_metadata_shrink_requires_query_and_metadata_overrides(missing):
    pack, bind = _metadata_pack(enabled=missing != "enabled")
    uids, shapes = [1, 7], [[2, 4, 128, 64], [3, 1, 1, 1]]
    if missing != "enabled":
        i = 0 if missing == "query" else 1
        del uids[i]
        del shapes[i]
    pack._set_backend_overrides(uids, shapes, [[] for _ in uids])
    bind(16388)
    pack._validate_backend_capacity()
    bind(12)
    with pytest.raises(ValueError, match="requires a 16388-byte buffer"):
        pack._validate_backend_capacity()


def test_shared_metadata_keeps_every_query_dependency():
    pack, bind = _metadata_pack(rules=[(1, 4096, 1), (2, 8, 1)])
    pack._set_backend_overrides([1, 2, 7], [[2, 4, 128, 64], [5, 4, 128, 64], [3, 1, 1, 1]], [[], [], []])
    bind(24)
    pack._validate_backend_capacity()
    bind(12)
    with pytest.raises(ValueError, match="requires a 24-byte buffer"):
        pack._validate_backend_capacity()


@pytest.mark.parametrize("shape", [[2, 4, 64], [-1, 4, 128, 64], [2**63 - 1, 4, 128, 64]])
def test_metadata_query_extent_is_validated_before_launch(shape):
    pack, bind = _metadata_pack()
    bind(16388)
    pack._set_backend_overrides([1, 7], [shape, [1, 1, 1, 1]], [[], []])
    with pytest.raises(ValueError, match="rank-4 query|batch extent overflows"):
        pack._validate_backend_capacity()


def _packed_graph(handle, dtype, cache, offset_dtype=torch.int32):
    batch, h, hk, d, sq, sk = 3 if cache == 4 else 2, 4, 2, 64, 128, 2048
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(
        handle=handle, io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=True
    )
    values, shapes, strides = {}, {}, {}

    def metadata(uid, step):
        meta_dtype = offset_dtype if uid < 100 else torch.int32
        t = g.tensor(
            dim=[cache + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64 if meta_dtype == torch.int64 else cudnn.data_type.INT32
        ).set_uid(uid)
        values[uid] = torch.arange(batch + 1, device="cuda", dtype=meta_dtype) * step
        shapes[uid], strides[uid] = [batch + 1, 1, 1, 1], [1, 1, 1, 1]
        return t

    qoff, koff, voff, ooff, soff = [metadata(uid, step) for uid, step in [(50, 2), (51, 5), (52, 5), (53, 2), (54, 2)]]
    qlen, klen = metadata(100, 2), metadata(101, 5)
    tensors = []
    for uid, heads, seq, off in [(1, h, sq, qoff), (2, hk, sk, koff), (3, hk, sk, voff)]:
        stride = [heads * seq * d, d, heads * d, 1]
        t = g.tensor(dim=[cache, heads, seq, d], stride=stride, data_type=dt).set_uid(uid)
        t.set_ragged_offset(off).set_ragged_offset_multiplier(heads * d)
        tensors.append(t)
        shapes[uid], strides[uid] = [batch, heads, seq, d], stride
        values[uid] = torch.zeros((batch * (2 if uid == 1 else 5), heads, d), device="cuda", dtype=dtype)
    for b in range(batch):
        values[3][b * 5 : (b + 1) * 5].fill_(b + 1)
    out, stats = g.sdpa(
        *tensors,
        generate_stats=True,
        attn_scale=0.125,
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        cu_seq_len_q=qlen,
        cu_seq_len_kv=klen,
        implementation=cudnn.attention_implementation.UNIFIED,
    )
    for t, uid, off, width in [(out, 10, ooff, d), (stats, 11, soff, 1)]:
        stride = [sq * h * width, width, h * width, 1]
        t.set_uid(uid).set_output(True).set_dim([cache, h, sq, width]).set_stride(stride)
        t.set_data_type(dt if width == d else cudnn.data_type.FLOAT).set_ragged_offset(off).set_ragged_offset_multiplier(h * width)
        shapes[uid], strides[uid] = [batch, h, sq, width], stride
        values[uid] = torch.empty((batch * 2, h, width), device="cuda", dtype=dtype if width == d else torch.float32)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    for i, cfg in enumerate(g.plans):
        if is_backend_engine(cfg.engine_id):
            g.build_plan_at_index(i)
            break
    else:
        pytest.fail("No native packed SDPA plan")
    order = sorted(values)
    overrides = dict(override_uids=order, override_shapes=[shapes[u] for u in order], override_strides=[strides[u] for u in order])
    return g, values, overrides, batch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cache", [4, 4096])
@pytest.mark.parametrize("entry", ["mapping", "ordered"])
@pytest.mark.parametrize("lifecycle", ["fresh", "loaded", "reserialized"])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_packed_metadata_uses_actual_batch_after_loading(dtype, cache, entry, lifecycle, offset_dtype, cudnn_handle):
    if cudnn.backend_version() < 92800 or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Unified cumulative-length packed SDPA requires cuDNN 9.28+ and Ampere+")
    g, values, overrides, batch = _packed_graph(cudnn_handle, dtype, cache, offset_dtype)
    for _ in range({"fresh": 0, "loaded": 1, "reserialized": 2}[lifecycle]):
        blob = g.serialize(as_bytes=True)
        g = cudnn.pygraph(handle=cudnn_handle)
        g.deserialize(blob)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    workspace = torch.empty(max(1, g.get_workspace_size_plan_at_index(g._plan_index, handle=cudnn_handle, **overrides)), device="cuda", dtype=torch.uint8)
    order = tuple(sorted(values))

    def execute():
        if entry == "mapping":
            g.execute(values, workspace, handle=cudnn_handle, **overrides)
        else:
            g.execute(tuple(values[u] for u in order), workspace, tensor_uids=order, handle=cudnn_handle, **overrides)

    def check(delta):
        for b in range(batch):
            torch.testing.assert_close(values[10][b * 2 : (b + 1) * 2], torch.full_like(values[10][:2], b + 1 + delta), atol=0, rtol=0)
        expected = torch.tensor([math.log(4), math.log(5)], device="cuda").repeat(batch).reshape(-1, 1, 1).expand_as(values[11])
        torch.testing.assert_close(values[11], expected, atol=0.002, rtol=0.002)

    execute()
    check(0)
    # Short views retain fully sized owned storage for safe old-code RED runs.
    for uid in (50, 100, 101):
        full = values[uid]
        values[uid] = full[:-1]
        pos = overrides["override_uids"].index(uid)
        original_shape = overrides["override_shapes"][pos]
        overrides["override_shapes"][pos] = [1, 1, 1, 1]
        values[10].fill_(123)
        workspace.fill_(165)
        with pytest.raises(ValueError, match=f"tensor uid {uid} requires a {(batch + 1) * full.element_size()}-byte buffer"):
            execute()
        torch.cuda.synchronize()
        assert torch.all(values[10] == 123).item()
        assert torch.all(workspace == 165).item()
        values[uid], overrides["override_shapes"][pos] = full, original_shape
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    replay = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(capture_stream):
            cudnn.set_stream(cudnn_handle, capture_stream.cuda_stream)
            with torch.cuda.graph(replay, stream=capture_stream):
                execute()
        torch.cuda.current_stream().wait_stream(capture_stream)
        values[3].add_(2)
        values[10].fill_(float("nan"))
        replay.replay()
        torch.cuda.synchronize()
        check(2)
    finally:
        torch.cuda.synchronize()
        replay.reset()
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)


def test_malformed_loaded_override_flag_does_not_replace_live_plan(cudnn_handle):
    if cudnn.backend_version() < 92800 or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Unified packed SDPA requires cuDNN 9.28+ and Ampere+")
    live, values, overrides, _ = _packed_graph(cudnn_handle, torch.float16, 4)
    incoming, *_ = _packed_graph(cudnn_handle, torch.float16, 4096)
    before = live._lowered_graph._get_buffer_bounds()
    payload = incoming.serialize(as_bytes=True)
    # UBJSON T is true; U\x01 is the integer 1. Keep the blob syntactically
    # valid while invalidating only the frontend override flag's type.
    field = b"is_override_shape_enabledT"
    assert payload.count(field) == 1
    payload = payload.replace(field, b"is_override_shape_enabledU\x01")
    with pytest.raises((ValueError, RuntimeError), match="boolean"):
        live.deserialize(payload)
    assert live._lowered_graph._get_buffer_bounds() == before
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    ws = torch.empty(max(1, live.get_workspace_size_plan_at_index(live._plan_index, **overrides)), device="cuda", dtype=torch.uint8)
    live.execute(values, ws, handle=cudnn_handle, **overrides)
    torch.cuda.synchronize()
    for b in range(3):
        torch.testing.assert_close(values[10][b * 2 : (b + 1) * 2], torch.full_like(values[10][:2], b + 1), atol=0, rtol=0)
