# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled DSA cache, scoring, selection, and expansion contracts."""

import cuda.bindings.driver as cuda
import pytest
import torch

import cudnn
from cudnn.deepseek_sparse_attention import pooled
from cudnn.ops import dsa
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(torch.cuda.get_device_capability() not in ((10, 0), (10, 3)), reason="Pooled DSA requires Blackwell"),
    pytest.mark.skipif(cutedsl_too_old(cutedsl_state()[1]), reason="Pooled DSA requires a supported CuTe DSL version"),
]


def _reference(ids, positions, pool_size, output_shape, row_map=None):
    ids, positions = ids.cpu(), positions.cpu()
    row_map = row_map.cpu() if row_map is not None else torch.arange(positions.numel())
    out = torch.full(output_shape, -1, dtype=torch.int32)
    topk = ids.shape[1] * pool_size
    for row, position in enumerate(positions.tolist()):
        src = int(row_map[row])
        if not 0 <= src < ids.shape[0] or not 0 <= position < 2**31 - 1:
            continue
        length = position + 1
        if length <= topk:
            out[row, :length] = torch.arange(length)
            continue
        completed, tail = divmod(length, pool_size)
        for i, pool in enumerate(ids[src].tolist()):
            if 0 <= pool < completed:
                out[row, i * pool_size : (i + 1) * pool_size] = torch.arange(pool * pool_size, (pool + 1) * pool_size)
        out[row, topk : topk + tail] = torch.arange(completed * pool_size, length)
    return out


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mapped", [False, True])
def test_expand_indices_causal_tail_invalid_ids_and_padding(pool_size, id_dtype, position_dtype, mapped):
    lengths = [0, 1, pool_size - 1, pool_size, 32, *range(4096, 4096 + pool_size)]
    rows, groups = len(lengths), 8
    ids = torch.arange(rows * (groups + 5), device="cuda", dtype=id_dtype).reshape(rows, groups + 5)[:, :groups]
    ids[:, 1] = -1
    ids[:, 2] = 100000
    pos_storage = torch.empty(rows * 2, dtype=position_dtype, device="cuda")
    positions = pos_storage[::2]
    positions.copy_(torch.tensor(lengths, device="cuda") - 1)
    row_map = torch.arange(rows - 1, -1, -1, device="cuda", dtype=torch.int32) if mapped else None
    if mapped:
        row_map[-1] = -1
        row_map[-2] = rows
    width = groups * pool_size + pool_size + 7
    storage = torch.full((rows + 3, width + 11), 999, dtype=torch.int32, device="cuda")
    out = storage[:, :width]
    result = dsa.pooled_expand_forward(ids, positions, pool_size, output=out, row_map=row_map)
    torch.testing.assert_close(result["indices"].cpu(), _reference(ids, positions, pool_size, out.shape, row_map), rtol=0, atol=0)
    assert (storage[:, width:] == 999).all().item()
    assert result["indices"].data_ptr() == out.data_ptr()


def test_expand_indices_rebinds_dynamic_shapes_and_replays_without_allocations():
    from cudnn.deepseek_sparse_attention.pooled.indices import _compile_indices

    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(2, 1)
    pos = torch.tensor([8191, 8194], device="cuda", dtype=torch.int64)
    out = torch.empty(2, 35, dtype=torch.int32, device="cuda")
    plan = pooled.ExpandIndices(ids, pos, out)
    plan.compile()
    ids2 = torch.arange(16, device="cuda", dtype=torch.int32).repeat(5, 1)[:, :7]
    pos2 = torch.tensor([4096, 4097, 4098, -1, 12], device="cuda", dtype=torch.int64)
    storage = torch.empty(8, 48, dtype=torch.int32, device="cuda")
    out2 = storage[:, :40]
    plan.execute(ids2, pos2, out2)
    torch.cuda.synchronize()
    misses = _compile_indices.cache_info().misses
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            plan.execute(ids2, pos2, out2)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    assert _compile_indices.cache_info().misses == misses
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(ids2, pos2, out2)
    pos2.add_(1)
    graph.replay()
    torch.testing.assert_close(out2.cpu(), _reference(ids2, pos2, 4, out2.shape), rtol=0, atol=0)


def test_expand_indices_uses_explicit_stream_and_rejects_contract_changes():
    ids = torch.arange(8, device="cuda", dtype=torch.int32).view(1, -1)
    pos = torch.zeros(1, device="cuda", dtype=torch.int64)
    out = torch.empty(1, 35, device="cuda", dtype=torch.int32)
    plan = pooled.ExpandIndices(ids, pos, out)
    plan.compile()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        pos.fill_(8194)
    plan.execute(ids, pos, out, current_stream=cuda.CUstream(stream.cuda_stream))
    stream.synchronize()
    torch.testing.assert_close(out.cpu(), _reference(ids, pos, 4, out.shape), rtol=0, atol=0)
    with pytest.raises(ValueError, match="dtypes"):
        plan.execute(ids.long(), pos, out)
    with pytest.raises(ValueError, match="row_map presence"):
        plan.execute(ids, pos, out, row_map=torch.zeros(1, device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="cover"):
        plan.execute(ids, pos, out[:, :32])
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(out[:, :8], pos, out)


def _pool_reference(keys, gates, ape):
    value = (keys.float() * (gates.float() + ape).softmax(dim=0)).sum(0).to(torch.bfloat16).float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = value.reshape(-1, 2, stride)
        value = torch.stack((pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]), dim=1).flatten()
    return (value * 128**-0.5).to(torch.bfloat16)


def _quantized_pool(value, dtype):
    if dtype == "bf16":
        return value.view(torch.uint8), None
    if dtype == "fp8":
        scale = (value.float().abs().max().clamp_min(1e-4) / 448).log2().ceil().exp2()
        return (value.float() / scale).to(torch.float8_e4m3fn).view(torch.uint8), scale.reshape(1).view(torch.uint8)
    blocks = value.float().reshape(4, 32)
    exponent = (blocks.abs().amax(-1).clamp_min(6 * 2**-126) / 6).log2().ceil()
    scaled = blocks / exponent.exp2()[:, None]
    code = torch.zeros_like(scaled, dtype=torch.uint8)
    for i, threshold in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5), 1):
        code = torch.where(scaled.abs() > threshold if i % 2 else scaled.abs() >= threshold, i, code)
    code = (code | (scaled.signbit().byte() << 3)).flatten()
    return code[::2] | (code[1::2] << 4), (exponent + 127).byte()


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_pool_cache_update_matches_reference_and_preserves_padding(pool_size, cache_dtype, mode):
    torch.manual_seed(31)
    page_size, blocks = 64, 3
    counts = [13, 19, 0] if mode == "prefill" else [1, 0, 7]
    starts = [4096, 4096, 4096] if mode == "prefill" else [4095, 0, 4093]
    total = sum(counts)
    keys = torch.randn(total, 136, device="cuda", dtype=torch.bfloat16)[:, :128]
    gates = torch.zeros_like(keys)
    ape = torch.zeros(pool_size, 128, device="cuda", dtype=torch.float32)
    value_bytes = {"bf16": 256, "fp8": 128, "fp4": 64}[cache_dtype]
    page_bytes = page_size * (value_bytes + (0 if cache_dtype == "bf16" else 4))
    backing = torch.full((blocks, page_bytes + 128), 0xA5, device="cuda", dtype=torch.uint8)
    cache = backing[:, :page_bytes]
    tail_backing = torch.randn(blocks, 2, pool_size + 1, 128, device="cuda", dtype=torch.bfloat16)
    tail = tail_backing[:, :, :pool_size]
    tail[:, 1].zero_()
    slots, tail_slots, pos, offsets = [], [], [], [0]
    for request, (count, start) in enumerate(zip(counts, starts)):
        for position in range(start, start + count):
            slot = request * page_size + position // pool_size - start // pool_size
            slots.append(slot if (position + 1) % pool_size == 0 else -1)
            tail_slots.append(request * pool_size + position % pool_size)
            pos.append(position)
        offsets.append(offsets[-1] + count)
    expected_cache = backing.cpu().clone()
    expected_tail = tail_backing.cpu().clone()
    host_keys, host_gates, host_ape = keys.cpu(), gates.cpu(), ape.cpu()
    for row, (slot, tslot) in enumerate(zip(slots, tail_slots)):
        block, phase = divmod(tslot, pool_size)
        if slot >= 0:
            if mode == "prefill":
                pool_k = host_keys[row - pool_size + 1 : row + 1]
                pool_g = host_gates[row - pool_size + 1 : row + 1]
            else:
                pool_k = expected_tail[block, 0, :pool_size].clone()
                pool_g = expected_tail[block, 1, :pool_size].clone()
                pool_k[phase], pool_g[phase] = host_keys[row], host_gates[row]
            values, scales = _quantized_pool(_pool_reference(pool_k, pool_g, host_ape), cache_dtype)
            page, offset = divmod(slot, page_size)
            expected_cache[page, offset * value_bytes : (offset + 1) * value_bytes] = values
            if scales is not None:
                expected_cache[page, page_size * value_bytes + offset * 4 : page_size * value_bytes + (offset + 1) * 4] = scales
        expected_tail[block, 0, phase] = host_keys[row]
        expected_tail[block, 1, phase] = host_gates[row]
    slots = torch.tensor(slots, device="cuda", dtype=torch.int64)
    tail_slots = torch.tensor(tail_slots, device="cuda", dtype=torch.int64)
    positions = torch.tensor(pos, device="cuda", dtype=torch.int64) if mode == "decode" else None
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32) if mode == "decode" else None
    args = (keys, gates, ape, cache, tail, slots, tail_slots, positions, cu_seqlens)
    plan = pooled.PoolCacheUpdate(*args, pool_size=pool_size, page_size=page_size, cache_dtype=cache_dtype, mode=mode)
    plan.compile()
    plan.execute(*args)
    torch.testing.assert_close(tail_backing.cpu(), expected_tail, rtol=0, atol=0)
    torch.testing.assert_close(backing.cpu(), expected_cache, rtol=0, atol=0)


def _cache_args(rows=4, padded=False):
    stride = 136 if padded else 128
    keys = torch.zeros(rows, stride, device="cuda", dtype=torch.bfloat16)[:, :128]
    gates = torch.zeros_like(keys)
    ape = torch.zeros(4, 128, device="cuda", dtype=torch.float32)
    cache = torch.zeros(1, 64 * 256 + (16 if padded else 0), device="cuda", dtype=torch.uint8)[:, : 64 * 256]
    tail = torch.zeros(1, 2, 5 if padded else 4, 128, device="cuda", dtype=torch.bfloat16)[:, :, :4]
    slots = torch.full((rows,), -1, device="cuda", dtype=torch.int64)
    slots[3::4] = torch.arange(rows // 4, device="cuda", dtype=torch.int64)
    tail_slots = torch.arange(rows, device="cuda", dtype=torch.int64) % 4
    return keys, gates, ape, cache, tail, slots, tail_slots


def test_pool_cache_update_rebinds_shapes_and_replays_without_allocations():
    from cudnn.deepseek_sparse_attention.pooled.cache import _compile_cache_update

    plan = pooled.PoolCacheUpdate(*_cache_args(), cache_dtype="bf16")
    plan.compile()
    args = _cache_args(rows=12, padded=True)
    plan.execute(*args)
    torch.cuda.synchronize()
    misses = _compile_cache_update.cache_info().misses
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            plan.execute(*args)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert _compile_cache_update.cache_info().misses == misses
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(*args)
    args[0].fill_(1)
    graph.replay()
    expected = _pool_reference(torch.ones(4, 128), torch.zeros(4, 128), torch.zeros(4, 128))
    actual = args[3].view(torch.bfloat16).reshape(64, 128)[:3]
    torch.testing.assert_close(actual.cpu(), expected.expand(3, 128), rtol=0, atol=0)
    torch.testing.assert_close(args[4][0, 0].cpu(), torch.ones(4, 128, dtype=torch.bfloat16), rtol=0, atol=0)


def test_pool_cache_update_explicit_stream_and_contract_errors():
    args = _cache_args()
    plan = pooled.PoolCacheUpdate(*args, cache_dtype="bf16")
    plan.compile()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        args[0].fill_(2)
    result = plan.execute(*args, current_stream=cuda.CUstream(stream.cuda_stream))
    stream.synchronize()
    expected = _pool_reference(torch.full((4, 128), 2), torch.zeros(4, 128), torch.zeros(4, 128))
    torch.testing.assert_close(args[3].view(torch.bfloat16)[0, :128].cpu(), expected, rtol=0, atol=0)
    assert result["cache"] is args[3] and result["tail"] is args[4]
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(args[4][0, 0], *args[1:])
    with pytest.raises(ValueError, match="dtype"):
        plan.execute(args[0].float(), *args[1:])
    with pytest.raises(ValueError, match="neither"):
        plan.execute(*args, positions=args[5])
    with pytest.raises(ValueError, match="autograd"):
        plan.execute(args[0].detach().requires_grad_(), *args[1:])
    with pytest.raises(ValueError, match="requires positions"):
        pooled.PoolCacheUpdate(*args, mode="decode", cache_dtype="bf16").check_support()


def test_pool_cache_update_fp4_exact_power_of_two_scale():
    args = list(_cache_args())
    args[0][:, 1] = 17
    args[3] = torch.zeros(1, 64 * 68, device="cuda", dtype=torch.uint8)
    dsa.pooled_cache_update_forward(*args, cache_dtype="fp4")
    # Hadamard rounds to +/-1.5: the scale must be 0.25, not 0.5.
    torch.testing.assert_close(args[3][0, :64].cpu(), torch.full((64,), 0xF7, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(args[3][0, 4096:4100].cpu(), torch.full((4,), 125, dtype=torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_pool_cache_update_nonuniform_gates_and_ape(mode):
    torch.manual_seed(73)
    args = list(_cache_args())
    args[0].normal_()
    args[1].normal_()
    args[2].normal_()
    host_keys, host_gates, host_ape = [t.cpu().double() for t in args[:3]]
    if mode == "decode":
        args.extend((torch.arange(4, device="cuda", dtype=torch.int64), torch.tensor([0, 4], device="cuda", dtype=torch.int32)))
    dsa.pooled_cache_update_forward(*args, cache_dtype="bf16", mode=mode)
    pooled = (host_keys * (host_gates + host_ape).softmax(0)).sum(0)
    expected = _pool_reference(host_keys, host_gates, host_ape).double()
    actual = args[3].view(torch.bfloat16)[0, :128].cpu().double()
    # Both BF16 rounding boundaries can differ by one ulp near a midpoint.
    error_bound = 2 * torch.finfo(torch.bfloat16).eps * pooled.norm()
    assert (actual - expected).norm() <= error_bound


def _quantize_index_values(value, dtype):
    if dtype == "bf16":
        return value, None, value.float()
    if dtype == "fp8":
        scale = (value.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil().exp2()
        data = (value.float() / scale).to(torch.float8_e4m3fn)
        return data, scale, data.float() * scale
    blocks = value.float().reshape(*value.shape[:-1], 4, 32)
    exponent = (blocks.abs().amax(-1).clamp_min(6 * 2**-126) / 6).log2().ceil()
    scaled = blocks / exponent.exp2()[..., None]
    code = torch.zeros_like(scaled, dtype=torch.uint8)
    for i, threshold in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5), 1):
        code = torch.where(scaled.abs() > threshold if i % 2 else scaled.abs() >= threshold, i, code)
    code = (code | (scaled.signbit().byte() << 3)).reshape(value.shape)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    decoded = levels[(code & 7).long()] * torch.where(code < 8, 1, -1) * exponent.exp2().repeat_interleave(32, -1)
    return code[..., ::2] | (code[..., 1::2] << 4), (exponent + 127).byte(), decoded


def _indexer_case(dtype, packed_scales=False, selected_pools=32, pool_size=4, page_size=64):
    pytest.importorskip("triton", minversion="3.7.0")
    torch.manual_seed(109)
    requests, pages = 3, 13
    positions = torch.tensor([1, 128, 3071, 511, 512, 1537, 2000, -1, 3074], device="cuda", dtype=torch.int64)
    positions = (positions // 4) * pool_size + positions % 4
    rows, capacity = len(positions), pages * page_size
    q, qs, decoded_q = _quantize_index_values(torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16), dtype)
    k, ks, decoded_k = _quantize_index_values(torch.randn(requests, capacity, 128, device="cuda", dtype=torch.bfloat16), dtype)
    weights = torch.randn(rows, 32, device="cuda") / 64
    reference_weights = weights.clone()
    if dtype == "fp8":
        weights *= qs.squeeze(-1)
        qs = None
    elif dtype == "fp4" and packed_scales:
        qs = qs.view(torch.int32).squeeze(-1)
    value_bytes = {"bf16": 256, "fp8": 128, "fp4": 64}[dtype]
    width = page_size * (value_bytes + (0 if dtype == "bf16" else 4))
    backing = torch.full((requests * pages, width + 64), 0xA5, device="cuda", dtype=torch.uint8)
    cache = backing[:, :width]
    table_storage = torch.full((requests, pages + 3), -1, device="cuda", dtype=torch.int32)
    table = table_storage[:, :pages]
    table.copy_(torch.randperm(requests * pages, device="cuda").reshape(requests, pages))
    physical = table.flatten().long()
    cache[physical, : page_size * value_bytes] = k.view(torch.uint8).reshape(requests * pages, -1)
    if ks is not None:
        cache[physical, page_size * value_bytes :] = ks.view(torch.uint8).reshape(requests * pages, -1)
    cu = torch.tensor([0, 3, 3, 9], device="cuda", dtype=torch.int32)
    output_storage = torch.full((rows + 2, selected_pools * pool_size + pool_size + 8), 999, device="cuda", dtype=torch.int32)
    output = output_storage[:, : selected_pools * pool_size + pool_size + 3]
    args = q, weights, cache, table, cu, positions, output, qs
    reference = decoded_q.double(), decoded_k.double(), reference_weights.double()
    return args, reference, backing, output_storage


def _check_indexer_result(args, reference, workspace, selected_pools=32, pool_size=4, page_size=64):
    q, weights, cache, table, cu, positions, output, scales = args
    decoded_q, decoded_k, ref_weights = reference
    capacity = table.shape[1] * page_size
    scores = workspace[: q.shape[0] * capacity * 4].view(torch.float32).view(q.shape[0], capacity)
    offsets, host_positions = cu.cpu().tolist(), positions.cpu().tolist()
    topk = selected_pools * pool_size
    for request, (start, end) in enumerate(zip(offsets, offsets[1:])):
        for row in range(start, end):
            length = host_positions[row] + 1
            pools = max(length // pool_size, 0)
            expected = ((decoded_q[row] @ decoded_k[request].T).relu() * ref_weights[row, :, None]).sum(0)
            torch.testing.assert_close(scores[row, :pools].double(), expected[:pools], rtol=3e-5, atol=2e-4)
            assert scores[row, pools:].isneginf().all().item()
            actual = output[row].cpu()
            if length <= topk:
                wanted = torch.full_like(actual, -1)
                wanted[: max(length, 0)] = torch.arange(max(length, 0))
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            else:
                groups = actual[:topk].reshape(selected_pools, pool_size)
                assert torch.equal(groups, groups[:, :1] + torch.arange(pool_size))
                selected = groups[:, 0] // pool_size
                assert len(selected.unique()) == selected_pools and selected.min() >= 0 and selected.max() < pools
                cutoff = expected[:pools].topk(selected_pools).values[-1]
                assert (expected[selected.to(expected.device)] >= cutoff - 2e-4).all().item()
                wanted_tail = torch.full_like(actual[topk:], -1)
                wanted_tail[: length % pool_size] = torch.arange(pools * pool_size, length)
                torch.testing.assert_close(actual[topk:], wanted_tail, rtol=0, atol=0)
    assert (output[q.shape[0] :] == -1).all().item()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("columns", [65, 8192, 32768])
@pytest.mark.parametrize("pool_size", [4, 16])
def test_fused_select_boundaries_ties_and_replay(mode, score_dtype, columns, pool_size):
    topk = 32 if columns == 65 else 512
    boundary = topk * pool_size
    positions = torch.tensor(
        [
            boundary - 1,
            boundary,
            boundary + 1,
            boundary + 2,
            boundary + 3,
            columns * pool_size - 1,
            columns * pool_size + 1,
            -1,
            2**31 - 1,
            columns * pool_size + 44,
        ],
        device="cuda",
        dtype=torch.int64,
    )
    rows = positions.numel()
    backing = torch.full((rows, columns + 8), float("nan"), dtype=score_dtype, device="cuda")
    scores = backing[:, :columns].zero_()
    workspace = torch.empty(dsa.pooled_select_workspace_size(scores), dtype=torch.uint8, device="cuda")
    width = boundary + pool_size + 3
    storage = torch.full((rows + 2, width + 4), 12345, dtype=torch.int32, device="cuda")
    output = storage[:, :width]

    def run():
        result = dsa.pooled_select_forward(scores, positions, selected_pools=topk, pool_size=pool_size, mode=mode, output=output, workspace=workspace)
        assert result["indices"] is output

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for shift in (0, -1):
        positions.add_(shift)
        if shift:
            scores.normal_()
        output.fill_(-9)
        graph.replay()
        for row, position in enumerate(positions.cpu().tolist()):
            length = position + 1 if 0 <= position < 2**31 - 1 else 0
            actual = output[row].cpu()
            if length <= boundary:
                expected = torch.full_like(actual, -1)
                expected[:length] = torch.arange(length)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                groups = actual[:boundary].view(topk, pool_size)
                ids = groups[:, 0] // pool_size
                assert ids.unique().numel() == topk
                assert ids.min() >= 0 and ids.max() < min(length // pool_size, columns)
                values = scores[row].float().cpu()
                torch.testing.assert_close(
                    values[ids].sort().values, values[: min(length // pool_size, columns)].topk(topk).values.sort().values, rtol=0, atol=0
                )
                torch.testing.assert_close(groups, ids[:, None] * pool_size + torch.arange(pool_size, dtype=torch.int32), rtol=0, atol=0)
                expected = torch.full_like(actual[boundary:], -1)
                expected[: length % pool_size] = torch.arange(length // pool_size * pool_size, length)
                torch.testing.assert_close(actual[boundary:], expected, rtol=0, atol=0)
        assert (output[rows:] == -1).all()
        assert (storage[:, width:] == 12345).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pooled_select_public_dynamic_stream_and_no_allocation(dtype, monkeypatch):
    from cudnn.deepseek_sparse_attention.pooled.forward import _compile_pooled_select

    stream = torch.cuda.Stream()
    scores = torch.randn(8, 264, device="cuda", dtype=dtype)[:, :256]
    positions = torch.full((16,), 1026, device="cuda", dtype=torch.int32)[::2]
    output = torch.empty(10, 135, device="cuda", dtype=torch.int32)[:, :131]
    workspace = torch.empty(dsa.pooled_select_workspace_size(scores, max_pools=8192), device="cuda", dtype=torch.uint8)
    kwargs = dict(selected_pools=32, max_pools=8192, output=output, workspace=workspace)
    dsa.pooled_select_forward(scores, positions, **kwargs)
    misses = _compile_pooled_select.cache_info().misses
    stream.wait_stream(torch.cuda.current_stream())
    for rows, columns in ((3, 129), (0, 65), (8, 256)):
        x, p = scores[:rows, :columns], positions[:rows]
        expected_ids = x.float().topk(32).indices.cpu()
        expected = _reference(expected_ids, p, 4, output.shape)
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                raise AssertionError("public selector allocated device storage")

            patch.setattr(torch, "empty", forbidden)
            patch.setattr(torch, "zeros", forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                dsa.pooled_select_forward(x, p, current_stream=stream.cuda_stream, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        stream.synchronize()
        # Ties can change selected IDs; compare selected score values instead.
        for row in range(rows):
            ids = output[row, :128:4].long() // 4
            torch.testing.assert_close(x[row, ids].sort().values, x[row].topk(32).values.sort().values, rtol=0, atol=0)
        torch.testing.assert_close(output[:rows, 128:].cpu(), expected[:rows, 128:], rtol=0, atol=0)
        assert (output[rows:] == -1).all()
    assert _compile_pooled_select.cache_info().misses == misses
    allocated = dsa.pooled_select_forward(scores, positions, selected_pools=32, current_stream=0)
    assert allocated["indices"].shape == (8, 131)


def test_pooled_select_rejects_invalid_buffers_before_launch(monkeypatch):
    from cudnn.deepseek_sparse_attention.pooled import selection

    scores = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    positions = torch.full((4,), 1023, device="cuda", dtype=torch.int64)
    output = torch.empty(4, 131, device="cuda", dtype=torch.int32)
    workspace = torch.empty(dsa.pooled_select_workspace_size(scores), device="cuda", dtype=torch.uint8)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached compilation")

    monkeypatch.setattr(selection, "_compile_pooled_select", forbidden)
    bad = (
        {"selected_pools": 31},
        {"pool_size": 8},
        {"mode": "bad"},
        {"max_pools": 255},
        {"logits": scores.half()},
        {"logits": scores[:, ::2]},
        {"logits": scores.detach().requires_grad_()},
    )
    for override in bad:
        with pytest.raises(ValueError):
            dsa.pooled_select_forward(**(dict(logits=scores, positions=positions, selected_pools=32, output=output, workspace=workspace) | override))
    for override in (
        {"positions": positions[:3]},
        {"positions": positions.float()},
        {"positions": positions.cpu()},
        {"output": output[:, :128]},
        {"output": output[:3]},
        {"output": scores.view(torch.int32)[:, :131]},
        {"workspace": workspace[:4]},
        {"workspace": workspace[1:]},
        {"workspace": scores.view(torch.uint8).flatten()},
    ):
        with pytest.raises(ValueError):
            dsa.pooled_select_forward(**(dict(logits=scores, positions=positions, selected_pools=32, output=output, workspace=workspace) | override))


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("selected_pools", [32, 512])
def test_pooled_indexer_paged_scores_topk_and_expansion(dtype, mode, selected_pools):
    args, reference, backing, output_storage = _indexer_case(dtype, packed_scales=mode == "decode", selected_pools=selected_pools)
    plan = pooled.IndexerForward(*args, selected_pools=selected_pools, cache_dtype=dtype, mode=mode)
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    plan.execute(*args[:7], workspace, args[7], max_query_tokens=6)
    _check_indexer_result(args, reference, workspace, selected_pools)
    if selected_pools == 32:
        result = dsa.pooled_indexer_forward(
            *args[:6], args[7], output=args[6], workspace=workspace, selected_pools=32, cache_dtype=dtype, mode=mode, max_query_tokens=6
        )
        assert result["indices"].data_ptr() == args[6].data_ptr()
        _check_indexer_result(args, reference, workspace, selected_pools)
    assert (backing[:, args[2].shape[1] :] == 0xA5).all().item()
    assert (output_storage[:, args[6].shape[1] :] == 999).all().item()


@pytest.mark.parametrize("dtype,page_size", [("bf16", 128), ("fp8", 256), ("fp4", 128)])
def test_pooled_indexer_large_pools_and_pages(dtype, page_size):
    args, reference, _, _ = _indexer_case(dtype, pool_size=16, page_size=page_size)
    plan = pooled.IndexerForward(*args, pool_size=16, page_size=page_size, selected_pools=32, cache_dtype=dtype, mode="prefill")
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    plan.execute(*args[:7], workspace, args[7], max_query_tokens=6)
    _check_indexer_result(args, reference, workspace, pool_size=16, page_size=page_size)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("padding", [1, 8])
def test_pooled_indexer_dynamic_layouts_graph_and_explicit_stream(dtype, padding):
    from cudnn.deepseek_sparse_attention.pooled.forward import _compile_pooled_select, _compile_scores

    args, reference, _, _ = _indexer_case(dtype)
    plan = pooled.IndexerForward(*args, selected_pools=32, cache_dtype=dtype, mode="prefill")
    plan.compile()
    changed = list(args)
    q = torch.empty(6, 32, args[0].shape[2] + padding, device="cuda", dtype=args[0].dtype)[:, :, : args[0].shape[2]]
    q.copy_(args[0][:6])
    changed[0], changed[1], changed[5] = q, args[1][:6], args[5][:6]
    changed[4] = torch.tensor([0, 3, 3, 6], device="cuda", dtype=torch.int32)
    changed[7] = args[7][:6] if args[7] is not None else None
    workspace = torch.empty(plan.scratch_workspace_bytes(num_queries=6), device="cuda", dtype=torch.uint8)
    execute = lambda **kw: plan.execute(*changed[:7], workspace, changed[7], max_query_tokens=3, **kw)
    execute()
    torch.cuda.synchronize()
    caches = (_compile_scores, _compile_pooled_select)
    misses = [fn.cache_info().misses for fn in caches]
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        execute()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert [fn.cache_info().misses for fn in caches] == misses
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    changed[5].sub_(1)
    graph.replay()
    _check_indexer_result(changed, reference, workspace)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        changed[5].fill_(3)
    execute(current_stream=cuda.CUstream(stream.cuda_stream))
    stream.synchronize()
    _check_indexer_result(changed, reference, workspace)
    with pytest.raises(ValueError, match="presence"):
        execute(keys=changed[0])
    with pytest.raises(ValueError, match="cover"):
        plan.execute(*changed[:7], workspace, changed[7], max_query_tokens=1)
    with pytest.raises(ValueError, match="large enough"):
        plan.execute(*changed[:7], workspace[:1], changed[7], max_query_tokens=3)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_pooled_indexer_updates_cache_before_scoring(dtype, mode):
    args, reference, _, _ = _indexer_case(dtype)
    args[5].copy_(torch.tensor([3072, 3073, 3074, 2048, 2049, 2050, 2051, 2052, 2053], device="cuda"))
    torch.manual_seed(179)
    keys = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.zeros_like(keys)
    ape = torch.zeros(4, 128, device="cuda")
    tail = torch.randn(3, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
    request_ids = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 2], device="cuda")
    pools = args[5] // 4
    slots = args[3][request_ids, pools // 64].long() * 64 + pools % 64
    slots = torch.where((args[5] + 1) % 4 == 0, slots, -1)
    tail_slots = request_ids * 4 + args[5] % 4
    updates = dict(keys=keys, gates=gates, ape=ape, tail=tail, pool_slots=slots, tail_slots=tail_slots)
    value = _pool_reference(keys[3:7].cpu(), gates[3:7].cpu(), ape.cpu())
    values, scales = _quantized_pool(value, dtype)
    expected_cache = args[2].cpu().clone()
    page = int(args[3][2, 8].item())
    expected_cache[page, : values.numel()] = values
    if scales is not None:
        expected_cache[page, 64 * values.numel() : 64 * values.numel() + 4] = scales
    decoded = _quantize_index_values(value[None, :], dtype)[2][0]
    reference[1][2, 512] = decoded.to("cuda", dtype=torch.float64)
    expected_tail = tail.cpu().clone()
    for row, slot in enumerate(tail_slots.cpu().tolist()):
        block, phase = divmod(slot, 4)
        expected_tail[block, 0, phase] = keys[row].cpu()
        expected_tail[block, 1, phase] = gates[row].cpu()
    plan = pooled.IndexerForward(*args, selected_pools=32, cache_dtype=dtype, mode=mode, **updates)
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    plan.execute(*args[:7], workspace, args[7], max_query_tokens=6, **updates)
    torch.testing.assert_close(args[2].cpu(), expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(tail.cpu(), expected_tail, rtol=0, atol=0)
    _check_indexer_result(args, reference, workspace)
    with pytest.raises(ValueError, match="presence"):
        plan.execute(*args[:7], workspace, args[7], max_query_tokens=6)
