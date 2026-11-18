# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sage FP8 block sparse attention forward (``block_sparse_attention_fp8_forward`` and the private quantized launches).

Each FP8 kernel is architecture specific, so tests are grouped by kernel: SM90 blk64/blk128, SM100/SM110 blk64,
SM100/SM103 blk128 and SM120 blk64/blk128.  Most compare against ``fp8_online_reference``, which models the
kernels' FP8 rounding of P.  The quantizer recipes are in ``test_BSA_fp8_quantize.py``, argument checks and dispatch
in ``test_BSA_api.py``.

The SM100/SM103 blk128 numerical cases are adapted from NVIDIA-JerryChen/Block-Sparse-Attention f953680.
"""

import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask, fp8_online_reference
from block_sparse_attention.cutedsl.bsa_utils import cuda_graph, import_bsa, no_host_sync, require_sm90_fp8, require_sm120_fp8

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

# ---- SM90 blk64 and blk128


def _sm90_reference(q, k, v, qs, ks, vs, indices, count, beta, block_size=64):
    """``fp8_online_reference`` for the SM90 operand layout (V as BHDS), in FP64."""
    return fp8_online_reference((q, k, v, qs, ks, vs), indices, count, beta, block_size=block_size, v_layout="bhds", dtype=torch.float64)


with_block = pytest.mark.parametrize("block", [pytest.param(64, id="blk64"), pytest.param(128, id="blk128")])


def _sm90_inputs(batch=1, heads=1, sq=64, sk=192):
    torch.manual_seed(529)
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    return q, k, torch.randn_like(k) * 0.5


def _sm90_ids(batch, heads, sq, sk, block=64):
    """Every KV block once, rotated so the first slot is not block 0."""
    order = torch.roll(torch.arange(sk // block, device="cuda", dtype=torch.int32), 1)
    return order.expand(batch, heads, sq // block, -1).contiguous()


def _sm90_launch(block):
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    return getattr(interface, f"_bsa_attn_fwd_sm90_fp8_blk{block}")


def _sm90_quantized(batch=1, heads=1, sq=64, sk=192):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    return _quantize_sage_bhsd(*_sm90_inputs(batch, heads, sq, sk), v_layout="bhds")


def _sm90_assert_reference(block, tensors, ids, count, beta=128**-0.5, atol=0.003, rtol=0.015):
    out, lse = _sm90_launch(block)(*tensors, ids, count, beta)
    ref, lse_ref = _sm90_reference(*tensors, ids, count, beta, block_size=block)
    torch.testing.assert_close(out.double(), ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse.double(), lse_ref, atol=1e-4, rtol=1e-5)
    return out, lse


@pytest.mark.L0
@with_block
@pytest.mark.parametrize("beta", [128**-0.5, 0.0, -0.125])
@pytest.mark.parametrize("batch,heads,q_blocks,k_blocks,count", [(1, 1, 1, 1, 1), (2, 3, 2, 3, 3), (1, 1, 1, 10, 5)])
def test_sm90_quantized_forward_and_centered_lse(block, batch, heads, q_blocks, k_blocks, count, beta):
    require_sm90_fp8(block)
    tensors = _sm90_quantized(batch, heads, q_blocks * block, k_blocks * block)
    _sm90_assert_reference(block, tensors, _sm90_ids(batch, heads, q_blocks * block, k_blocks * block, block), count, beta)


@with_block
@pytest.mark.parametrize(
    "count", [pytest.param(count, marks=pytest.mark.L0 if count in (1, 2, 17, 256) else pytest.mark.L1) for count in (1, 2, 3, 5, 8, 17, 33, 65, 256)]
)
def test_sm90_pipeline_lengths(block, count):
    """List lengths across the pipeline prologue, steady state and drain.  The softmax-scale sign does not depend on the
    length; test_sm90_quantized_forward_and_centered_lse covers zero and negative scales."""
    require_sm90_fp8(block)
    sk = (count + 2) * block
    _sm90_assert_reference(block, _sm90_quantized(sq=block, sk=sk), _sm90_ids(1, 1, block, sk, block), count)


@with_block
@pytest.mark.parametrize(
    "count", [pytest.param(count, marks=pytest.mark.L0 if count in (32, 33, 64, 65) else pytest.mark.L1) for count in (2, 31, 32, 33, 63, 64, 65, 97)]
)
def test_sm90_pipeline_stage_wrap(block, count):
    require_sm90_fp8(block)
    sk = (count + 3) * block
    ids = torch.randperm(sk // block, device="cuda")[:count].int().reshape(1, 1, 1, count)
    _sm90_assert_reference(block, _sm90_quantized(sq=block, sk=sk), ids, count)


@pytest.mark.L0
@pytest.mark.parametrize("scale_pattern", ["positive", "signed_q", "signed_k", "zero_k"])
@pytest.mark.parametrize(
    "block,dispatch",
    [pytest.param(64, "pipelined", id="blk64-pipelined"), pytest.param(64, "compact", id="blk64-compact"), pytest.param(128, "default", id="blk128")],
)
def test_sm90_k_scales_per_16_tokens_and_v_channel_scales(block, dispatch, scale_pattern):
    require_sm90_fp8(block)
    batch, heads, q_blocks, k_blocks, count = 2, 3, 2, 10, 5
    sq, sk = q_blocks * block, k_blocks * block
    q, k, v = _sm90_inputs(batch, heads, sq, sk)
    q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    v = v.transpose(-1, -2).contiguous().to(q.dtype)
    qs = torch.linspace(0.02, 0.3, sq, device="cuda").expand(batch, heads, -1).contiguous()
    # One distinct scale per 16-token group of a block, varying again from block to block.
    ks = torch.logspace(-5, 1, block // 16, base=2, device="cuda").repeat(batch, heads, k_blocks)
    ks *= torch.linspace(0.5, 2.0, sk // 16, device="cuda")
    if scale_pattern == "signed_q":
        qs[..., ::2] *= -1
    elif scale_pattern == "signed_k":
        ks[..., ::3] *= -1
    elif scale_pattern == "zero_k":
        ks[..., ::2] = 0
    vs = torch.linspace(0.25, 2.0, 128, device="cuda").expand(heads, -1).contiguous()
    ids = _sm90_ids(batch, heads, sq, sk, block)
    reference, lse_ref = _sm90_reference(q, k, v, qs, ks, vs, ids, count, 128**-0.5, block_size=block)
    if dispatch == "compact":
        # Enough query tiles to take the large-grid schedule.
        repeats = 2 * torch.cuda.get_device_properties(0).multi_processor_count // (batch * heads * q_blocks) + 1
        q, qs, ids = q.repeat(1, 1, repeats, 1), qs.repeat(1, 1, repeats), ids.repeat(1, 1, repeats, 1)
        reference, lse_ref = reference.repeat(1, 1, repeats, 1), lse_ref.repeat(1, 1, repeats)
    out, lse = _sm90_launch(block)(q, k, v, qs, ks, vs, ids, count)
    torch.testing.assert_close(out.double(), reference, atol=0.003, rtol=0.015)
    torch.testing.assert_close(lse.double(), lse_ref, atol=1e-4, rtol=1e-5)


@pytest.mark.L0
@with_block
def test_sm90_public_forward_and_nondefault_stream_graph(block):
    require_sm90_fp8(block)
    from cudnn import block_sparse_attention_fp8_forward

    launch = _sm90_launch(block)
    q, k, v = _sm90_inputs(2, 3, 2 * block, 4 * block)
    ids = _sm90_ids(2, 3, 2 * block, 4 * block, block)
    reference = ((q.double() @ k.double().transpose(-1, -2)) * 128**-0.5).softmax(-1) @ v.double()
    with cuda_graph() as graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)
            assert list(result.keys()) == ["o_tensor"]
            out = result["o_tensor"]
            assert out.dtype == torch.bfloat16 and out.is_contiguous()
            entries = dict(launch.compile_cache)
            # A runtime count and a different scale reuse the compiled kernel without a host read.
            with no_host_sync():
                block_sparse_attention_fp8_forward(q, k, v, ids, block_sparse_num=1, softmax_scale=0.0625, sparse_block_size=block)
            # capture_begin itself synchronizes in PyTorch; only the API launch above is covered by sync_debug_mode.
            with torch.cuda.graph(graph, stream=stream):
                captured = block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)["o_tensor"]
            graph.replay()
            assert launch.compile_cache == entries
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(captured, out, atol=0, rtol=0)
        delta = (out.double() - reference).abs()
        assert delta.max() < 0.02
        assert delta.mean() / reference.abs().mean().clamp_min(1e-8) < 0.04


@pytest.mark.L0
@with_block
def test_sm90_sync_detector_rejects_injected_device_read(block, monkeypatch):
    require_sm90_fp8(block)
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention import _interface

    q, k, v = _sm90_inputs(sq=block, sk=3 * block)
    ids = _sm90_ids(1, 1, block, 3 * block, block)
    block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)

    def synchronizing_launch(*args, **kwargs):
        args[3][0, 0, 0].item()  # Deliberate D2H: the detector must reject it.
        pytest.fail("synchronizing launch escaped the detector")

    monkeypatch.setattr(_interface, f"_bsa_attn_fwd_sm90_fp8_blk{block}", synchronizing_launch)
    with no_host_sync():
        with pytest.raises(RuntimeError, match="synchroniz"):
            block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)


@pytest.mark.L0
@with_block
@pytest.mark.parametrize("constant", [0.0, 2.0])
def test_sm90_zero_queries_and_constant_keys(block, constant):
    require_sm90_fp8(block)
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q, k, v = _sm90_inputs(sq=block, sk=3 * block)
    q.zero_()
    k.fill_(constant)
    v.zero_()
    quantized = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    out, lse = _sm90_launch(block)(*quantized, _sm90_ids(1, 1, block, 3 * block, block), 3)
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(3 * block)), atol=1e-6, rtol=0)


@pytest.mark.L0
@with_block
def test_sm90_large_topk_accumulation(block):
    require_sm90_fp8(block)
    sk = 128 * block
    tensors = _sm90_quantized(sq=block, sk=sk)
    _sm90_assert_reference(block, tensors, _sm90_ids(1, 1, block, sk, block), 128, atol=0.001, rtol=0.02)


@pytest.mark.L0
@with_block
def test_sm90_changing_max_requires_add_before_rescale(block):
    require_sm90_fp8(block)
    # Reverse traversal visits blocks 0,1,2; each block raises the maximum.
    # Keeping prior TileO in the previous max basis gives a large error.
    q = torch.zeros((1, 1, block, 128), device="cuda")
    q[..., 0] = 1
    k = torch.zeros((1, 1, 3 * block, 128), device="cuda")
    k[..., 0] = torch.arange(3, device="cuda").repeat_interleave(block) * 4
    v = torch.arange(3 * block, device="cuda").remainder(7).float().expand(128, -1).contiguous()[None, None]
    tensors = (
        q.to(torch.float8_e4m3fn),
        k.to(torch.float8_e4m3fn),
        v.to(torch.float8_e4m3fn),
        torch.ones((1, 1, block), device="cuda"),
        torch.ones((1, 1, 3 * block // 16), device="cuda"),
        torch.ones((1, 128), device="cuda"),
    )
    ids = torch.tensor([2, 1, 0], device="cuda", dtype=torch.int32).reshape(1, 1, 1, 3)
    _sm90_assert_reference(block, tensors, ids, 3, 1.0)


@pytest.mark.L0
@with_block
def test_sm90_dynamic_geometry_and_scale_reuse(block, monkeypatch):
    require_sm90_fp8(block)
    from cudnn.block_sparse_attention import _interface

    _sm90_launch(block)(*_sm90_quantized(sq=block, sk=4 * block), _sm90_ids(1, 1, block, 4 * block, block), 1)

    def unexpected(*args, **kwargs):
        pytest.fail("dynamic launch arguments triggered recompilation")

    for batch, heads, q_blocks, k_blocks, count, beta in ((2, 3, 2, 6, 5, -0.125), (1, 2, 1, 1, 1, 0.0)):
        sq, sk = q_blocks * block, k_blocks * block
        with monkeypatch.context() as patch:
            patch.setattr(_interface.cute, "compile", unexpected)
            _sm90_assert_reference(block, _sm90_quantized(batch, heads, sq, sk), _sm90_ids(batch, heads, sq, sk, block), count, beta)


@pytest.mark.L0
@with_block
def test_sm90_large_grid_multiple_waves(block):
    require_sm90_fp8(block)
    tensors = _sm90_quantized(sq=block, sk=6 * block)
    ids = _sm90_ids(1, 1, block, 6 * block, block)
    expected, expected_lse = _sm90_assert_reference(block, tensors, ids, 5)
    tiles = 2 * torch.cuda.get_device_properties(0).multi_processor_count + 1
    q, k, v, qs, ks, vs = tensors
    large = (q.repeat(1, 1, tiles, 1), k, v, qs.repeat(1, 1, tiles), ks, vs)
    out, lse = _sm90_launch(block)(*large, ids.repeat(1, 1, tiles, 1), 5)
    torch.testing.assert_close(out, expected.repeat(1, 1, tiles, 1), atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse.repeat(1, 1, tiles), atol=0, rtol=0)


@pytest.mark.L0
@with_block
@pytest.mark.parametrize("count", [1, 3, 17])
def test_sm90_uniform_logits_ring_without_blas(block, count):
    require_sm90_fp8(block)
    # Analytic sparse mean: exercise single-block/drain and repeated ring reuse
    # under memcheck without cuBLAS's internal library-symbol probes.
    blocks = count + 3
    q = torch.zeros((1, 1, block, 128), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros((1, 1, blocks * block, 128), device="cuda", dtype=q.dtype)
    values = ((torch.arange(128, device="cuda")[:, None] + 3 * torch.arange(blocks, device="cuda")[None, :]) % 7 - 3).float()
    v = values.repeat_interleave(block, dim=1)[None, None].to(q.dtype)
    qs = torch.ones((1, 1, block), device="cuda")
    ks = torch.ones((1, 1, blocks * block // 16), device="cuda")
    vs = torch.linspace(0.25, 2.0, 128, device="cuda")[None]
    ids = (torch.arange(count, device="cuda", dtype=torch.int32) * 7 % blocks).reshape(1, 1, 1, count)
    out, lse = _sm90_launch(block)(q, k, v, qs, ks, vs, ids, count)
    expected = (values[:, ids.flatten()].double().mean(-1) * vs[0].double())[None, None, None].expand_as(out)
    torch.testing.assert_close(out.double(), expected, atol=0.008, rtol=0.005)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(count * block)), atol=1e-5, rtol=0)


@pytest.mark.L0
@with_block
@pytest.mark.parametrize("layout", ["prefix", "stepped", "permuted", "broadcast"])
@pytest.mark.parametrize("tiles", [2, 96])
def test_sm90_strided_sparse_indices_forward_and_graph(block, layout, tiles, monkeypatch):
    require_sm90_fp8(block)
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    launch = _sm90_launch(block)
    # tiles=96 exceeds the large-grid threshold of the blk64 schedule, so both schedules run on the test H100.
    batch, heads, capacity = 2, 3, 5
    q, k, v = _sm90_inputs(batch, heads, tiles * block, (capacity + 3) * block)
    quantized = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    entries = None
    original_contiguous = torch.Tensor.contiguous

    def forbid_index_copy(tensor, *args, **kwargs):
        if tensor.dtype == torch.int32:
            pytest.fail("sparse indices must be read with their native strides")
        return original_contiguous(tensor, *args, **kwargs)

    graphs = []
    try:
        with torch.cuda.stream(stream):
            for spacing, count in ((2, 3), (3, capacity)):
                ids = _strided_sparse_indices(layout, batch, heads, tiles, capacity, spacing)
                packed = ids.contiguous()
                expected, expected_lse = launch(*quantized, packed, count)
                with monkeypatch.context() as patch:
                    patch.setattr(torch.Tensor, "contiguous", forbid_index_copy)
                    out, lse = launch(*quantized, ids, count)
                    public = block_sparse_attention_fp8_forward(q, k, v, ids, block_sparse_num=count, sparse_block_size=block)["o_tensor"]
                    if entries is None:
                        entries = dict(launch.compile_cache)
                    else:
                        assert launch.compile_cache == entries
                    with no_host_sync():
                        launch(*quantized, ids, count)
                    graph = torch.cuda.CUDAGraph()
                    graphs.append(graph)
                    with torch.cuda.graph(graph, stream=stream):
                        captured = block_sparse_attention_fp8_forward(q, k, v, ids, block_sparse_num=count, sparse_block_size=block)["o_tensor"]
                    graph.replay()
                    assert launch.compile_cache == entries
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
                torch.testing.assert_close(public, expected, atol=0, rtol=0)
                torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        torch.cuda.current_stream().wait_stream(stream)
    finally:
        for graph in graphs:
            graph.reset()


def _strided_sparse_indices(layout, batch, heads, tiles, capacity, spacing):
    # Guard slots contain invalid IDs so accidentally reading packed offsets
    # cannot silently produce the same result.
    shape = (batch, heads, tiles, capacity)
    if layout == "prefix":
        view = torch.full((*shape[:3], capacity + spacing + 1), -1, device="cuda", dtype=torch.int32)[..., 1 : capacity + 1]
    elif layout == "stepped":
        view = torch.full((*shape[:3], capacity * spacing + 1), -1, device="cuda", dtype=torch.int32)[..., 1::spacing]
    elif layout == "permuted":
        view = torch.empty((batch, capacity, tiles, heads), device="cuda", dtype=torch.int32).permute(0, 3, 2, 1)
    elif layout == "broadcast":
        view = torch.empty((1, 1, tiles, capacity), device="cuda", dtype=torch.int32).expand(shape)
    else:
        raise AssertionError(layout)
    # Different query rows select different physical K blocks. Broadcast only
    # batch/head on purpose, so a wrong query stride still changes the result.
    values = (torch.arange(capacity, device="cuda") + torch.arange(tiles, device="cuda")[:, None]) % (capacity + 3)
    if layout == "broadcast":
        view[0, 0].copy_(values)
    else:
        view.copy_(values)
    assert not view.is_contiguous()
    return view


@pytest.mark.L0
@pytest.mark.parametrize("count", [1, 3, 65, 256])
def test_sm90_blk64_schedule_boundary_reuses_compile_and_matches_reference(count):
    require_sm90_fp8(64)
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
    from cudnn.block_sparse_attention._interface import _bsa_attn_fwd_sm90_fp8_blk64 as launch

    # Identical query tiles let us check every output against an independent
    # online reference without doing hundreds of redundant FP64 matmuls.
    limit = 2 * torch.cuda.get_device_properties(0).multi_processor_count
    small = _quantize_sage_bhsd(*_sm90_inputs(sk=(count + 2) * 64), v_layout="bhds")
    ids = _sm90_ids(1, 1, 64, (count + 2) * 64)
    ref, lse_ref = _sm90_reference(*small, ids, count, 128**-0.5)
    compiled = None
    for tiles in (limit, limit + 1):
        q, k, v, qs, ks, vs = small
        tensors = (q.repeat(1, 1, tiles, 1), k, v, qs.repeat(1, 1, tiles), ks, vs)
        tile_ids = ids.repeat(1, 1, tiles, 1)
        # Warm both geometries; all extents stay dynamic in one compiled host
        # dispatcher, including the branch across the occupancy boundary.
        launch(*tensors, tile_ids, count)
        if compiled is None:
            compiled = dict(launch.compile_cache)
        else:
            assert launch.compile_cache == compiled
        with no_host_sync():
            out, lse = launch(*tensors, tile_ids, count)
        torch.testing.assert_close(out.double(), ref.repeat(1, 1, tiles, 1), atol=0.003, rtol=0.015)
        torch.testing.assert_close(lse.double(), lse_ref.repeat(1, 1, tiles), atol=1e-4, rtol=1e-5)
        with cuda_graph() as graph:
            with torch.cuda.graph(graph):
                graph_out, graph_lse = launch(*tensors, tile_ids, count)
            graph.replay()
            torch.testing.assert_close(graph_out, out, atol=0, rtol=0)
            torch.testing.assert_close(graph_lse, lse, atol=0, rtol=0)
            assert launch.compile_cache == compiled


@pytest.mark.L0
def test_sm90_blk128_basis_probability_and_v_layout():
    require_sm90_fp8(128)
    from cudnn.block_sparse_attention._interface import _bsa_attn_fwd_sm90_fp8_blk128 as launch

    # Every query selects a different key, including both WGs and every
    # 16/32/64-column fragment boundary. Off-diagonal P rounds to FP8 zero.
    q = torch.eye(128, device="cuda").reshape(1, 1, 128, 128).to(torch.float8_e4m3fn)
    k = q.clone()
    values = ((torch.arange(128, device="cuda")[:, None] * 3 + torch.arange(128, device="cuda")[None, :]) % 16 - 8).float()
    v = values.T.contiguous().reshape(1, 1, 128, 128).to(q.dtype)
    qs = torch.full((1, 1, 128), 64.0, device="cuda")
    ks = torch.ones((1, 1, 8), device="cuda")
    vs = torch.ones((1, 128), device="cuda")
    ids = torch.zeros((1, 1, 1, 1), dtype=torch.int32, device="cuda")
    out, lse = launch(q, k, v, qs, ks, vs, ids, 1, 1.0)
    torch.testing.assert_close(out.float(), values[None, None], atol=0, rtol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, 64), atol=1e-5, rtol=0)


# ---- SM100/SM110 blk64


def _make_block_index(batch: int, heads: int, seqlen_q: int, seqlen_k: int, topk: int) -> torch.Tensor:
    num_q_blocks = math.ceil(seqlen_q / 64)
    num_kv_blocks = math.ceil(seqlen_k / 64)
    result = torch.empty((batch, heads, num_q_blocks, topk), device="cuda", dtype=torch.int32)
    block_ids = torch.arange(num_kv_blocks, device="cuda")
    for batch_idx in range(batch):
        for head in range(heads):
            for q_block in range(num_q_blocks):
                selected = torch.roll(block_ids, shifts=batch_idx * 7 + head + q_block)[:topk].sort().values
                result[batch_idx, head, q_block] = selected.to(torch.int32)
    return result


@pytest.mark.L0
@pytest.mark.parametrize(("batch", "heads"), ((1, 4), (2, 3)))
@torch_fork_set_rng(seed=2026)
def test_sm100_blk64_matches_reference(batch, heads):
    ops = import_bsa(families=(10, 11), min_cutedsl=(4, 6, 1), feature="this numerical Sage FP8 test")
    seqlen_q, seqlen_k, head_dim, topk = 128, 640, 128, 5
    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k) * 0.5
    q2k = _make_block_index(batch, heads, seqlen_q, seqlen_k, topk)

    block_sizes = torch.full((seqlen_k // 64,), 64, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, topk, block_sizes, seqlen_q, seqlen_k, 64)
    softmax_scale = head_dim**-0.5
    reference, _ = attention_reference(q, k, v, mask, softmax_scale)

    result = ops.block_sparse_attention_fp8_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=topk,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    assert list(result.keys()) == ["o_tensor"]
    actual = result["o_tensor"]
    assert actual.dtype == torch.bfloat16
    assert actual.shape == q.shape
    difference = (actual.float() - reference).abs()
    assert difference.max().item() < 0.2
    assert (difference.mean() / reference.abs().mean().clamp_min(1e-8)).item() < 0.04


# ---- SM100/SM103 blk128


@pytest.fixture
def require_sm100():
    import_bsa(archs=(100, 103), min_cutedsl=(4, 6, 2), reason="requires SM100/SM103")


def quantize_sage_bhsd(*args):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    return _quantize_sage_bhsd(*args)


def bsa_fp8_fwd(q, k, v, qs, ks, vs, index, topk, scale=None, *, sparse_block_size=128, block_sizes=None, q2k_block_nums=None, kv_splits=1):
    from cudnn.block_sparse_attention._interface import _bsa_attn_fwd_sm100_fp8_blk128

    assert sparse_block_size == 128
    return _bsa_attn_fwd_sm100_fp8_blk128(q, k, v, qs, ks, vs, index, topk, scale, block_sizes, q2k_block_nums, kv_splits)


def make_inputs(batch, heads, sq, sk, *, gain=1.0):
    return tuple((torch.randn(batch, heads, seq, 128, device="cuda") * gain).bfloat16() for seq in (sq, sk, sk))


def make_metadata(batch, heads, sq, sk, topk, *, variable=False, sizes_mode=0):
    """Unsorted, distinct sparse indices; inactive slots deliberately invalid."""
    nq, nk = math.ceil(sq / 128), math.ceil(sk / 128)
    index = torch.stack([torch.randperm(nk)[:topk] for _ in range(batch * heads * nq)]).reshape(batch, heads, nq, topk).to(device="cuda", dtype=torch.int32)
    nums = None
    if variable:
        nums = (torch.arange(batch * heads * nq) % (topk + 1)).reshape(batch, heads, nq)
        nums = nums.to(device="cuda", dtype=torch.int32)
        inactive = torch.arange(topk, device="cuda") >= nums[..., None]
        index.masked_fill_(inactive, 0x3FFFFFFF)
    sizes = None
    if sizes_mode:
        shape = {1: (nk,), 2: (batch, nk), 3: (batch, heads, nk)}[sizes_mode]
        sizes = torch.randint(1, 129, shape, device="cuda", dtype=torch.int32)
        sizes[..., -1].clamp_(max=sk - (nk - 1) * 128)
    return index, nums, sizes


def sparse_reference(q, k, v, index, topk, nums=None, sizes=None, scale=None):
    """FP32 torch matmul/softmax over exactly the selected valid KV tokens."""
    batch, heads, sq, dim = q.shape
    sk = k.shape[2]
    scale = dim**-0.5 if scale is None else scale
    out = torch.empty_like(q, dtype=torch.float32)
    offsets = torch.arange(128, device=q.device)
    for qb in range(index.shape[2]):
        blocks = index[:, :, qb].long()
        count = topk if nums is None else nums[:, :, qb, None]
        live = torch.arange(blocks.shape[-1], device=q.device) < count
        # Empty and unused slots must never address the underlying tensors.
        blocks = blocks.masked_fill(~live, 0)
        tokens = blocks[..., None] * 128 + offsets
        valid = live[..., None] & (tokens < sk)
        if sizes is not None:
            if sizes.ndim == 1:
                extent = sizes[blocks]
            elif sizes.ndim == 2:
                extent = sizes[torch.arange(batch, device=q.device)[:, None, None], blocks]
            else:
                extent = sizes.gather(2, blocks)
            valid = valid & (offsets < extent[..., None])
        tokens = tokens.clamp(0, sk - 1).flatten(-2)
        gather = tokens[..., None].expand(-1, -1, -1, dim)
        kt = k.gather(2, gather).float()
        vt = v.gather(2, gather).float()
        begin, end = qb * 128, min((qb + 1) * 128, sq)
        scores = (q[:, :, begin:end].float() @ kt.transpose(-1, -2)) * scale
        scores.masked_fill_(~valid.flatten(-2).unsqueeze(-2), -torch.inf)
        probs = torch.softmax(scores, dim=-1).nan_to_num()
        out[:, :, begin:end] = probs @ vt
    return out


def dequantize(quantized):
    q, k, v, qs, ks, vs = quantized
    batch, heads, sq, dim = q.shape
    sk = k.shape[2]
    return (
        (q.float() * qs.reshape(batch, heads, sq, 1)).bfloat16(),
        (k.float() * ks.reshape(batch, heads, -1).repeat_interleave(16, dim=-1)[..., :sk, None]).bfloat16(),
        (v.float() * vs.reshape(1, heads, 1, dim)).bfloat16(),
    )


def assert_accuracy(out, ref, *, relative=0.029, absolute=0.15):
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    error = (out.float() - ref.float()).abs()
    rel = (error.mean() / ref.abs().mean().clamp_min(1e-8)).item()
    maximum = error.max().item()
    assert maximum < absolute, f"max absolute error {maximum} >= {absolute}"
    assert rel < relative, f"mean relative error {rel} >= {relative}"
    return rel, maximum


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("sizes_mode, topk", [(1, 1), (2, 2), (3, 1)])
def test_sm100_blk128_provided_extent_respects_physical_tail(sizes_mode, topk):
    """Provided extents cannot make padded tokens participate in softmax.  The block_sizes rank only changes how the
    extent is indexed, independently of the list length, so topk alternates."""
    q = torch.zeros(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 129, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    index = torch.tensor([1] if topk == 1 else [1, 0], device="cuda", dtype=torch.int32).reshape(1, 1, 1, topk)
    # The public API accepts an extent larger than the physical one-token tail.
    sizes = torch.full(
        {1: (2,), 2: (1, 2), 3: (1, 1, 2)}[sizes_mode],
        128,
        device="cuda",
        dtype=torch.int32,
    )
    quantized = quantize_sage_bhsd(q, k, v)
    out = bsa_fp8_fwd(*quantized, index, topk, sparse_block_size=128, block_sizes=sizes)
    # Constant V has output one regardless of the number of valid KV tokens.
    # Use this analytic oracle instead of requiring exact FP32 reduction sums.
    assert_accuracy(out, torch.ones_like(out, dtype=torch.float32))


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("topk", [1, 3, 5, 17, 33])
def test_sm100_blk128_scales_and_online_rescaling(topk):
    torch.manual_seed(3000 + topk)
    batch, heads, sq, sk = 2, 3, 257, 4353
    # Unit FP8 data, nonuniform independent scale factors. The K scales vary
    # within each 128-token block, so block-wide scaling would fail this test.
    q, k, v = [t.to(torch.float8_e4m3fn) for t in make_inputs(batch, heads, sq, sk, gain=0.5)]
    qs = torch.linspace(0.25, 3.0, batch * heads * sq, device="cuda").reshape(batch, heads, sq)
    ks = torch.linspace(0.125, 4.0, batch * heads * math.ceil(sk / 16), device="cuda").reshape(batch, heads, -1)
    vs = torch.linspace(0.2, 1.5, heads * 128, device="cuda").reshape(heads, 128)
    index, nums, sizes = make_metadata(batch, heads, sq, sk, topk, sizes_mode=3)
    scale = 0.13
    out = bsa_fp8_fwd(q, k, v, qs, ks, vs, index, topk, scale, sparse_block_size=128, block_sizes=sizes)
    ref = sparse_reference(*dequantize((q, k, v, qs, ks, vs)), index, topk, sizes=sizes, scale=scale)
    assert_accuracy(out, ref)


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
def test_sm100_blk128_channel_scales_and_empty_tiles():
    """Check every shuffled channel against an independent constant-V oracle."""
    torch.manual_seed(3811)
    batch, heads, sq, sk, topk = 2, 3, 129, 257, 3
    q = torch.zeros(batch, heads, sq, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros(batch, heads, sk, 128, device="cuda", dtype=torch.float8_e4m3fn)
    channels = torch.arange(128, device="cuda")
    values = torch.where(channels % 2 == 0, 0.5, -2.0)
    v = values.view(1, 1, 1, 128).expand(batch, heads, sk, 128).contiguous().to(k.dtype)
    qs = torch.ones(batch, heads, sq, device="cuda")
    ks = torch.ones(batch, heads, math.ceil(sk / 16), device="cuda")
    vs = (torch.randperm(heads * 128, device="cuda") % 128 + 1).float().reshape(heads, 128) / 256
    index = torch.arange(topk, device="cuda", dtype=torch.int32)
    index = index.view(1, 1, 1, topk).expand(batch, heads, 2, topk).contiguous()
    nums = (torch.arange(batch * heads * 2, device="cuda") % 4).int().reshape(batch, heads, 2)
    sizes = torch.tensor([128, 128, 1], device="cuda", dtype=torch.int32)
    out = bsa_fp8_fwd(
        q,
        k,
        v,
        qs,
        ks,
        vs,
        index,
        topk,
        sparse_block_size=128,
        block_sizes=sizes,
        q2k_block_nums=nums,
    )
    expected = (values[None, :] * vs).bfloat16()[None, :, None, :]
    expected = expected.expand(batch, heads, sq, 128).clone()
    empty = nums.repeat_interleave(128, dim=-1)[..., :sq] == 0
    expected[empty] = 0
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
def test_sm100_blk128_mixed_row_rescaling():
    """Mix constant and increasing logits across rows, tiles and K streams."""
    torch.manual_seed(3900)
    batch, heads, sq, sk, topk = 1, 2, 80 * 128 + 17, 17 * 128 + 1, 17
    q = torch.zeros(batch, heads, sq, 128, device="cuda")
    q[..., 0] = ((torch.arange(sq, device="cuda") // 32) % 2).float()
    k = torch.zeros(batch, heads, sk, 128, device="cuda")
    k[..., 0] = (torch.arange(sk, device="cuda") // 128).float() * 0.625 - 4.0
    v = torch.randn_like(k) * 0.25
    q, k, v = (x.to(torch.float8_e4m3fn) for x in (q, k, v))
    qs = torch.ones(batch, heads, sq, device="cuda")
    ks = torch.ones(batch, heads, math.ceil(sk / 16), device="cuda")
    vs = torch.ones(heads, 128, device="cuda")
    quantized = (q, k, v, qs, ks, vs)
    # Keep the full GPU workload under racecheck without instrumenting the
    # oracle's many GEMMs, which can exhaust the sanitizer's host memory.
    reference_inputs = tuple(t.cpu() for t in dequantize(quantized))
    nq, nk = math.ceil(sq / 128), math.ceil(sk / 128)
    # The kernel walks these indices in reverse: nonzero query rows see
    # increasing logits, while zero query rows keep the same normalization.
    index = torch.arange(nk - 1, nk - topk - 1, -1, device="cuda", dtype=torch.int32)
    index = index.expand(batch, heads, nq, topk).contiguous()
    sizes = (sk - torch.arange(nk, device="cuda") * 128).clamp(0, 128).int()
    for counts in ((17, 0, 7, 3), (0, 17, 3, 7)):
        nums = torch.tensor(counts, device="cuda", dtype=torch.int32)
        nums = nums.repeat(math.ceil(batch * heads * nq / 4))[: batch * heads * nq]
        nums = nums.reshape(batch, heads, nq)
        selected = index.masked_fill(torch.arange(topk, device="cuda") >= nums[..., None], 0x3FFFFFFF)
        out = bsa_fp8_fwd(
            *quantized,
            selected,
            topk,
            1.0,
            sparse_block_size=128,
            block_sizes=sizes,
            q2k_block_nums=nums,
        )
        ref = sparse_reference(*reference_inputs, selected.cpu(), topk, nums.cpu(), sizes.cpu(), scale=1.0).to(out.device)
        assert_accuracy(out, ref)
        for _ in range(3):
            repeat = bsa_fp8_fwd(
                *quantized,
                selected,
                topk,
                1.0,
                sparse_block_size=128,
                block_sizes=sizes,
                q2k_block_nums=nums,
            )
            assert torch.equal(out, repeat)


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
def test_sm100_blk128_unused_capacity_and_fully_masked_blocks():
    torch.manual_seed(4500)
    q, k, v = make_inputs(2, 3, 257, 641, gain=0.5)
    quantized = quantize_sage_bhsd(q, k, v)
    index, _, sizes = make_metadata(2, 3, 257, 641, 5, sizes_mode=3)
    # Fixed top-k must ignore capacity beyond its first three entries.
    index[..., 3:] = 0x3FFFFFFF
    sizes.zero_()
    sizes[0, 0, 1] = 128
    out = bsa_fp8_fwd(*quantized, index, 3, sparse_block_size=128, block_sizes=sizes)
    ref = sparse_reference(*dequantize(quantized), index, 3, sizes=sizes)
    assert_accuracy(out, ref)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[0, 1:]) == 0


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("flatten", [False, True])
def test_sm100_blk128_scale_views_and_graph_stream(flatten):
    q, k, v = make_inputs(2, 3, 129, 385)
    q8, k8, v8, qs, ks, vs = quantize_sage_bhsd(q, k, v)
    index, nums, sizes = make_metadata(2, 3, 129, 385, 3, variable=True, sizes_mode=2)

    def attention(scales):
        return bsa_fp8_fwd(q8, k8, v8, *scales, index, 3, sparse_block_size=128, q2k_block_nums=nums, block_sizes=sizes)

    expected = attention((qs, ks, vs))
    views = []
    for scale in (qs, ks, vs):
        storage = torch.empty((*scale.shape, 2), device=scale.device, dtype=scale.dtype)
        storage[..., 0].copy_(scale)
        view = storage[..., 0]
        views.append(view.reshape(-1) if flatten else view)
    with cuda_graph() as graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = attention(views)
            with torch.cuda.graph(graph, stream=stream):
                captured = attention(views)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.equal(captured, expected)


_SPLITS = (1, 2, 4, 8)
_L0_SPLIT_CASES = ((1, 0), (2, 2), (4, 1), (8, 3))  # every split count and block_sizes rank once, both count modes


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize(
    "kv_splits, sizes_mode, variable",
    [
        # block_sizes rank and the count mode are independent of the split count: all split x rank pairs, the count
        # mode alternating so that every split count and every rank sees both.
        pytest.param(kv_splits, sizes_mode, (i + sizes_mode) % 2 == 1, marks=pytest.mark.L0 if (kv_splits, sizes_mode) in _L0_SPLIT_CASES else pytest.mark.L1)
        for i, kv_splits in enumerate(_SPLITS)
        for sizes_mode in (0, 1, 2, 3)
    ],
)
def test_sm100_blk128_dynamic_metadata_and_splits(kv_splits, sizes_mode, variable):
    """One compiled variant per (splits, block_sizes rank, count mode) serves every dynamic extent: one-token tails, odd
    and unit topk, batch/head strides that are not multiples of 16, empty splits and rows, and poisoned inactive
    indices.  Scales are passed flattened for half of the split counts (the blk64 API contract)."""
    torch.manual_seed(3700 + sizes_mode)
    for batch, heads, sq, sk, topk in [
        (1, 1, 1, 1, 1),
        (2, 3, 129, 641, 5),
        (1, 5, 257, 259, 3),
        (3, 1, 63, 145, 1),
        (2, 3, 257, 913, 7),
        (1, 2, 389, 1153, 9),
    ]:
        q, k, v = make_inputs(batch, heads, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(batch, heads, sq, sk, topk, variable=variable, sizes_mode=sizes_mode)
        quantized = list(quantize_sage_bhsd(q, k, v))
        if kv_splits in (1, 4):
            quantized[3:] = [t.flatten() for t in quantized[3:]]
        kwargs = dict(
            sparse_block_size=128,
            kv_splits=kv_splits,
            block_sizes=sizes,
            q2k_block_nums=nums,
        )
        out = bsa_fp8_fwd(*quantized, index, topk, **kwargs)
        ref = sparse_reference(*dequantize(quantized), index, topk, nums, sizes)
        assert_accuracy(out, ref)
        original_ref = sparse_reference(q, k, v, index, topk, nums, sizes)
        assert_accuracy(out, original_ref, relative=0.08, absolute=0.3)
        repeated = bsa_fp8_fwd(*quantized, index, topk, **kwargs)
        assert torch.equal(out, repeated)
        if nums is not None:
            empty = nums.repeat_interleave(128, dim=-1)[..., :sq] == 0
            assert torch.count_nonzero(out[empty]) == 0
        # A cached split entry must consume new inputs rather than old outputs.
        replacement = list(quantized)
        replacement[2] = torch.zeros_like(quantized[2])
        zero = bsa_fp8_fwd(*replacement, index, topk, **kwargs)
        assert torch.count_nonzero(zero) == 0


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("kv_splits", [2, 3, 4, 8])
def test_sm100_blk128_split_rescaling_and_zero_extents(kv_splits):
    torch.manual_seed(3821)
    q, k, v = make_inputs(1, 2, 257, 1665, gain=0.5)
    # Unequal split normalizers and mixed masked blocks exercise log-sum-exp
    # combine weights, including the producer's rounded-P factor of 128.
    k[:, :, 0:512] *= 4
    q[:, :, 0:128] *= 2
    index, nums, sizes = make_metadata(1, 2, 257, 1665, 13, variable=True, sizes_mode=3)
    nums.fill_(13)
    nums[0, 0, 0] = 0
    nums[0, 1, 1] = 1
    # Rebuild indices because the variable helper poisons initially dead slots.
    index.copy_(torch.arange(13, device="cuda", dtype=torch.int32))
    sizes[..., ::3] = 0
    sizes[0, 0].zero_()
    quantized = quantize_sage_bhsd(q, k, v)
    out = bsa_fp8_fwd(
        *quantized,
        index,
        13,
        sparse_block_size=128,
        kv_splits=kv_splits,
        block_sizes=sizes,
        q2k_block_nums=nums,
    )
    ref = sparse_reference(*dequantize(quantized), index, 13, nums, sizes)
    assert_accuracy(out, ref)
    assert torch.count_nonzero(out[0, 0]) == 0
    repeated = bsa_fp8_fwd(
        *quantized,
        index,
        13,
        sparse_block_size=128,
        kv_splits=kv_splits,
        block_sizes=sizes,
        q2k_block_nums=nums,
    )
    assert torch.equal(out, repeated)


@pytest.mark.L0
@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("splits", [1, 3])
def test_sm100_blk128_public_cache_reuse_and_graph(monkeypatch, splits):
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention import _interface

    cases = []
    for b, h, sq, sk, topk in [(1, 1, 129, 259, 3), (2, 3, 257, 641, 5)]:
        q, k, v = make_inputs(b, h, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(b, h, sq, sk, topk, variable=True, sizes_mode=3)
        cases.append((q, k, v, index, nums, sizes, topk))

    def run(case):
        q, k, v, index, nums, sizes, topk = case
        return block_sparse_attention_fp8_forward(q, k, v, index, topk, sizes, nums, sparse_block_size=128, kv_splits=splits)["o_tensor"]

    _interface._bsa_attn_fwd_sm100_fp8_blk128.compile_cache.clear()
    run(cases[0])
    cache_size = len(_interface._bsa_attn_fwd_sm100_fp8_blk128.compile_cache)

    def no_compile(*args, **kwargs):
        pytest.fail("warmed calls recompiled")

    monkeypatch.setattr(_interface.cute, "compile", no_compile)
    for case in cases:
        actual = run(case)
        q, k, v, index, nums, sizes, topk = case
        assert_accuracy(actual, sparse_reference(q, k, v, index, topk, nums, sizes), relative=0.08, absolute=0.3)
    with cuda_graph() as graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with no_host_sync():
                run(cases[0])
                run(cases[1])
            with torch.cuda.graph(graph, stream=stream):
                captured = run(cases[1])
        torch.cuda.current_stream().wait_stream(stream)
        cases[1][2].zero_()
        graph.replay()
        assert torch.count_nonzero(captured) == 0
        assert len(_interface._bsa_attn_fwd_sm100_fp8_blk128.compile_cache) == cache_size


TEST_ROOT = Path(__file__).resolve().parents[2]  # test/python, where the child process resolves `block_sparse_attention`


def _compile_quantizer():
    import cutlass
    import cutlass.cute as cute
    from cudnn.block_sparse_attention._fp8_quant import _SageFp8Quantizer

    sm120 = os.environ["CUTE_DSL_ARCH"] == "sm_120a"
    tensors = [
        cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), stride_order=(0,), assumed_align=16)
        for dtype in [cutlass.BFloat16] * 3 + [cutlass.Float8E4M3FN] * 3 + [cutlass.Float32] * 6
    ]
    cute.compile(
        _SageFp8Quantizer(is_sm120=sm120, v_block_size=128 if sm120 else 0),
        *tensors,
        *[cutlass.Int32(x) for x in (2, 3, 129, 259, 2, 4)],
        cutlass.Float32(259),
        cute.runtime.make_fake_stream(),
        options=f"--gpu-arch {os.environ['CUTE_DSL_ARCH']}",
    )


def _compile_variants():
    """Compile both architectural pipelines without executing on the host GPU."""
    import cutlass
    import cutlass.cute as cute
    from cudnn.block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100_fp8_fused_pv import BlockSparseAttnForwardFp8FusedPvSm100Blk128

    def tensor(dtype, rank, dim=False, align=4):
        shape = tuple(cute.sym_int() for _ in range(rank - 1)) + (128 if dim else cute.sym_int(),)
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(rank))), assumed_align=align)

    _compile_quantizer()
    for split, sizes, counts in [(1, 0, False), (1, 3, True), (3, 2, True), (2, 1, False)]:
        kernel = BlockSparseAttnForwardFp8FusedPvSm100Blk128(
            128,
            allow_empty_block_nums=counts,
            has_block_sizes=bool(sizes),
            block_sizes_mode=sizes,
            num_splits=split,
        )
        args = (
            *[tensor(cutlass.Float8E4M3FN, 4, True, 16) for _ in range(3)],
            tensor(cutlass.Float32 if split > 1 else cutlass.BFloat16, 4, True, 16),
            tensor(cutlass.Float32, 3) if split > 1 else None,
            tensor(cutlass.Float32, 3),
            tensor(cutlass.Float32, 3),
            tensor(cutlass.Float32, 2),
            cutlass.Float32(128**-0.5),
            tensor(cutlass.Int32, 4),
            tensor(cutlass.Int32, sizes) if sizes else None,
            cutlass.Int32(3),
            tensor(cutlass.Int32, 3) if counts else None,
            cute.runtime.make_fake_stream(),
        )
        cute.compile(kernel, *args, options=f"--gpu-arch {os.environ['CUTE_DSL_ARCH']} --enable-tvm-ffi")


@pytest.mark.L0
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_sm100_blk128_offline_compile(arch):
    from cudnn.block_sparse_attention import _interface

    if _interface._cutlass_dsl_version() < (4, 6, 2):
        pytest.skip("requires CuTe DSL >= 4.6.2")
    result = subprocess.run(
        [sys.executable, "-c", "from block_sparse_attention.cutedsl.test_BSA_fp8_forward import _compile_variants; _compile_variants()"],
        env={**os.environ, "CUTE_DSL_ARCH": arch},
        cwd=TEST_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.L0
def test_sm120_blk128_quantizer_offline_compile():
    from cudnn.block_sparse_attention import _interface

    if _interface._cutlass_dsl_version() < (4, 6, 2):
        pytest.skip("requires CuTe DSL >= 4.6.2")
    result = subprocess.run(
        [sys.executable, "-c", "from block_sparse_attention.cutedsl.test_BSA_fp8_forward import _compile_quantizer; _compile_quantizer()"],
        env={**os.environ, "CUTE_DSL_ARCH": "sm_120a"},
        cwd=TEST_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---- SM120 blk64 and blk128


@pytest.mark.L0
@pytest.mark.parametrize("block_size", (64, 128))
@pytest.mark.parametrize("tail", (0, 19))
@torch_fork_set_rng(seed=120128)
def test_sm120_native_forward(block_size, tail):
    ops = require_sm120_fp8()
    batch, heads, dim, topk = 1, 2, 128, 3
    sq, sk = 2 * block_size + tail, 5 * block_size + tail
    q = torch.randn((batch, heads, sq, dim), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((batch, heads, sk, dim), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k) * 0.5
    nq, nk = (sq + block_size - 1) // block_size, (sk + block_size - 1) // block_size
    indices = torch.tensor([nk - 1, 0, 2], device="cuda", dtype=torch.int32).expand(batch, heads, nq, topk).contiguous()
    sizes = torch.full((nk,), block_size, device="cuda", dtype=torch.int32)
    sizes[-1] = sk - (nk - 1) * block_size
    mask = block_sparse_mask(indices, topk, sizes, sq, sk, block_size)
    expected, _ = attention_reference(q, k, v, mask)
    result = ops.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=block_size)
    torch.cuda.synchronize()
    actual = result["o_tensor"]
    assert actual.dtype == torch.bfloat16 and actual.shape == q.shape and actual.is_contiguous()
    difference = (actual.float() - expected).abs()
    if block_size == 128 and tail == 0:
        indices64 = torch.stack((indices * 2, indices * 2 + 1), dim=-1).flatten(-2).repeat_interleave(2, dim=2)
        baseline = ops.block_sparse_attention_fp8_forward(q, k, v, indices64)["o_tensor"]
        baseline_error = (baseline.float() - expected).abs().mean() / expected.abs().mean()
        assert difference.mean() / expected.abs().mean() <= baseline_error * 1.03 + 1e-4
    assert difference.max().item() < 0.2 * max(expected.abs().max().item(), 1.0)
    # The matched blk64 baseline also exceeds 4% for the aligned 384-key case.
    # Keep the original tests' tolerance unchanged and bound this case at 5%,
    # with the tighter matched-baseline regression check above.
    assert (difference.mean() / expected.abs().mean().clamp_min(1e-8)).item() < 0.05


@pytest.mark.L0
@pytest.mark.parametrize(
    "size_rank, v_block_size, k_quant_block_size",
    # The block_sizes rank (metadata indexing), the V layout and the K scale granularity feed independent paths: every
    # rank x V layout pair, the K granularity alternating so that every rank and both V layouts see both.
    [(rank, v_block, 1 if (rank + v_block // 128) % 2 else 16) for rank in (1, 2, 3) for v_block in (0, 128)],
)
@torch_fork_set_rng(seed=8128)
def test_sm120_blk128_scales_counts_and_partial_blocks(size_rank, v_block_size, k_quant_block_size):
    require_sm120_fp8(per_token=k_quant_block_size == 1)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    batch, heads, sq, sk, dim = 2, 2, 147, 531, 128
    q8 = torch.randint(-4, 5, (batch, heads, sq, dim), device="cuda").to(torch.float8_e4m3fn)
    k8 = torch.randint(-4, 5, (batch, heads, sk, dim), device="cuda").to(torch.float8_e4m3fn)
    v8 = torch.randint(-4, 5, (batch, heads, sk, dim), device="cuda").to(torch.float8_e4m3fn)
    qs = torch.rand((batch, heads, sq), device="cuda") * 0.1 + 0.01
    ks = torch.exp2(torch.linspace(-2, 2, (sk + k_quant_block_size - 1) // k_quant_block_size, device="cuda")).expand(batch, heads, -1).contiguous()
    if k_quant_block_size == 1:
        # Adjacent tokens must differ: a per-16 reader cannot pass this oracle.
        ks[..., 1::2] *= 0.25
    vs = torch.linspace(0.125, 2, heads * dim, device="cuda").view(heads, dim)
    quantized = (q8, k8, v8, qs, ks, vs)
    indices = torch.tensor([4, 0, 2], device="cuda", dtype=torch.int32).expand(batch, heads, 2, 3).contiguous()
    counts = torch.tensor([3, 0, 1, 2, 2, 3, 0, 1], device="cuda", dtype=torch.int32).view(batch, heads, 2)
    sizes = torch.tensor([128, 128, 73, 128, 19], device="cuda", dtype=torch.int32)
    if size_rank >= 2:
        sizes = sizes.repeat(batch, 1)
        sizes[1, 0] = 91
    if size_rank == 3:
        sizes = sizes[:, None].repeat(1, heads, 1)
        sizes[:, 1, 2] = 0
    expanded_sizes = sizes.expand(batch, heads, -1) if size_rank == 1 else sizes[:, None].expand(-1, heads, -1) if size_rank == 2 else sizes
    expected, expected_lse = fp8_online_reference(
        quantized, indices, counts, 0.0625, block_size=128, block_sizes=expanded_sizes, k_quant_block_size=k_quant_block_size
    )
    if v_block_size:
        padded = torch.zeros((batch, heads, (sk + 127) // 128 * 128, dim), device="cuda", dtype=v8.dtype)
        padded[:, :, :sk] = v8
        v8 = padded.view(batch, heads, -1, 128, dim).transpose(-1, -2).contiguous()
        quantized = (q8, k8, v8, qs, ks, vs)
    actual, lse = interface._bsa_attn_fwd_sm120_fp8(
        *quantized,
        indices,
        3,
        0.0625,
        block_sizes=sizes,
        q2k_block_nums=counts,
        sparse_block_size=128,
        v_block_size=v_block_size,
        k_quant_block_size=k_quant_block_size,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=5e-3, atol=2e-3)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)
    empty_rows = counts.repeat_interleave(128, dim=-1)[..., :sq] == 0
    assert torch.count_nonzero(actual[empty_rows]) == 0
    assert torch.isneginf(lse[empty_rows]).all()


@pytest.mark.L0
@torch_fork_set_rng(seed=120128)
def test_sm120_blk128_compile_options_are_scoped_and_cached(monkeypatch):
    """Compiler tuning must neither affect blk64 nor alias an untuned callable."""
    ops = require_sm120_fp8()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    native = importlib.import_module("cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8")
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    kernel = native.BlockSparseAttnForwardFp8Sm120Blk128
    tuned_options = '--ptxas-options="--register-usage-level=2"'
    assert getattr(kernel(v_block_size=0), "_compile_options", "") == ""
    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 16384, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = torch.arange(128, device="cuda", dtype=torch.int32).expand(1, 1, 2, 128).contiguous()
    quantized = _quantize_sage_bhsd(q, k, v, v_block_size=128)
    monkeypatch.setattr(interface._bsa_attn_fwd_sm120_fp8, "compile_cache", {})
    original_compile = interface.cute.compile
    compiled_options = []
    executed_options = []

    def record_compile(instance, *args, **kwargs):
        if isinstance(instance, kernel):
            compiled_options.append(kwargs.get("options", ""))
        elif instance.__class__.__name__ == "BlockSparseAttnForwardFp8Sm120Blk64":
            assert not kwargs.get("options", "")
        compiled = original_compile(instance, *args, **kwargs)
        if isinstance(instance, kernel):

            def record_execute(*runtime_args):
                executed_options.append(kwargs.get("options", ""))
                return compiled(*runtime_args)

            return record_execute
        return compiled

    monkeypatch.setattr(interface.cute, "compile", record_compile)
    results = []
    for options in (tuned_options, "", tuned_options):
        monkeypatch.setattr(kernel, "_compile_options", options, raising=False)
        results.append(interface._bsa_attn_fwd_sm120_fp8(*quantized, indices, 128, 128**-0.5, sparse_block_size=128, v_block_size=128))
    torch.cuda.synchronize()
    assert compiled_options == [tuned_options, ""]
    assert executed_options == [tuned_options, "", tuned_options]
    assert len(interface._bsa_attn_fwd_sm120_fp8.compile_cache) == 2
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    monkeypatch.setattr(kernel, "_compile_min_blocks", 128, raising=False)
    for topk in (1, 127):
        short_indices = indices[..., :topk].contiguous()
        interface._bsa_attn_fwd_sm120_fp8(*quantized, short_indices, topk, 128**-0.5, sparse_block_size=128, v_block_size=128)
        assert executed_options[-1] == ""
    counts = torch.full((1, 1, 2), 128, device="cuda", dtype=torch.int32)
    variable_result = interface._bsa_attn_fwd_sm120_fp8(*quantized, indices, 128, 128**-0.5, q2k_block_nums=counts, sparse_block_size=128, v_block_size=128)
    assert executed_options[-1] == ""
    for actual, expected in zip(variable_result, results[0]):
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    indices64 = torch.zeros((1, 1, 4, 1), device="cuda", dtype=torch.int32)
    ops.block_sparse_attention_fp8_forward(q, k, v, indices64, sparse_block_size=64)
    torch.cuda.synchronize()


@pytest.mark.L0
def test_sm120_blk128_never_launches_blk64(monkeypatch):
    ops = require_sm120_fp8()
    old = importlib.import_module("cudnn.block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8")
    native = importlib.import_module("cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8")

    def forbid_blk64(*args, **kwargs):
        raise AssertionError("native blk128 must not construct a blk64 attention kernel")

    monkeypatch.setattr(old.BlockSparseAttnForwardFp8Sm120Blk64, "__init__", forbid_blk64)
    kernel = native.BlockSparseAttnForwardFp8Sm120Blk128()
    assert kernel.tile_shape_qk == (128, 128, 128)
    assert kernel.tile_shape_pv == (128, 128, 128)
    assert kernel.num_threads == 384
    assert kernel.num_mma_warps == 8
    assert kernel.num_compute_threads == 256
    assert kernel.compute_registers == 240 and kernel.load_registers == 24
    assert 256 * kernel.compute_registers + 128 * kernel.load_registers == 64512
    q = torch.ones((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    actual = ops.block_sparse_attention_fp8_forward(q, q, q, indices, sparse_block_size=128)["o_tensor"]
    torch.testing.assert_close(actual, q, atol=0.01, rtol=0.01)


@pytest.mark.L0
@torch_fork_set_rng(seed=128)
def test_sm120_blk128_graph_replay_uses_runtime_metadata(monkeypatch):
    ops = require_sm120_fp8()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k, v = torch.randn_like(q), torch.randn_like(q)
    indices = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)

    def run():
        return ops.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=128)["o_tensor"]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    def forbid_compile(*args, **kwargs):
        raise AssertionError("warmed native FP8 must reuse its compile cache")

    monkeypatch.setattr(interface.cute, "compile", forbid_compile)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run()
    graph.replay()
    first = actual.clone()
    indices.fill_(1)
    graph.replay()
    expected = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, first)


@pytest.mark.L0
@pytest.mark.parametrize("v_block_size", (0, 128))
@torch_fork_set_rng(seed=12817)
def test_sm120_blk128_producer_replay_with_mixed_counts(v_block_size):
    """Exercise startup handoffs, empty CTAs, and repeated pipeline reuse."""
    require_sm120_fp8()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    sq, sk, heads, topk = 512, 20 * 128 + 19, 2, 17
    q8 = torch.zeros((1, heads, sq, 128), device="cuda").to(torch.float8_e4m3fn)
    k8 = torch.zeros((1, heads, sk, 128), device="cuda").to(torch.float8_e4m3fn)
    v8 = torch.randint(-3, 4, k8.shape, device="cuda").to(torch.float8_e4m3fn)
    qs = torch.ones((1, heads, sq), device="cuda")
    ks = torch.ones((1, heads, (sk + 15) // 16), device="cuda")
    vs = torch.linspace(0.5, 1.5, heads * 128, device="cuda").view(heads, 128)
    indices = torch.stack([torch.randperm(21, device="cuda")[:topk] for _ in range(heads * 4)]).view(1, heads, 4, topk).int()
    counts = torch.tensor([0, 1, 2, 17, 17, 2, 1, 0], device="cuda", dtype=torch.int32).view(1, heads, 4)
    kernel_v = v8
    if v_block_size:
        padded = torch.zeros((1, heads, (sk + 127) // 128 * 128, 128), device="cuda", dtype=v8.dtype)
        padded[:, :, :sk] = v8
        kernel_v = padded.view(1, heads, -1, 128, 128).transpose(-1, -2).contiguous()

    def run():
        return interface._bsa_attn_fwd_sm120_fp8(
            q8, k8, kernel_v, qs, ks, vs, indices, topk, 128**-0.5, q2k_block_nums=counts, sparse_block_size=128, v_block_size=v_block_size
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, lse = run()

    for iteration in range(4):
        counts.copy_(counts.flip(-1))
        for _ in range(5):
            graph.replay()
        expected = torch.zeros_like(actual, dtype=torch.float32)
        expected_lse = torch.full_like(lse, -torch.inf)
        for head in range(heads):
            for qb in range(4):
                count = int(counts[0, head, qb])
                if count:
                    blocks = indices[0, head, qb, :count].long()
                    tokens = (blocks[:, None] * 128 + torch.arange(128, device="cuda")).flatten()
                    tokens = tokens[tokens < sk]
                    expected[0, head, qb * 128 : (qb + 1) * 128] = v8[0, head].float()[tokens].mean(dim=0) * vs[head]
                    expected_lse[0, head, qb * 128 : (qb + 1) * 128] = torch.log(torch.tensor(tokens.numel(), device="cuda", dtype=torch.float32))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected, atol=1e-4, rtol=5e-3)
        torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.L0
@pytest.mark.parametrize("input_kind", ("levels", "rounding"))
def test_sm120_blk128_shared_p_fragment_bitwise(input_kind):
    """Check native STSM/LDSM remapping at FP8 values and rounding boundaries."""
    require_sm120_fp8()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from cudnn._cutlass_compat import SmemAllocator
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import _make_acc_into_fp8_op
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8 import _make_acc_into_fp8_smem

    @cute.kernel
    def fragment_kernel(mP: cute.Tensor, mRef: cute.Tensor, mGot: cute.Tensor, tiled_mma: cute.TiledMma, smem_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        sP = SmemAllocator().allocate_tensor(cutlass.Float8E4M3FN, smem_layout.outer, byte_alignment=128, swizzle=smem_layout.inner)
        thr_mma = tiled_mma.get_slice(tidx)
        tPrP = cute.make_rmem_tensor(thr_mma.partition_shape_C((128, 128)), cutlass.Float32)
        tPrP.store(thr_mma.partition_C(mP).load())
        tRef = _make_acc_into_fp8_op(tPrP, tiled_mma.tv_layout_A, cutlass.Float8E4M3FN)
        tGot = _make_acc_into_fp8_smem(tPrP, sP, tiled_mma, tiled_mma)
        tRef32 = cute.recast_tensor(tRef, cutlass.Int32)
        tGot32 = cute.recast_tensor(tGot, cutlass.Int32)
        for i in cutlass.range_constexpr(cute.size(tRef32)):
            mRef[tidx, i] = tRef32[i]
            mGot[tidx, i] = tGot32[i]

    @cute.jit
    def run(mP: cute.Tensor, mRef: cute.Tensor, mGot: cute.Tensor):
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (16, 8, 32)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=(128, 16, 32),
        )
        smem_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64, cutlass.Float8E4M3FN)
        smem_layout = cute.tile_to_shape(smem_atom, (128, 128), order=(0, 1))
        fragment_kernel(mP, mRef, mGot, tiled_mma, smem_layout).launch(grid=(1, 1, 1), block=(256, 1, 1), smem=16384)

    # The scaled softmax probabilities span [0, 256], including subnormals.
    levels = torch.arange(121, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    values = levels
    if input_kind == "rounding":
        midpoint = (levels[:-1] + levels[1:]) * 0.5
        values = torch.stack(
            (torch.nextafter(midpoint, torch.full_like(midpoint, -torch.inf)), midpoint, torch.nextafter(midpoint, torch.full_like(midpoint, torch.inf))),
            dim=-1,
        ).flatten()
    probabilities = values.repeat((128 * 128 + values.numel() - 1) // values.numel())[: 128 * 128].reshape(128, 128)
    reference = torch.empty((256, 16), device="cuda", dtype=torch.int32)
    actual = torch.empty_like(reference)
    tensors = [from_dlpack(tensor, assumed_align=16) for tensor in (probabilities, reference, actual)]
    compiled = cute.compile(run, *tensors)
    compiled(*tensors)
    torch.cuda.synchronize()
    assert torch.equal(actual, reference)


@pytest.mark.L0
@pytest.mark.parametrize("stream_chunk", (0, 32, 64))
def test_sm120_blk128_ordered_softmax_and_rescale_bitwise(stream_chunk):
    """Preserve ordered FP32 sums and identity scaling, including subnormals."""
    require_sm120_fp8()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import (
        _online_softmax_fp8,
        _rescale_o_for_next_acc_fp8,
    )
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8 import (
        _normalize_softmax_chunk_fp8,
        _online_softmax_ordered_fp8,
        _prepare_softmax_streaming_fp8,
        _rescale_o_if_needed_fp8,
    )
    from cudnn.block_sparse_attention.csrc.utils import layout_utils

    @cute.kernel
    def helper_kernel(
        mS: cute.Tensor,
        mO: cute.Tensor,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mPResult: cute.Tensor,
        mOResult: cute.Tensor,
        mStats: cute.Tensor,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        case, _, _ = cute.arch.block_idx()
        thr_mma = tiled_mma.get_slice(tidx)
        tScS = thr_mma.partition_C(cute.make_identity_tensor((128, 128)))
        tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
        for implementation in cutlass.range_constexpr(2):
            tSrS = cute.make_rmem_tensor(thr_mma.partition_shape_C((128, 128)), cutlass.Float32)
            tOrO = cute.make_rmem_tensor_like(tSrS)
            tSrS.store(thr_mma.partition_C(mS[case, None, None]).load())
            tOrO.store(thr_mma.partition_C(mO).load())
            tMax = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Float32)
            tSum = cute.make_rmem_tensor_like(tMax)
            tQScale = cute.make_rmem_tensor_like(tMax)
            tKScale = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
            for m in cutlass.range_constexpr(2):
                row = tScS_mn[m, 0][0]
                tMax[m] = mMax[case, row]
                tSum[m] = mSum[case, row, tidx % 4]
                tQScale[m] = mQScale[row]
            for kg in cutlass.range_constexpr(8):
                tKScale[kg] = mKScale[case, kg]
            if cutlass.const_expr(implementation == 0):
                tScale = _online_softmax_fp8(tiled_mma, tSrS, tMax, tSum, tQScale, 8.0, tKScale)
                _rescale_o_for_next_acc_fp8(tiled_mma, tOrO, tScale)
            else:
                if cutlass.const_expr(stream_chunk == 0):
                    tScale = _online_softmax_ordered_fp8(tiled_mma, tSrS, tMax, tSum, tQScale, 8.0, tKScale)
                else:
                    tScale, scaled_max = _prepare_softmax_streaming_fp8(tSrS, tMax, tSum, tQScale, 8.0)
                    for chunk in cutlass.range_constexpr(128 // stream_chunk):
                        groups = cute.size(tSrS, mode=[2]) // (128 // stream_chunk)
                        scores_part = cute.make_tensor(
                            tSrS.iterator + chunk * groups * tSrS.stride[2],
                            cute.make_layout((tSrS.shape[0], tSrS.shape[1], groups), stride=tSrS.stride),
                        )
                        _normalize_softmax_chunk_fp8(scores_part, scaled_max, tSum, tQScale)
                _rescale_o_if_needed_fp8(tOrO, tScale)
            thr_mma.partition_C(mPResult[case, implementation, None, None]).store(tSrS.load())
            thr_mma.partition_C(mOResult[case, implementation, None, None]).store(tOrO.load())
            for m in cutlass.range_constexpr(2):
                mStats[case, implementation, tidx, m, 0] = tMax[m]
                mStats[case, implementation, tidx, m, 1] = tSum[m]
                mStats[case, implementation, tidx, m, 2] = tScale[m]

    @cute.jit
    def run(
        mS: cute.Tensor,
        mO: cute.Tensor,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mPResult: cute.Tensor,
        mOResult: cute.Tensor,
        mStats: cute.Tensor,
    ):
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (16, 8, 32)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=(128, 16, 32),
        )
        helper_kernel(mS, mO, mMax, mSum, mQScale, mKScale, mPResult, mOResult, mStats, tiled_mma).launch(grid=(4, 1, 1), block=(256, 1, 1))

    generator = torch.Generator(device="cuda").manual_seed(120128)
    scores = torch.randn((4, 128, 128), device="cuda", generator=generator) * 128
    scores[2, :, 93:] = -torch.inf
    scores[3] = -torch.inf
    previous_max = torch.full((4, 128), -torch.inf, device="cuda")
    previous_max[0] = 1e6  # Every lane can take the identity-rescale path.
    previous_max[2, ::2] = 1000  # Mixed changed/unchanged rows in every warp.
    previous_max[2, 1::2] = 256
    previous_sum = torch.rand((4, 128, 4), device="cuda", generator=generator) * 8192
    previous_sum[1] = 0
    previous_sum[3] = 0
    q_scale = torch.rand((128,), device="cuda", generator=generator) * 0.01 + 0.001
    k_scale = torch.rand((4, 8), device="cuda", generator=generator) + 0.125
    if stream_chunk:
        # The streaming helper sees already-descaled per-token scores.
        k_scale.fill_(1.0)
    bit_patterns = torch.tensor(
        [0, -2147483648, 1, -2147483647, 8388607, -2139095041, 8388608, -2139095040, 1065353216, -1082130432],
        device="cuda",
        dtype=torch.int32,
    )
    output_values = bit_patterns.view(torch.float32).repeat((128 * 128 + 9) // 10)[: 128 * 128].reshape(128, 128)
    probabilities = torch.empty((4, 2, 128, 128), device="cuda")
    outputs = torch.empty_like(probabilities)
    statistics = torch.empty((4, 2, 256, 2, 3), device="cuda")
    tensors = [
        from_dlpack(tensor, assumed_align=16)
        for tensor in (scores, output_values, previous_max, previous_sum, q_scale, k_scale, probabilities, outputs, statistics)
    ]
    compiled = cute.compile(run, *tensors)
    compiled(*tensors)
    torch.cuda.synchronize()
    for result in (probabilities, outputs, statistics):
        # Integer views also check the sign of zero and exact subnormal bits.
        assert torch.equal(result[:, 0].view(torch.int32), result[:, 1].view(torch.int32))


@pytest.mark.L0
@torch_fork_set_rng(seed=28128)
def test_sm120_per_token_public_api_and_replay():
    ops = require_sm120_fp8(per_token=True)
    q, k, v = [torch.randn((1, 2, 257, 128), device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(3)]
    indices = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32).expand(1, 2, 3, 3).contiguous()
    kwargs = dict(sparse_block_size=128, k_quant_block_size=1)
    result = ops.block_sparse_attention_fp8_forward(q, k, v, indices, **kwargs)["o_tensor"]
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(result.float(), reference.float(), rtol=0.05, atol=0.015)
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = ops.block_sparse_attention_fp8_forward(q, k, v, indices, **kwargs)["o_tensor"]
        k.mul_(0.5)
        v.mul_(-2)
        captured.fill_(float("nan"))
        graph.replay()
        expected = ops.block_sparse_attention_fp8_forward(q, k, v, indices, **kwargs)["o_tensor"]
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="SM120 blk128"):
        ops.block_sparse_attention_fp8_forward(q, k, v, indices, k_quant_block_size=1)
    for invalid in (0, 2, True, 1.0):
        with pytest.raises(ValueError, match="k_quant_block_size"):
            ops.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=128, k_quant_block_size=invalid)


@pytest.mark.L0
@pytest.mark.parametrize("sk", (256, 257, 258))
@pytest.mark.parametrize("scale_layout", ("offset", "padded_heads"))
@torch_fork_set_rng(seed=28129)
def test_sm120_per_token_scale_views(sk, scale_layout):
    """Vectorized loads must preserve supported views with four-byte alignment."""
    require_sm120_fp8(per_token=True)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q = torch.randn((2, 2, 147, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k, v = [torch.randn((2, 2, sk, 128), device="cuda", dtype=q.dtype) * 0.5 for _ in range(2)]
    packed = list(_quantize_sage_bhsd(q, k, v, v_block_size=128, k_quant_block_size=1))
    topk = (sk + 127) // 128
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).expand(2, 2, 2, topk).contiguous()
    kwargs = dict(sparse_block_size=128, v_block_size=128, k_quant_block_size=1)
    expected = interface._bsa_attn_fwd_sm120_fp8(*packed, indices, topk, 128**-0.5, **kwargs)
    scales = packed[4]
    if scale_layout == "offset":
        storage = torch.empty(scales.numel() + 1, device=q.device, dtype=scales.dtype)
        view = storage[1:].view_as(scales)
    else:
        storage = torch.empty((*scales.shape[:-1], sk + 1), device=q.device, dtype=scales.dtype)
        view = storage[..., :sk]
    view.copy_(scales)
    packed[4] = view
    actual = interface._bsa_attn_fwd_sm120_fp8(*packed, indices, topk, 128**-0.5, **kwargs)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize(
    "sk, scale_layout, variable_counts",
    # The K tail and the scale view decide together whether the scales take TMA: every pair; the metadata mode
    # alternates so that every tail and every view see both.
    [(sk, layout, (i + j) % 2 == 1) for i, sk in enumerate((256, 320)) for j, layout in enumerate(("contiguous", "padded_aligned", "broadcast"))],
)
@torch_fork_set_rng(seed=28132)
def test_sm120_per_token_tma_scales_match_fallback_and_replay(monkeypatch, sk, scale_layout, variable_counts):
    """TMA scale loads share K's barrier and preserve tails and scalar views."""
    require_sm120_fp8(per_token=True)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    batch, heads, sq = 2, 2, 147
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn((batch, heads, sk, 128), device=q.device, dtype=q.dtype) for _ in range(2)]
    packed = list(_quantize_sage_bhsd(q, k, v, v_block_size=128, k_quant_block_size=1))
    scales = packed[4]
    if scale_layout == "padded_aligned":
        storage = torch.empty((*scales.shape[:-1], sk + 4), device=q.device, dtype=scales.dtype)
        scales = storage[..., :sk].copy_(scales)
    elif scale_layout == "broadcast":
        scales = scales[:1, :1].expand(batch, heads, sk)
    packed[4] = scales
    reference_storage = torch.empty(scales.numel() + 1, device=q.device, dtype=scales.dtype)
    reference_scales = reference_storage[1:].view_as(scales).copy_(scales)
    reference_packed = list(packed)
    reference_packed[4] = reference_scales
    nk = (sk + 127) // 128
    indices = torch.arange(nk - 1, -1, -1, device=q.device, dtype=torch.int32).expand(batch, heads, 2, nk).contiguous()
    kwargs = dict(sparse_block_size=128, v_block_size=128, k_quant_block_size=1)
    if variable_counts:
        counts = torch.full((batch, heads, 2), nk, device=q.device, dtype=torch.int32)
        counts[0, 0, 0] = 0
        sizes = torch.full((nk,), 73, device=q.device, dtype=torch.int32)
        sizes[0] = 0
        sizes[-1] = min(41, sk - (nk - 1) * 128)
        kwargs.update(q2k_block_nums=counts, block_sizes=sizes)
    monkeypatch.setattr(interface._bsa_attn_fwd_sm120_fp8, "compile_cache", {})
    original_compile = interface.cute.compile
    compiled_flags = []

    def record_compile(instance, *args, **kw):
        compiled_flags.append(instance.k_scale_tma)
        return original_compile(instance, *args, **kw)

    monkeypatch.setattr(interface.cute, "compile", record_compile)

    def run(inputs):
        return interface._bsa_attn_fwd_sm120_fp8(*inputs, indices, nk, 128**-0.5, **kwargs)

    expected = run(reference_packed)
    actual = run(packed)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert compiled_flags[0] is False
    assert compiled_flags[-1] is (scale_layout != "broadcast")
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = run(packed)
        # A broadcast view must update its non-overlapping source only once.
        if scale_layout == "broadcast":
            scales[:1, :1].mul_(0.75)
        else:
            scales.mul_(0.75)
        packed[5].mul_(1.5)
        reference_scales.copy_(scales)
        for result in captured:
            result.fill_(float("nan"))
        graph.replay()
        expected = run(reference_packed)
        for result, reference in zip(captured, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize(
    "tail, offset_scales",
    # An offset scale view only changes the load alignment, independently of the query tail: it alternates.
    [(0, False), (19, True)],
)
@torch_fork_set_rng(seed=28130)
def test_sm120_per_token_m64_matches_m128_and_replays(monkeypatch, tail, offset_scales):
    """M64 shares Q/P/O storage without changing KV128 softmax or metadata."""
    require_sm120_fp8(per_token=True)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    batch, heads, sq, sk = 2, 2, 128 + tail, 256 + tail
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k, v = [torch.randn((batch, heads, sk, 128), device="cuda", dtype=q.dtype) * 0.5 for _ in range(2)]
    packed = list(_quantize_sage_bhsd(q, k, v, v_block_size=128, k_quant_block_size=1))
    if offset_scales:
        scales = packed[4]
        storage = torch.empty(scales.numel() + 1, device=q.device, dtype=scales.dtype)
        packed[4] = storage[1:].view_as(scales).copy_(scales)
    nq, nk = (sq + 127) // 128, (sk + 127) // 128
    indices = torch.arange(nk - 1, -1, -1, device="cuda", dtype=torch.int32).expand(batch, heads, nq, nk).contiguous()
    kwargs = dict(sparse_block_size=128, v_block_size=128, k_quant_block_size=1)
    monkeypatch.setattr(interface._bsa_attn_fwd_sm120_fp8, "compile_cache", {})
    original_compile = interface.cute.compile
    compiled_tiles = []

    def record_compile(instance, *args, **kw):
        compiled_tiles.append(instance.tile_m)
        return original_compile(instance, *args, **kw)

    monkeypatch.setattr(interface.cute, "compile", record_compile)

    def run():
        return interface._bsa_attn_fwd_sm120_fp8(*packed, indices, nk, 128**-0.5, **kwargs)

    monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: 10000)
    expected = run()
    monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: batch * heads * nq)
    actual = run()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert compiled_tiles == [128, 64]
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = run()
        packed[4].mul_(0.75)
        packed[5].mul_(1.5)
        for result in captured:
            result.fill_(float("nan"))
        graph.replay()
        monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: 10000)
        expected = run()
        for result, reference in zip(captured, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert compiled_tiles == [128, 64]


@pytest.mark.L0
@pytest.mark.parametrize(
    "tail, offset_scales",
    # An offset scale view only changes the load alignment, independently of the wave tail: it alternates.
    [(0, False), (19, True), (64, False), (65, True)],
)
@torch_fork_set_rng(seed=28131)
def test_sm120_per_token_wave_tail_matches_m128_and_replays(monkeypatch, tail, offset_scales):
    """The two launches must cover query tails, heads and batches exactly once."""
    require_sm120_fp8(per_token=True)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    batch, heads, sq, sk = 2, 2, 128 + tail, 2176 + tail
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k, v = [torch.randn((batch, heads, sk, 128), device="cuda", dtype=q.dtype) * 0.5 for _ in range(2)]
    packed = list(_quantize_sage_bhsd(q, k, v, v_block_size=128, k_quant_block_size=1))
    if offset_scales:
        scales = packed[4]
        storage = torch.empty(scales.numel() + 1, device=q.device, dtype=scales.dtype)
        packed[4] = storage[1:].view_as(scales).copy_(scales)
    nq, nk = (sq + 127) // 128, (sk + 127) // 128
    offsets = torch.arange(batch * heads * nq, device=q.device, dtype=torch.int32).view(batch, heads, nq, 1)
    indices = ((torch.arange(16, device=q.device, dtype=torch.int32) + offsets) % nk).contiguous()
    kwargs = dict(sparse_block_size=128, v_block_size=128, k_quant_block_size=1)
    monkeypatch.setattr(interface._bsa_attn_fwd_sm120_fp8, "compile_cache", {})
    original_compile = interface.cute.compile
    compiled_dispatch = []

    def record_compile(instance, *args, **kw):
        compiled_dispatch.append((instance.tile_m, instance.wave_split))
        return original_compile(instance, *args, **kw)

    monkeypatch.setattr(interface.cute, "compile", record_compile)

    def run(**extra):
        return interface._bsa_attn_fwd_sm120_fp8(*packed, indices, 16, 128**-0.5, **kwargs, **extra)

    monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: 10000)
    expected = run()
    sms = batch * heads * nq * 3 // 4
    monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: sms)
    actual = run()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert compiled_dispatch == [(128, 0), (128, sms)]
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = run()
        packed[4].mul_(0.75)
        packed[5].mul_(1.5)
        indices.copy_((indices + 1) % nk)
        for result in captured:
            result.fill_(float("nan"))
        graph.replay()
        monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: 10000)
        expected = run()
        for result, reference in zip(captured, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert compiled_dispatch == [(128, 0), (128, sms)]

    if tail:
        # Different device wave sizes must not reuse a callable with stale
        # segment boundaries, even though tensor layouts are unchanged.
        monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: sms + 1)
        actual = run()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert compiled_dispatch[-1] == (128, sms + 1)

    counts = torch.full((batch, heads, nq), 16, device=q.device, dtype=torch.int32)
    monkeypatch.setattr(interface, "_get_device_sm_count_for_device", lambda _: sms)
    actual = run(q2k_block_nums=counts)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert compiled_dispatch[-1] == (128, 0)


@pytest.mark.L0
@torch_fork_set_rng(seed=120128)
def test_sm120_per_token_probability_chunk_cache_and_parity(monkeypatch):
    """K32/K64 schedules use distinct cache entries and identical arithmetic."""
    require_sm120_fp8(per_token=True)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    native = importlib.import_module("cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8")
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    kernel = native.BlockSparseAttnForwardFp8Sm120Blk128
    # Hold compiler flags constant so they cannot hide a missing chunk key.
    monkeypatch.setattr(kernel, "_compile_options", "")
    monkeypatch.setattr(interface._bsa_attn_fwd_sm120_fp8, "compile_cache", {})
    original_compile = interface.cute.compile
    compiled_chunks = []
    executed_chunks = []

    def record_compile(instance, *args, **kwargs):
        chunk = instance.p_tile_size
        compiled_chunks.append(chunk)
        compiled = original_compile(instance, *args, **kwargs)

        def record_execute(*runtime_args):
            executed_chunks.append(chunk)
            return compiled(*runtime_args)

        return record_execute

    q = torch.randn((1, 1, 147, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 16384, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quantized = _quantize_sage_bhsd(q, k, v, v_block_size=128, k_quant_block_size=1)
    indices = torch.arange(128, device="cuda", dtype=torch.int32).expand(1, 1, 2, 128).contiguous()
    monkeypatch.setattr(interface.cute, "compile", record_compile)
    results = []
    for topk in (128, 127, 128, 127):
        results.append(interface._bsa_attn_fwd_sm120_fp8(*quantized, indices, topk, 128**-0.5, sparse_block_size=128, v_block_size=128, k_quant_block_size=1))
    torch.cuda.synchronize()
    assert compiled_chunks == [64, 32]
    assert executed_chunks == [64, 32, 64, 32]
    for actual, expected in zip(results[2:], results[:2]):
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)

    # Runtime-count dispatch uses K32 even for 128 blocks. Compare both O and
    # LSE against K64 with exactly the same quantized operands and block list.
    counts = torch.full((1, 1, 2), 128, device="cuda", dtype=torch.int32)
    variable = interface._bsa_attn_fwd_sm120_fp8(
        *quantized,
        indices,
        128,
        128**-0.5,
        q2k_block_nums=counts,
        sparse_block_size=128,
        v_block_size=128,
        k_quant_block_size=1,
    )
    torch.cuda.synchronize()
    assert compiled_chunks == [64, 32, 32]
    for output, reference in zip(variable, results[0]):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
