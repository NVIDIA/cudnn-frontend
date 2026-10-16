# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper Sage FP8 (blk64 and blk128) layout, numerics, and asynchronous launch contracts.

Tests that hold for both block sizes are parametrized by `block`; the blk64 schedule choice and the blk128 fragment
layout checks stay separate because only one kernel has that behavior.
"""

import importlib
import math

import pytest
import torch

from block_sparse_attention.cutedsl.bsa_reference import assert_fp8_quantization_matches_fp32

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

with_block = pytest.mark.parametrize("block", [pytest.param(64, id="blk64"), pytest.param(128, id="blk128")])


@pytest.fixture(autouse=True)
def require_sm90(request):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper SM90")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    block = request.node.callspec.params.get("block", 64) if hasattr(request.node, "callspec") else 64
    # blk128 needs a newer DSL than blk64.
    required = (4, 6, 2) if block == 128 else (4, 6, 1)
    if interface._cutlass_dsl_version() < required:
        pytest.skip(f"Sage FP8 blk{block} requires nvidia-cutlass-dsl>={'.'.join(map(str, required))}")


def _inputs(batch=1, heads=1, sq=64, sk=192):
    torch.manual_seed(529)
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    return q, k, torch.randn_like(k) * 0.5


def _ids(batch, heads, sq, sk, block=64):
    """Every KV block once, rotated so the first slot is not block 0."""
    order = torch.roll(torch.arange(sk // block, device="cuda", dtype=torch.int32), 1)
    return order.expand(batch, heads, sq // block, -1).contiguous()


def _launch(block):
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    return getattr(interface, f"_bsa_attn_fwd_sm90_fp8_blk{block}")


def _quantized(batch=1, heads=1, sq=64, sk=192):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    return _quantize_sage_bhsd(*_inputs(batch, heads, sq, sk), v_layout="bhds")


def _assert_reference(block, tensors, ids, count, beta=128**-0.5, atol=0.003, rtol=0.015):
    out, lse = _launch(block)(*tensors, ids, count, beta)
    ref, lse_ref = _online_reference(*tensors, ids, count, beta, block_size=block)
    torch.testing.assert_close(out.double(), ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse.double(), lse_ref, atol=1e-4, rtol=1e-5)
    return out, lse


def _online_reference(q, k, v, qs, ks, vs, indices, count, beta, block_size=64):
    """Simulate online P rounding; host reads here are reference-only."""
    b, h, sq, _ = q.shape
    output = torch.empty_like(q, dtype=torch.float64)
    lse = torch.empty((b, h, sq), device=q.device, dtype=torch.float64)
    ids = indices.cpu().tolist()
    for bi in range(b):
        for hi in range(h):
            for qb in range(sq // block_size):
                rows = slice(qb * block_size, (qb + 1) * block_size)
                maximum = torch.full((block_size, 1), -math.inf, device=q.device, dtype=torch.float64)
                total = torch.zeros_like(maximum)
                acc = torch.zeros((block_size, 128), device=q.device, dtype=torch.float64)
                for slot in reversed(range(count)):
                    block = ids[bi][hi][qb][slot]
                    cols = slice(block * block_size, (block + 1) * block_size)
                    x = q[bi, hi, rows].double() @ k[bi, hi, cols].double().T
                    x *= qs[bi, hi, rows].double()[:, None]
                    x *= ks[bi, hi, block * (block_size // 16) : (block + 1) * (block_size // 16)].double().repeat_interleave(16)[None, :]
                    x *= beta
                    new_max = torch.maximum(maximum, x.amax(-1, keepdim=True))
                    alpha = (maximum - new_max).exp()
                    p = (x - new_max).exp() * 256
                    acc = acc * alpha + p.to(torch.float8_e4m3fn).double() @ v[bi, hi, :, cols].double().T
                    total = total * alpha + p.sum(-1, keepdim=True)
                    maximum = new_max
                output[bi, hi, rows] = acc / total * vs[hi].double()[None, :]
                lse[bi, hi, rows] = (maximum + total.log() - math.log(256)).squeeze(-1)
    return output, lse


@pytest.mark.parametrize("batch,heads,sk", [(1, 1, 64), (2, 3, 128), (2, 3, 192), (1, 1, 640), (2, 3, 576), (2, 1, 32832)])
def test_v_quantization_bhds_bitwise_recipe(batch, heads, sk):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q, k, v = _inputs(batch, heads, sk=sk)
    bhsd = _quantize_sage_bhsd(q, k, v)
    bhds = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    for i in (0, 1, 3, 4, 5):
        assert torch.equal(bhsd[i].view(torch.uint8), bhds[i].view(torch.uint8))
    assert bhds[2].shape == (batch, heads, 128, sk) and bhds[2].is_contiguous()
    actual = bhds[2].transpose(-1, -2).contiguous().view(torch.uint8)
    assert torch.equal(bhsd[2].view(torch.uint8), actual)
    scale = v.float().abs().amax(dim=(0, 2)).clamp_min(1e-3) / 448
    scaled = v.float() * scale.reciprocal()[None, :, None, :]
    assert torch.equal(scale, bhds[5])
    assert_fp8_quantization_matches_fp32(bhds[2].transpose(-1, -2), scaled)


@with_block
@pytest.mark.parametrize("beta", [128**-0.5, 0.0, -0.125])
@pytest.mark.parametrize("batch,heads,q_blocks,k_blocks,count", [(1, 1, 1, 1, 1), (2, 3, 2, 3, 3), (1, 1, 1, 10, 5)])
def test_quantized_forward_and_centered_lse(block, batch, heads, q_blocks, k_blocks, count, beta):
    tensors = _quantized(batch, heads, q_blocks * block, k_blocks * block)
    _assert_reference(block, tensors, _ids(batch, heads, q_blocks * block, k_blocks * block, block), count, beta)


@with_block
@pytest.mark.parametrize("beta", [128**-0.5, 0.0, -0.125])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 17, 33, 65, 256])
def test_pipeline_lengths_and_scales(block, count, beta):
    sk = (count + 2) * block
    _assert_reference(block, _quantized(sq=block, sk=sk), _ids(1, 1, block, sk, block), count, beta)


@with_block
@pytest.mark.parametrize("count", [2, 31, 32, 33, 63, 64, 65, 97])
def test_pipeline_stage_wrap(block, count):
    sk = (count + 3) * block
    ids = torch.randperm(sk // block, device="cuda")[:count].int().reshape(1, 1, 1, count)
    _assert_reference(block, _quantized(sq=block, sk=sk), ids, count)


@pytest.mark.parametrize("scale_pattern", ["positive", "signed_q", "signed_k", "zero_k"])
@pytest.mark.parametrize(
    "block,dispatch",
    [pytest.param(64, "pipelined", id="blk64-pipelined"), pytest.param(64, "compact", id="blk64-compact"), pytest.param(128, "default", id="blk128")],
)
def test_k_scales_per_16_tokens_and_v_channel_scales(block, dispatch, scale_pattern):
    batch, heads, q_blocks, k_blocks, count = 2, 3, 2, 10, 5
    sq, sk = q_blocks * block, k_blocks * block
    q, k, v = _inputs(batch, heads, sq, sk)
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
    ids = _ids(batch, heads, sq, sk, block)
    reference, lse_ref = _online_reference(q, k, v, qs, ks, vs, ids, count, 128**-0.5, block_size=block)
    if dispatch == "compact":
        # Enough query tiles to take the large-grid schedule.
        repeats = 2 * torch.cuda.get_device_properties(0).multi_processor_count // (batch * heads * q_blocks) + 1
        q, qs, ids = q.repeat(1, 1, repeats, 1), qs.repeat(1, 1, repeats), ids.repeat(1, 1, repeats, 1)
        reference, lse_ref = reference.repeat(1, 1, repeats, 1), lse_ref.repeat(1, 1, repeats)
    out, lse = _launch(block)(q, k, v, qs, ks, vs, ids, count)
    torch.testing.assert_close(out.double(), reference, atol=0.003, rtol=0.015)
    torch.testing.assert_close(lse.double(), lse_ref, atol=1e-4, rtol=1e-5)


@with_block
def test_public_forward_and_nondefault_stream_graph(block):
    from cudnn import block_sparse_attention_fp8_forward

    launch = _launch(block)
    q, k, v = _inputs(2, 3, 2 * block, 4 * block)
    ids = _ids(2, 3, 2 * block, 4 * block, block)
    reference = ((q.double() @ k.double().transpose(-1, -2)) * 128**-0.5).softmax(-1) @ v.double()
    graph = torch.cuda.CUDAGraph()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)
            assert list(result.keys()) == ["o_tensor"]
            out = result["o_tensor"]
            assert out.dtype == torch.bfloat16 and out.is_contiguous()
            entries = dict(launch.compile_cache)
            # A runtime count and a different scale reuse the compiled kernel without a host read.
            torch.cuda.set_sync_debug_mode("error")
            try:
                block_sparse_attention_fp8_forward(q, k, v, ids, block_sparse_num=1, softmax_scale=0.0625, sparse_block_size=block)
            finally:
                torch.cuda.set_sync_debug_mode("default")
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
    finally:
        graph.reset()


COMMON_REJECT_CASES = ["q_tail", "k_tail", "block_sizes", "counts", "scale_nan", "scale_device", "count_device", "block_size", "prefix"]


@pytest.mark.parametrize(
    "block,case",
    [pytest.param(block, case, id=f"blk{block}-{case}") for block in (64, 128) for case in COMMON_REJECT_CASES]
    + [pytest.param(128, case, id=f"blk128-{case}") for case in ("arch", "version")],
)
def test_unsupported_inputs_rejected_before_quantization(block, case, monkeypatch):
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention import api, _interface, _fp8_quant

    q, k, v = _inputs(sq=block, sk=2 * block)
    ids = _ids(1, 1, block, 2 * block, block)
    kwargs = {"sparse_block_size": block}
    if case == "q_tail":
        q = q[:, :, : block - 1].contiguous()
    elif case == "k_tail":
        k, v = k[:, :, : block + block // 2].contiguous(), v[:, :, : block + block // 2].contiguous()
    elif case == "block_sizes":
        kwargs["block_sizes"] = torch.full((2,), block, dtype=torch.int32, device="cuda")
    elif case == "counts":
        kwargs["q2k_block_nums"] = torch.ones((1, 1, 1), dtype=torch.int32, device="cuda")
    elif case == "scale_nan":
        kwargs["softmax_scale"] = math.nan
    elif case == "scale_device":
        kwargs["softmax_scale"] = torch.tensor(0.1, device="cuda")
    elif case == "count_device":
        kwargs["block_sparse_num"] = torch.tensor(1, device="cuda")
    elif case == "block_size":
        kwargs["sparse_block_size"] = 256
    elif case == "arch":
        monkeypatch.setattr(api, "_device_arch", lambda q: 110)
    elif case == "version":
        monkeypatch.setattr(_interface, "_cutlass_dsl_version", lambda: (4, 6, 1))
    else:  # prefix: more index rows than query blocks
        ids = ids.repeat(1, 1, 2, 1)

    def unexpected(*args, **kw):
        pytest.fail("unsupported inputs reached quantization")

    monkeypatch.setattr(_fp8_quant, "_quantize_sage_bhsd", unexpected)
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        block_sparse_attention_fp8_forward(q, k, v, ids, **kwargs)


@with_block
def test_sync_detector_rejects_injected_device_read(block, monkeypatch):
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention import _interface

    q, k, v = _inputs(sq=block, sk=3 * block)
    ids = _ids(1, 1, block, 3 * block, block)
    block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)

    def synchronizing_launch(*args, **kwargs):
        args[3][0, 0, 0].item()  # Deliberate D2H: the detector must reject it.
        pytest.fail("synchronizing launch escaped the detector")

    monkeypatch.setattr(_interface, f"_bsa_attn_fwd_sm90_fp8_blk{block}", synchronizing_launch)
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match="synchroniz"):
            block_sparse_attention_fp8_forward(q, k, v, ids, sparse_block_size=block)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@with_block
@pytest.mark.parametrize("constant", [0.0, 2.0])
def test_zero_queries_and_constant_keys(block, constant):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q, k, v = _inputs(sq=block, sk=3 * block)
    q.zero_()
    k.fill_(constant)
    v.zero_()
    quantized = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    out, lse = _launch(block)(*quantized, _ids(1, 1, block, 3 * block, block), 3)
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(3 * block)), atol=1e-6, rtol=0)


@with_block
def test_large_topk_accumulation(block):
    sk = 128 * block
    tensors = _quantized(sq=block, sk=sk)
    _assert_reference(block, tensors, _ids(1, 1, block, sk, block), 128, atol=0.001, rtol=0.02)


@with_block
def test_changing_max_requires_add_before_rescale(block):
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
    _assert_reference(block, tensors, ids, 3, 1.0)


@with_block
def test_dynamic_geometry_and_scale_reuse(block, monkeypatch):
    from cudnn.block_sparse_attention import _interface

    _launch(block)(*_quantized(sq=block, sk=4 * block), _ids(1, 1, block, 4 * block, block), 1)

    def unexpected(*args, **kwargs):
        pytest.fail("dynamic launch arguments triggered recompilation")

    for batch, heads, q_blocks, k_blocks, count, beta in ((2, 3, 2, 6, 5, -0.125), (1, 2, 1, 1, 1, 0.0)):
        sq, sk = q_blocks * block, k_blocks * block
        with monkeypatch.context() as patch:
            patch.setattr(_interface.cute, "compile", unexpected)
            _assert_reference(block, _quantized(batch, heads, sq, sk), _ids(batch, heads, sq, sk, block), count, beta)


@with_block
def test_large_grid_multiple_waves(block):
    tensors = _quantized(sq=block, sk=6 * block)
    ids = _ids(1, 1, block, 6 * block, block)
    expected, expected_lse = _assert_reference(block, tensors, ids, 5)
    tiles = 2 * torch.cuda.get_device_properties(0).multi_processor_count + 1
    q, k, v, qs, ks, vs = tensors
    large = (q.repeat(1, 1, tiles, 1), k, v, qs.repeat(1, 1, tiles), ks, vs)
    out, lse = _launch(block)(*large, ids.repeat(1, 1, tiles, 1), 5)
    torch.testing.assert_close(out, expected.repeat(1, 1, tiles, 1), atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse.repeat(1, 1, tiles), atol=0, rtol=0)


@with_block
@pytest.mark.parametrize("count", [1, 3, 17])
def test_uniform_logits_ring_without_blas(block, count):
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
    out, lse = _launch(block)(q, k, v, qs, ks, vs, ids, count)
    expected = (values[:, ids.flatten()].double().mean(-1) * vs[0].double())[None, None, None].expand_as(out)
    torch.testing.assert_close(out.double(), expected, atol=0.008, rtol=0.005)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(count * block)), atol=1e-5, rtol=0)


@with_block
@pytest.mark.parametrize("layout", ["prefix", "stepped", "permuted", "broadcast"])
@pytest.mark.parametrize("tiles", [2, 96])
def test_strided_sparse_indices_forward_and_graph(block, layout, tiles, monkeypatch):
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    launch = _launch(block)
    # tiles=96 exceeds the large-grid threshold of the blk64 schedule, so both schedules run on the test H100.
    batch, heads, capacity = 2, 3, 5
    q, k, v = _inputs(batch, heads, tiles * block, (capacity + 3) * block)
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
                    torch.cuda.set_sync_debug_mode("error")
                    try:
                        launch(*quantized, ids, count)
                    finally:
                        torch.cuda.set_sync_debug_mode("default")
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


def test_blk128_basis_probability_and_v_layout():
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


def test_private_layout_tag_required_even_when_sk_equals_dim():
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
    from cudnn.block_sparse_attention._interface import _bsa_fp8_blk64_fwd_quantized

    quantized = _quantize_sage_bhsd(*_inputs(sk=128), v_layout="bhds")
    with pytest.raises(ValueError, match="v_layout='bhds'"):
        _bsa_fp8_blk64_fwd_quantized(*quantized, _ids(1, 1, 64, 128), 2)


def test_v_quantization_signed_midpoints_non_power_two_scales():
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q, k, v = _inputs(sk=64)
    scales = torch.tensor([1.0, 3.0, 5.0, 7.0], device="cuda").repeat(32)
    midpoints = torch.tensor([0.53125, -0.53125, 1.0625, -1.0625], device="cuda")
    v.zero_()
    v[0, 0, 0] = scales * 448
    v[0, 0, 1:5] = midpoints[:, None] * scales[None, :]
    quantized = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    expected_scale = v.float().abs().amax(dim=(0, 2)) / 448
    expected = (v.float() * expected_scale.reciprocal()[None, :, None, :]).to(torch.float8_e4m3fn)
    assert torch.equal(quantized[5], expected_scale)
    assert torch.equal(quantized[2].transpose(-1, -2).contiguous().view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("sk", [192, 576, 32832])
def test_hopper_qk_quantizer_matches_independent_recipe(sk):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q, k, v = _inputs(2, 3, 128, sk)
    q8, k8, _, qs, ks, _ = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    q_scale = q.float().abs().amax(-1).clamp_min(1e-3) / 448
    q_ref = (q * q_scale.bfloat16().reciprocal().bfloat16()[..., None]).bfloat16().to(torch.float8_e4m3fn)
    assert torch.equal(qs, q_scale)
    assert torch.equal(q8.view(torch.uint8), q_ref.view(torch.uint8))
    centered = k.float() - k.float().mean(2, keepdim=True)
    blocks = centered.reshape(2, 3, sk // 16, 16, 128)
    k_scale = blocks.abs().amax((-1, -2)).clamp_min(1e-3) / 448
    scaled = (blocks / k_scale[..., None, None]).reshape_as(k8)
    torch.testing.assert_close(ks, k_scale, atol=1e-6, rtol=0)
    _assert_k_codes_at_midpoints(k8, scaled)


def _assert_k_codes_at_midpoints(actual, scaled_reference):
    # FP32 mean reduction order can flip an exact FP8 tie. Require a single
    # adjacent code AND evidence from the unquantized reference, rather than
    # a fixed absolute tolerance (one E4M3 step is not always <= 4).
    reference = scaled_reference.to(torch.float8_e4m3fn)
    mismatch = actual.view(torch.uint8) != reference.view(torch.uint8)
    actual_codes = actual.view(torch.uint8)[mismatch].short()
    reference_codes = reference.view(torch.uint8)[mismatch].short()
    delta = (actual_codes - reference_codes).abs()
    torch.testing.assert_close(delta, torch.ones_like(delta), rtol=0, atol=0)
    midpoint = (actual.float()[mismatch] + reference.float()[mismatch]) * 0.5
    distance = (scaled_reference[mismatch] - midpoint).abs()
    bound = 4 * torch.finfo(torch.float32).eps * midpoint.abs().clamp_min(1)
    assert torch.all(distance <= bound), "K-code mismatch away from an FP8 midpoint"


@pytest.mark.parametrize("actual,scaled,accepted", [(88, 92, True), (-128, -136, True), (88, 92.001, False), (80, 92, False)])
def test_k_midpoint_detector(actual, scaled, accepted):
    actual = torch.tensor([actual], device="cuda").to(torch.float8_e4m3fn)
    scaled = torch.tensor([scaled], device="cuda", dtype=torch.float32)
    if accepted:
        _assert_k_codes_at_midpoints(actual, scaled)
    else:
        with pytest.raises(AssertionError):
            _assert_k_codes_at_midpoints(actual, scaled)


@pytest.mark.parametrize("count", [1, 3, 65, 256])
def test_blk64_schedule_boundary_reuses_compile_and_matches_reference(count):
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
    from cudnn.block_sparse_attention._interface import _bsa_attn_fwd_sm90_fp8_blk64 as launch

    # Identical query tiles let us check every output against an independent
    # online reference without doing hundreds of redundant FP64 matmuls.
    limit = 2 * torch.cuda.get_device_properties(0).multi_processor_count
    small = _quantize_sage_bhsd(*_inputs(sk=(count + 2) * 64), v_layout="bhds")
    ids = _ids(1, 1, 64, (count + 2) * 64)
    ref, lse_ref = _online_reference(*small, ids, count, 128**-0.5)
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
        torch.cuda.set_sync_debug_mode("error")
        try:
            out, lse = launch(*tensors, tile_ids, count)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.testing.assert_close(out.double(), ref.repeat(1, 1, tiles, 1), atol=0.003, rtol=0.015)
        torch.testing.assert_close(lse.double(), lse_ref.repeat(1, 1, tiles), atol=1e-4, rtol=1e-5)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                graph_out, graph_lse = launch(*tensors, tile_ids, count)
            graph.replay()
            torch.testing.assert_close(graph_out, out, atol=0, rtol=0)
            torch.testing.assert_close(graph_lse, lse, atol=0, rtol=0)
            assert launch.compile_cache == compiled
        finally:
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
