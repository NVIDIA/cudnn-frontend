# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Sage FP8 quantizer (``_fp8_quant._quantize_sage_bhsd``) against independent torch recipes, bit for bit.

Q is scaled per row, K is mean-centered and scaled per 16 keys (per key with ``k_quant_block_size=1``), V per channel;
the SM90 kernels take V as BHDS and the SM120 blk128 kernel as 128 x 128 blocks.  An FP8 code may differ from the
recipe only by one adjacent code at an exact FP32 midpoint, where the reduction order decides the tie.
"""

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import assert_fp8_quantization_matches_fp32
from block_sparse_attention.cutedsl.bsa_utils import import_bsa, require_sm90_fp8, require_sm120_fp8

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _inputs(batch=1, heads=1, sq=64, sk=192):
    torch.manual_seed(529)
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    return q, k, torch.randn_like(k) * 0.5


@pytest.mark.L0
@pytest.mark.parametrize(("batch", "heads"), ((1, 4), (2, 3)))
@torch_fork_set_rng(seed=20260709)
def test_sm100_sm120_quantizer_matches_recipe(batch, heads):
    import_bsa(families=(10, 11, 12), min_cutedsl=(4, 6, 1), feature="this numerical Sage FP8 test")
    quantizer = importlib.import_module("cudnn.block_sparse_attention._fp8_quant")

    seqlen_q, seqlen_k, head_dim = 128, 192, 128
    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    actual = quantizer._quantize_sage_bhsd(q, k, v)
    torch.cuda.synchronize()

    q_scale = q.float().abs().amax(dim=-1).clamp_min(1.0e-3) / 448.0
    q_reciprocal = q_scale.to(torch.bfloat16).reciprocal().to(torch.bfloat16)
    q_fp8 = (q * q_reciprocal.unsqueeze(-1)).to(torch.bfloat16).to(torch.float8_e4m3fn)

    k_mean = k.float().mean(dim=2, keepdim=True)
    k_centered = k.float() - k_mean
    k_blocks = k_centered.reshape(batch, heads, seqlen_k // 16, 16, head_dim)
    k_scale = k_blocks.abs().amax(dim=(-1, -2)).clamp_min(1.0e-3) / 448.0
    k_fp8 = (k_blocks / k_scale[..., None, None]).to(torch.float8_e4m3fn).reshape_as(k)

    v_scale = v.abs().float().amax(dim=(0, 2)).clamp_min(1.0e-3) / 448.0
    v_reciprocal = v_scale.reciprocal()
    v_fp8 = (v.float() * v_reciprocal[None, :, None, :]).to(torch.float8_e4m3fn)

    expected = (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)
    assert all(tensor.is_contiguous() for tensor in actual)
    assert torch.equal(actual[0].float(), expected[0].float())
    assert torch.allclose(actual[1].float(), expected[1].float(), atol=4.0, rtol=0.0)
    assert_fp8_quantization_matches_fp32(actual[2], v.float() * v_reciprocal[None, :, None, :])
    assert torch.equal(actual[3], expected[3])
    assert torch.allclose(actual[4], expected[4], atol=1.0e-6, rtol=0.0)
    assert torch.equal(actual[5], expected[5])


@pytest.mark.L0
@pytest.mark.parametrize("batch,heads,sk", [(1, 1, 64), (2, 3, 128), (2, 3, 192), (1, 1, 640), (2, 3, 576), (2, 1, 32832)])
def test_sm90_v_bhds_layout_matches_bhsd_bitwise(batch, heads, sk):
    require_sm90_fp8()
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


@pytest.mark.L0
def test_sm90_v_signed_midpoints_non_power_two_scales():
    require_sm90_fp8()
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


@pytest.mark.L0
@pytest.mark.parametrize("sk", [192, 576, 32832])
def test_sm90_qk_matches_independent_recipe(sk):
    require_sm90_fp8()
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


@pytest.mark.L0
@pytest.mark.parametrize("actual,scaled,accepted", [(88, 92, True), (-128, -136, True), (88, 92.001, False), (80, 92, False)])
def test_k_midpoint_detector(actual, scaled, accepted):
    actual = torch.tensor([actual], device="cuda").to(torch.float8_e4m3fn)
    scaled = torch.tensor([scaled], device="cuda", dtype=torch.float32)
    if accepted:
        _assert_k_codes_at_midpoints(actual, scaled)
    else:
        with pytest.raises(AssertionError):
            _assert_k_codes_at_midpoints(actual, scaled)


@pytest.mark.L0
@pytest.mark.parametrize("sk", (1, 147, 1024))
@torch_fork_set_rng(seed=27128)
def test_sm120_per_token_k_quantization(sk):
    require_sm120_fp8(per_token=True)
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q = torch.randn((2, 3, 147, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 3, sk, 128), device="cuda", dtype=torch.bfloat16)
    k[..., 1::2, :] *= 0.01
    v = torch.randn_like(k)
    old = _quantize_sage_bhsd(q, k, v, v_block_size=128)
    new = _quantize_sage_bhsd(q, k, v, v_block_size=128, k_quant_block_size=1)
    # Only K's numerical recipe changes; Q, V, and their scales stay exact.
    for index in (0, 2, 3, 5):
        assert torch.equal(old[index].view(torch.uint8), new[index].view(torch.uint8))
    centered = k.float() - k.float().mean(-2, keepdim=True)
    scale = centered.abs().amax(-1).clamp_min(1e-3) / 448.0
    torch.testing.assert_close(new[4], scale, rtol=2e-6, atol=1e-8)
    expected = (centered / scale[..., None]).to(torch.float8_e4m3fn)
    actual_codes, expected_codes = new[1].float(), expected.float()
    different = actual_codes != expected_codes
    if different.any():
        # FP32 chunked mean reduction can move an exact E4M3 midpoint.
        # Prove each disagreement is one adjacent code at that midpoint;
        # a generic FP8 relative tolerance would miss a wrong scale or row.
        a, e = actual_codes[different], expected_codes[different]
        midpoint = (a + e) * 0.5
        step = torch.exp2(torch.floor(torch.log2(midpoint.abs())) - 3).clamp_min(2**-9)
        torch.testing.assert_close((a - e).abs(), step, rtol=0, atol=0)
        normalized = (centered / scale[..., None])[different]
        torch.testing.assert_close(normalized, midpoint, rtol=0, atol=1e-3)


@pytest.mark.L0
@pytest.mark.parametrize(("batch", "heads", "sq", "sk"), ((1, 1, 128, 128), (2, 3, 147, 659), (1, 8, 257, 2176)))
@torch_fork_set_rng(seed=120128)
def test_sm120_blk128_fused_v_quantization_exact(batch, heads, sq, sk):
    require_sm120_fp8()
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    expected = _quantize_sage_bhsd(q, k, v)
    actual = _quantize_sage_bhsd(q, k, v, v_block_size=128)
    torch.cuda.synchronize()
    for index, (a, b) in enumerate(zip(actual, expected)):
        if index == 2:
            assert a.shape == (batch, heads, (sk + 127) // 128, 128, 128)
            a = a.transpose(-1, -2).reshape(batch, heads, -1, 128)
            assert torch.count_nonzero(a[:, :, sk:].view(torch.uint8)) == 0
            a = a[:, :, :sk]
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    # Reuse both compiled layouts under capture with no post-quantization copy.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = _quantize_sage_bhsd(q, k, v, v_block_size=128)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(actual, replayed):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.L0
def test_sm120_blk128_v_quantization_rounding():
    require_sm120_fp8()
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.sage_v_quant import SageFp8VQuantizerSm120Blk128

    levels = torch.arange(0, 127, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoint = ((levels[:-1] + levels[1:]) * 0.5).to(torch.bfloat16)
    neighbors = torch.stack(
        (
            torch.nextafter(midpoint, torch.full_like(midpoint, -torch.inf)),
            midpoint,
            torch.nextafter(midpoint, torch.full_like(midpoint, torch.inf)),
        )
    ).flatten()
    values = torch.cat((neighbors, -neighbors, torch.tensor([0.0, -0.0], device="cuda", dtype=torch.bfloat16)))
    scales = torch.linspace(0.73, 1.37, 128, device="cuda").view(1, 128)
    v = values.repeat((128 * 128 + values.numel() - 1) // values.numel())[: 128 * 128].view(1, 1, 128, 128)
    expected = (v.float() * scales.reciprocal()).to(torch.float8_e4m3fn)
    actual = torch.empty((1, 1, 1, 128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    tensors = tuple(from_dlpack(t.view(-1), assumed_align=16) for t in (v, actual, scales))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (*tensors, cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(128), stream)
    compiled = cute.compile(SageFp8VQuantizerSm120Blk128(), *args)
    compiled(*args)
    torch.cuda.synchronize()
    unpacked = actual.transpose(-1, -2).reshape_as(v)
    assert torch.equal(unpacked.contiguous().view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.L0
def test_sm120_blk128_quantizer_has_no_repack_launch():
    require_sm120_fp8()
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd

    q = torch.ones((1, 1, 257, 128), device="cuda", dtype=torch.bfloat16)
    for block in (0, 128):
        _quantize_sage_bhsd(q, q, q, v_block_size=block)
    torch.cuda.synchronize()
    for block in (0, 128):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            _quantize_sage_bhsd(q, q, q, v_block_size=block)
            torch.cuda.synchronize()
        launches = [event for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        assert len(launches) == 7, [event.name for event in launches]
