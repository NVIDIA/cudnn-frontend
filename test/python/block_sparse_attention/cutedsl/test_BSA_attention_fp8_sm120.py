# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SM120 Sage FP8 correctness and dispatch regression tests."""

import importlib

import pytest
import torch

from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask
from test_utils import torch_fork_set_rng

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _require_sm120(*, per_token=False):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native Sage FP8 tests require SM120")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("SM120 FP8 BSA tests")
    if error is not None:
        pytest.skip(error)
    if interface._cutlass_dsl_version() < (4, 6, 1):
        pytest.skip("Sage FP8 requires CuTe DSL >= 4.6.1")
    if per_token and interface._cutlass_dsl_version() < (4, 8, 0):
        pytest.skip("Per-token SM120 FP8 BSA requires CuTe DSL >= 4.8.0")
    from cudnn import ops

    return ops


@pytest.mark.L0
@torch_fork_set_rng(seed=120128)
def test_sm120_fp8_blk128_compile_options_are_scoped_and_cached(monkeypatch):
    """Compiler tuning must neither affect blk64 nor alias an untuned callable."""
    ops = _require_sm120()
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
@pytest.mark.parametrize("block_size", (64, 128))
@pytest.mark.parametrize("tail", (0, 19))
@torch_fork_set_rng(seed=120128)
def test_sm120_fp8_native_forward(block_size, tail):
    ops = _require_sm120()
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


def _online_fp8_reference(quantized, indices, counts, sizes, softmax_scale, k_quant_block_size=16):
    """Model the native KV128 online softmax including the FP8 P cast."""
    q8, k8, v8, qs, ks, vs = quantized
    q = q8.float() * qs[..., None]
    k = k8.float() * ks.repeat_interleave(k_quant_block_size, dim=-1)[..., : k8.shape[2], None]
    v = v8.float()
    out = torch.zeros_like(q)
    lse = torch.full(q.shape[:-1], -torch.inf, device=q.device)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for qb in range(indices.shape[2]):
                begin, end = qb * 128, min((qb + 1) * 128, q.shape[2])
                q_tile = q[b, h, begin:end]
                row_max = torch.full((end - begin, 1), -torch.inf, device=q.device)
                row_sum = torch.zeros_like(row_max)
                acc = torch.zeros_like(q_tile)
                for slot in reversed(range(int(counts[b, h, qb]))):
                    kb = int(indices[b, h, qb, slot])
                    valid = min(int(sizes[b, h, kb]), k.shape[2] - kb * 128)
                    if valid == 0:
                        continue
                    scores = (q_tile @ k[b, h, kb * 128 : kb * 128 + valid].T) * softmax_scale
                    new_max = torch.maximum(row_max, scores.amax(dim=-1, keepdim=True))
                    rescale = torch.exp(row_max - new_max)
                    p = torch.exp(scores - new_max) * 256.0
                    acc = acc * rescale + p.to(torch.float8_e4m3fn).float() @ v[b, h, kb * 128 : kb * 128 + valid]
                    row_sum = row_sum * rescale + p.sum(dim=-1, keepdim=True)
                    row_max = new_max
                out[b, h, begin:end] = torch.where(row_sum > 0, acc / row_sum, 0) * vs[h]
                lse[b, h, begin:end] = (row_max + torch.log(row_sum) - torch.log(torch.tensor(256.0, device=q.device))).squeeze(-1)
    return out, lse


@pytest.mark.L0
@pytest.mark.parametrize("size_rank", (1, 2, 3))
@pytest.mark.parametrize("v_block_size", (0, 128))
@pytest.mark.parametrize("k_quant_block_size", (1, 16))
@torch_fork_set_rng(seed=8128)
def test_sm120_fp8_blk128_scales_counts_and_partial_blocks(size_rank, v_block_size, k_quant_block_size):
    _require_sm120(per_token=k_quant_block_size == 1)
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
    expected, expected_lse = _online_fp8_reference(quantized, indices, counts, expanded_sizes, 0.0625, k_quant_block_size)
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
@pytest.mark.parametrize("sk", (1, 147, 1024))
@torch_fork_set_rng(seed=27128)
def test_sm120_fp8_per_token_quantization(sk):
    _require_sm120(per_token=True)
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
@torch_fork_set_rng(seed=28128)
def test_sm120_fp8_per_token_public_api_and_replay():
    ops = _require_sm120(per_token=True)
    q, k, v = [torch.randn((1, 2, 257, 128), device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(3)]
    indices = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32).expand(1, 2, 3, 3).contiguous()
    kwargs = dict(sparse_block_size=128, k_quant_block_size=1)
    result = ops.block_sparse_attention_fp8_forward(q, k, v, indices, **kwargs)["o_tensor"]
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(result.float(), reference.float(), rtol=0.05, atol=0.015)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = ops.block_sparse_attention_fp8_forward(q, k, v, indices, **kwargs)["o_tensor"]
        k.mul_(0.5)
        v.mul_(-2)
        captured.fill_(float("nan"))
        graph.replay()
        expected = ops.block_sparse_attention_fp8_forward(q, k, v, indices, **kwargs)["o_tensor"]
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    finally:
        graph.reset()
    with pytest.raises(NotImplementedError, match="SM120 blk128"):
        ops.block_sparse_attention_fp8_forward(q, k, v, indices, k_quant_block_size=1)
    for invalid in (0, 2, True, 1.0):
        with pytest.raises(ValueError, match="k_quant_block_size"):
            ops.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=128, k_quant_block_size=invalid)


@pytest.mark.L0
@pytest.mark.parametrize("sk", (256, 257, 258))
@pytest.mark.parametrize("scale_layout", ("offset", "padded_heads"))
@torch_fork_set_rng(seed=28129)
def test_sm120_fp8_per_token_scale_views(sk, scale_layout):
    """Vectorized loads must preserve supported views with four-byte alignment."""
    _require_sm120(per_token=True)
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
def test_sm120_fp8_per_token_dsl_gate(monkeypatch):
    ops = _require_sm120(per_token=True)
    from cudnn.frost import buffers
    from cudnn.block_sparse_attention import _fp8_quant

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))

    def forbid_quantization(*args, **kwargs):
        raise AssertionError("quantization must not run below the DSL floor")

    monkeypatch.setattr(_fp8_quant, "_quantize_sage_bhsd", forbid_quantization)
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"4\.8\.0.*4\.7\.0"):
        ops.block_sparse_attention_fp8_forward(q, q, q, indices, sparse_block_size=128, k_quant_block_size=1)


@pytest.mark.L0
@pytest.mark.parametrize("tail", (0, 19))
@pytest.mark.parametrize("offset_scales", (False, True))
@torch_fork_set_rng(seed=28130)
def test_sm120_fp8_per_token_m64_matches_m128_and_replays(monkeypatch, tail, offset_scales):
    """M64 shares Q/P/O storage without changing KV128 softmax or metadata."""
    _require_sm120(per_token=True)
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
    graph = torch.cuda.CUDAGraph()
    try:
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
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("tail", (0, 19, 64, 65))
@pytest.mark.parametrize("offset_scales", (False, True))
@torch_fork_set_rng(seed=28131)
def test_sm120_fp8_per_token_wave_tail_matches_m128_and_replays(monkeypatch, tail, offset_scales):
    """The two launches must cover query tails, heads and batches exactly once."""
    _require_sm120(per_token=True)
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
    graph = torch.cuda.CUDAGraph()
    try:
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
    finally:
        graph.reset()

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
@pytest.mark.parametrize(("batch", "heads", "sq", "sk"), ((1, 1, 128, 128), (2, 3, 147, 659), (1, 8, 257, 2176)))
@torch_fork_set_rng(seed=120128)
def test_sm120_fp8_blk128_fused_v_quantization_exact(batch, heads, sq, sk):
    _require_sm120()
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
def test_sm120_fp8_blk128_v_quantization_rounding():
    _require_sm120()
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
def test_sm120_fp8_blk128_quantizer_has_no_repack_launch():
    _require_sm120()
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


@pytest.mark.L0
def test_sm120_fp8_blk128_never_launches_blk64(monkeypatch):
    ops = _require_sm120()
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
@pytest.mark.parametrize(
    ("arch", "block_size", "error", "message"),
    ((110, 128, NotImplementedError, "requires SM90, SM100/SM103, or SM120"), (120, 32, ValueError, "64 or 128")),
)
def test_sm120_fp8_blk128_support_validation(monkeypatch, arch, block_size, error, message):
    ops = _require_sm120()
    api = importlib.import_module("cudnn.block_sparse_attention.api")
    monkeypatch.setattr(api, "_device_arch", lambda tensor: arch)
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(error, match=message):
        ops.block_sparse_attention_fp8_forward(q, q, q, indices, sparse_block_size=block_size)


@pytest.mark.L0
def test_sm120_fp8_blk128_dsl_gate_precedes_quantizer_import(monkeypatch):
    _require_sm120()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.cutlass, "__version__", "4.6.0")
    # The entry dispatches on Q's device first (SM90 / SM100 / SM120 share it); the remaining
    # operands stay opaque, so the SM120 DSL gate must still raise before any of them is read.
    q = torch.empty((1,), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"requires nvidia-cutlass-dsl>=4\.6\.1"):
        interface.bsa_fp8_blk128_fwd(q, object(), object(), object(), 1)


@pytest.mark.L0
@torch_fork_set_rng(seed=128)
def test_sm120_fp8_blk128_graph_replay_uses_runtime_metadata(monkeypatch):
    ops = _require_sm120()
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
def test_sm120_fp8_blk128_producer_replay_with_mixed_counts(v_block_size):
    """Exercise startup handoffs, empty CTAs, and repeated pipeline reuse."""
    _require_sm120()
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
def test_sm120_fp8_blk128_shared_p_fragment_bitwise(input_kind):
    """Check native STSM/LDSM remapping at FP8 values and rounding boundaries."""
    _require_sm120()
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
def test_sm120_fp8_blk128_ordered_softmax_and_rescale_bitwise():
    """Preserve ordered FP32 sums and identity scaling, including subnormals."""
    _require_sm120()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import (
        _online_softmax_fp8,
        _rescale_o_for_next_acc_fp8,
    )
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8 import (
        _online_softmax_ordered_fp8,
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
                tScale = _online_softmax_ordered_fp8(tiled_mma, tSrS, tMax, tSum, tQScale, 8.0, tKScale)
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
