# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100/SM103 FP8 blk128 integration and upstream numerical regressions.

Numerical cases adapted from NVIDIA-JerryChen/Block-Sparse-Attention f953680.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture
def require_sm100():
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("requires SM100/SM103")
    from cudnn.block_sparse_attention import _interface

    if _interface._cutlass_dsl_version() < (4, 6, 2):
        pytest.skip("requires CuTe DSL >= 4.6.2")


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


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("sizes_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_fp8_blk128_native_tails_and_metadata(sizes_mode, variable):
    torch.manual_seed(2100 + sizes_mode)
    # Alternate all dynamic extents in one process, including one-token tails,
    # odd topk, topk=1, and batch/head strides that are not multiples of 16.
    for batch, heads, sq, sk, topk in [
        (1, 1, 1, 1, 1),
        (2, 3, 129, 641, 5),
        (1, 5, 257, 259, 3),
        (3, 1, 63, 145, 1),
    ]:
        q, k, v = make_inputs(batch, heads, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(batch, heads, sq, sk, topk, variable=variable, sizes_mode=sizes_mode)
        quantized = list(quantize_sage_bhsd(q, k, v))
        # Flattened scales are part of the blk64 API contract too.
        quantized[3:] = [t.flatten() for t in quantized[3:]]
        out = bsa_fp8_fwd(*quantized, index, topk, sparse_block_size=128, block_sizes=sizes, q2k_block_nums=nums)
        ref = sparse_reference(*dequantize(quantized), index, topk, nums, sizes)
        assert_accuracy(out, ref)


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("sizes_mode", [1, 2, 3])
@pytest.mark.parametrize("topk", [1, 2])
def test_fp8_blk128_provided_extent_respects_physical_tail(sizes_mode, topk):
    """Provided extents cannot make padded tokens participate in softmax."""
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


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("topk", [1, 3, 5, 17, 33])
def test_fp8_blk128_scales_and_online_rescaling(topk):
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


@pytest.mark.usefixtures("require_sm100")
def test_fp8_blk128_channel_scales_and_empty_tiles():
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


@pytest.mark.usefixtures("require_sm100")
def test_fp8_blk128_mixed_row_rescaling():
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


@pytest.mark.usefixtures("require_sm100")
def test_fp8_blk128_unused_capacity_and_fully_masked_blocks():
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


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("flatten", [False, True])
def test_fp8_blk128_scale_views_and_graph_stream(flatten):
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
    graph = torch.cuda.CUDAGraph()
    try:
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
    finally:
        graph.reset()


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("kv_splits", [1, 2, 4, 8])
@pytest.mark.parametrize("sizes_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_split_dynamic_metadata(kv_splits, sizes_mode, variable):
    torch.manual_seed(3700 + sizes_mode)
    # Change every dynamic extent while reusing the compiled variant. Include
    # empty splits, odd counts, physical tails, and poisoned inactive indices.
    for batch, heads, sq, sk, topk in [
        (1, 1, 1, 1, 1),
        (2, 3, 257, 913, 7),
        (1, 2, 389, 1153, 9),
    ]:
        q, k, v = make_inputs(batch, heads, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(batch, heads, sq, sk, topk, variable=variable, sizes_mode=sizes_mode)
        quantized = quantize_sage_bhsd(q, k, v)
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


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("kv_splits", [2, 3, 4, 8])
def test_split_rescaling_and_zero_extents(kv_splits):
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


@pytest.mark.parametrize("arch", [100, 103])
@pytest.mark.parametrize("splits", [1, 3])
def test_public_dispatch_and_metadata(monkeypatch, arch, splits):
    from cudnn.block_sparse_attention import api, _interface, _fp8_quant

    monkeypatch.setattr(api, "_device_arch", lambda tensor: arch)
    monkeypatch.setattr(_interface, "_get_device_arch_for_device", lambda device: arch)
    q, k, v = make_inputs(2, 3, 129, 259)
    index, nums, sizes = make_metadata(2, 3, 129, 259, 3, variable=True, sizes_mode=3)
    quantized = tuple(object() for _ in range(6))
    expected = object()
    calls = []

    def quantize(a, b, c, **kwargs):
        assert a is q and b is k and c is v and not kwargs
        calls.append("quantize")
        return quantized

    def launch(*args):
        assert args[:6] == quantized
        assert args[6] is index and args[7:9] == (3, 0.125)
        assert args[9] is sizes and args[10] is nums and args[11] == splits
        calls.append("launch")
        return expected

    monkeypatch.setattr(_fp8_quant, "_quantize_sage_bhsd", quantize)
    monkeypatch.setattr(_interface, "_bsa_attn_fwd_sm100_fp8_blk128", launch)
    result = api.block_sparse_attention_fp8_forward(
        q,
        k,
        v,
        index,
        3,
        sizes,
        nums,
        softmax_scale=0.125,
        sparse_block_size=128,
        kv_splits=splits,
    )
    assert result["o_tensor"] is expected
    assert calls == ["quantize", "launch"]


@pytest.mark.parametrize("case", ["arch", "split_type", "split_range", "split_other_path", "scale_zero", "scale_negative", "scale_nan", "old_dsl"])
def test_public_rejects_before_quantization(monkeypatch, case):
    from cudnn.block_sparse_attention import api, _interface, _fp8_quant

    monkeypatch.setattr(api, "_device_arch", lambda tensor: 100)
    monkeypatch.setattr(_interface, "_get_device_arch_for_device", lambda device: 100)
    q, k, v = make_inputs(1, 1, 128, 256)
    index, _, _ = make_metadata(1, 1, 128, 256, 1)
    kwargs = {"sparse_block_size": 128}
    if case == "arch":
        monkeypatch.setattr(api, "_device_arch", lambda tensor: 107)
    elif case == "split_type":
        kwargs["kv_splits"] = 2.0
    elif case == "split_range":
        kwargs["kv_splits"] = 257
    elif case == "split_other_path":
        kwargs.update(sparse_block_size=64, kv_splits=2)
    elif case == "old_dsl":
        monkeypatch.setattr(_interface, "_cutlass_dsl_version", lambda: (4, 6, 1))
    else:
        kwargs["softmax_scale"] = {"scale_zero": 0.0, "scale_negative": -1.0, "scale_nan": float("nan")}[case]

    def unexpected(*args, **kwargs):
        pytest.fail("invalid arguments reached quantization")

    monkeypatch.setattr(_fp8_quant, "_quantize_sage_bhsd", unexpected)
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        api.block_sparse_attention_fp8_forward(q, k, v, index, 1, **kwargs)


@pytest.mark.usefixtures("require_sm100")
@pytest.mark.parametrize("splits", [1, 3])
def test_public_e2e_cache_reuse_and_graph(monkeypatch, splits):
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
    graph = torch.cuda.CUDAGraph()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda.set_sync_debug_mode("error")
            try:
                run(cases[0])
                run(cases[1])
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                captured = run(cases[1])
        torch.cuda.current_stream().wait_stream(stream)
        cases[1][2].zero_()
        graph.replay()
        assert torch.count_nonzero(captured) == 0
        assert len(_interface._bsa_attn_fwd_sm100_fp8_blk128.compile_cache) == cache_size
    finally:
        graph.reset()


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


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_offline_compile(arch):
    from cudnn.block_sparse_attention import _interface

    if _interface._cutlass_dsl_version() < (4, 6, 2):
        pytest.skip("requires CuTe DSL >= 4.6.2")
    result = subprocess.run(
        [sys.executable, "-c", "from block_sparse_attention.cutedsl.test_BSA_attention_fp8_sm100_blk128 import _compile_variants; _compile_variants()"],
        env={**os.environ, "CUTE_DSL_ARCH": arch},
        cwd=TEST_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sm120_blocked_v_offline_compile():
    from cudnn.block_sparse_attention import _interface

    if _interface._cutlass_dsl_version() < (4, 6, 2):
        pytest.skip("requires CuTe DSL >= 4.6.2")
    result = subprocess.run(
        [sys.executable, "-c", "from block_sparse_attention.cutedsl.test_BSA_attention_fp8_sm100_blk128 import _compile_quantizer; _compile_quantizer()"],
        env={**os.environ, "CUTE_DSL_ARCH": "sm_120a"},
        cwd=TEST_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
