# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SM120 Sage FP8 correctness and dispatch regression tests."""

import importlib

import pytest
import torch

from fe_api.bsa.bsa_reference import attention_reference, block_sparse_mask
from test_utils import torch_fork_set_rng

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native Sage FP8 tests require SM120")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    if interface._cutlass_dsl_version() < (4, 6, 1):
        pytest.skip("Sage FP8 requires CuTe DSL >= 4.6.1")
    from cudnn import BSA

    return BSA


@pytest.mark.L0
def test_sm120_fp8_revision_benchmark_preserves_baseline_layout(monkeypatch, capsys):
    """An archived blocked-V revision must not be timed in legacy BHSD mode."""
    _require_sm120()
    import json
    import sys
    from pathlib import Path

    native = importlib.import_module("cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmark" / "bsa"))
    benchmark = importlib.import_module("benchmark_sm120_fp8_revision")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_sm120_fp8_revision",
            "--baseline-kernel",
            native.__file__,
            "--sequence",
            "256",
            "--heads",
            "1",
            "--densities",
            "1",
            "--patterns",
            "strided",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ],
    )
    assert benchmark.main() == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert records[0]["v_block_sizes"] == [128, 128]
    assert len(records) == 3
    assert all(all(record["bitwise_equal"]) for record in records[1:])


@pytest.mark.L0
@pytest.mark.parametrize("block_size", (64, 128))
@pytest.mark.parametrize("tail", (0, 19))
@torch_fork_set_rng(seed=120128)
def test_sm120_fp8_native_forward(block_size, tail):
    BSA = _require_sm120()
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
    result = BSA.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=block_size)
    torch.cuda.synchronize()
    actual = result["o_tensor"]
    assert actual.dtype == torch.bfloat16 and actual.shape == q.shape and actual.is_contiguous()
    difference = (actual.float() - expected).abs()
    if block_size == 128 and tail == 0:
        indices64 = torch.stack((indices * 2, indices * 2 + 1), dim=-1).flatten(-2).repeat_interleave(2, dim=2)
        baseline = BSA.block_sparse_attention_fp8_forward(q, k, v, indices64)["o_tensor"]
        baseline_error = (baseline.float() - expected).abs().mean() / expected.abs().mean()
        assert difference.mean() / expected.abs().mean() <= baseline_error * 1.03 + 1e-4
    assert difference.max().item() < 0.2 * max(expected.abs().max().item(), 1.0)
    # The matched blk64 baseline also exceeds 4% for the aligned 384-key case.
    # Keep the original tests' tolerance unchanged and bound this case at 5%,
    # with the tighter matched-baseline regression check above.
    assert (difference.mean() / expected.abs().mean().clamp_min(1e-8)).item() < 0.05


def _online_fp8_reference(quantized, indices, counts, sizes, softmax_scale):
    """Model the native KV128 online softmax including the FP8 P cast."""
    q8, k8, v8, qs, ks, vs = quantized
    q = q8.float() * qs[..., None]
    k = k8.float() * ks.repeat_interleave(16, dim=-1)[..., : k8.shape[2], None]
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
@torch_fork_set_rng(seed=8128)
def test_sm120_fp8_blk128_scales_counts_and_partial_blocks(size_rank, v_block_size):
    _require_sm120()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    batch, heads, sq, sk, dim = 2, 2, 147, 531, 128
    q8 = torch.randint(-4, 5, (batch, heads, sq, dim), device="cuda").to(torch.float8_e4m3fn)
    k8 = torch.randint(-4, 5, (batch, heads, sk, dim), device="cuda").to(torch.float8_e4m3fn)
    v8 = torch.randint(-4, 5, (batch, heads, sk, dim), device="cuda").to(torch.float8_e4m3fn)
    qs = torch.rand((batch, heads, sq), device="cuda") * 0.1 + 0.01
    ks = torch.exp2(torch.linspace(-2, 2, (sk + 15) // 16, device="cuda")).expand(batch, heads, -1).contiguous()
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
    expected, expected_lse = _online_fp8_reference(quantized, indices, counts, expanded_sizes, 0.0625)
    if v_block_size:
        padded = torch.zeros((batch, heads, (sk + 127) // 128 * 128, dim), device="cuda", dtype=v8.dtype)
        padded[:, :, :sk] = v8
        v8 = padded.view(batch, heads, -1, 128, dim).transpose(-1, -2).contiguous()
        quantized = (q8, k8, v8, qs, ks, vs)
    actual, lse = interface._bsa_attn_fwd_sm120_fp8(
        *quantized, indices, 3, 0.0625, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, v_block_size=v_block_size
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=5e-3, atol=2e-3)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)
    empty_rows = counts.repeat_interleave(128, dim=-1)[..., :sq] == 0
    assert torch.count_nonzero(actual[empty_rows]) == 0
    assert torch.isneginf(lse[empty_rows]).all()


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
    expected = (v * scales.reciprocal().to(torch.bfloat16)).to(torch.float8_e4m3fn)
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
    BSA = _require_sm120()
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
    actual = BSA.block_sparse_attention_fp8_forward(q, q, q, indices, sparse_block_size=128)["o_tensor"]
    torch.testing.assert_close(actual, q, atol=0.01, rtol=0.01)


@pytest.mark.L0
@pytest.mark.parametrize(("arch", "block_size", "error", "message"), ((100, 128, NotImplementedError, "requires SM120"), (120, 32, ValueError, "64 or 128")))
def test_sm120_fp8_blk128_support_validation(monkeypatch, arch, block_size, error, message):
    BSA = _require_sm120()
    api = importlib.import_module("cudnn.block_sparse_attention.api")
    monkeypatch.setattr(api, "_device_arch", lambda tensor: arch)
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(error, match=message):
        BSA.block_sparse_attention_fp8_forward(q, q, q, indices, sparse_block_size=block_size)


@pytest.mark.L0
def test_sm120_fp8_blk128_dsl_gate_precedes_quantizer_import(monkeypatch):
    _require_sm120()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.cutlass, "__version__", "4.6.0")
    with pytest.raises(RuntimeError, match=r"requires nvidia-cutlass-dsl>=4\.6\.1"):
        interface.bsa_fp8_blk128_fwd(object(), object(), object(), object(), 1)


@pytest.mark.L0
@torch_fork_set_rng(seed=128)
def test_sm120_fp8_blk128_graph_replay_uses_runtime_metadata(monkeypatch):
    BSA = _require_sm120()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k, v = torch.randn_like(q), torch.randn_like(q)
    indices = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)

    def run():
        return BSA.block_sparse_attention_fp8_forward(q, k, v, indices, sparse_block_size=128)["o_tensor"]

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

    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import _make_acc_into_fp8_op
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fp8 import _make_acc_into_fp8_smem

    @cute.kernel
    def fragment_kernel(mP: cute.Tensor, mRef: cute.Tensor, mGot: cute.Tensor, tiled_mma: cute.TiledMma, smem_layout: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        sP = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float8E4M3FN, smem_layout.outer, byte_alignment=128, swizzle=smem_layout.inner)
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
