# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SM100 MXFP8 scale-layout contract."""

from types import SimpleNamespace

import pytest
import torch
from cudnn.deepseek_sparse_attention.utils.sm100 import mxfp8_scale_utils
from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference

pytestmark = pytest.mark.L0


@pytest.fixture
def cuda_scale_packer():
    """Require a supported CUDA and CuTe DSL scale-packer environment."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
    from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("requires CuTe DSL")
    requirement_error = cutedsl_requirement_error("SM100 DSA MXFP8 scale packing")
    if requirement_error:
        pytest.skip(requirement_error)
    try:
        gpu_arch_flag()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_mxfp8_scale_reference_roundtrip_and_datapath_order_host_only():
    """Verify oracle round-tripping and the UMMA data-path byte order."""
    logical = torch.arange(130 * 5 * 3, dtype=torch.int64).reshape(130, 5, 3)
    packed = mxfp8_scale_reference.pack_blockscaled_scale_mkl(logical)

    assert packed.shape == (3, 256, 8)
    assert torch.equal(mxfp8_scale_reference.unpack_blockscaled_scale_mkl(packed, 130, 5), logical)
    tile = mxfp8_scale_reference.pack_blockscaled_scale_mkl(logical[:128, :4, :1]).reshape(-1)
    assert torch.equal(tile[:16], logical[[0, 32, 64, 96], :4, 0].reshape(-1))
    assert torch.equal(tile[16:32], logical[[1, 33, 65, 97], :4, 0].reshape(-1))


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="requires torch.float8_e8m0fnu")
def test_mxfp8_scale_reference_preserves_e8m0_bits_host_only():
    """Verify that the oracle preserves the raw E8M0 encoding."""
    logical_bytes = torch.arange(130 * 5 * 3, dtype=torch.int64).to(torch.uint8).reshape(130, 5, 3)
    expected = mxfp8_scale_reference.pack_blockscaled_scale_mkl(logical_bytes)
    actual = mxfp8_scale_reference.pack_blockscaled_scale_mkl(logical_bytes.view(torch.float8_e8m0fnu))
    assert actual.dtype == torch.float8_e8m0fnu
    assert torch.equal(actual.view(torch.uint8), expected)


@pytest.mark.parametrize("qhead_per_kv_head,seqlen_q", [(64, 2), (32, 4), (4, 33)])
def test_mxfp8_scale_reference_bshd_order_host_only(qhead_per_kv_head, seqlen_q):
    """Verify BSHD-to-MKL ordering for supported query-head groups."""
    batch, heads_kv, groups = 2, 2, 5
    heads_q = heads_kv * qhead_per_kv_head
    q = torch.arange(batch * seqlen_q * heads_q * groups).reshape(batch, seqlen_q, heads_q, groups)
    k = torch.arange(batch * 7 * heads_kv * groups).reshape(batch, 7, heads_kv, groups)
    q_mkl = mxfp8_scale_reference.logical_q_scale_to_mkl(q, qhead_per_kv_head)
    k_mkl = mxfp8_scale_reference.logical_k_scale_to_mkl(k)
    assert q_mkl.shape == (seqlen_q * qhead_per_kv_head, groups, batch * heads_kv)
    assert k_mkl.shape == (7, groups, batch * heads_kv)
    assert torch.equal(q_mkl[0, :, 0], q[0, 0, 0])
    assert torch.equal(q_mkl[qhead_per_kv_head, :, 0], q[0, 1, 0])
    assert torch.equal(k_mkl[6, :, 3], k[1, 6, 1])


def test_mxfp8_scale_reference_thd_padding_host_only():
    """Verify that the THD oracle inserts aligned per-sequence padding."""
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    padded_q = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_q, 32)
    padded_k = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_k, 128)
    q = torch.arange(3 * 8 * 5).reshape(3, 8, 5)
    k = torch.arange(5 * 2 * 5).reshape(5, 2, 5)
    q_mkl = mxfp8_scale_reference.logical_q_scale_to_mkl_thd(q, cu_q, padded_q, 4)
    k_mkl = mxfp8_scale_reference.logical_k_scale_to_mkl_thd(k, cu_k, padded_k)
    assert q_mkl.shape == (256, 5, 2)
    assert k_mkl.shape == (256, 5, 2)
    assert torch.equal(q_mkl[0, :, 0], q[0, 0])
    assert torch.equal(q_mkl[128, :, 0], q[2, 0])
    assert torch.equal(k_mkl[128, :, 1], k[3, 1])
    assert torch.count_nonzero(q_mkl[8:128]) == 0
    assert torch.count_nonzero(k_mkl[3:128]) == 0


def test_mxfp8_scale_reference_preserves_custom_thd_padding_host_only():
    """Verify that explicit THD padded offsets determine output capacity."""
    prefix = torch.tensor([0, 1, 2], dtype=torch.int32)
    padded = torch.tensor([0, 256, 512], dtype=torch.int32)
    q = torch.arange(2 * 64 * 4).reshape(2, 64, 4)
    k = torch.arange(2 * 4).reshape(2, 1, 4)
    q_packed = mxfp8_scale_reference.pack_q_scale_thd(q, prefix, padded, 64)
    k_packed = mxfp8_scale_reference.pack_k_scale_thd(k, prefix, padded)
    assert q_packed.shape == (1, 512 * 64, 4)
    assert k_packed.shape == (1, 512, 4)


@pytest.mark.parametrize(
    "total_tokens,num_sequences,alignment,expected",
    [
        (0, 4, 128, 0),
        (1, 4, 128, 128),
        (255, 2, 128, 384),
        (3, 2, 32, 64),
        (8192, 1, 2, 8192),
    ],
)
def test_minimal_padded_token_capacity_host_only(total_tokens, num_sequences, alignment, expected):
    """Verify the minimum capacity bound for padded THD prefixes."""
    assert mxfp8_scale_utils.minimal_padded_token_capacity(total_tokens, num_sequences, alignment) == expected


def test_runtime_packer_rejects_cpu_inputs_before_launch(monkeypatch):
    """Verify that CPU input is rejected before launcher lookup."""

    def unexpected_launch():
        """Fail if invalid CPU input reaches launcher lookup."""
        raise AssertionError("invalid input reached the CUDA launcher")

    monkeypatch.setattr(mxfp8_scale_utils, "_cuda_pack_launchers", unexpected_launch)
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        mxfp8_scale_utils.pack_q_scale_bshd(torch.empty((1, 2, 4, 4), dtype=torch.uint8), 4)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_runtime_packer_rejects_nonbyte_and_strided_inputs_before_mutation(monkeypatch):
    """Verify dtype and stride rejection without mutating caller storage."""
    output = torch.full((1, 128, 4), 0xA5, dtype=torch.uint8, device="cuda")

    def unexpected_launch():
        """Fail if invalid source storage reaches launcher lookup."""
        raise AssertionError("invalid input reached the CUDA launcher")

    monkeypatch.setattr(mxfp8_scale_utils, "_cuda_pack_launchers", unexpected_launch)
    with pytest.raises(TypeError, match="one-byte"):
        mxfp8_scale_utils.pack_k_scale_bshd(torch.empty((1, 3, 1, 4), dtype=torch.float32, device="cuda"))
    storage = torch.empty((1, 3, 1, 8), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        mxfp8_scale_utils.pack_k_scale_bshd(storage[..., ::2], out=output)
    assert torch.all(output == 0xA5)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_runtime_packer_validates_caller_storage_before_launch(monkeypatch):
    """Verify caller-output shape, dtype, and alias checks before launch."""
    q = torch.empty((1, 32, 4, 4), dtype=torch.uint8, device="cuda")
    k = torch.empty((1, 128, 1, 4), dtype=torch.uint8, device="cuda")

    def unexpected_launch():
        """Fail if invalid output storage reaches launcher lookup."""
        raise AssertionError("invalid output reached the CUDA launcher")

    monkeypatch.setattr(mxfp8_scale_utils, "_cuda_pack_launchers", unexpected_launch)
    with pytest.raises(ValueError, match="out must have shape"):
        mxfp8_scale_utils.pack_q_scale_bshd(q, 4, out=torch.empty((1, 64, 4), dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="out must have dtype"):
        mxfp8_scale_utils.pack_q_scale_bshd(q, 4, out=torch.empty((1, 128, 4), dtype=torch.int8, device="cuda"))
    with pytest.raises(ValueError, match="must not alias"):
        mxfp8_scale_utils.pack_q_scale_bshd(q, 4, out=q.view(1, 128, 4))
    with pytest.raises(ValueError, match="q_out must not alias either source"):
        mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 4, q_out=k.view(1, 128, 4))
    shared = torch.empty((1, 128, 4), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="q_out and k_out must not alias"):
        mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 4, q_out=shared, k_out=shared)
    thd_out = torch.full((1, 128, 4), 0xA5, dtype=torch.uint8, device="cuda")
    prefix = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="must be int32"):
        mxfp8_scale_utils.pack_q_scale_thd(q[0], prefix.to(torch.int64), prefix, 4, out=thd_out)
    assert torch.all(thd_out == 0xA5)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_fused_packer_rejects_known_alias_before_allocating(monkeypatch):
    """Verify known fused-output aliasing is rejected before allocation."""
    q = torch.empty((1, 32, 4, 4), dtype=torch.uint8, device="cuda")
    k = torch.empty((1, 128, 1, 4), dtype=torch.uint8, device="cuda")

    def unexpected_allocation(*_args, **_kwargs):
        """Fail if known-invalid output triggers convenience allocation."""
        raise AssertionError("known-invalid output triggered an allocation")

    monkeypatch.setattr(mxfp8_scale_utils.torch, "empty", unexpected_allocation)
    with pytest.raises(ValueError, match="q_out must not alias either source"):
        mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 4, q_out=k.view(1, 128, 4))


def test_thd_runtime_packer_requires_caller_output_host_only():
    """Verify that THD packing requires caller-owned output storage."""
    q = torch.empty((3, 8, 4), dtype=torch.uint8)
    prefix = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="out is required"):
        mxfp8_scale_utils.pack_q_scale_thd(q, prefix, prefix, 4)


@pytest.mark.usefixtures("cuda_scale_packer")
@pytest.mark.parametrize("batch,sq,sk,hkv,ratio,groups", [(2, 131, 259, 1, 64, 5), (1, 3, 7, 2, 4, 4)])
def test_cuda_bshd_packers_match_reference_and_reuse_outputs(batch, sq, sk, hkv, ratio, groups):
    """Compare CUDA BSHD packers with the oracle and verify output reuse."""
    q_cpu = torch.arange(batch * sq * hkv * ratio * groups).to(torch.uint8).reshape(batch, sq, hkv * ratio, groups)
    k_cpu = torch.arange(batch * sk * hkv * groups).to(torch.uint8).reshape(batch, sk, hkv, groups)
    q_ref, k_ref = mxfp8_scale_reference.pack_qk_scale_bshd(q_cpu, k_cpu, ratio)
    q, k = q_cpu.cuda(), k_cpu.cuda()
    q_out = torch.empty_like(q_ref, device="cuda")
    k_out = torch.empty_like(k_ref, device="cuda")

    assert mxfp8_scale_utils.pack_q_scale_bshd(q, ratio, out=q_out) is q_out
    assert mxfp8_scale_utils.pack_k_scale_bshd(k, out=k_out) is k_out
    actual_q, actual_k = mxfp8_scale_utils.pack_qk_scale_bshd(q, k, ratio, q_out=q_out, k_out=k_out)
    torch.cuda.synchronize()
    assert actual_q is q_out and actual_k is k_out
    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_cuda_mkl_packer_matches_reference():
    """Compare the CUDA MKL layout packer with the PyTorch oracle."""
    logical_cpu = torch.arange(130 * 5 * 3).to(torch.uint8).reshape(130, 5, 3)
    expected = mxfp8_scale_reference.pack_blockscaled_scale_mkl(logical_cpu)
    output = torch.empty_like(expected, device="cuda")
    assert mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical_cpu.cuda(), out=output) is output
    torch.cuda.synchronize()
    assert torch.equal(output.cpu(), expected)


@pytest.mark.usefixtures("cuda_scale_packer")
@pytest.mark.parametrize("sq,sk", [(0, 0), (0, 5), (3, 0)])
def test_cuda_fused_packer_accepts_empty_extents(sq, sk):
    """Verify fused CUDA packing for empty query or key extents."""
    q_cpu = torch.empty((1, sq, 64, 5), dtype=torch.uint8)
    k_cpu = torch.empty((1, sk, 1, 5), dtype=torch.uint8)
    q_ref, k_ref = mxfp8_scale_reference.pack_qk_scale_bshd(q_cpu, k_cpu, 64)
    q_out = torch.empty_like(q_ref, device="cuda")
    k_out = torch.empty_like(k_ref, device="cuda")
    mxfp8_scale_utils.pack_qk_scale_bshd(q_cpu.cuda(), k_cpu.cuda(), 64, q_out=q_out, k_out=k_out)
    torch.cuda.synchronize()
    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)


@pytest.mark.usefixtures("cuda_scale_packer")
@pytest.mark.parametrize("extra_rows", [0, 128])
def test_cuda_thd_packers_match_reference_and_reuse_outputs(extra_rows):
    """Compare CUDA THD packers with the oracle and preserve capacity tails."""
    ratio, hkv, groups = 4, 2, 5
    cu_q_cpu = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k_cpu = torch.tensor([0, 3, 5], dtype=torch.int32)
    padded_q_cpu = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_q_cpu, 32)
    padded_k_cpu = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_k_cpu, 128)
    q_cpu = torch.arange(3 * hkv * ratio * groups).to(torch.uint8).reshape(3, hkv * ratio, groups)
    k_cpu = torch.arange(5 * hkv * groups).to(torch.uint8).reshape(5, hkv, groups)
    q_ref = mxfp8_scale_reference.pack_q_scale_thd(q_cpu, cu_q_cpu, padded_q_cpu, ratio)
    k_ref = mxfp8_scale_reference.pack_k_scale_thd(k_cpu, cu_k_cpu, padded_k_cpu)
    q_out = torch.full(
        (q_ref.shape[0], q_ref.shape[1] + extra_rows, q_ref.shape[2]),
        0xA5,
        dtype=q_ref.dtype,
        device="cuda",
    )
    k_out = torch.full(
        (k_ref.shape[0], k_ref.shape[1] + extra_rows, k_ref.shape[2]),
        0xA5,
        dtype=k_ref.dtype,
        device="cuda",
    )

    assert mxfp8_scale_utils.pack_q_scale_thd(q_cpu.cuda(), cu_q_cpu.cuda(), padded_q_cpu.cuda(), ratio, out=q_out) is q_out
    assert mxfp8_scale_utils.pack_k_scale_thd(k_cpu.cuda(), cu_k_cpu.cuda(), padded_k_cpu.cuda(), out=k_out) is k_out
    torch.cuda.synchronize()
    assert torch.equal(q_out[:, : q_ref.shape[1]].cpu(), q_ref)
    assert torch.equal(k_out[:, : k_ref.shape[1]].cpu(), k_ref)
    assert torch.count_nonzero(q_out[:, q_ref.shape[1] :]) == 0
    assert torch.count_nonzero(k_out[:, k_ref.shape[1] :]) == 0


@pytest.mark.usefixtures("cuda_scale_packer")
def test_cuda_fused_packer_is_async_stream_safe_and_graph_capturable():
    """Verify fused packing on a side stream and in CUDA Graph capture."""
    cuda = pytest.importorskip("cuda.bindings.driver")
    q = torch.arange(129 * 64 * 5, device="cuda").to(torch.uint8).reshape(1, 129, 64, 5)
    k = torch.arange(259 * 5, device="cuda").to(torch.uint8).reshape(1, 259, 1, 5)
    q_ref, k_ref = mxfp8_scale_reference.pack_qk_scale_bshd(q.cpu(), k.cpu(), 64)
    q_out = torch.empty_like(q_ref, device="cuda")
    k_out = torch.empty_like(k_ref, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    torch.cuda.set_sync_debug_mode("error")
    try:
        mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 64, q_out=q_out, k_out=k_out, stream=cuda.CUstream(side.cuda_stream))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 64, q_out=q_out, k_out=k_out)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(q_out.cpu(), q_ref)
        assert torch.equal(k_out.cpu(), k_ref)
    finally:
        graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unavailable", ["old_dsl", "no_dsl", "unsupported_arch"])
def test_runtime_packer_reports_unavailable_fast_path_without_mutating_output(monkeypatch, unavailable):
    """Verify unavailable fast paths fail without mutating caller output."""
    from cudnn.deepseek_sparse_attention.utils import runtime
    from cudnn.frost import buffers

    q = torch.zeros((1, 3, 8, 5), dtype=torch.uint8, device="cuda")
    output = torch.full((2, 128, 8), 0xA5, dtype=torch.uint8, device="cuda")
    state = (
        (False, None)
        if unavailable == "no_dsl"
        else (
            True,
            ("nvidia-cutlass-dsl", "4.6.2" if unavailable == "old_dsl" else "4.7.1"),
        )
    )
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: state)
    if unavailable == "unsupported_arch":
        monkeypatch.setattr(runtime, "device_capability", lambda _device: (8, 0))
    mxfp8_scale_utils._cuda_packer_requirement_error.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="requires|does not support"):
            mxfp8_scale_utils.pack_q_scale_bshd(q, 4, out=output)
        assert torch.all(output == 0xA5)
    finally:
        mxfp8_scale_utils._cuda_packer_requirement_error.cache_clear()


def test_blockscaled_umma_descriptor_bitfields_host_only():
    """Verify block-scaled UMMA instruction-descriptor bit fields."""
    cutlass = pytest.importorskip("cutlass")
    mma_desc = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mma_desc")
    desc = mma_desc.make_blockscaled_instr_desc(
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
        M=256,
        N=64,
        a_major=mma_desc.Major.MN,
        b_major=mma_desc.Major.K,
        a_sf_id=3,
        b_sf_id=2,
        scale_format=1,
        is_sparse=True,
    )
    assert (desc >> 2) & 0x1 == 1
    assert (desc >> 4) & 0x3 == 2
    assert (desc >> 7) & 0x7 == mma_desc.MXF8F6F4Format.E4M3
    assert (desc >> 10) & 0x7 == mma_desc.MXF8F6F4Format.E5M2
    assert (desc >> 15) & 0x1 == mma_desc.Major.MN
    assert (desc >> 16) & 0x1 == mma_desc.Major.K
    assert (desc >> 17) & 0x3F == 64 // 8
    assert (desc >> 23) & 0x1 == 1
    assert (desc >> 24) & 0x1F == 256 // 16
    assert (desc >> 29) & 0x3 == 3
    reserved_mask = 0x3 | (1 << 3) | (1 << 6) | (0x3 << 13) | (1 << 31)
    assert desc & reserved_mask == 0


def test_blockscaled_mma_op_adapter_maps_scale_and_major_modes_host_only():
    """Verify MMA adapter mapping for scale IDs and operand major modes."""
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    mma_desc = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mma_desc")
    op = SimpleNamespace(
        a_dtype=cutlass.Float8E4M3FN,
        b_dtype=cutlass.Float8E5M2,
        shape_mnk=(128, 256, 64),
        a_major_mode=cute.nvgpu.tcgen05.mma.OperandMajorMode.K,
        b_major_mode=cute.nvgpu.tcgen05.mma.OperandMajorMode.MN,
    )
    desc = mma_desc.blockscaled_mma_op_to_idesc(op, sf_id=2)
    assert (desc >> 4) & 0x3 == 2
    assert (desc >> 15) & 0x1 == mma_desc.Major.K
    assert (desc >> 16) & 0x1 == mma_desc.Major.MN
    assert (desc >> 29) & 0x3 == 2
    assert desc == mma_desc.make_blockscaled_instr_desc(
        op.a_dtype,
        op.b_dtype,
        M=128,
        N=256,
        a_major=mma_desc.Major.K,
        b_major=mma_desc.Major.MN,
        a_sf_id=2,
        b_sf_id=2,
    )
