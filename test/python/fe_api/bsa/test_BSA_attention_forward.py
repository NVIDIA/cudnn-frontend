# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.bsa.bsa_reference import attention_reference, block_sparse_mask
from fe_api.bsa.bsa_utils import make_fixed_metadata, make_variable_metadata, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")

        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bsa_attention_forward_fixed_blocks():
    BSA = _import_bsa()
    block_size = supported_block_size()
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
    )
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    major, _ = torch.cuda.get_device_capability()
    if major == 9:
        split_result = BSA.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k,
            block_sparse_num,
            block_sizes,
            sparse_block_size=block_size,
            kv_splits=2,
        )
        torch.testing.assert_close(split_result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(split_result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
def test_bsa_attention_forward_variable_blocks_and_layout():
    BSA = _import_bsa()
    block_size = supported_block_size()
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_nums, block_sizes = make_variable_metadata(batch, heads, seqlen_q, seqlen_k, block_size)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        0,
        block_sizes,
        q2k_block_nums=block_nums,
        sparse_block_size=block_size,
    )
    mask = block_sparse_mask(q2k, 0, block_sizes, seqlen_q, seqlen_k, block_size, block_nums)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    result_bshd = BSA.block_sparse_attention_forward(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        q2k,
        0,
        block_sizes,
        q2k_block_nums=block_nums,
        sparse_block_size=block_size,
        layout="bshd",
    )
    torch.testing.assert_close(result_bshd["o_tensor"].transpose(1, 2), result["o_tensor"], atol=0, rtol=0)
    torch.testing.assert_close(result_bshd["lse_tensor"], result["lse_tensor"], atol=0, rtol=0)

    major, _ = torch.cuda.get_device_capability()
    if major in {9, 10, 11, 12}:
        empty_block_nums = block_nums.clone()
        empty_block_nums[..., 0] = 0
        empty_result = BSA.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k,
            0,
            block_sizes,
            q2k_block_nums=empty_block_nums,
            sparse_block_size=block_size,
            allow_empty_block_nums=True,
        )
        empty_mask = block_sparse_mask(q2k, 0, block_sizes, seqlen_q, seqlen_k, block_size, empty_block_nums)
        empty_o_ref, empty_lse_ref = attention_reference(q, k, v, empty_mask)
        torch.testing.assert_close(empty_result["o_tensor"].float(), empty_o_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(empty_result["lse_tensor"], empty_lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_bsa_attention_forward_gqa_without_block_sizes():
    BSA = _import_bsa()
    block_size = supported_block_size()
    batch, q_heads, kv_heads, seqlen_q, seqlen_k, dim = 1, 4, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, q_heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, kv_heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, full_block_sizes = make_fixed_metadata(batch, q_heads, seqlen_q, seqlen_k, block_size)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes=None,
        sparse_block_size=block_size,
        pack_gqa=False,
    )
    mask = block_sparse_mask(q2k, block_sparse_num, full_block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    major, _ = torch.cuda.get_device_capability()
    if major in {10, 11}:
        gqa_ratio = q_heads // kv_heads
        packed_q2k, packed_count, _ = make_fixed_metadata(batch, kv_heads, seqlen_q * gqa_ratio, seqlen_k, block_size)
        packed_result = BSA.block_sparse_attention_forward(
            q,
            k,
            v,
            packed_q2k,
            packed_count,
            block_sizes=None,
            sparse_block_size=block_size,
        )
        torch.testing.assert_close(packed_result["o_tensor"], result["o_tensor"], atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(packed_result["lse_tensor"], result["lse_tensor"], atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=4)
def test_bsa_attention_forward_sm100_blk64():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("explicit blk64 forward is specific to SM100/SM110")

    BSA = _import_bsa()
    block_size = 64
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        use_clc=False,
    )
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    split_result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        use_clc=False,
        kv_splits=2,
    )
    torch.testing.assert_close(split_result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(split_result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)
    assert result["o_tensor"].is_contiguous()
    assert split_result["o_tensor"].is_contiguous()
    assert split_result["o_tensor"].stride() == result["o_tensor"].stride()
    assert result["lse_tensor"].is_contiguous()
    assert split_result["lse_tensor"].is_contiguous()
    assert split_result["lse_tensor"].stride() == result["lse_tensor"].stride()

    q_bshd, k_bshd, v_bshd = (tensor.transpose(1, 2) for tensor in (q, k, v))
    result_bshd = BSA.block_sparse_attention_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        layout="bshd",
        use_clc=False,
    )
    split_result_bshd = BSA.block_sparse_attention_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        layout="bshd",
        use_clc=False,
        kv_splits=2,
    )
    torch.testing.assert_close(split_result_bshd["o_tensor"], result_bshd["o_tensor"], atol=3e-2, rtol=3e-2)
    assert result_bshd["o_tensor"].is_contiguous()
    assert split_result_bshd["o_tensor"].is_contiguous()
    assert split_result_bshd["o_tensor"].stride() == result_bshd["o_tensor"].stride()


@pytest.mark.L0
@torch_fork_set_rng(seed=6)
@pytest.mark.parametrize("seqlen_q", [1, 63, 65])
def test_bsa_attention_forward_sm100_blk64_combine_partial_tail_rows(seqlen_q):
    """Exercise a partial Q tile through both the producer and split combine.

    A non-multiple-of-64 sequence makes the blk64 producer's final tile partial
    and leaves invalid rows in the split-combine tile. The former must keep every
    correction warp converged through its exchange barriers; the latter must
    initialize invalid shared-LSE rows to ``-inf``.
    """
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk64 split combine is specific to SM100/SM110")

    BSA = _import_bsa()
    block_size = 64
    batch, heads, seqlen_k, dim = 1, 1, 256, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
        use_clc=False,
    )
    split_result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
        use_clc=False,
        kv_splits=2,
    )

    for actual in (result, split_result):
        torch.testing.assert_close(actual["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(actual["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
def test_bsa_attention_forward_sm100_blk64_workspace_fallback(monkeypatch):
    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    gib = 1 << 30
    fake_q = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * gib, 16 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(
        interface,
        "_blk64_split_workspace_bytes",
        lambda q, value_dim, kv_splits: kv_splits * gib // 2,
    )

    assert interface._resolve_blk64_split_workspace(fake_q, 128, 8, allow_fallback=True) == 2
    with pytest.raises(RuntimeError, match="requires about"):
        interface._resolve_blk64_split_workspace(fake_q, 128, 8, allow_fallback=False)


@pytest.mark.L0
def test_bsa_attention_forward_sm100_blk64_auto_uses_workspace_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("auto split workspace fallback is specific to SM100/SM110 blk64")

    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    q2k = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    block_sizes = torch.full((4,), 64, device="cuda", dtype=torch.int32)

    class WorkspaceFallbackCalled(Exception):
        pass

    monkeypatch.setattr(interface, "_sm100_blk64_auto_kv_splits", lambda *args, **kwargs: 2)

    def workspace_fallback(q_arg, value_dim, kv_splits, allow_fallback):
        assert q_arg is not None
        assert value_dim == 128
        assert kv_splits == 2
        assert allow_fallback is True
        raise WorkspaceFallbackCalled

    monkeypatch.setattr(interface, "_resolve_blk64_split_workspace", workspace_fallback)
    with pytest.raises(WorkspaceFallbackCalled):
        interface.bsa_attn_fwd_blk64_cutedsl(
            q,
            k,
            v,
            q2k,
            block_sizes,
            block_sparse_num=2,
            use_clc=False,
            kv_splits="auto",
        )


@pytest.mark.L0
def test_bsa_attention_forward_sm120_static_compile_key_tracks_tensor_type():
    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    contiguous = torch.empty((2, 3), dtype=torch.float32)
    different_shape = torch.empty((2, 4), dtype=torch.float32)
    different_stride = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    different_dtype = torch.empty((2, 3), dtype=torch.float16)

    key = interface._tensor_static_compile_key(contiguous)
    assert key != interface._tensor_static_compile_key(different_shape)
    assert key != interface._tensor_static_compile_key(different_stride)
    assert key != interface._tensor_static_compile_key(different_dtype)


@pytest.mark.L0
def test_bsa_attention_forward_sm120_builds_static_compile_key(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")

    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    q2k = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)

    class CompileReached(Exception):
        pass

    def fail_compile(*args, **kwargs):
        raise CompileReached

    monkeypatch.setattr(interface, "_get_device_arch", lambda: 120)
    monkeypatch.setattr(interface.bsa_attn_fwd, "compile_cache", {})
    monkeypatch.setattr(interface.cute, "compile", fail_compile)
    with pytest.raises(CompileReached):
        interface._bsa_attn_fwd_sm120_blk64(
            q,
            k,
            v,
            q2k,
            block_sparse_num=1,
        )


@pytest.mark.L0
def test_bsa_attention_forward_rejects_unsupported_sm100_head_dims(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("SM100/SM110-specific head dimension validation")

    BSA = _import_bsa()
    batch, heads, seqlen_q, seqlen_k = 1, 1, 128, 256
    q = torch.empty((batch, heads, seqlen_q, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, heads, seqlen_k, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((batch, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16)
    q2k = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    block_sizes = torch.full((2,), 128, device="cuda", dtype=torch.int32)
    error = r"SM100/SM110 blk128 forward supports .*got \(192, 128\)"

    with pytest.raises(NotImplementedError, match=error):
        BSA.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k,
            block_sparse_num=2,
            block_sizes=block_sizes,
            sparse_block_size=128,
        )

    interface = importlib.import_module("cudnn.block_sparse_attention._interface")

    def fail_if_compiled(*args, **kwargs):
        pytest.fail("unsupported head dimensions reached CuTe JIT compilation")

    monkeypatch.setattr(interface.cute, "compile", fail_if_compiled)
    with pytest.raises(NotImplementedError, match=error):
        interface.bsa_attn_fwd(
            q,
            k,
            v,
            q2k,
            block_sparse_num=2,
            block_sizes=block_sizes,
            return_lse=True,
        )


@pytest.mark.L0
def test_bsa_attention_forward_rejects_invalid_metadata():
    BSA = _import_bsa()
    block_size = supported_block_size()
    batch, heads, seqlen_q, seqlen_k, dim = 1, 1, 2 * block_size, 4 * block_size, 128
    q = torch.empty((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)

    with pytest.raises(ValueError, match="shape prefix"):
        BSA.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k.repeat(1, 2, 1, 1),
            block_sparse_num,
            block_sizes,
            sparse_block_size=block_size,
        )

    with pytest.raises(ValueError, match="same CUDA device"):
        BSA.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k.cpu(),
            block_sparse_num,
            block_sizes,
            sparse_block_size=block_size,
        )
