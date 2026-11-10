# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side contracts of the BSA entry points: argument checks that raise before any kernel is built or launched,
CuTe DSL version gates, architecture dispatch, split/bucket policies, workspace fallbacks, compile keys and lazy
imports.  Most patch the device architecture or the launch and run on any GPU; numerics live in the other files.
"""

import builtins
import importlib
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from cudnn.api_base import TupleDict
from block_sparse_attention.cutedsl.bsa_utils import blk128_case, import_bsa, make_fixed_metadata, require_sm90_fp8, require_sm120_fp8, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

# ---- BF16/FP16 forward


@pytest.mark.L0
def test_forward_rejects_invalid_metadata():
    ops = import_bsa()
    block_size = supported_block_size()
    batch, heads, seqlen_q, seqlen_k, dim = 1, 1, 2 * block_size, 4 * block_size, 128
    q = torch.empty((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)

    with pytest.raises(ValueError, match="shape prefix"):
        ops.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k.repeat(1, 2, 1, 1),
            block_sparse_num,
            block_sizes,
            sparse_block_size=block_size,
        )

    with pytest.raises(ValueError, match="same CUDA device"):
        ops.block_sparse_attention_forward(
            q,
            k,
            v,
            q2k.cpu(),
            block_sparse_num,
            block_sizes,
            sparse_block_size=block_size,
        )


@pytest.mark.L0
def test_forward_rejects_unsupported_sm100_head_dims_before_compile(monkeypatch):
    ops = import_bsa(families=(10, 11), reason="SM100/SM110-specific head dimension validation")
    batch, heads, seqlen_q, seqlen_k = 1, 1, 128, 256
    q = torch.empty((batch, heads, seqlen_q, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, heads, seqlen_k, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((batch, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16)
    q2k = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    block_sizes = torch.full((2,), 128, device="cuda", dtype=torch.int32)
    error = r"SM100/SM110 blk128 forward supports .*got \(192, 128\)"

    with pytest.raises(NotImplementedError, match=error):
        ops.block_sparse_attention_forward(
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
@pytest.mark.parametrize("kind", ["fp16_backward", "gqa_backward", "head_specific_backward", "unaligned_rows"])
def test_sm90_blk128_rejects_unsupported_contracts(kind):
    """The SM90 blk128 backward takes BF16 MHA with per-batch block_sizes; the forward needs 16-byte aligned rows."""
    ops = import_bsa(families=(9,), reason="the native SM90 blk128 kernels are under test")
    q, k, v, ids, counts, sizes, count = blk128_case(
        dtype=torch.float16 if kind == "fp16_backward" else torch.bfloat16,
        ratio=2 if kind == "gqa_backward" else 1,
        sizes_rank=3 if kind == "head_specific_backward" else 0,
    )
    if kind == "unaligned_rows":
        q = torch.empty((*q.shape[:-1], 129), device=q.device, dtype=q.dtype)[..., :128]
        with pytest.raises(NotImplementedError, match="aligned"):
            ops.block_sparse_attention_forward(q, k, v, ids, count, sparse_block_size=128)
    else:
        with pytest.raises((NotImplementedError, ValueError)):
            ops.block_sparse_attention_backward(
                torch.empty_like(q),
                q,
                k,
                v,
                torch.empty_like(q),
                torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
                ids,
                count,
                block_sizes=sizes,
                sparse_block_size=128,
            )


@pytest.mark.L0
def test_sm120_blk128_rejects_old_dsl_before_kernel_import(monkeypatch):
    """Reject an unsupported installed DSL before importing the FA4 kernel."""
    ops = import_bsa(archs=(120,), reason="FA4-style blk128 is specific to SM120")
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.6.2")))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """Fail if the unsupported-DSL path reaches the specialized kernel import."""
        if name == "cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fa4":
            raise AssertionError("The FA4 kernel must not be imported with an unsupported DSL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"requires nvidia-cutlass-dsl >= 4\.7\.0; found 4\.6\.2"):
        ops.block_sparse_attention_forward(q, q, q, indices, block_sparse_num=1, sparse_block_size=128)


@pytest.mark.L0
def test_sm90_blk128_rejects_old_dsl_before_kernel_import(monkeypatch):
    """Reject an unsupported installed DSL before importing the SM90 blk128 kernels."""
    ops = import_bsa(families=(9,), reason="the native SM90 blk128 kernels are under test")
    from cudnn.block_sparse_attention import _interface

    monkeypatch.setattr(_interface, "_cutlass_dsl_version", lambda: (4, 6, 1))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert "sm90_blk128.bsa_" not in name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    q, k, v, ids, _, _, count = blk128_case()
    with pytest.raises(RuntimeError, match="requires nvidia-cutlass-dsl"):
        ops.block_sparse_attention_forward(q, k, v, ids, count, sparse_block_size=128)


@pytest.mark.L0
@pytest.mark.parametrize(("seqlen_k", "expected"), [(4 * 64, False), (4 * 64 - 1, True)])
def test_sm100_blk64_detects_partial_kv_tail(seqlen_k, expected):
    import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    k = torch.empty((1, 2, seqlen_k, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)

    assert interface._sm100_blk64_has_partial_kv_tail(k, v) is expected


@pytest.mark.L0
def test_static_compile_key_tracks_tensor_type():
    import_bsa()
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
def test_sm120_blk64_builds_static_compile_key(monkeypatch):

    import_bsa()
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
@pytest.mark.parametrize(
    ("heads", "seqlen", "count", "expected_splits"),
    (
        (40, 131072, 256, 4),  # 64 MiB of K/V per head vs 50 MiB L2; about L2 / 2 per split
        (20, 262144, 512, 8),
        (10, 524288, 1024, 16),
        (4, 262144, 128, 4),  # a split keeps at least 32 blocks of each row
        (8, 131072, 256, 1),  # 64 MiB pays only with 16+ heads
        (2, 262144, 512, 1),  # fewer than 4 heads need more than 4x L2
        (2, 524288, 256, 8),  # fewer than 4 heads: above 4x L2
        (40, 65536, 128, 1),  # 32 MiB per head fits in L2
        (40, 131072, 2048, 1),  # dense lists lost 12% when split
        (40, 8192, 64, 1),  # fewer Q tiles per head than resident CTAs
        (40, 131072, 16, 1),  # too few blocks per split
    ),
)
def test_sm90_blk64_auto_split_policy(monkeypatch, heads, seqlen, count, expected_splits):
    """The SM90 policy splits a head's K/V into about L2 / 2 once it clearly exceeds L2, and declines unmeasured regimes (metadata only)."""
    import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=50 * 1024**2, multi_processor_count=132))
    q = SimpleNamespace(is_cuda=True, device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, heads, seqlen, 128))
    kv = SimpleNamespace(shape=(1, heads, seqlen, 128), element_size=lambda: 2)
    assert interface._sm90_blk64_auto_kv_splits(q, kv, kv, count) == expected_splits
    # Unmeasured regimes: GQA, non-16-bit dtype.
    gqa_kv = SimpleNamespace(shape=(1, max(heads // 2, 1), seqlen, 128), element_size=lambda: 2)
    assert interface._sm90_blk64_auto_kv_splits(q, gqa_kv, gqa_kv, count) == 1
    q_fp32 = SimpleNamespace(is_cuda=True, device=q.device, dtype=torch.float32, shape=q.shape)
    assert interface._sm90_blk64_auto_kv_splits(q_fp32, kv, kv, count) == 1


@pytest.mark.L0
def test_sm100_blk64_auto_split_counts_stay_valid(monkeypatch):
    """Automatic counts stay within the supported domain and the caller's cap."""
    import_bsa(families=(10, 11), reason="large-Q scheduler policy is specific to SM100/SM110 blk64")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q_large = torch.empty((1, 40, 131072, 1), device="cuda", dtype=torch.int8)
    q_small = torch.empty((1, 4, 128, 1), device="cuda", dtype=torch.int8)
    q2k_block_index = torch.empty((1, 1, 1, 2048), device="cuda", dtype=torch.int32)

    for q in (q_large, q_small):
        for count in (0, 1, 256, 2048):
            for cap in (1, 2, 4, 8):
                splits = interface._sm100_blk64_auto_kv_splits(q, q2k_block_index, count, max_kv_splits=cap)
                assert 1 <= splits <= min(cap, count or q2k_block_index.shape[-1])
                assert splits in {1, 2, 4, 8, 16}

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=126 * 1024**2))
    for heads in (1, 2, 4, 8):
        q_metadata = SimpleNamespace(is_cuda=True, device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, heads, 349440, 128))
        for count in (548, 900, 1089):
            kv_metadata = SimpleNamespace(shape=(1, heads, count * 10 * 64, 128), element_size=lambda: 2)
            for cap in (1, 2, 4, 8):
                splits = interface._sm100_blk64_auto_kv_splits(q_metadata, q2k_block_index, count, cap, k=kv_metadata, v=kv_metadata)
                assert 1 <= splits <= min(cap, count)
                assert splits in {1, 2, 4, 8, 16}


@pytest.mark.L0
def test_sm100_blk64_split_workspace_fallback(monkeypatch):
    import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    gib = 1 << 30
    fake_q = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * gib, 16 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(
        interface,
        "_blk64_split_workspace_bytes",
        lambda q, value_dim, kv_splits, output_dtype=None: kv_splits * gib // 2,
    )

    assert interface._resolve_blk64_split_workspace(fake_q, 128, 8, allow_fallback=True) == 2
    with pytest.raises(RuntimeError, match="requires about"):
        interface._resolve_blk64_split_workspace(fake_q, 128, 8, allow_fallback=False)


@pytest.mark.L0
def test_sm100_blk64_auto_split_uses_workspace_fallback(monkeypatch):
    import_bsa(families=(10, 11), reason="auto split workspace fallback is specific to SM100/SM110 blk64")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    q2k = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    block_sizes = torch.full((4,), 64, device="cuda", dtype=torch.int32)

    class WorkspaceFallbackCalled(Exception):
        pass

    def choose_splits(q_arg, index_arg, count_arg, *, k, v):
        assert q_arg is q
        assert index_arg is q2k
        assert count_arg == 2
        assert k.shape == v.shape == (1, 1, 256, 128)
        return 2

    monkeypatch.setattr(interface, "_sm100_blk64_auto_kv_splits", choose_splits)

    def workspace_fallback(q_arg, value_dim, kv_splits, allow_fallback, output_dtype=None):
        assert q_arg is not None
        assert value_dim == 128
        assert kv_splits == 2
        assert allow_fallback is True
        assert output_dtype is torch.bfloat16
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


# ---- backward


@pytest.mark.L0
@pytest.mark.parametrize("arch", [107, 110])
def test_backward_blk128_block_sizes_rejects_unsupported_arch_before_dispatch(monkeypatch, arch):
    """Partial blk128 KV blocks in the backward are SM100/SM103 only; other archs raise before dispatch."""
    ops = import_bsa()
    from cudnn.block_sparse_attention import _interface, api

    q = torch.zeros(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(1, 1, 128, device="cuda", dtype=torch.float32)
    ids = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.int32)
    counts = torch.ones(1, 1, 1, device="cuda", dtype=torch.int32)
    sizes = torch.ones(1, device="cuda", dtype=torch.int32)
    monkeypatch.setattr(api, "_device_arch", lambda _: arch)

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Unsupported padding reached kernel dispatch")

    monkeypatch.setattr(_interface, "bsa_attn_bwd", unexpected_dispatch)
    with pytest.raises(NotImplementedError, match=f"SM{arch} blk128 backward does not yet support block_sizes"):
        ops.block_sparse_attention_backward(q, q, q, q, q, lse, ids, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, layout="bshd")


@pytest.mark.L0
@pytest.mark.parametrize("num_q_blocks", [1, 2999, 3000, 4097, 8192, 65536])
def test_backward_sm100_blk64_auto_bucket_is_a_valid_choice(num_q_blocks):
    """The automatic bucket is a performance heuristic, so only its contract is asserted: the choice is either
    the default (``None``) or a usable bucket size."""
    from cudnn.block_sparse_attention.csrc.bwd.sm100_blk64.bsa_bwd_sm100 import sm100_bwd_auto_bucketed_k2q_size_blocks

    choice = sm100_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks)
    assert choice is None or (isinstance(choice, int) and choice >= 1)


# ---- Sage FP8 forward


@pytest.mark.L0
def test_fp8_public_exports_do_not_eagerly_import_quantizer():
    private_module = "cudnn.block_sparse_attention._fp8_quant"
    sys.modules.pop(private_module, None)

    import cudnn

    package = importlib.import_module("cudnn.block_sparse_attention")
    assert cudnn.block_sparse_attention_fp8_forward is package.block_sparse_attention_fp8_forward
    assert cudnn.BSA.block_sparse_attention_fp8_forward is package.block_sparse_attention_fp8_forward
    assert not hasattr(cudnn, "quantize_sage_bhsd")
    assert not hasattr(cudnn.BSA, "quantize_sage_bhsd")
    assert not hasattr(package, "quantize_sage_bhsd")
    assert private_module not in sys.modules


@pytest.mark.L0
@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ("4.6.1", (4, 6, 1)),
        ("4.6.1+cu13", (4, 6, 1)),
        ("4.7.0.dev0", (4, 7, 0)),
    ),
)
def test_fp8_cutedsl_version_parser(monkeypatch, version, expected):
    import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.cutlass, "__version__", version)
    assert interface._cutlass_dsl_version() == expected


@pytest.mark.L0
@pytest.mark.parametrize("version", ("4.5.0", "4.6.0"))
def test_fp8_blk64_dsl_gate_precedes_quantizer_import(monkeypatch, version):
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    private_module = "cudnn.block_sparse_attention._fp8_quant"
    sys.modules.pop(private_module, None)
    monkeypatch.setattr(interface.cutlass, "__version__", version)

    with pytest.raises(RuntimeError, match=r"nvidia-cutlass-dsl>=4\.6\.1"):
        interface.bsa_fp8_blk64_fwd(object(), object(), object(), object(), 1)
    assert private_module not in sys.modules


@pytest.mark.L0
def test_fp8_blk128_dsl_gate_precedes_quantizer_import(monkeypatch):
    require_sm120_fp8()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.cutlass, "__version__", "4.6.0")
    # The entry dispatches on Q's device first (SM90 / SM100 / SM120 share it); the remaining
    # operands stay opaque, so the SM120 DSL gate must still raise before any of them is read.
    q = torch.empty((1,), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"requires nvidia-cutlass-dsl>=4\.6\.1"):
        interface.bsa_fp8_blk128_fwd(q, object(), object(), object(), 1)


@pytest.mark.L0
def test_fp8_sm120_per_token_dsl_gate(monkeypatch):
    ops = require_sm120_fp8(per_token=True)
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
def test_fp8_interface_quantizes_before_private_launch(monkeypatch):
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface, "_require_sage_fp8_cutedsl", lambda: None)
    monkeypatch.setattr(interface, "_get_device_arch_for_device", lambda device: 100)

    q, k, v = SimpleNamespace(device=SimpleNamespace(index=0)), object(), object()
    q2k, block_sizes, q2k_nums = object(), object(), object()
    quantized = tuple(object() for _ in range(6))
    expected = object()

    fake_quantizer = ModuleType("cudnn.block_sparse_attention._fp8_quant")

    def fake_quantize(actual_q, actual_k, actual_v):
        assert (actual_q, actual_k, actual_v) == (q, k, v)
        return quantized

    fake_quantizer._quantize_sage_bhsd = fake_quantize
    monkeypatch.setitem(sys.modules, fake_quantizer.__name__, fake_quantizer)

    def fake_launch(*args, **kwargs):
        assert args[:6] == quantized
        assert args[6:] == (q2k, 7, 0.125)
        assert kwargs == {"block_sizes": block_sizes, "q2k_block_nums": q2k_nums}
        return expected

    monkeypatch.setattr(interface, "_bsa_fp8_blk64_fwd_quantized", fake_launch)
    result = interface.bsa_fp8_blk64_fwd(
        q,
        k,
        v,
        q2k,
        7,
        0.125,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_nums,
    )
    assert result is expected


@pytest.mark.L0
@pytest.mark.parametrize(("batch", "heads"), ((1, 4), (1, 3), (2, 3)))
def test_fp8_forward_accepts_positive_batch_and_head_counts(monkeypatch, batch, heads):
    ops = import_bsa()
    api = importlib.import_module("cudnn.block_sparse_attention.api")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(api, "_device_arch", lambda tensor: 100)

    shape = (batch, heads, 64, 128)
    q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    q2k = torch.zeros((batch, heads, 1, 1), device="cuda", dtype=torch.int32)
    expected = torch.empty(shape, device="cuda", dtype=torch.bfloat16)

    def fake_forward(*args, **kwargs):
        expected_args = (q, k, v, q2k)
        assert all(actual is expected_arg for actual, expected_arg in zip(args[:4], expected_args))
        assert args[4] == 1
        assert args[5] is None
        assert kwargs == {"block_sizes": None, "q2k_block_nums": None}
        return expected

    monkeypatch.setattr(interface, "bsa_fp8_blk64_fwd", fake_forward)
    result = ops.block_sparse_attention_fp8_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=1,
    )

    assert isinstance(result, TupleDict)
    assert list(result.keys()) == ["o_tensor"]
    assert result["o_tensor"] is expected
    assert result[0] is expected


@pytest.mark.L0
@pytest.mark.parametrize("arch", (90, 120))
def test_fp8_blk128_dispatch_preserves_v_layout_and_metadata(monkeypatch, arch):
    """Pin each blk128 path's V layout, K scaling and sparse metadata."""
    api = importlib.import_module("cudnn.block_sparse_attention.api")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(api, "_device_arch", lambda tensor: arch)
    monkeypatch.setattr(interface, "_get_device_arch_for_device", lambda device: arch)
    q = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    sizes = torch.full((1,), 128, device="cuda", dtype=torch.int32) if arch == 120 else None
    counts = torch.ones((1, 1, 1), device="cuda", dtype=torch.int32) if arch == 120 else None
    quantized = tuple(object() for _ in range(6))
    expected = object()
    calls = []
    fake_quantizer = ModuleType("cudnn.block_sparse_attention._fp8_quant")

    def quantize(actual_q, actual_k, actual_v, **kwargs):
        assert actual_q is actual_k is actual_v is q
        assert kwargs == ({"v_layout": "bhds"} if arch == 90 else {"v_block_size": 128, "k_quant_block_size": 16})
        calls.append("quantize")
        return quantized

    def launch(*args, **kwargs):
        assert args[:6] == quantized
        assert args[6] is indices and args[7:] == (1, 0.125)
        assert kwargs == ({} if arch == 90 else dict(block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, v_block_size=128, k_quant_block_size=16))
        calls.append("launch")
        return expected, None

    def wrong_arch(*args, **kwargs):
        pytest.fail("blk128 dispatched to the wrong architecture")

    fake_quantizer._quantize_sage_bhsd = quantize
    monkeypatch.setitem(sys.modules, fake_quantizer.__name__, fake_quantizer)
    monkeypatch.setattr(interface, "_bsa_attn_fwd_sm90_fp8_blk128", launch if arch == 90 else wrong_arch)
    monkeypatch.setattr(interface, "_bsa_attn_fwd_sm120_fp8", launch if arch == 120 else wrong_arch)
    result = api.block_sparse_attention_fp8_forward(q, q, q, indices, 1, sizes, counts, sparse_block_size=128, softmax_scale=0.125)
    assert result["o_tensor"] is expected
    assert calls == ["quantize", "launch"]


@pytest.mark.L0
@pytest.mark.parametrize("arch", [100, 103])
@pytest.mark.parametrize("splits", [1, 3])
def test_fp8_sm100_blk128_dispatch_and_metadata(monkeypatch, arch, splits):
    """The SM100/SM103 blk128 path quantizes, then launches with the caller's metadata tensors and kv_splits."""
    import_bsa()
    from cudnn.block_sparse_attention import api, _interface, _fp8_quant

    monkeypatch.setattr(api, "_device_arch", lambda tensor: arch)
    monkeypatch.setattr(_interface, "_get_device_arch_for_device", lambda device: arch)
    q = torch.zeros((2, 3, 129, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros((2, 3, 259, 128), device="cuda", dtype=torch.bfloat16), torch.zeros((2, 3, 259, 128), device="cuda", dtype=torch.bfloat16)
    index = torch.zeros((2, 3, 2, 3), device="cuda", dtype=torch.int32)
    nums = torch.full((2, 3, 2), 3, device="cuda", dtype=torch.int32)
    sizes = torch.full((2, 3, 3), 128, device="cuda", dtype=torch.int32)
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


_FP8_SM90_CASES = ("q_tail", "k_tail", "block_sizes", "counts", "scale_nan", "scale_device", "count_device", "block_size", "prefix")
_FP8_SCALE_CASES = ("scale_zero", "scale_negative", "scale_nan")
_FP8_REJECT_MESSAGES = {
    "unsupported_arch": "requires SM90, SM100/SM103, or SM120",
    "block_size_32": "64 or 128",
    **{f"sm100-blk64-{case}": "softmax_scale must be positive" for case in _FP8_SCALE_CASES},
}


@pytest.mark.L0
@pytest.mark.parametrize(
    "arch, block, case",
    [pytest.param(90, block, case, id=f"sm90-blk{block}-{case}") for block in (64, 128) for case in _FP8_SM90_CASES]
    + [pytest.param(90, 128, case, id=f"sm90-blk128-{case}") for case in ("arch", "old_dsl")]
    + [pytest.param(100, 128, case, id=f"sm100-blk128-{case}") for case in ("arch", "split_type", "split_range", *_FP8_SCALE_CASES, "old_dsl")]
    + [pytest.param(100, 64, case, id=f"sm100-blk64-{case}") for case in ("split", *_FP8_SCALE_CASES)]
    + [pytest.param(110, 128, "unsupported_arch", id="sm110-unsupported_arch"), pytest.param(120, 32, "block_size_32", id="sm120-block_size_32")],
)
def test_fp8_rejects_before_quantization(request, monkeypatch, arch, block, case):
    """Unsupported arguments raise before the quantizer runs.  The architecture is patched, so every row runs on any GPU.

    Sage FP8 folds the softmax scale into its Q*K dequantization scales; a nonpositive scale would turn fully masked
    groups into NaN/+inf, so it is rejected instead of returning non-finite output.
    """
    import_bsa()
    from cudnn import block_sparse_attention_fp8_forward
    from cudnn.block_sparse_attention import api, _interface, _fp8_quant

    monkeypatch.setattr(api, "_device_arch", lambda tensor: arch)
    monkeypatch.setattr(_interface, "_get_device_arch_for_device", lambda device: arch)
    rows = block if block in (64, 128) else 128
    q = torch.randn((1, 1, rows, 128), device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn((1, 1, 2 * rows, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    ids = torch.tensor([1, 0], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    kwargs = {"sparse_block_size": block}
    if case == "q_tail":
        q = q[:, :, : block - 1].contiguous()
    elif case == "k_tail":
        k, v = k[:, :, : block + block // 2].contiguous(), v[:, :, : block + block // 2].contiguous()
    elif case == "block_sizes":
        kwargs["block_sizes"] = torch.full((2,), block, dtype=torch.int32, device="cuda")
    elif case == "counts":
        kwargs["q2k_block_nums"] = torch.ones((1, 1, 1), dtype=torch.int32, device="cuda")
    elif case == "scale_device":
        kwargs["softmax_scale"] = torch.tensor(0.1, device="cuda")
    elif case == "count_device":
        kwargs["block_sparse_num"] = torch.tensor(1, device="cuda")
    elif case == "block_size":
        kwargs["sparse_block_size"] = 256
    elif case == "prefix":  # more index rows than query blocks
        ids = ids.repeat(1, 1, 2, 1)
    elif case == "arch":
        monkeypatch.setattr(api, "_device_arch", lambda tensor: 110 if arch == 90 else 107)
    elif case == "old_dsl":
        monkeypatch.setattr(_interface, "_cutlass_dsl_version", lambda: (4, 6, 1))
    elif case == "split_type":
        kwargs["kv_splits"] = 2.0
    elif case == "split_range":
        kwargs["kv_splits"] = 257
    elif case == "split":  # explicit kv_splits is an SM100/SM103 blk128 option
        kwargs["kv_splits"] = 2
    elif case in _FP8_SCALE_CASES:
        kwargs["softmax_scale"] = {"scale_zero": 0.0, "scale_negative": -(128**-0.5), "scale_nan": float("nan")}[case]

    def unexpected(*args, **kw):
        pytest.fail("unsupported inputs reached quantization")

    monkeypatch.setattr(_fp8_quant, "_quantize_sage_bhsd", unexpected)
    message = _FP8_REJECT_MESSAGES.get(case) or _FP8_REJECT_MESSAGES.get(request.node.callspec.id)
    with pytest.raises((ValueError, NotImplementedError, RuntimeError), match=message):
        block_sparse_attention_fp8_forward(q, k, v, ids, **kwargs)


@pytest.mark.L0
def test_fp8_sm90_private_v_layout_tag_is_required():
    """The SM90 kernels read V as BHDS; the private launch requires the tag even when V's shape cannot tell (sk == d)."""
    require_sm90_fp8()
    from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
    from cudnn.block_sparse_attention._interface import _bsa_fp8_blk64_fwd_quantized

    q, k, v = (torch.randn((1, 1, n, 128), device="cuda", dtype=torch.bfloat16) for n in (64, 128, 128))
    quantized = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
    ids = torch.tensor([1, 0], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    with pytest.raises(ValueError, match="v_layout='bhds'"):
        _bsa_fp8_blk64_fwd_quantized(*quantized, ids, 2)


@pytest.mark.L0
def test_fp8_quantized_inputs_require_fully_contiguous_bhsd():
    import_bsa(families=(10, 11), min_cutedsl=(4, 6, 1), feature="this numerical Sage FP8 test")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")

    batch, heads, seqlen, head_dim = 1, 4, 64, 128

    def make_padded_fp8():
        storage = torch.empty((batch, heads, seqlen, head_dim + 1), device="cuda", dtype=torch.float8_e4m3fn)
        return storage[..., :head_dim]

    q = make_padded_fp8()
    k = make_padded_fp8()
    v = make_padded_fp8()
    assert all(tensor.stride(-1) == 1 and not tensor.is_contiguous() for tensor in (q, k, v))

    q_scale = torch.empty((batch, heads, seqlen), device="cuda", dtype=torch.float32)
    k_scale = torch.empty((batch, heads, seqlen // 16), device="cuda", dtype=torch.float32)
    v_scale = torch.empty((heads, head_dim), device="cuda", dtype=torch.float32)
    q2k = torch.zeros((batch, heads, 1, 1), device="cuda", dtype=torch.int32)

    with pytest.raises(AssertionError, match="fully contiguous BHSD"):
        interface._bsa_fp8_blk64_fwd_quantized(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q2k,
            1,
        )


@pytest.mark.L0
@pytest.mark.parametrize(
    ("heads", "seqlen_q", "topk", "expected_splits"),
    (
        (4, 64, 127, 1),
        (4, 64, 128, 16),
        (4, 64, 224, 8),
        (4, 64, 400, 16),
        (8, 64, 128, 4),
        (8, 64, 256, 8),
        (8, 64, 500, 16),
        (4, 8192, 899, 1),
        (4, 8192, 900, 4),
    ),
)
def test_fp8_sm100_blk64_auto_kv_splits(heads, seqlen_q, topk, expected_splits):
    import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    assert interface._sm100_blk64_auto_fp8_kv_splits(topk, heads, seqlen_q) == expected_splits


@pytest.mark.L0
def test_fp8_sm100_blk64_auto_split_uses_workspace_fallback(monkeypatch):
    import_bsa(families=(10, 11), min_cutedsl=(4, 6, 1), feature="this numerical Sage FP8 test")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface, "_get_device_arch", lambda: 100)

    batch, heads, seqlen_q, seqlen_k, head_dim, topk = 2, 3, 64, 64, 128, 128
    q = torch.empty((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    q_scale = torch.empty((batch, heads, seqlen_q), device="cuda", dtype=torch.float32)
    k_scale = torch.empty((batch, heads, seqlen_k // 16), device="cuda", dtype=torch.float32)
    v_scale = torch.empty((heads, head_dim), device="cuda", dtype=torch.float32)
    q2k = torch.zeros((batch, heads, 1, topk), device="cuda", dtype=torch.int32)

    class WorkspaceFallbackCalled(Exception):
        pass

    def workspace_fallback(q_arg, value_dim, kv_splits, allow_fallback, output_dtype=None):
        assert q_arg is q
        assert value_dim == head_dim
        assert kv_splits == 16
        assert allow_fallback is True
        assert output_dtype is torch.bfloat16
        raise WorkspaceFallbackCalled

    monkeypatch.setattr(interface, "_resolve_blk64_split_workspace", workspace_fallback)
    with pytest.raises(WorkspaceFallbackCalled):
        interface._bsa_fp8_blk64_fwd_quantized(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q2k,
            topk,
        )


@pytest.mark.L0
def test_fp8_split_workspace_estimate_uses_bf16_output_size():
    import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")

    batch, heads, seqlen_q, value_dim, kv_splits = 1, 4, 64, 128, 2
    q = torch.empty((batch, heads, seqlen_q, value_dim), dtype=torch.float8_e4m3fn)
    rows = batch * heads * seqlen_q
    num_q_blocks = math.ceil(seqlen_q / 64)
    expected_bytes = (
        kv_splits * rows * (value_dim + 1) * 4 + rows * (value_dim * torch.bfloat16.itemsize + 4) + batch * heads * num_q_blocks * (kv_splits + 1) * 4
    )

    assert interface._blk64_split_workspace_bytes(q, value_dim, kv_splits, output_dtype=torch.bfloat16) == expected_bytes
