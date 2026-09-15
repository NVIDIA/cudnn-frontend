# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""BSA forward correctness tests, including native SM120 blk128.

Install this checkout with its CUDA/cuDNN and Python test dependencies as
described in the repository agent guide. SM120 FA4-style tests require a
CUDA-enabled PyTorch build and CuTe DSL >= 4.7.0. Verify the actual imported
kernel path before testing, especially with multiple editable checkouts.
From the repository root:

    python -c 'import cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fa4 as k; print(k.__file__)'
    (cd test/python && CUDA_VISIBLE_DEVICES=0 python -m pytest -q fe_api/bsa/test_BSA_attention_forward.py -k sm120)
    (cd test/python && CUDA_VISIBLE_DEVICES=0 python -m pytest -q fe_api/bsa)

The full directory also contains other BSA tests. Unsupported configurations
skip; legacy-comparison cases require SM120_PR1010_SOURCE_DIR, whose preparation
and full-workload commands are in test_sm120_three_paths_benchmark.py.
test_sm120_blk128_pair_benchmark.py documents saved-kernel A/B measurements.

For a fresh single-GPU comparison with PyTorch's cuDNN dense SDPA backend:

    CUDA_VISIBLE_DEVICES=0 python benchmark/bsa/benchmark_sm120_blk128.py \
      --sequence 142720 --heads 8 --densities 0.15 0.20 \
      --patterns strided local --warmup 5 --repeats 21 --fail-below-target

This benchmark uses BF16 BHSD [1, 8, 142720, 128] and native KV128; its 4.5x
gate applies to the slowest 20%-density case. A missed performance gate is
not itself an accuracy failure. Measure on an idle GPU without a profiler.

Production blk128 consumes original sparse metadata with no KV128-to-KV64
expansion. FP16/BF16 reference tests cover fixed/variable counts, partial Q/KV,
GQA, layouts, and full/tail scheduling. The FA4-style path keeps FP32 accumulators
and BF16 probabilities for BF16 inputs; it does not use FP8/INT8 quantization.
These are single-GPU tests; multi-GPU overlap prototypes are not in this PR.
"""

import builtins
import importlib
from types import SimpleNamespace

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.bsa.bsa_reference import attention_reference, block_sparse_mask
from fe_api.bsa.bsa_utils import make_fixed_metadata, make_variable_metadata, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.L0
def test_sm120_fa4_blk128_rejects_old_dsl_before_kernel_import(monkeypatch):
    """Reject an unsupported installed DSL before importing the FA4 kernel."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("FA4-style blk128 is specific to SM120")
    BSA = _import_bsa()
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
        BSA.block_sparse_attention_forward(q, q, q, indices, block_sparse_num=1, sparse_block_size=128)


def _import_bsa(require_fa4=False):
    """Load BSA or skip unavailable dependencies, optionally checking the FA4 floor."""
    if require_fa4:
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

        installed, version = cutedsl_state()
        if not installed or cutedsl_too_old(version):
            pytest.skip("FA4-style blk128 requires a supported CuTe DSL version")
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
@torch_fork_set_rng(seed=17)
def test_bsa_attention_forward_sm120_native_blk128():
    """Compare native blk128 forward output and LSE with the FP32 reference."""
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip("native blk128 forward is specific to SM120")

    BSA = _import_bsa()
    block_size = 128
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


@pytest.mark.L0
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("block_sparse_num", (1, 2, 3, 4, 5))
@torch_fork_set_rng(seed=20)
def test_bsa_attention_forward_sm120_fa4_blk128_fixed_topk(dtype, block_sparse_num):
    """Cover fixed top-k 1-5, GQA, partial Q, and query-dependent KV selections."""
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip("FA4-style blk128 forward is specific to SM120")

    BSA = _import_bsa(require_fa4=True)
    block_size = 128
    batch, q_heads, kv_heads, seqlen_q, seqlen_k, dim = 2, 4, 2, block_size + 1, 6 * block_size, 128
    q = torch.randn((batch, q_heads, seqlen_q, dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, kv_heads, seqlen_k, dim), device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    q_block = torch.arange(2, device="cuda", dtype=torch.int32).view(1, 1, 2, 1)
    batch_head = torch.arange(batch * q_heads, device="cuda", dtype=torch.int32).view(batch, q_heads, 1, 1)
    slots = torch.tensor([0, 2, 3, 5, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 5)
    q2k = ((q_block + batch_head + slots) % 6).contiguous()

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=block_sparse_num,
        sparse_block_size=block_size,
    )
    block_sizes = torch.full((6,), block_size, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("wave_kind,block_sparse_num", (("tail_only", 3), ("unsplit", 1), ("full", 5), ("mixed", 3)))
@torch_fork_set_rng(seed=20)
def test_bsa_attention_forward_sm120_blk128_wave_boundaries(dtype, wave_kind, block_sparse_num):
    """Validate all query rows across full, split-tail, mixed, and unsplit waves."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("native blk128 wave scheduling requires SM120")
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    full_wave_q_blocks = sm_count // 2 if sm_count % 2 == 0 else sm_count
    q_blocks = {
        "tail_only": 2,
        "unsplit": sm_count // 4 + 1,
        "full": full_wave_q_blocks,
        "mixed": full_wave_q_blocks + 2,
    }[wave_kind]
    sequence = (q_blocks - 1) * 128 + 1
    BSA = _import_bsa(require_fa4=True)
    q = torch.randn((1, 2, sequence, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 1, 768, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    q_block = torch.arange(q_blocks, device="cuda", dtype=torch.int32).view(1, 1, -1, 1)
    head = torch.arange(2, device="cuda", dtype=torch.int32).view(1, 2, 1, 1)
    slots = torch.tensor([0, 2, 3, 5, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 5)
    q2k = ((3 * q_block + 7 * head + slots) % 6).contiguous()
    result = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num=block_sparse_num, sparse_block_size=128)

    # Exercise every output row, including the Q64 boundary inside blk128 and
    # the final partial Q tile. Per-query metadata prevents accidental reuse
    # of the first sparse row from passing this check.
    block_sizes = torch.full((6,), 128, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, sequence, 768, 128)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
def test_bsa_sm120_wave_planning_is_not_repeated_on_cache_hits(monkeypatch):
    """Ensure cached launches reuse the SM-count decision made during tracing."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("native blk128 wave scheduling requires SM120")
    BSA = _import_bsa(require_fa4=True)
    from cudnn.block_sparse_attention import _interface
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128 import bsa_fwd_sm120_fa4

    query_count = 0
    original_query = bsa_fwd_sm120_fa4._device_sm_count

    def counted_query():
        """Count device-metadata queries while preserving the actual SM count."""
        nonlocal query_count
        query_count += 1
        return original_query()

    monkeypatch.setattr(bsa_fwd_sm120_fa4, "_device_sm_count", counted_query)
    monkeypatch.setattr(_interface.bsa_attn_fwd, "compile_cache", {})
    q = torch.zeros((1, 1, 129, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    indices = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)
    first = BSA.block_sparse_attention_forward(q, k, v, indices, block_sparse_num=1, sparse_block_size=128)
    assert query_count > 0
    trace_query_count = query_count
    second = BSA.block_sparse_attention_forward(q, k, v, indices, block_sparse_num=1, sparse_block_size=128)
    assert query_count == trace_query_count
    torch.testing.assert_close(first["o_tensor"], torch.ones_like(q), atol=0.0, rtol=0.0)
    torch.testing.assert_close(second["o_tensor"], first["o_tensor"], atol=0.0, rtol=0.0)


@pytest.mark.L0
@torch_fork_set_rng(seed=18)
def test_bsa_attention_forward_sm120_native_blk128_variable_blocks_and_layout():
    """Check variable counts, valid block sizes, empty rows, and BHSD/BSHD layouts."""
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip("native blk128 forward is specific to SM120")

    BSA = _import_bsa()
    block_size = 128
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
        block_sizes=block_sizes,
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
        block_sizes=block_sizes,
        q2k_block_nums=block_nums,
        sparse_block_size=block_size,
        layout="bshd",
    )
    torch.testing.assert_close(result_bshd["o_tensor"].transpose(1, 2), result["o_tensor"], atol=0, rtol=0)
    torch.testing.assert_close(result_bshd["lse_tensor"], result["lse_tensor"], atol=0, rtol=0)

    empty_block_nums = block_nums.clone()
    empty_block_nums[..., 0] = 0
    empty_result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sizes=block_sizes,
        q2k_block_nums=empty_block_nums,
        sparse_block_size=block_size,
        allow_empty_block_nums=True,
    )
    empty_mask = block_sparse_mask(q2k, 0, block_sizes, seqlen_q, seqlen_k, block_size, empty_block_nums)
    empty_o_ref, empty_lse_ref = attention_reference(q, k, v, empty_mask)
    torch.testing.assert_close(empty_result["o_tensor"].float(), empty_o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(empty_result["lse_tensor"], empty_lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
def test_bsa_attention_forward_sm120_native_blk128_partial_q_and_kv_tiles():
    """Verify native128 masking for partial final query and key/value tiles."""
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip("native blk128 forward is specific to SM120")

    BSA = _import_bsa()
    block_size = 128
    batch, heads, seqlen_q, seqlen_k, dim = 1, 1, block_size + 1, 2 * block_size + 1, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = torch.tensor([0, 2], device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 1, 2, 2).contiguous()
    block_sizes = torch.tensor([block_size, block_size, 1], device="cuda", dtype=torch.int32)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=2,
        block_sizes=block_sizes,
        sparse_block_size=block_size,
    )
    mask = block_sparse_mask(q2k, 2, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


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
@pytest.mark.parametrize(("seqlen_k", "expected"), [(4 * 64, False), (4 * 64 - 1, True)])
def test_bsa_attention_forward_sm100_blk64_detects_partial_kv_tail(seqlen_k, expected):
    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    k = torch.empty((1, 2, seqlen_k, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)

    assert interface._sm100_blk64_has_partial_kv_tail(k, v) is expected


@pytest.mark.L0
@torch_fork_set_rng(seed=859)
def test_bsa_attention_forward_sm100_blk64_partial_kv_tail_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("partial-tail exact KV layout is specific to SM100/SM110 blk64")

    BSA = _import_bsa()
    block_size = 64
    batch, heads, seqlen_q, seqlen_k, dim = 1, 1, block_size, block_size + 1, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2)
    block_sparse_num = 2
    block_sizes = torch.tensor([block_size, 1], device="cuda", dtype=torch.int32)

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
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    assert torch.isfinite(result["o_tensor"]).all()
    assert torch.isfinite(result["lse_tensor"]).all()
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    result_bshd = BSA.block_sparse_attention_forward(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
        layout="bshd",
        use_clc=False,
    )
    assert torch.isfinite(result_bshd["o_tensor"]).all()
    assert torch.isfinite(result_bshd["lse_tensor"]).all()
    torch.testing.assert_close(result_bshd["o_tensor"].transpose(1, 2), result["o_tensor"], atol=0, rtol=0)
    torch.testing.assert_close(result_bshd["lse_tensor"], result["lse_tensor"], atol=0, rtol=0)


@pytest.mark.L0
@torch_fork_set_rng(seed=2029)
def test_bsa_attention_forward_sm100_blk64_split_kv_clc_persistent_tiles():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk64 split-KV CLC scheduling is specific to SM100/SM110")

    BSA = _import_bsa()
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    batch, heads, num_kv_blocks, head_dim = 2, 3, 8, 128
    kv_splits = 3
    num_q_blocks = max(32, 2 * sm_count // (batch * heads * kv_splits) + 1)
    block_size = 64
    seqlen_q, seqlen_k = num_q_blocks * block_size, num_kv_blocks * block_size
    assert batch * heads * num_q_blocks * kv_splits > 2 * sm_count

    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = (
        torch.arange(num_kv_blocks, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, num_kv_blocks)
        .expand(batch, heads, num_q_blocks, num_kv_blocks)
        .contiguous()
    )
    block_sizes = torch.full((num_kv_blocks,), block_size, device="cuda", dtype=torch.int32)

    reference = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        num_kv_blocks,
        block_sizes,
        sparse_block_size=block_size,
        use_clc=False,
        kv_splits=kv_splits,
    )
    torch.cuda.synchronize()
    actual = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        num_kv_blocks,
        block_sizes,
        sparse_block_size=block_size,
        use_clc=True,
        kv_splits=kv_splits,
    )
    torch.cuda.synchronize()
    repeated = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        num_kv_blocks,
        block_sizes,
        sparse_block_size=block_size,
        use_clc=True,
        kv_splits=kv_splits,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual["o_tensor"], reference["o_tensor"], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual["lse_tensor"], reference["lse_tensor"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(repeated["o_tensor"], actual["o_tensor"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated["lse_tensor"], actual["lse_tensor"], rtol=0.0, atol=0.0)


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
        lambda q, value_dim, kv_splits, output_dtype=None: kv_splits * gib // 2,
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


@pytest.mark.L0
def test_bsa_attention_forward_sm100_blk64_large_q_auto_scheduler_policy():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("large-Q scheduler policy is specific to SM100/SM110 blk64")

    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q_large = torch.empty((1, 40, 131072, 1), device="cuda", dtype=torch.int8)
    q_small = torch.empty((1, 4, 128, 1), device="cuda", dtype=torch.int8)
    q2k_block_index = torch.empty((1, 1, 1, 2048), device="cuda", dtype=torch.int32)

    assert interface._sm100_blk64_auto_kv_splits(q_large, q2k_block_index, 2048) == 1
    assert interface._sm100_blk64_auto_kv_splits(q_small, q2k_block_index, 2048) == 8
    assert interface.choose_blk64_use_clc(q_large, 256)
    assert interface.choose_blk64_use_clc(q_large, 2048)


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
