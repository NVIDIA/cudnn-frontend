# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import dsa_init, with_dsa_indexer_top_k_params
from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_indexer_top_k


def _allocate_inputs(cfg, next_n: int, input_pattern: str):
    """Allocate inputs with the kernel's ``n_rows == batch_size * next_n``
    invariant held: treat every row as its own batch for ``next_n=1``,
    otherwise group ``next_n`` consecutive rows per batch.
    """
    b = cfg["b"]
    s_kv = cfg["s_kv"]
    s_q = cfg["s_q"]
    dtype = cfg["dtype"]
    n_rows = b * s_q
    device = "cuda"

    assert n_rows % next_n == 0, f"n_rows={n_rows} must be divisible by next_n={next_n}"
    batch_size = n_rows // next_n

    if input_pattern == "random":
        input_values = torch.randn(n_rows, s_kv, dtype=dtype, device=device)
        seq_lens = torch.randint(max(1, s_kv // 2), s_kv + 1, (batch_size,), dtype=torch.int32, device=device)
        return input_values, seq_lens

    columns = torch.arange(s_kv, device=device)
    patterns = torch.empty(8, s_kv, dtype=dtype, device=device)
    patterns[0] = (columns % 7 - 3).to(dtype)  # Many ties at each score.
    patterns[1].fill_(1.0)
    patterns[2].fill_(0.0)
    patterns[3].fill_(-0.0)
    patterns[4] = torch.where(columns % 2 == 0, 0.0, -0.0)
    patterns[5] = 1.0 + ((columns * 17) % 8).to(dtype) * torch.finfo(dtype).eps  # Adjacent representable scores.
    patterns[6] = torch.where(columns % 2 == 0, float("inf"), 1.0)
    patterns[7].fill_(float("-inf"))
    patterns[7, : min(cfg["topk"] // 2, s_kv // 4)] = 1.0
    input_values = patterns[(torch.arange(n_rows, device=device) // next_n) % len(patterns)].contiguous()

    # Each pattern gets a full-length group and a ragged group. next_n > 1
    # exercises the speculative stagger, including short and K-boundary rows.
    lengths = [s_kv] * len(patterns) + [2, 3, max(2, cfg["topk"] - 1), max(2, cfg["topk"]), cfg["topk"] + 1, s_kv // 2, s_kv - 1, s_kv]
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)[torch.arange(batch_size, device=device) % len(lengths)]
    seq_lens.clamp_(max=s_kv)
    return input_values, seq_lens


def _allocate_outputs(op, input_values):
    """The caller owns the outputs and the radix-scratch workspace (Rule 8 / R2)."""
    n_rows = input_values.shape[0]
    device = input_values.device
    indices = torch.empty(n_rows, op.top_k, dtype=torch.int32, device=device)
    values = torch.empty(n_rows, op.top_k, dtype=input_values.dtype, device=device) if op.return_val else None
    workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
    return indices, values, workspace


def _import_dsa():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Indexer top-k requires compute capability 9.0 or newer")
    return DSA


@torch_fork_set_rng(seed=0)
@with_dsa_indexer_top_k_params
def test_DSA_indexer_top_k_compile_execute(
    dtype,
    acc_dtype,
    top_k,
    next_n,
    s_kv,
    input_pattern,
    tie_break,
    return_val,
    request,
    compile_allocates_nothing,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        top_k=top_k,
        next_n=next_n,
        return_val=return_val,
        min_compute_capability=90,
        s_q_default=1024 if input_pattern == "random" else 32,
        s_kv_default=s_kv,
    )
    top_k = cfg["topk"]
    input_values, seq_lens = _allocate_inputs(cfg, next_n=next_n, input_pattern=input_pattern)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    op = DSA.IndexerTopK(
        sample_input_values=input_values,
        sample_seq_lens=seq_lens,
        top_k=top_k,
        next_n=next_n,
        return_val=return_val,
        tie_break=tie_break,
    )
    assert op.check_support()
    compile_allocates_nothing(op)
    indices, values, workspace = _allocate_outputs(op, input_values)
    op.execute(input_values, seq_lens, out_indices=indices, out_values=values, current_stream=stream, workspace=workspace)

    if not cfg["skip_ref"]:
        check_ref_indexer_top_k(
            input_values,
            seq_lens,
            top_k,
            next_n,
            indices,
            values,
            return_val,
            tie_break=tie_break,
        )


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("return_val", [True, False])
def test_DSA_indexer_top_k_execute_allocates_nothing(dtype, return_val, compile_allocates_nothing):
    """R9: warm executes into caller-owned outputs and workspace allocate nothing; the
    fe_api conftest arms torch's sync debug mode around execute(), so they never sync either."""
    DSA = _import_dsa()
    from cudnn.api_base import ws_align

    n_rows, num_cols, top_k = 64, 4096, 512
    input_values = torch.randn(n_rows, num_cols, dtype=dtype, device="cuda")
    seq_lens = torch.randint(num_cols // 2, num_cols + 1, (n_rows,), dtype=torch.int32, device="cuda")

    op = DSA.IndexerTopK(sample_input_values=input_values, sample_seq_lens=seq_lens, top_k=top_k, return_val=return_val)
    assert op.check_support()
    compile_allocates_nothing(op)
    assert op.scratch_workspace_bytes() == ws_align(n_rows * (2 if dtype == torch.float32 else 1) * num_cols * 4)

    indices, values, workspace = _allocate_outputs(op, input_values)
    op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "IndexerTopK.execute allocated device memory"

    check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, indices, values, return_val)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_top_k_execute_requires_outputs_and_workspace():
    """Rule 1 both directions on the outputs; R2 on the workspace: execute() never allocates on the caller's behalf."""
    DSA = _import_dsa()

    n_rows, num_cols, top_k = 32, 1024, 128
    input_values = torch.randn(n_rows, num_cols, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")

    op = DSA.IndexerTopK(sample_input_values=input_values, sample_seq_lens=seq_lens, top_k=top_k, return_val=True)
    assert op.check_support()
    op.compile()
    indices, values, workspace = _allocate_outputs(op, input_values)
    required = op.scratch_workspace_bytes()

    with pytest.raises(TypeError):
        op.execute(input_values, seq_lens)
    with pytest.raises(ValueError, match="out_indices is required"):
        op.execute(input_values, seq_lens, out_indices=None, out_values=values, workspace=workspace)
    with pytest.raises(ValueError, match="out_indices dtype mismatch"):
        op.execute(input_values, seq_lens, out_indices=indices.to(torch.int64), out_values=values, workspace=workspace)
    with pytest.raises(ValueError, match="out_indices tensor shape mismatch"):
        op.execute(input_values, seq_lens, out_indices=indices[:, : top_k - 1], out_values=values, workspace=workspace)
    with pytest.raises(ValueError, match="out_indices must be contiguous"):
        op.execute(
            input_values, seq_lens, out_indices=torch.empty(n_rows, 2 * top_k, dtype=torch.int32, device="cuda")[:, ::2], out_values=values, workspace=workspace
        )
    with pytest.raises(ValueError, match="out_values is required"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=None, workspace=workspace)
    with pytest.raises(ValueError, match="out_values dtype mismatch"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values.to(torch.float32), workspace=workspace)
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace but execute\(\) received none"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values)
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=torch.empty(required - 1, dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="32-byte aligned"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=torch.empty(required + 16, dtype=torch.uint8, device="cuda")[16:])
    with pytest.raises(ValueError, match="workspace must be on input_values' device"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=torch.empty(required, dtype=torch.uint8))
    op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)

    # return_val=False: a provided-but-uncompiled out_values must raise, not be ignored.
    op_indices_only = DSA.IndexerTopK(sample_input_values=input_values, sample_seq_lens=seq_lens, top_k=top_k, return_val=False)
    assert op_indices_only.check_support()
    op_indices_only.compile()
    with pytest.raises(ValueError, match="out_values must be None"):
        op_indices_only.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)
    op_indices_only.execute(input_values, seq_lens, out_indices=indices, out_values=None, workspace=workspace)
    check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, indices, None, False)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_top_k_params
def test_DSA_indexer_top_k_wrapper(
    dtype,
    acc_dtype,
    top_k,
    next_n,
    s_kv,
    input_pattern,
    tie_break,
    return_val,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        top_k=top_k,
        next_n=next_n,
        return_val=return_val,
        min_compute_capability=90,
        s_q_default=1024 if input_pattern == "random" else 32,
        s_kv_default=s_kv,
    )
    top_k = cfg["topk"]
    input_values, seq_lens = _allocate_inputs(cfg, next_n=next_n, input_pattern=input_pattern)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        next_n=next_n,
        return_val=return_val,
        stream=stream,
        tie_break=tie_break,
    )

    indices = result["indices"]
    values = result["values"]
    if not cfg["skip_ref"]:
        check_ref_indexer_top_k(
            input_values,
            seq_lens,
            top_k,
            next_n,
            indices,
            values,
            return_val,
            tie_break=tie_break,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tie_break", [0, 1, 2], ids=["ties-none", "ties-small", "ties-large"])
@pytest.mark.parametrize("return_val", [True, False])
def test_DSA_indexer_top_k_wrapper_ignores_vector_padding_with_negative_infinity(tie_break, return_val):
    """OOB vector lanes must not join a real -inf threshold bin."""
    DSA = _import_dsa()

    num_rows = 633
    num_cols = 768
    seq_len = 633
    finite_values = 475
    top_k = 512

    input_values = torch.full(
        (num_rows, num_cols),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    input_values[:, :finite_values] = torch.randn(
        num_rows,
        finite_values,
        dtype=torch.float32,
        device="cuda",
    )
    seq_lens = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")

    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k=top_k,
        next_n=1,
        return_val=return_val,
        tie_break=tie_break,
    )
    check_ref_indexer_top_k(
        input_values,
        seq_lens,
        top_k,
        1,
        result["indices"],
        result["values"],
        return_val,
        tie_break=tie_break,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
def test_DSA_indexer_top_k_wrapper_accepts_strided_inputs():
    """The eager wrapper copies strided ``input_values`` / ``seq_lens`` contiguous on the launch stream; the class declines them."""
    DSA = _import_dsa()
    n_rows, num_cols, top_k = 16, 512, 64
    base = torch.randn(n_rows, 2 * num_cols, dtype=torch.float32, device="cuda")
    input_values = base[:, ::2]
    seq_lens = torch.full((2 * n_rows,), num_cols, dtype=torch.int32, device="cuda")[::2]
    assert not input_values.is_contiguous() and not seq_lens.is_contiguous()

    with pytest.raises((ValueError, NotImplementedError)):
        DSA.IndexerTopK(input_values, seq_lens, top_k).check_support()

    got = DSA.indexer_top_k_wrapper(input_values, seq_lens, top_k)
    expected = DSA.indexer_top_k_wrapper(input_values.contiguous(), seq_lens.contiguous(), top_k)
    torch.cuda.synchronize()
    torch.testing.assert_close(torch.sort(got["values"], dim=1).values, torch.sort(expected["values"], dim=1).values, atol=0.0, rtol=0.0)
    assert torch.equal(torch.sort(got["indices"], dim=1).values, torch.sort(expected["indices"], dim=1).values)


@pytest.mark.L0
@torch_fork_set_rng(seed=6)
def test_DSA_indexer_top_k_wrapper_staging_outlives_the_released_original():
    """R1 staging: the strided-input copy on an explicit side stream reads the caller's tensor
    asynchronously. The wrapper records the original on that stream, so a caller that releases it
    right after the call, while the copy is still queued, cannot hand the block to its next
    same-size allocation (poisoned here) before the copy has read it."""
    DSA = _import_dsa()
    from cuda.bindings import driver as cuda

    n_rows, num_cols, top_k = 64, 4096, 64
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")
    base = torch.randn(n_rows, 2 * num_cols, dtype=torch.float32, device="cuda")
    expected = DSA.indexer_top_k_wrapper(base[:, ::2].contiguous(), seq_lens, top_k)
    # Warm the window's kernels first: under CUDA lazy loading a first launch waits for the device to drain (no race).
    DSA.indexer_top_k_wrapper(base[:, ::2], seq_lens, top_k)
    torch.empty(1, dtype=torch.float32, device="cuda").fill_(float("-inf"))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # no other cached block of this size: the released one is the only candidate for reuse
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(1_000_000_000)  # the wrapper's staging copy queues behind this
    got = DSA.indexer_top_k_wrapper(base[:, ::2], seq_lens, top_k, stream=cuda.CUstream(side.cuda_stream))
    del base  # the caller releases its reference while the copy is still pending
    poison = torch.empty(n_rows, 2 * num_cols, dtype=torch.float32, device="cuda")
    # Same size as the released block: with the original recorded, the allocator defers the block's reuse
    # (or waits for the pending copy first) instead of letting this fill race it.
    poison.fill_(float("-inf"))
    torch.cuda.synchronize()
    torch.testing.assert_close(torch.sort(got["values"], dim=1).values, torch.sort(expected["values"], dim=1).values, atol=0.0, rtol=0.0)
    assert torch.equal(torch.sort(got["indices"], dim=1).values, torch.sort(expected["indices"], dim=1).values)


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("dtype,num_cols", [(torch.bfloat16, 17), (torch.float32, 9)], ids=["bf16", "fp32"])
def test_DSA_indexer_top_k_wrapper_short_misaligned_row(dtype, num_cols):
    """A row shorter than its misaligned prologue must not read columns past its length."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Indexer top-k requires compute capability 9.0 or newer")

    seq_len = 3
    top_k = 1
    # Row 1 starts num_cols elements in, which is not 32-byte aligned; its prologue would span past seq_len.
    input_values = torch.randn(2, num_cols, dtype=torch.float32, device="cuda").to(dtype)
    input_values[1, seq_len:] = 100.0
    seq_lens = torch.tensor([num_cols, seq_len], dtype=torch.int32, device="cuda")

    result = DSA.indexer_top_k_wrapper(input_values, seq_lens, top_k=top_k, next_n=1, return_val=True, tie_break=1)
    assert int(result["indices"][1, 0]) < seq_len
    check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, result["indices"], result["values"], True, tie_break=1)
