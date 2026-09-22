# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_indexer_top_k_params
from fe_api.dsa.dsa_reference import check_ref_indexer_top_k


def _allocate_inputs(cfg, next_n: int):
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

    input_values = torch.randn(n_rows, s_kv, dtype=dtype, device=device)
    # Random-but-reasonable seq_lens (each in [s_kv // 2, s_kv]).
    lo = max(1, s_kv // 2)
    seq_lens = torch.randint(
        lo,
        s_kv + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_top_k_params
def test_DSA_indexer_top_k_compile_execute(
    dtype,
    acc_dtype,
    top_k,
    next_n,
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
    )
    input_values, seq_lens = _allocate_inputs(cfg, next_n=next_n)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        op = DSA.IndexerTopK(
            sample_input_values=input_values,
            sample_seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=return_val,
        )
        assert op.check_support()
        compile_allocates_nothing(op)
        indices, values, workspace = _allocate_outputs(op, input_values)
        op.execute(input_values, seq_lens, indices, values, current_stream=stream, workspace=workspace)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    if not cfg["skip_ref"]:
        check_ref_indexer_top_k(
            input_values,
            seq_lens,
            top_k,
            next_n,
            indices,
            values,
            return_val,
        )


@pytest.mark.L0
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
    op.execute(input_values, seq_lens, indices, values, workspace=workspace)
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        op.execute(input_values, seq_lens, indices, values, workspace=workspace)
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
        op.execute(input_values, seq_lens, None, values, workspace=workspace)
    with pytest.raises(ValueError, match="out_indices dtype mismatch"):
        op.execute(input_values, seq_lens, indices.to(torch.int64), values, workspace=workspace)
    with pytest.raises(ValueError, match="out_indices tensor shape mismatch"):
        op.execute(input_values, seq_lens, indices[:, : top_k - 1], values, workspace=workspace)
    with pytest.raises(ValueError, match="out_indices must be contiguous"):
        op.execute(input_values, seq_lens, torch.empty(n_rows, 2 * top_k, dtype=torch.int32, device="cuda")[:, ::2], values, workspace=workspace)
    with pytest.raises(ValueError, match="out_values is required"):
        op.execute(input_values, seq_lens, indices, None, workspace=workspace)
    with pytest.raises(ValueError, match="out_values dtype mismatch"):
        op.execute(input_values, seq_lens, indices, values.to(torch.float32), workspace=workspace)
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace but execute\(\) received none"):
        op.execute(input_values, seq_lens, indices, values)
    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace"):
        op.execute(input_values, seq_lens, indices, values, workspace=torch.empty(required - 1, dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="32-byte aligned"):
        op.execute(input_values, seq_lens, indices, values, workspace=torch.empty(required + 16, dtype=torch.uint8, device="cuda")[16:])
    op.execute(input_values, seq_lens, indices, values, workspace=workspace)

    # return_val=False: a provided-but-uncompiled out_values must raise, not be ignored.
    op_indices_only = DSA.IndexerTopK(sample_input_values=input_values, sample_seq_lens=seq_lens, top_k=top_k, return_val=False)
    assert op_indices_only.check_support()
    op_indices_only.compile()
    with pytest.raises(ValueError, match="out_values must be None"):
        op_indices_only.execute(input_values, seq_lens, indices, values, workspace=workspace)
    op_indices_only.execute(input_values, seq_lens, indices, None, workspace=workspace)
    check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, indices, None, False)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_top_k_params
def test_DSA_indexer_top_k_wrapper(
    dtype,
    acc_dtype,
    top_k,
    next_n,
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
    )
    input_values, seq_lens = _allocate_inputs(cfg, next_n=next_n)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        result = DSA.indexer_top_k_wrapper(
            input_values,
            seq_lens,
            top_k,
            next_n=next_n,
            return_val=return_val,
            stream=stream,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

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
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_indexer_top_k_wrapper_ignores_vector_padding_with_negative_infinity():
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
        return_val=False,
    )
    torch.cuda.synchronize()

    indices = result["indices"]
    assert torch.all((indices >= 0) & (indices < seq_len)).item()

    selected_values = torch.gather(input_values, 1, indices.to(torch.int64))
    expected_values = torch.topk(input_values[:, :seq_len], top_k, dim=1).values
    torch.testing.assert_close(
        torch.sort(selected_values, dim=1).values,
        torch.sort(expected_values, dim=1).values,
        atol=0.0,
        rtol=0.0,
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
