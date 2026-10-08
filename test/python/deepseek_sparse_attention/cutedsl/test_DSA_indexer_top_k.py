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
    op.compile()
    indices, values = op.execute(input_values, seq_lens, current_stream=stream)

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
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Indexer top-k requires compute capability 9.0 or newer")

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
