# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_indexer_top_k
from deepseek_sparse_attention.cutedsl.dsa_utils import (
    dsa_init,
    with_dsa_indexer_top_k_params,
)
from test_utils import torch_fork_set_rng


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
        from cuda.bindings import driver as cuda
        from cudnn import DSA
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
def test_DSA_indexer_top_k_rejects_zero_extents_and_nonpositive_next_n():
    """Invalid launch extents must fail in check_support(), before compilation."""
    DSA = _import_dsa()

    cases = [
        (
            torch.empty((0, 16), dtype=torch.float32, device="cuda"),
            torch.empty((0,), dtype=torch.int32, device="cuda"),
            1,
            "n_rows must be positive",
        ),
        (
            torch.empty((1, 0), dtype=torch.float32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            1,
            "num_cols must be positive",
        ),
        (
            torch.empty((0, 16), dtype=torch.float32, device="cuda"),
            torch.zeros((1,), dtype=torch.int32, device="cuda"),
            0,
            "next_n must be positive",
        ),
    ]
    for input_values, seq_lens, next_n, message in cases:
        with pytest.raises(ValueError, match=message):
            DSA.IndexerTopK(input_values, seq_lens, 1, next_n=next_n).check_support()

    sample_input = torch.empty((1, 16), dtype=torch.float32, device="cuda")
    sample_seq_lens = torch.full((1,), 16, dtype=torch.int32, device="cuda")
    op = DSA.IndexerTopK(sample_input, sample_seq_lens, 1)
    assert op.check_support()
    with pytest.raises(ValueError, match="input_values must be on cuda"):
        op.execute(
            sample_input.cpu(),
            sample_seq_lens.cpu(),
            out_indices=torch.empty((1, 1), dtype=torch.int32),
            out_values=torch.empty((1, 1), dtype=torch.float32),
            workspace=torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8),
        )
    assert op._compiled_kernel is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "dtype,max_num_cols",
    [(torch.float32, 16384), (torch.float16, 32768), (torch.bfloat16, 32768)],
    ids=["fp32", "fp16", "bf16"],
)
def test_DSA_indexer_top_k_sm120_deterministic_smem_limit(monkeypatch, dtype, max_num_cols):
    """SM120 rejects deterministic shapes that exceed its per-block SMEM budget."""
    DSA = _import_dsa()
    from cudnn.deepseek_sparse_attention.indexer_top_k import api as topk_api

    monkeypatch.setattr(topk_api, "device_capability", lambda _device: (12, 0))
    seq_lens = torch.tensor([max_num_cols], dtype=torch.int32, device="cuda")

    supported_input = torch.empty((1, max_num_cols), dtype=dtype, device="cuda")
    supported = DSA.IndexerTopK(supported_input, seq_lens, top_k=512, deterministic=True)
    assert supported.check_support()

    unsupported_input = torch.empty((1, max_num_cols + 1), dtype=dtype, device="cuda")
    unsupported = DSA.IndexerTopK(unsupported_input, seq_lens, top_k=512, deterministic=True)
    with pytest.raises(RuntimeError, match=rf"SM120 supports at most {max_num_cols} columns"):
        unsupported.check_support()
    assert unsupported._compiled_kernel is None

    # The new admission check is scoped to the opt-in deterministic specialization.
    default = DSA.IndexerTopK(unsupported_input, seq_lens, top_k=512)
    assert default.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=48)
@pytest.mark.parametrize("dtype,num_cols", [(torch.float32, 16384), (torch.float16, 32768), (torch.bfloat16, 32768)])
@pytest.mark.parametrize("top_k", [129, 2048])
def test_DSA_indexer_top_k_sm120_native_boundary(dtype, num_cols, top_k):
    """Admission must reach native compilation, execution, and changed-input replay."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Native SM120 boundary qualification")
    DSA = _import_dsa()
    n_rows = 6
    columns = torch.arange(num_cols, device="cuda")
    input_values = torch.randn(n_rows, num_cols, dtype=dtype, device="cuda")
    input_values[-1].copy_(((columns % 31) - 15).to(dtype))
    seq_lens = torch.tensor([0, 1, top_k - 1, top_k, num_cols, num_cols], dtype=torch.int32, device="cuda")
    op = DSA.IndexerTopK(input_values, seq_lens, top_k, deterministic=True)
    assert op.check_support()
    op.compile()
    indices, values, workspace = _allocate_outputs(op, input_values)

    def execute():
        op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)

    def check():
        check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, indices, values, True, tie_break=1)
        for row, length in enumerate(seq_lens.tolist()):
            valid = min(length, top_k)
            assert torch.all(indices[row, 1:valid] >= indices[row, : max(0, valid - 1)])
            assert torch.all(indices[row, valid:] == -1)
            assert torch.all(torch.isneginf(values[row, valid:]))

    execute()
    check()
    baseline_indices, baseline_values = indices.clone(), values.clone()
    for _ in range(3):
        workspace.fill_(0xA5)
        indices.fill_(-99)
        values.fill_(123)
        execute()
        assert torch.equal(indices.view(torch.uint8), baseline_indices.view(torch.uint8))
        assert torch.equal(values.view(torch.uint8), baseline_values.view(torch.uint8))

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            execute()
        input_values.copy_(torch.randn_like(input_values))
        input_values[0].copy_(((columns * 13 % 37) - 18).to(dtype))
        seq_lens.copy_(torch.tensor([num_cols, num_cols, top_k + 1, top_k, 1, 0], dtype=torch.int32, device="cuda"))
        workspace.fill_(0x5A)
        indices.fill_(-98)
        values.fill_(456)
        graph.replay()
        torch.cuda.synchronize()
        check()
    finally:
        graph.reset()


@pytest.mark.L0
@torch_fork_set_rng(seed=49)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_DSA_indexer_top_k_sm120_default_spill(dtype):
    """Small row counts must still fit the SM120 shared-memory budget."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Native SM120 spill qualification")
    DSA = _import_dsa()
    num_cols, top_k = 131072, 2048
    input_values = torch.randn(3, num_cols, dtype=dtype, device="cuda")
    seq_lens = torch.tensor([top_k - 1, top_k + 1, num_cols], dtype=torch.int32, device="cuda")
    result = DSA.indexer_top_k_wrapper(input_values, seq_lens, top_k, return_val=True, tie_break=1)
    check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, result["indices"], result["values"], True, tie_break=1)


@pytest.mark.L0
def test_DSA_indexer_top_k_deterministic_compact_index_storage():
    """Determinism widens indices only for a colliding bitonic sentinel."""
    import cutlass

    from cudnn.deepseek_sparse_attention.indexer_top_k.indexer_top_k_varlen_util import (
        IndexerTopKKernelVarlen,
    )

    bitmap = IndexerTopKKernelVarlen(
        cutlass.Float32,
        max_num_cols=16384,
        top_k=512,
        deterministic=True,
        deterministic_ordering="bitmap",
    )
    assert bitmap.index_type == cutlass.Uint16
    input_index_bytes = bitmap.num_buffer_smem_input_idx * bitmap.indexer_topk_smem_input_size * bitmap.index_type.width // 8
    assert input_index_bytes == 64 * 1024

    bitmap_boundary = IndexerTopKKernelVarlen(
        cutlass.Float32,
        max_num_cols=65536,
        top_k=512,
        deterministic=True,
        deterministic_ordering="bitmap",
    )
    assert bitmap_boundary.index_type == cutlass.Uint16

    bitonic_compact = IndexerTopKKernelVarlen(
        cutlass.Float32,
        max_num_cols=32768,
        top_k=128,
        deterministic=True,
        deterministic_ordering="bitonic",
    )
    assert bitonic_compact.index_type == cutlass.Uint16
    assert bitonic_compact.index_sentinel == 0xFFFF

    bitonic_boundary = IndexerTopKKernelVarlen(
        cutlass.Float32,
        max_num_cols=65536,
        top_k=128,
        deterministic=True,
        deterministic_ordering="bitonic",
    )
    assert bitonic_boundary.index_type == cutlass.Uint32
    assert bitonic_boundary.index_sentinel == 0xFFFFFFFF


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_DSA_indexer_top_k_rejects_overlapping_writable_ranges(monkeypatch):
    """Reject exact byte-range overlap before launch, but allow disjoint views of one allocation."""
    DSA = _import_dsa()
    from cudnn.deepseek_sparse_attention.indexer_top_k import (
        indexer_top_k_decode_varlen as kernel_module,
    )

    n_rows, num_cols, top_k = 2, 64, 8
    input_values = torch.randn(n_rows, num_cols, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")
    op = DSA.IndexerTopK(input_values, seq_lens, top_k)
    assert op.check_support()
    op._compiled_kernel = object()

    workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    indices = torch.empty((n_rows, top_k), dtype=torch.int32, device="cuda")
    values = torch.empty((n_rows, top_k), dtype=torch.float32, device="cuda")
    launch_count = 0

    def launch_stub(*args, **kwargs):
        nonlocal launch_count
        launch_count += 1

    monkeypatch.setattr(kernel_module, "launch_topk_kernel", launch_stub)

    output_storage = torch.empty(n_rows * top_k * 4, dtype=torch.uint8, device="cuda")
    overlapping_indices = output_storage.view(torch.int32).view(n_rows, top_k)
    overlapping_values = output_storage.view(torch.float32).view(n_rows, top_k)
    with pytest.raises(ValueError, match="out_values overlaps out_indices"):
        op.execute(
            input_values,
            seq_lens,
            out_indices=overlapping_indices,
            out_values=overlapping_values,
            workspace=workspace,
        )

    workspace_indices = workspace[: n_rows * top_k * 4].view(torch.int32).view(n_rows, top_k)
    with pytest.raises(ValueError, match="workspace scratch overlaps out_indices"):
        op.execute(input_values, seq_lens, out_indices=workspace_indices, out_values=values, workspace=workspace)

    input_values_output = input_values.view(-1)[: n_rows * top_k].view(n_rows, top_k)
    with pytest.raises(ValueError, match="out_values overlaps input_values"):
        op.execute(input_values, seq_lens, out_indices=indices, out_values=input_values_output, workspace=workspace)

    assert launch_count == 0

    # Sharing storage is valid when the byte ranges are disjoint. Stub the
    # launch so this test remains focused on validation rather than kernel math.
    required = op.scratch_workspace_bytes()
    shared = torch.empty(required + n_rows * top_k * 4, dtype=torch.uint8, device="cuda")
    disjoint_workspace = shared[:required]
    disjoint_indices = shared[required:].view(torch.int32).view(n_rows, top_k)
    op.execute(input_values, seq_lens, out_indices=disjoint_indices, out_values=values, workspace=disjoint_workspace)
    assert launch_count == 1


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
        from cuda.bindings import driver as cuda
        from cudnn import DSA
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

    # contiguous() returns an already-contiguous storage-offset view unchanged;
    # the wrapper must clone it to repair the kernel's 32-byte base alignment.
    misaligned_storage = torch.randn(n_rows * num_cols + 1, dtype=torch.float32, device="cuda")
    misaligned_input = misaligned_storage[1:].view(n_rows, num_cols)
    assert misaligned_input.is_contiguous() and misaligned_input.data_ptr() % 32 != 0
    misaligned_op = DSA.IndexerTopK(misaligned_input, seq_lens.contiguous(), top_k)
    assert misaligned_op.check_support()
    misaligned_op._compiled_kernel = object()
    misaligned_indices, misaligned_values, misaligned_workspace = _allocate_outputs(misaligned_op, misaligned_input)
    with pytest.raises(ValueError, match="input_values must be 32-byte aligned"):
        misaligned_op.execute(
            misaligned_input,
            seq_lens.contiguous(),
            out_indices=misaligned_indices,
            out_values=misaligned_values,
            workspace=misaligned_workspace,
        )

    got_misaligned = DSA.indexer_top_k_wrapper(misaligned_input, seq_lens.contiguous(), top_k)
    expected_misaligned = DSA.indexer_top_k_wrapper(misaligned_input.clone(), seq_lens.contiguous(), top_k)
    torch.testing.assert_close(
        torch.sort(got_misaligned["values"], dim=1).values,
        torch.sort(expected_misaligned["values"], dim=1).values,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(torch.sort(got_misaligned["indices"], dim=1).values, torch.sort(expected_misaligned["indices"], dim=1).values)


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


@torch_fork_set_rng(seed=0)
@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tie_break", [0, 2], ids=["implicit-small", "large"])
def test_DSA_indexer_top_k_deterministic_repeatability(dtype, tie_break):
    """Deterministic mode fixes both the cutoff-tie set and output slot order."""
    DSA = _import_dsa()

    n_rows, num_cols, top_k = 17, 1024, 129
    columns = torch.arange(num_cols, device="cuda")
    input_values = ((columns % 7) - 3).to(dtype).expand(n_rows, -1).contiguous()
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")

    baseline = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        return_val=True,
        tie_break=tie_break,
        deterministic=True,
    )
    assert torch.all(baseline["indices"][:, 1:] >= baseline["indices"][:, :-1])
    effective_tie_break = 1 if tie_break == 0 else tie_break
    check_ref_indexer_top_k(
        input_values,
        seq_lens,
        top_k,
        1,
        baseline["indices"],
        baseline["values"],
        True,
        tie_break=effective_tie_break,
    )

    for _ in range(5):
        result = DSA.indexer_top_k_wrapper(
            input_values,
            seq_lens,
            top_k,
            return_val=True,
            tie_break=tie_break,
            deterministic=True,
        )
        assert torch.equal(result["indices"], baseline["indices"])
        assert torch.equal(result["values"], baseline["values"])


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
def test_DSA_indexer_top_k_deterministic_indices_only_repeatability():
    """The deterministic indices-only specialization preserves canonical ordering."""
    DSA = _import_dsa()

    n_rows, num_cols, top_k = 3, 1024, 129
    columns = torch.arange(num_cols, device="cuda")
    input_values = ((columns % 7) - 3).to(torch.float32).expand(n_rows, -1).contiguous()
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")

    baseline = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        return_val=False,
        deterministic=True,
    )
    assert baseline["values"] is None
    assert torch.all(baseline["indices"][:, 1:] >= baseline["indices"][:, :-1])
    check_ref_indexer_top_k(
        input_values,
        seq_lens,
        top_k,
        1,
        baseline["indices"],
        None,
        False,
        tie_break=1,
    )

    for _ in range(5):
        result = DSA.indexer_top_k_wrapper(
            input_values,
            seq_lens,
            top_k,
            return_val=False,
            deterministic=True,
        )
        assert result["values"] is None
        assert torch.equal(result["indices"], baseline["indices"])


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
@pytest.mark.parametrize(
    "dtype,n_rows,num_cols,top_k,return_val",
    [
        (torch.float32, 2, 17, 1, True),
        (torch.float32, 16, 8192, 512, True),
        (torch.float32, 8, 8192, 2048, True),
        (torch.float16, 4, 32000, 1024, True),
        (torch.bfloat16, 4, 65536, 512, True),
        (torch.float32, 1, 262144, 512, True),
        (torch.float32, 1, 262145, 512, True),
    ],
    ids=[
        "fp32-k1",
        "fp32-k512",
        "fp32-k2048",
        "fp16-k1024",
        "bf16-k512",
        "fp32-256k-cols",
        "fp32-above-256k-cols",
    ],
)
def test_DSA_indexer_top_k_deterministic_random_exact(dtype, n_rows, num_cols, top_k, return_val):
    """Random rows catch compare-exchange ordering bugs hidden by equal-value inputs."""
    DSA = _import_dsa()
    input_values = torch.randn(n_rows, num_cols, dtype=dtype, device="cuda")
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")

    if torch.cuda.get_device_capability() == (12, 0) and num_cols > (16384 if dtype == torch.float32 else 32768):
        with pytest.raises(RuntimeError, match="SM120 supports at most"):
            DSA.IndexerTopK(input_values, seq_lens, top_k, return_val=return_val, deterministic=True).check_support()
        return

    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        return_val=return_val,
        deterministic=True,
    )
    assert torch.all(result["indices"][:, 1:] >= result["indices"][:, :-1])
    check_ref_indexer_top_k(
        input_values,
        seq_lens,
        top_k,
        1,
        result["indices"],
        result["values"],
        return_val,
        tie_break=1,
    )

    gathered = input_values.gather(1, result["indices"].to(torch.int64))
    expected = torch.topk(input_values, top_k, dim=1, sorted=False).values
    torch.testing.assert_close(
        torch.sort(gathered, dim=1).values,
        torch.sort(expected, dim=1).values,
        atol=0,
        rtol=0,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=44)
def test_DSA_indexer_top_k_deterministic_special_radix_values():
    """Canonical ties cover signed zero, infinities, and adjacent FP32 keys."""
    DSA = _import_dsa()
    n_rows, num_cols, top_k = 8, 2048, 513
    columns = torch.arange(num_cols, device="cuda")
    input_values = torch.empty(n_rows, num_cols, dtype=torch.float32, device="cuda")
    input_values[0].fill_(1.0)
    input_values[1].fill_(0.0)
    input_values[2].fill_(-0.0)
    input_values[3] = torch.where(columns % 2 == 0, 0.0, -0.0)
    input_values[4] = torch.where(columns % 3 == 0, float("inf"), 1.0)
    input_values[5].fill_(float("-inf"))
    input_values[5, :top_k] = 1.0
    input_values[6] = 1.0 + ((columns * 17) % 8).to(torch.float32) * torch.finfo(torch.float32).eps
    input_values[7] = ((columns % 7) - 3).to(torch.float32)
    seq_lens = torch.full((n_rows,), num_cols, dtype=torch.int32, device="cuda")

    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        return_val=True,
        deterministic=True,
    )
    assert torch.all(result["indices"][:, 1:] >= result["indices"][:, :-1])
    check_ref_indexer_top_k(
        input_values,
        seq_lens,
        top_k,
        1,
        result["indices"],
        result["values"],
        True,
        tie_break=1,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=45)
def test_DSA_indexer_top_k_deterministic_empty_speculative_rows():
    """A speculative stagger may yield empty rows; their padding is canonical."""
    DSA = _import_dsa()
    next_n, num_cols, top_k = 4, 64, 16
    input_values = torch.randn(next_n, num_cols, dtype=torch.float32, device="cuda")
    seq_lens = torch.tensor([2], dtype=torch.int32, device="cuda")

    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        next_n=next_n,
        return_val=True,
        deterministic=True,
    )
    indices = result["indices"]
    values = result["values"]
    assert torch.all(indices[:2] == -1)
    assert torch.all(torch.isneginf(values[:2]))
    assert torch.equal(indices[2, :1], torch.tensor([0], dtype=torch.int32, device="cuda"))
    assert torch.equal(indices[3, :2], torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    assert torch.all(indices[2, 1:] == -1)
    assert torch.all(indices[3, 2:] == -1)
    assert torch.all(torch.isneginf(values[2, 1:]))
    assert torch.all(torch.isneginf(values[3, 2:]))


@pytest.mark.L0
@torch_fork_set_rng(seed=46)
def test_DSA_indexer_top_k_deterministic_mixed_speculative_rows():
    """next_n groups cover long radix selection and short/equal K boundaries together."""
    DSA = _import_dsa()
    next_n, num_cols, top_k = 4, 1024, 129
    n_rows = 2 * next_n
    columns = torch.arange(num_cols, device="cuda")
    input_values = ((columns % 7) - 3).to(torch.float32).expand(n_rows, -1).contiguous()
    # Effective row lengths are 1021..1024 and 127..130.
    seq_lens = torch.tensor([num_cols, 130], dtype=torch.int32, device="cuda")

    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        next_n=next_n,
        return_val=True,
        deterministic=True,
    )
    check_ref_indexer_top_k(
        input_values,
        seq_lens,
        top_k,
        next_n,
        result["indices"],
        result["values"],
        True,
        tie_break=1,
    )
    effective_lengths = (1021, 1022, 1023, 1024, 127, 128, 129, 130)
    for row, length in enumerate(effective_lengths):
        valid = min(length, top_k)
        assert torch.all(result["indices"][row, 1:valid] >= result["indices"][row, : valid - 1])
        assert torch.all(result["indices"][row, valid:] == -1)
        assert torch.all(torch.isneginf(result["values"][row, valid:]))


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
def test_DSA_indexer_top_k_deterministic_short_rows_and_graph_replay():
    """Capture uses current data/lengths and preserves canonical valid slots plus padding."""
    DSA = _import_dsa()
    n_rows, num_cols, top_k = 4, 4096, 512
    input_values = torch.randn(n_rows, num_cols, dtype=torch.float32, device="cuda")
    seq_lens = torch.tensor([127, 512, 513, num_cols], dtype=torch.int32, device="cuda")

    op = DSA.IndexerTopK(input_values, seq_lens, top_k, deterministic=True)
    assert op.check_support()
    op.compile()
    indices, values, workspace = _allocate_outputs(op, input_values)
    op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)

        input_values.copy_(torch.randn_like(input_values))
        seq_lens.copy_(torch.tensor([num_cols, 513, 512, 127], dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()

        check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, indices, values, True, tie_break=1)
        for row, length in enumerate(seq_lens.tolist()):
            valid = min(length, top_k)
            assert torch.all(indices[row, 1:valid] >= indices[row, : valid - 1])
            assert torch.all(indices[row, valid:] == -1)
    finally:
        graph.reset()


@pytest.mark.L0
@torch_fork_set_rng(seed=47)
def test_DSA_indexer_top_k_deterministic_k2048_short_rows_and_graph_replay():
    """K=2048 covers empty, short, exact-K, and radix-selected rows under replay."""
    DSA = _import_dsa()
    n_rows, num_cols, top_k = 6, 4096, 2048
    columns = torch.arange(num_cols, device="cuda")
    input_values = ((columns % 31) - 15).to(torch.float32).expand(n_rows, -1).contiguous()
    seq_lens = torch.tensor([0, 1, top_k - 1, top_k, top_k + 1, num_cols], dtype=torch.int32, device="cuda")

    op = DSA.IndexerTopK(input_values, seq_lens, top_k, deterministic=True)
    assert op.check_support()
    op.compile()
    indices, values, workspace = _allocate_outputs(op, input_values)
    op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            op.execute(input_values, seq_lens, out_indices=indices, out_values=values, workspace=workspace)

        input_values.copy_(((columns * 13 % 37) - 18).to(torch.float32).expand(n_rows, -1))
        seq_lens.copy_(torch.tensor([num_cols, top_k + 1, top_k, top_k - 1, 1, 0], dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()

        check_ref_indexer_top_k(input_values, seq_lens, top_k, 1, indices, values, True, tie_break=1)
        for row, length in enumerate(seq_lens.tolist()):
            valid = min(length, top_k)
            if valid > 1:
                assert torch.all(indices[row, 1:valid] >= indices[row, : valid - 1])
            assert torch.all(indices[row, valid:] == -1)
            assert torch.all(torch.isneginf(values[row, valid:]))
    finally:
        graph.reset()
