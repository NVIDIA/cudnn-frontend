# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_indexer_top_k
from deepseek_sparse_attention.cutedsl.dsa_utils import (
    dsa_init,
    with_dsa_indexer_top_k_params,
)
from determinism_utils import assert_bitwise, assert_golden, assert_replays
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
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("Native SM120 / SM121 boundary qualification")
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
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("Native SM120 / SM121 spill qualification")
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
@pytest.mark.parametrize(
    "rows,num_cols,top_k,expected",
    [
        (1, 4096, 64, "bitmap"),  # narrow rows: bitmap at every top_k (the SM100 rule picks bitonic here)
        (64, 65536, 128, "bitmap"),
        (64, 131072, 128, "bitonic"),  # wide rows, small top_k
        (64, 131072, 129, "bitmap"),
        (512, 262144, 512, "bitonic"),  # large occupancy widens the bitonic band to top_k 512
        (64, 262144, 512, "bitmap"),
        (512, 262144, 2048, "bitmap"),
        (1, 300000, 2048, "bitonic"),  # past the bitmap column limit
    ],
)
def test_DSA_indexer_top_k_sm103_auto_ordering(monkeypatch, rows, num_cols, top_k, expected):
    """On SM103 the deterministic auto ordering follows the B300 measurements, not the SM100 bitmap limits."""
    from cudnn.deepseek_sparse_attention.indexer_top_k import indexer_top_k_decode_varlen as kernel_module

    class _Resolved(Exception):
        pass

    def capture(max_num_cols, deterministic, ordering):
        raise _Resolved(ordering)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(kernel_module, "resolve_deterministic_ordering", capture)
    with pytest.raises(_Resolved) as resolved:
        kernel_module.compile_topk_kernel(torch.float32, rows, num_cols, top_k, 1, deterministic=True)
    assert resolved.value.args[0] == expected
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    with pytest.raises(_Resolved) as resolved:
        kernel_module.compile_topk_kernel(torch.float32, rows, num_cols, top_k, 1, deterministic=True)
    bitmap_limit = kernel_module._sm100_deterministic_bitmap_max_cols(kernel_module.cutlass.Float32, top_k, rows > 148)
    assert resolved.value.args[0] == ("bitmap" if kernel_module._bucket_num_cols(num_cols) <= bitmap_limit else "bitonic")


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

    if torch.cuda.get_device_capability() in ((12, 0), (12, 1)) and num_cols > (16384 if dtype == torch.float32 else 32768):
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


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_val,deterministic,tie_break", [(False, False, 1), (True, False, 2), (True, True, 0), (False, True, 2)])
def test_indexer_top_k_starts(dtype, return_val, deterministic, tie_break):
    """Compare original column ids and values with a per-interval oracle, including cutoff ties."""
    dsa = _import_dsa()
    width, k = 2053, 33
    starts_list = [1900, 1970, 5, 7, 2030, width]
    ends_list = [2003, 39, width]
    next_n = 2
    starts = torch.tensor(starts_list, dtype=torch.int32, device="cuda").repeat_interleave(2)[::2]
    ends = torch.tensor(ends_list, dtype=torch.int32, device="cuda")
    scores = (torch.arange(width, device="cuda") % 11).to(dtype).repeat(len(starts_list), 1)
    expected = []
    for r, start in enumerate(starts_list):
        end = ends_list[r // next_n] - next_n + r % next_n + 1
        scores[r, :start] = torch.inf
        scores[r, end:] = torch.inf
        cols = torch.arange(start, max(start, end), device="cuda")
        if tie_break == 2:
            cols = cols.flip(0)
        chosen = cols[scores[r, cols].argsort(descending=True, stable=True)[:k]]
        expected.append(chosen.sort().values)

    result = dsa.indexer_top_k_wrapper(scores, ends, k, next_n, return_val, tie_break=tie_break, deterministic=deterministic, starts=starts)
    for r, cols in enumerate(expected):
        n = cols.numel()
        ids = result["indices"][r, :n].long()
        torch.testing.assert_close(ids if deterministic else ids.sort().values, cols)
        assert (result["indices"][r, n:] == -1).all()
        if return_val:
            torch.testing.assert_close(result["values"][r, :n], scores[r, ids])
            assert result["values"][r, n:].isneginf().all()
    if not return_val:
        assert result["values"] is None


@pytest.mark.L0
def test_indexer_top_k_starts_graph_replay():
    """Replay a prepared plan with changed lower bounds at the captured address."""
    dsa = _import_dsa()
    width, k = 1024, 32
    scores = -torch.arange(width, device="cuda", dtype=torch.float32).repeat(2, 1)
    ends = torch.full((2,), width, device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(ends)
    op = dsa.IndexerTopK(scores, ends, k, deterministic=True, has_starts=True)
    op.check_support()
    op.compile()
    ids, values, workspace = _allocate_outputs(op, scores)
    outputs = dict(out_indices=ids, out_values=values, workspace=workspace)
    with pytest.raises(ValueError, match="starts"):
        op.execute(scores, ends, **outputs)
    with pytest.raises(ValueError, match="overlaps starts"):
        op.execute(scores, ends, starts=ids.flatten()[:2], **outputs)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            op.execute(scores, ends, starts=starts, **outputs)
        for bounds in ((977, 1019), (width, 51), (977, 1019)):
            starts.copy_(torch.tensor(bounds, dtype=starts.dtype, device="cuda"))
            ids.fill_(-7)
            values.fill_(torch.nan)
            graph.replay()
            cols = starts[:, None] + torch.arange(k, device="cuda", dtype=torch.int32)
            expected = torch.where(cols < width, cols, -1)
            torch.testing.assert_close(ids, expected)
            torch.testing.assert_close(values, torch.where(cols < width, -cols.float(), -torch.inf))
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_indexer_top_k_starts_batch_invariance_and_compile_cache(monkeypatch, dtype):
    """Runtime lower bounds and row regrouping reuse one bounded specialization."""
    from unittest.mock import Mock

    _import_dsa()
    from cudnn.deepseek_sparse_attention.indexer_top_k import indexer_top_k_decode_varlen as kernel
    from cudnn.deepseek_sparse_attention.indexer_top_k import api

    monkeypatch.setattr(kernel, "_compile_cache", {})
    monkeypatch.setattr(api, "_cache_of_IndexerTopKObjects", {})
    compile_spy = Mock(wraps=kernel.cute.compile)
    monkeypatch.setattr(kernel.cute, "compile", compile_spy)

    scores = (torch.arange(2053, device="cuda") % 7).to(dtype).repeat(3, 1)
    ends = torch.tensor([2048, 137, 27], dtype=torch.int32, device="cuda")
    starts = torch.tensor([1931, 7, 33], dtype=torch.int32, device="cuda")
    kwargs = dict(top_k=32, deterministic=True)
    original = api.indexer_top_k_wrapper(scores, ends, starts=starts, **kwargs)
    for order in ([2, 0, 0], [1]):
        result = api.indexer_top_k_wrapper(scores[order], ends[order], starts=starts[order], **kwargs)
        for key in ("indices", "values"):
            assert torch.equal(result[key].view(torch.uint8), original[key][order].view(torch.uint8))
    assert compile_spy.call_count == 1
    api.indexer_top_k_wrapper(scores, ends, **kwargs)
    assert compile_spy.call_count == 2


def _starts_spill_case(rows):
    width, k = 65539, 2048
    columns = torch.arange(width, device="cuda", dtype=torch.int32)
    low = columns % 67
    # All scores share a coarse bin around 2.0. Eliminate one half of its
    # candidates at each of the first three exact radix digits.
    bits = torch.where(columns % 2 == 0, 0x3FFFFC00 | low, 0x40000000 | ((columns & 2) << 15) | ((columns & 4) << 6) | low)
    scores = bits.view(torch.float32).repeat(rows, 1)
    intervals = [(3, width), (139, width - 19)]
    intervals = (intervals * ((rows + 1) // 2))[:rows]
    starts = torch.tensor([start for start, _ in intervals], device="cuda", dtype=torch.int32)
    ends = torch.tensor([end for _, end in intervals], device="cuda", dtype=torch.int32)
    scores.masked_fill_((columns < starts[:, None]) | (columns >= ends[:, None]), torch.inf)

    expected_rows = []
    for row, (start, end) in enumerate(intervals[:2]):
        descending_columns = columns[start:end].flip(0)
        order = scores[row, descending_columns].argsort(descending=True, stable=True)
        expected_rows.append(descending_columns[order[:k]].sort().values)
    expected = torch.stack(expected_rows).repeat((rows + 1) // 2, 1)[:rows]
    return scores, starts, ends, k, {"indices": expected.to(torch.int32), "values": scores.gather(1, expected)}


@pytest.mark.L0
def test_indexer_top_k_starts_spills_across_refinement_rounds():
    """Check bitwise replay and regrouping across the large-occupancy boundary."""
    dsa = _import_dsa()
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Wide deterministic FP32 Top-K requires the SM90 or SM100 family")
    scores, starts, ends, k, expected = _starts_spill_case(2)
    original = {name: tensor.clone() for name, tensor in zip(("scores", "starts", "ends"), (scores, starts, ends))}

    def run(x, lo, hi):
        result = dsa.indexer_top_k_wrapper(x, hi, k, starts=lo, tie_break=2, deterministic=True)
        return {"indices": result["indices"], "values": result["values"]}

    replay = assert_replays(lambda: run(scores, starts, ends))
    assert_bitwise(replay, expected, "independent interval reference")
    neighbors = torch.arange(152, device="cuda") % 2
    grouped_scores = scores[neighbors]
    grouped_scores[1::2, 256:] = 3.0
    grouped = run(grouped_scores, starts[neighbors], ends[neighbors])
    assert_bitwise({name: payload[::2] for name, payload in grouped.items()}, {name: payload[:1].expand(76, -1) for name, payload in replay.items()})
    reordered = run(scores.flip(0), starts.flip(0), ends.flip(0))
    assert_bitwise(reordered, {name: payload.flip(0) for name, payload in replay.items()}, "reordered rows")
    assert_bitwise({"scores": scores, "starts": starts, "ends": ends}, original, "unmodified inputs")


@pytest.mark.L0
def test_indexer_top_k_starts_spill_warm_stress():
    from warm_stress import StressCase, assert_warm_stress

    dsa = _import_dsa()
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Warmed wide FP32 Top-K requires the SM90 or SM100 family")
    rows = 2 * torch.cuda.get_device_properties(0).multi_processor_count
    scores, starts, ends, k, expected = _starts_spill_case(rows)
    op = dsa.IndexerTopK(scores, ends, k, has_starts=True, tie_break=2, deterministic=True)
    op.compile()
    indices, values, workspace = _allocate_outputs(op, scores)

    def launch():
        op.execute(scores, ends, starts=starts, out_indices=indices, out_values=values, workspace=workspace)

    stress = StressCase(
        "standalone DSA Top-K interval candidate refill",
        launch,
        {"indices": indices, "values": values},
        thread_blocks=rows,
        refills_per_launch=3,
        reference=expected,
    )
    assert_warm_stress(stress, warm=128, poisoned=128)


@pytest.mark.L0
def test_indexer_top_k_starts_bf16_mixed_interval_widths():
    """Verify canonical BF16 indices across the bitmap capacity boundary."""
    dsa = _import_dsa()
    rows, width, k = 152, 65539, 512
    columns = torch.arange(width, device="cuda")
    scores = ((columns * 7) % 17).to(torch.bfloat16).repeat(rows, 1)
    intervals = [(11, 4104), (7, 32775), (13, 32782), (39, 6560)]
    pattern = torch.arange(rows, device="cuda") % len(intervals)
    starts = torch.tensor([start for start, _ in intervals], device="cuda", dtype=torch.int32)[pattern]
    ends = torch.tensor([end for _, end in intervals], device="cuda", dtype=torch.int32)[pattern]
    scores.masked_fill_((columns < starts[:, None]) | (columns >= ends[:, None]), torch.inf)
    expected_rows = []
    for row, (start, end) in enumerate(intervals):
        order = scores[row, start:end].argsort(descending=True, stable=True)[:k]
        expected_rows.append((order + start).sort().values)
    expected = torch.stack(expected_rows)[pattern]

    def run():
        result = dsa.indexer_top_k_wrapper(scores, ends, k, starts=starts, deterministic=True)
        return {"indices": result["indices"], "values": result["values"]}

    assert_bitwise(
        assert_replays(run),
        {"indices": expected.to(torch.int32), "values": scores.gather(1, expected)},
        "independent reference across both ordering paths",
    )


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tie_break", [1, 2])
def test_indexer_top_k_starts_signed_zero_cutoff(dtype, tie_break):
    dsa = _import_dsa()
    scores = torch.tensor(
        [[-torch.inf, torch.inf, torch.inf, -0.0, -0.0, 0.0, 1.0, -0.0, 0.0, -0.0, -1.0, torch.inf]],
        device="cuda",
        dtype=dtype,
    )
    starts = torch.tensor([3], dtype=torch.int32, device="cuda")
    ends = torch.tensor([11], dtype=torch.int32, device="cuda")
    # Positive zeros outrank negative zeros. The cutoff is between negative zeros.
    ids = torch.tensor([[3, 4, 5, 6, 8] if tie_break == 1 else [5, 6, 7, 8, 9]], device="cuda", dtype=torch.int32)

    def run():
        result = dsa.indexer_top_k_wrapper(scores, ends, 5, starts=starts, deterministic=True, tie_break=tie_break)
        return {"indices": result["indices"], "values": result["values"]}

    assert_bitwise(assert_replays(run), {"indices": ids, "values": scores.gather(1, ids.long())}, "signed-zero cutoff")


@pytest.mark.L0
@pytest.mark.parametrize("has_starts", [False, True], ids=["prefix", "offset"])
@pytest.mark.parametrize("tie_break", [0, 1, 2])
@pytest.mark.parametrize("deterministic,return_val", [(False, True), (True, True), (False, False)], ids=["values", "deterministic", "indices"])
def test_indexer_top_k_spilled_candidate_planes(has_starts, tie_break, deterministic, return_val):
    """Exercise reuse, filtering, and reuse after switching a candidate plane."""
    dsa = _import_dsa()
    width, k, rows = 32773, 2048, 156
    columns = torch.arange(width, device="cuda", dtype=torch.int32)
    low = columns % 67
    common = (0x3F800000 | low).view(torch.float32)
    two_bins = (0x3F800000 | ((columns % 2) << 16) | low).view(torch.float32)
    # Only the small group of ones enters refinement for this full-width row.
    shared = torch.where(columns < k - 1, 2.0, torch.where(columns < k + 31, 1.0, 0.0))
    patterns = torch.stack((common, two_bins, common, two_bins, common, two_bins, shared))
    order = torch.arange(rows, device="cuda") % len(patterns)
    scores = patterns[order]
    starts_list = [3, 139, 7, 21, 5, 17, 11] if has_starts else [0] * len(patterns)
    ends_list = [width, width - 19, starts_list[2] + 16383, starts_list[3] + 16384, starts_list[4] + 17, starts_list[5], width]
    starts = torch.tensor(starts_list, dtype=torch.int32, device="cuda")[order]
    ends = torch.tensor(ends_list, dtype=torch.int32, device="cuda")[order]
    scores.masked_fill_((columns < starts[:, None]) | (columns >= ends[:, None]), torch.inf)

    def run():
        result = dsa.indexer_top_k_wrapper(
            scores, ends, k, starts=starts if has_starts else None, tie_break=tie_break, deterministic=deterministic, return_val=return_val
        )
        if return_val:
            return {"indices": result["indices"], "values": result["values"]}
        assert result["values"] is None
        return {"indices": result["indices"]}

    actual = assert_replays(run, allowed_inf={"values": -torch.inf}) if deterministic else run()
    for pattern, (start, end) in enumerate(zip(starts_list, ends_list)):
        count = min(k, end - start)
        candidates = torch.arange(start, end, device="cuda")
        if tie_break == 2:
            candidates = candidates.flip(0)
        ranked = candidates[patterns[pattern, candidates].argsort(descending=True, stable=True)[:count]]
        rows_ids = actual["indices"][pattern :: len(patterns)]
        valid_ids = rows_ids[:, :count].long()
        assert ((valid_ids >= start) & (valid_ids < end)).all()
        picked_values = scores[pattern :: len(patterns)].gather(1, valid_ids)
        sorted_ids = valid_ids.sort(dim=1).values
        assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
        if deterministic or tie_break:
            expected_ids = ranked.sort().values.expand_as(valid_ids)
            torch.testing.assert_close(valid_ids if deterministic else sorted_ids, expected_ids, atol=0, rtol=0)
        else:
            expected_values = patterns[pattern, ranked].sort().values.expand_as(picked_values)
            torch.testing.assert_close(picked_values.sort(dim=1).values, expected_values, atol=0, rtol=0)
        assert (rows_ids[:, count:] == -1).all()
        if return_val:
            rows_values = actual["values"][pattern :: len(patterns)]
            torch.testing.assert_close(rows_values[:, :count], picked_values, atol=0, rtol=0)
            assert rows_values[:, count:].isneginf().all()


@pytest.mark.L0
@pytest.mark.parametrize("has_starts", [False, True], ids=["prefix", "offset"])
@pytest.mark.parametrize("tie_break", [0, 1, 2])
def test_indexer_top_k_fp32_coarse_bin_refinement(has_starts, tie_break):
    """Check FP32 cutoff keys at the FP16 rounding boundaries used by coarse filtering."""
    dsa = _import_dsa()
    width, k, rows = 32773, 2048, 156
    columns = torch.arange(width, device="cuda", dtype=torch.int32)
    delta = columns * 17 % 127 - 63
    # Below and above 1.0 share the FP32 high byte. Around 2.0 it changes.
    # 2.5 lies in an interior mantissa group. Subnormal and overflow bins
    # also need the full refinement.
    centers = torch.tensor([0x3F800000, 0x40000000, 0x40200000, 0x33800000, 0x477FF080], device="cuda", dtype=torch.int32)
    magnitudes = (centers[:, None] + delta).view(torch.float32)
    patterns = torch.cat((magnitudes, -magnitudes, torch.zeros_like(magnitudes[:1])))
    pattern_count = len(patterns)
    first = [3, 11, 8, 17, 30, 1, 19, 24, 7, 12, 5] if has_starts else [0] * pattern_count
    last = [width, width, width, width, width, width, width, first[7] + 16383, first[8] + 16384, width, width]
    patterns[-1, first[-1] : first[-1] + k] = 32
    order = torch.arange(rows, device="cuda") % pattern_count
    starts = torch.tensor(first, device="cuda", dtype=torch.int32)[order]
    ends = torch.tensor(last, device="cuda", dtype=torch.int32)[order]
    scores = patterns[order]
    scores.masked_fill_((columns < starts[:, None]) | (columns >= ends[:, None]), torch.inf)

    def run(data=scores, lower=starts, upper=ends):
        result = dsa.indexer_top_k_wrapper(data, upper, k, starts=lower if has_starts else None, tie_break=tie_break, deterministic=True)
        return {"indices": result["indices"], "values": result["values"]}

    expected_indices = []
    expected_values = []
    for pattern, (start, end) in zip(patterns, zip(first, last)):
        candidates = torch.arange(start, end, device="cuda")
        if tie_break == 2:
            candidates = candidates.flip(0)
        indices = candidates[pattern[candidates].argsort(descending=True, stable=True)[:k]].sort().values
        expected_indices.append(indices.int())
        expected_values.append(pattern[indices])
    expected = {"indices": torch.stack(expected_indices)[order], "values": torch.stack(expected_values)[order]}
    actual = assert_replays(run)
    assert_bitwise(actual, expected, "FP32 interval oracle")
    for rows_to_check in ([0], [7, 2, 1, 7, 6, 10]):
        assert_bitwise(
            run(scores[rows_to_check], starts[rows_to_check], ends[rows_to_check]),
            {name: values[rows_to_check] for name, values in expected.items()},
            "isolated and reordered rows",
        )
    if has_starts and tie_break == 2:
        assert_golden("dsa-indexer-top-k-fp32-coarse-bin-refinement", {"scores": scores, "starts": starts, "ends": ends}, actual)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_starts", [False, True], ids=["prefix", "offset"])
@pytest.mark.parametrize("tie_break", [0, 1, 2])
@pytest.mark.parametrize("return_val", [False, True], ids=["indices", "values"])
def test_indexer_top_k_zero_scratch(dtype, has_starts, tie_break, return_val, compile_allocates_nothing):
    """The workspace-free plan copies original values and pads ragged rows."""
    dsa = _import_dsa()
    k, next_n = 513, 3
    width = k - int(has_starts)
    ends = torch.tensor([0, 2, width, width - 1], device="cuda", dtype=torch.int32)
    rows = ends.numel() * next_n
    columns = torch.arange(width, device="cuda")
    scores = ((columns[None, :] + torch.arange(rows, device="cuda")[:, None]) % 11 - 5).to(dtype)
    scores[:, 0], scores[:, 1] = -0.0, 0.0
    starts = torch.tensor([0, 1, 0, 1, 0, 1, 0, 7, 1, width - 2, width, 5], device="cuda", dtype=torch.int32) if has_starts else None
    op = dsa.IndexerTopK(scores, ends, k, next_n=next_n, return_val=return_val, tie_break=tie_break, deterministic=True, has_starts=has_starts)
    assert op.check_support()
    compile_allocates_nothing(op)
    assert op.scratch_workspace_bytes() == 0
    indices = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    values = torch.empty((rows, k), dtype=dtype, device="cuda") if return_val else None

    def execute(out_indices, out_values):
        op.execute(scores, ends, out_indices=out_indices, out_values=out_values, starts=starts)
        return {"indices": out_indices, "values": out_values} if return_val else {"indices": out_indices}

    def run():
        return execute(torch.empty_like(indices), torch.empty_like(values) if return_val else None)

    def reference():
        expected = {"indices": torch.full_like(indices, -1)}
        if return_val:
            expected["values"] = torch.full_like(values, -torch.inf)
        for row in range(rows):
            start = int(starts[row]) if has_starts else 0
            end = int(ends[row // next_n]) - next_n + row % next_n + 1
            if end > start:
                expected["indices"][row, : end - start] = columns[start:end]
                if return_val:
                    expected["values"][row, : end - start] = scores[row, start:end]
        return expected

    allowed_inf = {"values": -torch.inf}
    expected = assert_replays(run, allowed_inf=allowed_inf)
    assert_bitwise(expected, reference(), "fixed input")
    if has_starts and return_val and tie_break == 1:
        assert_golden(
            f"dsa-indexer-top-k-zero-scratch-{str(dtype).split('.')[-1]}",
            {"scores": scores, "starts": starts, "ends": ends},
            expected,
            allowed_inf=allowed_inf,
        )
    for groups in ([2], [3, 2, 2, 0] * 13):
        order = (torch.tensor(groups, device="cuda")[:, None] * next_n + torch.arange(next_n, device="cuda")).flatten()
        result = dsa.indexer_top_k_wrapper(
            scores[order],
            ends[groups],
            k,
            next_n=next_n,
            return_val=return_val,
            tie_break=tie_break,
            deterministic=True,
            starts=starts[order] if has_starts else None,
        )
        assert_bitwise({name: result[name] for name in expected}, {name: tensor[order] for name, tensor in expected.items()}, f"batch {groups}")

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            execute(indices, values)
        scores.neg_()
        ends.copy_(ends.flip(0))
        if has_starts:
            starts.copy_(starts.flip(0))
        graph.replay()
        assert_bitwise({"indices": indices, "values": values} if return_val else {"indices": indices}, reference(), "changed-input replay")
    finally:
        graph.reset()


@pytest.mark.L0
def test_indexer_top_k_zero_scratch_cache_boundary():
    """Width K and K+1 in the same width bucket require different kernels."""
    dsa = _import_dsa()
    k = 513
    for width in (k, k + 1, k):
        scores = torch.arange(width, device="cuda", dtype=torch.float32)[None, :]
        ends = torch.full((1,), width, dtype=torch.int32, device="cuda")
        plan = dsa.IndexerTopK(scores, ends, k, deterministic=True)
        assert (plan.scratch_workspace_bytes() == 0) == (width == k)
        result = dsa.indexer_top_k_wrapper(scores, ends, k, deterministic=True)
        ids = torch.arange(width - k, width, dtype=torch.int32, device="cuda")[None, :]
        assert_bitwise({"indices": result["indices"], "values": result["values"]}, {"indices": ids, "values": scores[:, -k:]})


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_indexer_top_k_starts_golden(dtype):
    """Pin cutoff ties and padding for deterministic standalone selection."""
    dsa = _import_dsa()
    scores = (torch.arange(39, device="cuda") % 5).to(dtype).repeat(3, 1)
    intervals = [(3, 39), (8, 10), (22, 22)]
    starts = torch.tensor([start for start, _ in intervals], dtype=torch.int32, device="cuda")
    ends = torch.tensor([end for _, end in intervals], dtype=torch.int32, device="cuda")
    k = 4

    def run():
        result = dsa.indexer_top_k_wrapper(scores, ends, k, starts=starts, deterministic=True)
        return {"indices": result["indices"], "values": result["values"]}

    expected_ids = torch.full((len(intervals), k), -1, dtype=torch.int32, device="cuda")
    expected_values = torch.full((len(intervals), k), -torch.inf, dtype=dtype, device="cuda")
    for row, (start, end) in enumerate(intervals):
        indices = (scores[row, start:end].argsort(descending=True, stable=True)[:k] + start).sort().values
        expected_ids[row, : indices.numel()] = indices.to(torch.int32)
        expected_values[row, : indices.numel()] = scores[row, indices]
    allowed_inf = {"values": -torch.inf}
    outputs = assert_replays(run, allowed_inf=allowed_inf)
    assert_bitwise(outputs, {"indices": expected_ids, "values": expected_values}, "independent interval reference")
    assert_golden(f"dsa-indexer-top-k-starts-{str(dtype).split('.')[-1]}", {"scores": scores, "starts": starts, "ends": ends}, outputs, allowed_inf=allowed_inf)
