# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
from typing import Iterable, Optional, Tuple

import pytest
import torch


def _arch_name(cc: Tuple[int, int]) -> str:
    return f"SM{cc[0]}{cc[1]}"


def require_arch(*, families: Iterable[int] = (), archs: Iterable[int] = (), feature: str = "this test", reason: Optional[str] = None) -> Tuple[int, int]:
    """Skip unless CUDA is available and the current GPU matches.

    ``families`` are compute-capability majors (``9`` = any SM9x); ``archs`` are exact compute
    capabilities written as ``major * 10 + minor`` (``100`` = SM100, ``103`` = SM103).  A GPU
    passes if it matches either list.  Returns the compute capability.  ``reason`` replaces the
    generated skip message.
    """
    if not torch.cuda.is_available():
        pytest.skip(f"{feature} requires CUDA")
    cc = torch.cuda.get_device_capability()
    families, archs = tuple(families), tuple(archs)
    if cc[0] in families or cc[0] * 10 + cc[1] in archs:
        return cc
    wanted = [f"SM{f}x" for f in families] + [f"SM{a}" for a in archs]
    pytest.skip(reason or f"{feature} requires {' or '.join(wanted)}; found {_arch_name(cc)}")


def require_cutedsl(min_version: Tuple[int, ...], feature: str = "this test") -> None:
    """Skip when the installed CuTe DSL is older than ``min_version``."""
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    if interface._cutlass_dsl_version() < tuple(min_version):
        pytest.skip(f"{feature} requires nvidia-cutlass-dsl>={'.'.join(map(str, min_version))}")


def import_bsa(
    *,
    families: Iterable[int] = (),
    archs: Iterable[int] = (),
    min_cutedsl: Optional[Tuple[int, ...]] = None,
    feature: str = "block sparse attention",
    reason: Optional[str] = None,
):
    """Return ``cudnn.ops`` for the BSA API, or skip.

    Checks, in order: the GPU architecture (only when ``families`` or ``archs`` is given), the
    optional dependencies of ``cudnn.block_sparse_attention``, and the CuTe DSL version.
    """
    families, archs = tuple(families), tuple(archs)
    if families or archs:
        require_arch(families=families, archs=archs, feature=feature, reason=reason)
    try:
        from cudnn import ops

        importlib.import_module("cudnn.block_sparse_attention._interface")
    except (ImportError, OSError) as error:
        return pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")
    if min_cutedsl is not None:
        require_cutedsl(min_cutedsl, feature)
    return ops


def import_kpage_bsa(reason: str):
    """``import_bsa`` for the KPage tests: SM90 or SM100/SM103 and the KPage CuTe DSL floor."""
    ops = import_bsa(families=(9, 10), reason=reason)
    from cudnn.block_sparse_attention._kpage_api import KPAGE_MIN_CUTEDSL

    require_cutedsl(KPAGE_MIN_CUTEDSL, "kpage")
    return ops


def require_sm90_fp8(block: int = 64):
    """Skip unless an SM90 GPU and the CuTe DSL floor of the Sage FP8 blk``block`` kernel (blk128 needs a newer one)."""
    require_arch(families=(9,), reason="the SM90 Sage FP8 kernels are under test")
    require_cutedsl((4, 6, 2) if block == 128 else (4, 6, 1), f"Sage FP8 blk{block}")


def require_sm120_fp8(*, per_token: bool = False):
    """``import_bsa`` for the SM120 Sage FP8 kernels: an SM120 GPU and their CuTe DSL floor (per-token K scales need 4.8)."""
    ops = import_bsa(families=(12,), min_cutedsl=(4, 6, 1), feature="Sage FP8", reason="the SM120 Sage FP8 kernels are under test")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("SM120 FP8 BSA tests")
    if error is not None:
        pytest.skip(error)
    if per_token:
        require_cutedsl((4, 8, 0), "Per-token SM120 FP8 BSA")
    return ops


@contextlib.contextmanager
def no_host_sync():
    """Fail on any host/device synchronization inside the block (``torch.cuda.set_sync_debug_mode("error")``)."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


@contextlib.contextmanager
def cuda_graph():
    """Yield a ``torch.cuda.CUDAGraph`` and release its memory pool on exit, also when the test fails."""
    graph = torch.cuda.CUDAGraph()
    try:
        yield graph
    finally:
        graph.reset()


def valid_rows(block_sizes: torch.Tensor, block_size: int) -> torch.Tensor:
    """Bool mask over the padded rows ``[num_blocks * block_size]``: row ``r`` of block ``j`` is valid iff ``r < block_sizes[j]``."""
    return (torch.arange(block_size, device=block_sizes.device).view(1, block_size) < block_sizes.view(-1, 1)).reshape(-1)


def random_metadata(heads: int, num_q_blocks: int, num_kv_blocks: int, kmax: int):
    """Unique random KV block lists per (head, Q block) with variable counts in [1, kmax]; batch 1."""
    kmax = min(kmax, num_kv_blocks)
    q2k = (
        torch.stack([torch.randperm(num_kv_blocks, device="cuda")[:kmax] for _ in range(heads * num_q_blocks)])
        .view(1, heads, num_q_blocks, kmax)
        .to(torch.int32)
    )
    nums = torch.randint(1, kmax + 1, (1, heads, num_q_blocks), device="cuda", dtype=torch.int32)
    return q2k, nums


def supported_block_size(*, backward: bool = False) -> int:
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    supported = {9, 10, 11} if backward else {9, 10, 11, 12}
    if major not in supported:
        direction = "backward" if backward else "forward"
        pytest.skip(f"block sparse attention {direction} is not supported on SM{major}x")
    return 64 if major in {9, 12} else 128


def make_fixed_metadata(batch: int, heads: int, seqlen_q: int, seqlen_k: int, block_size: int):
    num_q_blocks = (seqlen_q + block_size - 1) // block_size
    num_kv_blocks = (seqlen_k + block_size - 1) // block_size
    selected = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 2).expand(batch, heads, num_q_blocks, 2).contiguous()
    block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    block_sizes[-1] = seqlen_k - (num_kv_blocks - 1) * block_size
    return q2k, 2, block_sizes


def make_variable_metadata(batch: int, heads: int, seqlen_q: int, seqlen_k: int, block_size: int):
    num_q_blocks = (seqlen_q + block_size - 1) // block_size
    num_kv_blocks = (seqlen_k + block_size - 1) // block_size
    q2k = torch.empty((batch, heads, num_q_blocks, 3), dtype=torch.int32, device="cuda")
    q2k[..., 0, :] = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    q2k[..., 1, :] = torch.tensor([2, 0, 3], dtype=torch.int32, device="cuda")
    q2k_block_nums = torch.empty((batch, heads, num_q_blocks), dtype=torch.int32, device="cuda")
    q2k_block_nums[..., 0] = 1
    q2k_block_nums[..., 1] = 3
    block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    block_sizes[2] = block_size // 2
    return q2k, q2k_block_nums, block_sizes


def blk128_case(
    sq=256, sk=384, d=128, dv=128, ratio=1, dtype=torch.bfloat16, layout="bhsd", variable=False, sizes_rank=0, active_count=None, valid_size=None, b=2, h=4
):
    """Random blk128 problem: distinct KV blocks per query block, optional variable counts (inactive slots poisoned,
    one empty row) and ``block_sizes`` of rank ``sizes_rank`` (0 = none).  Returns ``q, k, v, indices, counts, sizes,
    count`` with ``q/k/v`` in ``layout``."""
    # Seeds the global generator (the directory's conftest restores it after each test): the caller's later draws,
    # such as dO, are then deterministic too.
    torch.manual_seed(109)
    q = torch.randn((b, h, sq, d), dtype=dtype, device="cuda")
    k = torch.randn((b, h // ratio, sk, d), dtype=dtype, device="cuda")
    v = torch.randn((b, h // ratio, sk, dv), dtype=dtype, device="cuda")
    nq, nk = (sq + 127) // 128, (sk + 127) // 128
    c = min(nk, 3 if active_count is None else active_count)
    indices = torch.rand(b, h, nq, nk, device="cuda").argsort(-1)[..., :c].int().contiguous()
    counts = torch.randint(0, c + 1, (b, h, nq), dtype=torch.int32, device="cuda") if variable else None
    if counts is not None:
        counts[0, 0, 0] = 0
        indices.masked_fill_(torch.arange(c, device="cuda") >= counts[..., None], 999999)
    sizes = None
    if sizes_rank:
        shape = {1: (nk,), 2: (b, nk), 3: (b, h, nk)}[sizes_rank]
        sizes = torch.randint(1, 129, shape, device="cuda", dtype=torch.int32)
        if valid_size is not None:
            sizes.fill_(valid_size)
        sizes[..., -1].clamp_(max=sk - 128 * (nk - 1))
    if layout == "bshd":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return q, k, v, indices, counts, sizes, c


def blk128_reference(q, k, v, indices, counts, sizes, count, scale, layout, do=None):
    """FP64 reference for ``blk128_case`` problems: ``(o, lse, grads or None)`` in ``layout``."""
    from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask

    if layout == "bshd":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        do = do.transpose(1, 2) if do is not None else None
    mask = block_sparse_mask(indices, count, sizes, q.shape[2], k.shape[2], 128, counts)
    out, lse, *grads = attention_reference(q, k, v, mask, scale, do=do, dtype=torch.float64)
    grads = tuple(grads) if do is not None else None
    if layout == "bshd":
        out = out.transpose(1, 2)
        grads = tuple(t.transpose(1, 2) for t in grads) if grads is not None else None
    return out, lse, grads


def check_blk128(
    sq=256,
    sk=384,
    d=128,
    dv=128,
    ratio=1,
    dtype=torch.bfloat16,
    layout="bhsd",
    variable=False,
    sizes_rank=0,
    splits=1,
    backward=True,
    bucket=1,
    scale=0.137,
    active_count=None,
    valid_size=None,
    strided=False,
    b=2,
    h=4,
):
    """Run the public blk128 forward (and backward) on a ``blk128_case`` problem and compare with the FP64 reference.

    ``strided`` passes metadata with a non-unit last stride and q/k/v rows inside wider allocations; the backward
    writes into row-padded gradient buffers whose padding must stay untouched.
    """
    from cudnn import ops

    q, k, v, ids, counts, sizes, count = blk128_case(sq, sk, d, dv, ratio, dtype, layout, variable, sizes_rank, active_count, valid_size, b, h)
    if strided:

        def stride_view(t, step):
            storage = torch.empty((*t.shape[:-1], t.shape[-1] * step), dtype=t.dtype, device=t.device)
            view = storage[..., ::step]
            view.copy_(t)
            return view

        ids = stride_view(ids, 2)
        counts = stride_view(counts, 2) if counts is not None else None
        sizes = stride_view(sizes, 2) if sizes is not None else None

        def row_view(t):
            storage = torch.empty((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype, device=t.device)
            view = storage[..., : t.shape[-1]]
            view.copy_(t)
            return view

        q, k, v = (row_view(t) for t in (q, k, v))
    opts = dict(block_sparse_num=count, q2k_block_nums=counts, block_sizes=sizes, sparse_block_size=128, softmax_scale=scale, layout=layout)
    result = ops.block_sparse_attention_forward(q, k, v, ids, allow_empty_block_nums=variable, kv_splits=splits, **opts)
    assert result["o_tensor"].is_contiguous()
    assert result["lse_tensor"].is_contiguous()
    do = torch.randn_like(result["o_tensor"]) if backward else None
    ro, rl, rg = blk128_reference(q, k, v, ids, counts, sizes, count, scale, layout, do)
    torch.testing.assert_close(result["o_tensor"].double(), ro, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(result["lse_tensor"].double(), rl, atol=1e-4, rtol=1e-5)
    if backward:
        # Noncompact output rows exercise native writes, not temporary copies.
        backing = tuple(torch.full((*t.shape[:-1], t.shape[-1] * 2), 17.0, dtype=t.dtype, device=t.device) for t in (q, k, v))
        dest = tuple(storage[..., : t.shape[-1]] for storage, t in zip(backing, (q, k, v)))
        gradients = ops.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            result["o_tensor"],
            result["lse_tensor"],
            ids,
            dq_tensor=dest[0],
            dk_tensor=dest[1],
            dv_tensor=dest[2],
            bucket_size_blocks=bucket,
            **opts,
        )
        for got, expected, buffer, storage in zip(gradients, rg, dest, backing):
            assert got.data_ptr() == buffer.data_ptr()
            # TMA output descriptors must preserve row strides and never touch
            # the allocation's padding, including partial KV tiles.
            torch.testing.assert_close(storage[..., buffer.shape[-1] :], torch.full_like(buffer, 17.0), atol=0, rtol=0)
            # With one token/block, dK accumulates BF16 rounding across many rows.
            # Same-input blk64 reproduces the identical 0.0512 maximum dK error;
            # bound both elementwise and relative L2 error for this extreme case.
            torch.testing.assert_close(got.double(), expected, atol=6e-2 if valid_size == 1 else 3e-2, rtol=3e-2)
            error = torch.linalg.vector_norm(got.double() - expected)
            assert error <= 0.01 * torch.linalg.vector_norm(expected) + 1e-5
