# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
