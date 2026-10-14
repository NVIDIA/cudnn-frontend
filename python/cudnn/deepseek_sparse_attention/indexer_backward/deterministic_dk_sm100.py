# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-owner deterministic SM100-family indexer dK accumulation."""

from __future__ import annotations

from numbers import Integral

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from cudnn.api_base import ws_align
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

SUPPORTED_DK_PLANES = (96, 128, 148)
DEFAULT_DK_PLANES = SUPPORTED_DK_PLANES[0]
# Backward-compatible aliases for downstream tests and internal callers that
# inspect the default policy.
NUM_DK_PRODUCER_CTAS = DEFAULT_DK_PLANES
NUM_DK_PLANES = DEFAULT_DK_PLANES
INDEXER_BACKWARD_SM100_CAPABILITIES = ((10, 0), (10, 3))
_compile_cache: dict = {}


def validate_deterministic_dk_planes(num_planes: int) -> int:
    """Validate and normalize the deterministic producer-plane specialization."""
    if isinstance(num_planes, bool) or not isinstance(num_planes, Integral) or num_planes not in SUPPORTED_DK_PLANES:
        raise ValueError(f"deterministic_dk_planes must be one of {SUPPORTED_DK_PLANES}, got {num_planes!r}")
    return int(num_planes)


def validate_deterministic_dk_plane_policy(deterministic: bool, num_planes: int) -> int:
    """Reject an unsupported or inactive non-default plane specialization."""
    num_planes = validate_deterministic_dk_planes(num_planes)
    if not deterministic and num_planes != DEFAULT_DK_PLANES:
        raise ValueError("deterministic_dk_planes is only configurable when deterministic=True")
    return num_planes


def deterministic_dk_producer_ctas(total_rows: int, num_planes: int = DEFAULT_DK_PLANES) -> int:
    """Choose a fixed-owner grid in which every launched CTA has a row."""
    num_planes = validate_deterministic_dk_planes(num_planes)
    if isinstance(total_rows, bool) or not isinstance(total_rows, Integral) or total_rows < 1:
        raise ValueError(f"deterministic IndexerBackward requires at least one query row, got {total_rows!r}")
    return min(num_planes, int(total_rows))


def deterministic_dk_planes_bytes(batch: int, seqlen_k: int, head_dim: int, num_planes: int = DEFAULT_DK_PLANES) -> int:
    """Return the byte count of the producer-private FP32 dK planes."""
    return validate_deterministic_dk_planes(num_planes) * batch * seqlen_k * head_dim * torch.float32.itemsize


def deterministic_dk_workspace_bytes(batch: int, seqlen_k: int, head_dim: int, num_planes: int = DEFAULT_DK_PLANES) -> int:
    """Return the exact aligned scratch size for the FP32 planes."""
    return ws_align(deterministic_dk_planes_bytes(batch, seqlen_k, head_dim, num_planes))


class DeterministicDkReduceSm100:
    """Fold producer-private FP32 planes in numeric CTA order."""

    arch = 100

    def __init__(self, head_dim: int, num_planes: int):
        """Specialize the reducer for the contiguous dK row width."""
        self.head_dim = head_dim
        self.num_planes = validate_deterministic_dk_planes(num_planes)

    @cute.jit
    def __call__(
        self,
        mdK_planes: cute.Tensor,
        mdK: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch one fixed-order reduction block per flattened KV row."""
        # Both tensors are contiguous. Keep the plane-major scratch layout so
        # producer CTA i and reducer operand i address the same plane.
        seqlen_k = cute.size(mdK.shape[0]) * cute.size(mdK.shape[1])
        planes = cute.make_tensor(
            mdK_planes.iterator,
            cute.make_layout(
                (self.head_dim, seqlen_k * self.num_planes),
                stride=(1, cutlass.Int64(self.head_dim)),
            ),
        )
        out = cute.make_tensor(
            mdK.iterator,
            cute.make_layout(
                (self.head_dim, seqlen_k),
                stride=(1, cutlass.Int64(self.head_dim)),
            ),
        )
        self.store_output(planes, out, seqlen_k).launch(
            grid=(seqlen_k, 1, 1),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def store_output(self, mdK_planes: cute.Tensor, mdK: cute.Tensor, seqlen_k: Int32):
        """Fold the planes in fixed order and convert once to the output dtype."""
        row_idx, _, _ = cute.arch.block_idx()
        dim_idx, _, _ = cute.arch.thread_idx()
        if row_idx < seqlen_k and dim_idx < self.head_dim:
            acc = Float32(0.0)
            for plane_idx in cutlass.range(self.num_planes, unroll_full=True):
                acc += mdK_planes[dim_idx, row_idx + plane_idx * seqlen_k]
            mdK[dim_idx, row_idx] = mdK.element_type(acc)


def reduce_deterministic_dk(
    d_index_k_planes: torch.Tensor,
    d_index_k: torch.Tensor,
    num_planes: int,
    current_stream=None,
) -> None:
    """Reduce caller-owned planes into ``d_index_k`` without allocation."""
    num_planes = validate_deterministic_dk_planes(num_planes)
    if d_index_k_planes.shape[0] != num_planes:
        raise ValueError(f"d_index_k_planes has {d_index_k_planes.shape[0]} planes, expected {num_planes}")
    stream = resolve_stream(current_stream)
    key = (d_index_k.device, d_index_k.shape[-1], d_index_k.dtype, num_planes)
    fn = _compile_cache.get(key)
    if fn is None:
        kernel = DeterministicDkReduceSm100(d_index_k.shape[-1], num_planes)
        fn = cute.compile(
            kernel,
            to_cute_tensor(d_index_k_planes),
            to_cute_tensor(d_index_k),
            stream,
            options=compile_options("--opt-level 2"),
        )
        _compile_cache[key] = fn
    fn(d_index_k_planes, d_index_k, stream)
