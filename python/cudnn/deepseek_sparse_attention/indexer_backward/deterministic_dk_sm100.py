# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-owner deterministic SM100-family indexer dK accumulation."""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from cudnn.api_base import ws_align
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

NUM_DK_PRODUCER_CTAS = 96
NUM_DK_PLANES = NUM_DK_PRODUCER_CTAS
INDEXER_BACKWARD_SM100_CAPABILITIES = ((10, 0), (10, 3))
_compile_cache: dict = {}


def deterministic_dk_producer_ctas(total_rows: int) -> int:
    """Choose a fixed-owner grid in which every launched CTA has a row."""
    return min(NUM_DK_PRODUCER_CTAS, total_rows)


def deterministic_dk_planes_bytes(batch: int, seqlen_k: int, head_dim: int) -> int:
    """Return the byte count of the producer-private FP32 dK planes."""
    return NUM_DK_PLANES * batch * seqlen_k * head_dim * torch.float32.itemsize


def deterministic_dk_workspace_bytes(batch: int, seqlen_k: int, head_dim: int) -> int:
    """Return the exact aligned scratch size for the FP32 planes."""
    return ws_align(deterministic_dk_planes_bytes(batch, seqlen_k, head_dim))


class DeterministicDkReduceSm100:
    """Fold producer-private FP32 planes in numeric CTA order."""

    arch = 100

    def __init__(self, head_dim: int):
        """Specialize the reducer for the contiguous dK row width."""
        self.head_dim = head_dim

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
                (self.head_dim, seqlen_k * NUM_DK_PLANES),
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
            for plane_idx in cutlass.range(NUM_DK_PLANES, unroll_full=True):
                acc += mdK_planes[dim_idx, row_idx + plane_idx * seqlen_k]
            mdK[dim_idx, row_idx] = mdK.element_type(acc)


def reduce_deterministic_dk(
    d_index_k_planes: torch.Tensor,
    d_index_k: torch.Tensor,
    current_stream=None,
) -> None:
    """Reduce caller-owned planes into ``d_index_k`` without allocation."""
    stream = resolve_stream(current_stream)
    key = (d_index_k.device, d_index_k.shape[-1], d_index_k.dtype)
    fn = _compile_cache.get(key)
    if fn is None:
        kernel = DeterministicDkReduceSm100(d_index_k.shape[-1])
        fn = cute.compile(
            kernel,
            to_cute_tensor(d_index_k_planes),
            to_cute_tensor(d_index_k),
            stream,
            options=compile_options("--opt-level 2"),
        )
        _compile_cache[key] = fn
    fn(d_index_k_planes, d_index_k, stream)
