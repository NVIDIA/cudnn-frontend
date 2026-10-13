# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-relayed deterministic SM100 indexer dK accumulation."""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from cudnn.api_base import ws_align
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

NUM_DK_PRODUCER_CTAS = 128
DK_RELAY_CLUSTER_SIZE = 8
NUM_DK_PLANES = NUM_DK_PRODUCER_CTAS // DK_RELAY_CLUSTER_SIZE
NUM_DK_RELAY_COUNTERS = NUM_DK_PLANES
_compile_cache: dict = {}


def deterministic_dk_producer_ctas(total_rows: int) -> int:
    """Choose a full cluster grid that never contains an inactive CTA."""
    if total_rows < DK_RELAY_CLUSTER_SIZE:
        return total_rows
    return min(NUM_DK_PRODUCER_CTAS, total_rows // DK_RELAY_CLUSTER_SIZE * DK_RELAY_CLUSTER_SIZE)


def deterministic_dk_cluster_size(total_rows: int) -> int:
    """Use cluster-1 only when fewer than eight query rows exist."""
    return DK_RELAY_CLUSTER_SIZE if total_rows >= DK_RELAY_CLUSTER_SIZE else 1


def deterministic_dk_planes_bytes(batch: int, seqlen_k: int, head_dim: int) -> int:
    """Return the byte count of the 16 cluster-private FP32 dK planes."""
    return NUM_DK_PLANES * batch * seqlen_k * head_dim * torch.float32.itemsize


def deterministic_dk_relay_bytes() -> int:
    """Return the byte count of the per-cluster relay counters."""
    return NUM_DK_RELAY_COUNTERS * torch.int32.itemsize


def deterministic_dk_workspace_bytes(batch: int, seqlen_k: int, head_dim: int) -> int:
    """Return the exact aligned scratch size for planes plus relay counters."""
    return ws_align(deterministic_dk_planes_bytes(batch, seqlen_k, head_dim)) + ws_align(deterministic_dk_relay_bytes())


@dsl_user_op
def _ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def _red_release(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64],
        "red.release.gpu.global.add.s32 [$0], 1;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_dk_relay_turn(relay_ptr: cute.Pointer, thread_idx: Int32, cluster_idx: Int32, turn: Int32) -> None:
    """Have one thread acquire-wait for a cluster's fixed relay turn."""
    if thread_idx == 0:
        observed = Int32(-1)
        while observed != turn:
            observed = _ld_acquire(relay_ptr + cluster_idx)


@cute.jit
def advance_dk_relay_turn(relay_ptr: cute.Pointer, thread_idx: Int32, cluster_idx: Int32) -> None:
    """Release-publish completion of the current cluster relay turn."""
    if thread_idx == 0:
        _red_release(relay_ptr + cluster_idx)


class DeterministicDkReduceSm100:
    """Fold 16 cluster-private FP32 planes in numeric cluster order."""

    arch = 100

    def __init__(self, head_dim: int):
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        mdK_planes: cute.Tensor,
        mdK: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Both tensors are contiguous. Keep the plane-major scratch layout so
        # producer cluster i and reducer operand i address the same plane.
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
        """Fold the 16 cluster planes and convert once to the output dtype."""
        row_idx, _, _ = cute.arch.block_idx()
        dim_idx, _, _ = cute.arch.thread_idx()
        if row_idx < seqlen_k and dim_idx < self.head_dim:
            acc = Float32(0.0)
            for plane_idx in cutlass.range_constexpr(NUM_DK_PLANES):
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
