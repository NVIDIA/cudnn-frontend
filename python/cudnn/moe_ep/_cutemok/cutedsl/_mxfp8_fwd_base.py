# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Base of the MXFP8 forward megakernel: the BF16 collective of the shared expert, the CLC drain,
the counter / SwiGLU tile helpers and the BF16 argument builder ``make_mega_args_mxfp8`` wraps.
"""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from ._mxfp8_gemm_base import _Bf16Gemm
from .mxfp8_contract import SWIGLU_TILE_ROWS, SWIGLU_TILE_COLUMNS
from ._persistent_bf16_fwd_mega import routed_weight_view
from ._host_common import peer_address_table, row_major_view
from ._bwd_device_helpers import _mbar_spin_wait

CLUSTER_SHAPE = (2, 1, 1)
DISPATCH_ROW_CHUNK_BYTES = 1024
SWIGLU_TILE_M = SWIGLU_TILE_ROWS
SWIGLU_TILE_N = SWIGLU_TILE_COLUMNS
SWIGLU_TILE_BYTES = 128 * 128 * 2
# 232448 B: the SM103 opt-in dynamic shared-memory maximum.
DYNAMIC_SHARED_MEMORY_BYTES = 232448


@cute.jit
def _counter_wait_gpu(
    counter: cute.Tensor,
    index: Int32,
    required: Int32,
) -> None:
    """Wait until the monotonically increasing ``counter[index]`` reaches ``required``: relaxed polls
    with ``nanosleep(16)`` between them, then one acquire fence at GPU scope."""

    pointer = counter.iterator + index
    observed = cute.arch.load(pointer, Int32, sem="relaxed", scope="gpu")
    while observed < required:
        _nanosleep_compat(16)
        observed = cute.arch.load(pointer, Int32, sem="relaxed", scope="gpu")
    _fence_acquire_gpu()


@dsl_user_op
def _nanosleep_compat(sleep_time: int, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], f"nanosleep.u32 {int(sleep_time)};", "", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _fence_acquire_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "fence.acquire.gpu;", "~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@cute.jit
def _ceil_div_i32(value: Int32, divisor: Int32) -> Int32:
    return (value + divisor - Int32(1)) // divisor


@cute.jit
def _issue_swiglu_tma_load(
    atom: cute.CopyAtom, tensor: cute.Tensor, row_tile: Int32, column_tile: Int32, smem_tile: cute.Tensor, tma_barrier: cute.Pointer
) -> None:
    """Load one 128x128 BF16 tile into one SwiGLU arena stage."""
    global_tile = cute.local_tile(tensor, (SWIGLU_TILE_M, SWIGLU_TILE_N), (row_tile, column_tile))
    smem_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
    global_flat = cute.group_modes(global_tile, 0, cute.rank(global_tile))
    smem_part, global_part = cpasync.tma_partition(atom, 0, cute.make_layout(1), smem_flat, global_flat)
    cute.copy(atom, global_part, smem_part, tma_bar_ptr=tma_barrier)


@cute.jit
def _issue_swiglu_tma_store(atom: cute.CopyAtom, tensor: cute.Tensor, row_tile: Int32, column_tile: Int32, smem_tile: cute.Tensor) -> None:
    """Store one completed 128x128 BF16 tile from the reusable arena."""
    global_tile = cute.local_tile(tensor, (SWIGLU_TILE_M, SWIGLU_TILE_N), (row_tile, column_tile))
    smem_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
    global_flat = cute.group_modes(global_tile, 0, cute.rank(global_tile))
    smem_part, global_part = cpasync.tma_partition(atom, 0, cute.make_layout(1), smem_flat, global_flat)
    cute.copy(atom, smem_part, global_part)


class _Mxfp8FwdBase:
    """Base of the MXFP8 forward kernel classes: the clamp mode, the BF16 collective and the CLC drain."""

    def __init__(self, is_clamped: bool):
        # The clamp mode is compile time; the clamp value is a runtime Float32.
        self.is_clamped = bool(is_clamped)
        self.collective = _Bf16Gemm()

    @cute.jit
    def _drain_clc_capacity_tail(
        self, needs_drain: Int32, clc_drain_pipeline, drain_consumer_state, drain_producer_state, drain_scheduler, cta_rank: Int32
    ) -> None:
        """Drain capacity-only CLC responses after every role finishes.

        Lane 0 of each warp: each CTA arms its own full barrier, rank 0 issues the query,
        both read only the response's valid bit, both arrive on rank 0's empty barrier, and
        rank 0 waits for it before re-issuing.  Tight spins.
        """
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if needs_drain != Int32(0):
            if cute.arch.lane_idx() == Int32(0):
                full_bar = clc_drain_pipeline.sync_object_full.get_barrier(drain_consumer_state.index)
                empty_bar = clc_drain_pipeline.sync_object_empty.get_barrier(drain_producer_state.index)
                response = drain_scheduler._clc_response_ptr
                full_phase = drain_consumer_state.phase
                empty_phase = Int32(0)
                cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(16))
                if cta_rank == Int32(0):
                    cute.arch.issue_clc_query(full_bar, response)
                active = Int32(1)
                while active != Int32(0):
                    _mbar_spin_wait(full_bar, full_phase)
                    full_phase = full_phase ^ Int32(1)
                    _, _, _, drained_valid = cute.arch.clc_response(response)
                    # The generic-proxy read of the response must be ordered before the
                    # next async-proxy CLC write into the same shared memory.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    clc_drain_pipeline.sync_object_empty.arrive(drain_consumer_state.index, clc_drain_pipeline.consumer_mask)
                    if cta_rank == Int32(0):
                        _mbar_spin_wait(empty_bar, empty_phase)
                        empty_phase = empty_phase ^ Int32(1)
                    active = Int32(drained_valid)
                    if active != Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(16))
                        if cta_rank == Int32(0):
                            cute.arch.issue_clc_query(full_bar, response)
            cute.arch.sync_warp()


def make_mega_args(
    x_peer_addresses,
    combine_peer_addresses,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    x_shared,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state,
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    topk: int,
    macrobatch_size: int = 32768,
    minibatch_size: int = 4096,
    num_comm_sms: int = 42,
    swiglu_limit: float = 0.0,
    stream=None,
    launch: bool = False,
    peer_tables: dict | None = None,
):
    """Build the launch keyword arguments after the host gate succeeds.

    ``state`` is duck-typed to a nine-output/five-counter object (``make_mega_args_mxfp8`` passes
    the MXFP8 state through an adapter).  Routed weights stay
    as strided ``[N,K,E]`` views, never a packed or copied mirror.

    ``launch=False`` builds the CuTe tensors and DSL scalars that ``cute.compile``
    traces.  ``launch=True`` builds the arguments of a call to the compiled TVM FFI
    function: the same torch views and plain Python scalars.
    ``peer_tables`` is the workspace-owned cache of peer address tables (``None``: build them).
    """

    import torch

    tables = {} if peer_tables is None else peer_tables
    if stream is None:
        stream = torch.cuda.current_stream(x_shared.device)
    cuda_stream = stream if isinstance(stream, cuda.CUstream) else cuda.CUstream(stream.cuda_stream)

    def dynamic_tensor(tensor, leading_dim):
        if launch:
            return tensor
        return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)

    if launch:
        Int32, Float32 = int, float  # noqa: N806 -- plain scalars at the FFI call
    else:
        from cutlass import Float32, Int32  # noqa: N806

    routed_gate_nke = routed_weight_view(routed_gate_weights)
    routed_up_nke = routed_weight_view(routed_up_weights)
    routed_down_nke = routed_weight_view(routed_down_weights)
    # Keyword arguments named after the kernel's __call__ parameters.
    return dict(
        x_peer_table=dynamic_tensor(peer_address_table(x_peer_addresses, x_shared.device, tables), 0),
        combine_peer_table=dynamic_tensor(peer_address_table(combine_peer_addresses, x_shared.device, tables), 0),
        schedule_peer_rank=dynamic_tensor(row_major_view(schedule_peer_rank), 0),
        schedule_peer_token_idx=dynamic_tensor(row_major_view(schedule_peer_token_idx), 0),
        num_tokens=dynamic_tensor(row_major_view(num_tokens), 0),
        tokens_per_expert=dynamic_tensor(row_major_view(tokens_per_expert), 0),
        x_shared=dynamic_tensor(row_major_view(x_shared), 1),
        x_routed=dynamic_tensor(row_major_view(state.x_routed), 1),
        shared_gate_weights=dynamic_tensor(row_major_view(shared_gate_weights), 1),
        routed_gate_weights=dynamic_tensor(routed_gate_nke, 1),
        shared_up_weights=dynamic_tensor(row_major_view(shared_up_weights), 1),
        routed_up_weights=dynamic_tensor(routed_up_nke, 1),
        shared_down_weights=dynamic_tensor(row_major_view(shared_down_weights), 1),
        routed_down_weights=dynamic_tensor(routed_down_nke, 1),
        gate_shared=dynamic_tensor(row_major_view(state.gate_shared), 1),
        gate_routed=dynamic_tensor(row_major_view(state.gate_routed), 1),
        up_shared=dynamic_tensor(row_major_view(state.up_shared), 1),
        up_routed=dynamic_tensor(row_major_view(state.up_routed), 1),
        hidden_shared=dynamic_tensor(row_major_view(state.hidden_shared), 1),
        hidden_routed=dynamic_tensor(row_major_view(state.hidden_routed), 1),
        y_shared=dynamic_tensor(row_major_view(state.y_shared), 1),
        y_routed=dynamic_tensor(row_major_view(state.y_routed), 1),
        gate_up_tile_ready=dynamic_tensor(row_major_view(state.gate_up_tile_ready), 0),
        hidden_row_block_ready=dynamic_tensor(row_major_view(state.hidden_row_block_ready), 0),
        x_routed_ready=dynamic_tensor(row_major_view(state.x_routed_ready), 0),
        y_routed_ready=dynamic_tensor(row_major_view(state.y_routed_ready), 0),
        y_routed_done=dynamic_tensor(row_major_view(state.y_routed_done), 0),
        hidden_size=Int32(x_shared.shape[1]),
        intermediate_size=Int32(shared_gate_weights.shape[0]),
        # From the schedule, not from the weight: a routed weight is the whole [E, N, K]
        # parameter when the experts are grouped but one expert's own allocation when they are
        # split, where shape[0] is the intermediate size.  tokens_per_expert is E either way.
        num_local_experts=Int32(tokens_per_expert.shape[0]),
        topk=Int32(topk),
        num_local_tokens=Int32(num_local_tokens),
        schedule_capacity=Int32(schedule_capacity),
        macrobatch_size=Int32(macrobatch_size),
        minibatch_size=Int32(minibatch_size),
        num_comm_sms=Int32(num_comm_sms),
        swiglu_limit=Float32(swiglu_limit),
        stream=cuda_stream,
    )


__all__ = [
    "_Mxfp8FwdBase",
    "make_mega_args",
]
