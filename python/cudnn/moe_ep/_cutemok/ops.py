# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The operators around the megakernels: thin forwarding functions to the CuTe DSL kernels (route
all-gather, EP barrier, scheduler, MXFP8 quantize, routed-row copy, forward / backward epilogue) and
the per-expert address tables of split routed weights.  An implementation module: the public API is
``cudnn.moe_ep.cutemok``.

The functions import ``cudnn.moe_ep._cutemok.cutedsl`` inside the call rather than at module scope, so importing this module
(``cudnn.moe_ep.cutemok`` does) does not pull CUTLASS, the version check and the device probe of the kernel modules into a
caller that only wanted ``CuteMoKConfig``.

These are plain Python functions, not ``torch.library`` custom ops: there are no fake / meta
implementations.  The megakernels are called through ``cudnn.moe_ep._cutemok.functional``, which is not traceable.
"""

import torch  # noqa: F401


def all_gather_top_experts(
    top_experts: torch.Tensor,
    all_gather_top_experts_buffer: torch.Tensor,
    all_gather_top_experts_buffer_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
) -> None:
    """Gathers every rank's top-k expert ids (int32 ``[num_local_tokens, topk]``) into the symmetric
    ``[ep_size, num_local_tokens, topk]`` buffer, ``chunk_bytes`` at a time."""
    from .cutedsl.all_gather_top_experts import all_gather_top_experts as impl

    return impl(top_experts, all_gather_top_experts_buffer, all_gather_top_experts_buffer_multicast_ptr, rank, chunk_bytes)


def barrier_all(
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    target: torch.Tensor,
) -> None:
    """Device-side barrier across the EP group."""
    from .cutedsl.barrier_all import barrier_all as impl

    return impl(barrier_buffer, barrier_buffer_ptrs, barrier_buffer_multicast_ptr, target)


def schedule(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds this rank's routed-row schedule from the all-gathered routes
    (``topk_all``: ``[ep_size, num_local_tokens, topk]``).

    Returns five tensors: peer rank, peer token index, token count, padded rows per expert and
    valid (unpadded) rows per expert -- the fields of ``CuteMoKSchedule``.
    """
    # Checked before any launch: a capacity the routed rows cannot fit makes the schedule kernel
    # trap, which takes the CUDA context with it.
    if not isinstance(topk_all, torch.Tensor) or topk_all.ndim != 3:
        raise ValueError("topk_all must have shape (ep_size, num_local_tokens, topk)")
    ep_size, num_local_tokens, topk = topk_all.shape
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("topk_all ep_size must be one of 1, 4, 8, 16, 32, 64")
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("topk_all num_local_tokens must be at least 512 and divisible by 256")
    if not 0 < topk <= 255:
        raise ValueError("topk_all topk must be in [1, 255]")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if type(schedule_capacity) is not int or schedule_capacity <= 0 or schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be positive and divisible by 256")
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule_capacity must hold at least one rank's routed tokens")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")

    from .cutedsl.scheduler import scheduler as impl

    return impl(topk_all, num_local_experts, schedule_capacity, rank)


def mxfp8_quantize(
    x_bf16: torch.Tensor, return_normal: bool, return_transposed: bool
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """BF16 -> MXFP8 (e4m3 payload, one e8m0 scale per 32 elements along K).

    Returns (data, scale, transposed data, transposed scale); the pairs not asked for are ``None``.
    """
    from .cutedsl.mxfp8_quantize import mxfp8_quantize as impl

    return impl(x_bf16, return_normal, return_transposed)


def copy_routed_rows(
    top_experts: torch.Tensor,
    sources: list[torch.Tensor],
    destinations: list[torch.Tensor],
    router_weights: torch.Tensor | None = None,
    router_weight_buffer: torch.Tensor | None = None,
    zero_unrouted: torch.Tensor | None = None,
) -> None:
    """Copies the rows of one or two BF16 [T, H] sources that have at least one route
    (top_experts >= 0) into their destinations, leaving the other rows untouched, and the full FP32
    [T, topk] router weights when given.  ``zero_unrouted`` (FP32 [T, topk]): its slots with
    top_experts < 0 are set to zero, the others left untouched."""
    from .cutedsl.copy_routed_rows import copy_routed_rows as impl

    impl(top_experts, sources, destinations, router_weights, router_weight_buffer, zero_unrouted)


def fwd_epilogue(
    y_shared: torch.Tensor | None,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
    top_experts: torch.Tensor,
    shared_output_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """y_shared + sum over the used slots k of topk_weights[t, k] * combine_buffer[t * topk + k, :].

    ``top_experts`` marks a token's unused route slots with -1; their combine rows were never
    written and are skipped.  ``shared_output_gate`` (optional contiguous BF16 ``[T, 1]``) scales
    the shared output in FP32 before the routed terms are added.
    ``y_shared`` None: a routed-only layer, the routed sum alone (the gate then needs a shared output).
    """
    from .cutedsl.fwd_epilogue import fwd_epilogue_bf16 as impl

    return impl(y_shared, combine_buffer, topk_weights, top_experts, shared_output_gate)


def bwd_epilogue(d_x_shared: torch.Tensor | None, d_x_routed_buffer: torch.Tensor, top_experts: torch.Tensor) -> torch.Tensor:
    """d_x_shared + sum over the used slots k of d_x_routed_buffer[t * topk + k, :]; the routed
    sum alone when ``d_x_shared`` is None (a routed-only layer)."""
    from .cutedsl.bwd_epilogue import bwd_epilogue_bf16 as impl

    return impl(d_x_shared, d_x_routed_buffer, top_experts)


#: The per-expert address tables below are plain int64 CUDA tensors.  The kernels keep one TMA
#: descriptor per operand and retarget it (``cute.nvgpu.cpasync.update_tma_descriptor``); all an
#: address is needed for is to say where.  Since every expert allocation is identical in dtype,
#: shape and stride, the address is the only thing that varies, so the table is the whole of the
#: per-expert state.
def _storage_table(expert_tensors, *, what: str, check) -> "torch.Tensor":
    """Validate a list of per-expert allocations and return their base addresses.

    Requires a nonempty list of contiguous, 16-byte aligned CUDA tensors agreeing on dtype, shape
    and device.  Anything looser would silently produce a table whose entries the kernel cannot
    address with one descriptor schema.
    """
    if not isinstance(expert_tensors, (list, tuple)) or not expert_tensors:
        raise TypeError(f"{what} must be a nonempty list of tensors")
    first = expert_tensors[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"each {what} entry must be a torch.Tensor")
    check(first)
    for index, tensor in enumerate(expert_tensors):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != first.device
            or tensor.dtype != first.dtype
            or tensor.shape != first.shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"every {what} entry must be a contiguous tensor with the same dtype, shape and " f"device; entry {index} is not")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{what} entry {index} is not 16-byte aligned")
    return torch.tensor([tensor.data_ptr() for tensor in expert_tensors], dtype=torch.int64, device=first.device)


def _check_matrix(dtype):
    def check(tensor: "torch.Tensor") -> None:
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.ndim != 2:
            raise ValueError("each expert tensor must be a contiguous CUDA matrix")
        if tensor.dtype != dtype:
            raise TypeError(f"expert tensors must have dtype {dtype}")

    return check


def _check_scale(tensor: "torch.Tensor") -> None:
    if not tensor.is_cuda or tensor.dtype != torch.uint8 or not tensor.is_contiguous() or tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (32, 16):
        raise ValueError("each expert scale must be a contiguous CUDA uint8 tensor shaped " "[row_blocks, col_blocks, 32, 16]")


def make_routed_weight_storage_table_bf16(expert_tensors) -> "torch.Tensor":
    """Per-expert base addresses of a BF16 routed weight held in split allocations."""
    return _storage_table(expert_tensors, what="BF16 expert weight", check=_check_matrix(torch.bfloat16))


def make_routed_weight_storage_table_mxfp8(expert_tensors) -> "torch.Tensor":
    """Per-expert base addresses of an MXFP8 routed weight held in split allocations."""
    return _storage_table(expert_tensors, what="MXFP8 expert weight", check=_check_matrix(torch.float8_e4m3fn))


def make_routed_scale_storage_table(expert_tensors) -> "torch.Tensor":
    """Per-expert base addresses of the tcgen05-layout scales of a split routed weight."""
    return _storage_table(expert_tensors, what="expert scale", check=_check_scale)


def make_routed_d_weight_storage_table(expert_tensors) -> "torch.Tensor":
    """Per-expert base addresses of the main-grad buffers a split routed weight accumulates into."""

    def check(tensor: "torch.Tensor") -> None:
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.ndim != 2:
            raise ValueError("each expert main grad must be a contiguous CUDA matrix")
        if tensor.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("expert main grads must have dtype float32 or bfloat16")

    return _storage_table(expert_tensors, what="expert main grad", check=check)
