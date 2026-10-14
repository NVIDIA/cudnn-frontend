# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive adjacent-query KV sharing for SM100 DSA backward."""

from functools import lru_cache
import math

import cutlass
import cutlass.cute as cute
import torch

from cudnn.api_base import WorkspaceCarver
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream

from ._interface_sm100 import _align_workspace_bytes, _workspace_shapes_sm100
from ._qpair_topk_sm100 import QPairTopKTransformSm100
from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .dsa_bwd_sm100_h16 import FlashAttentionDSABackwardSm100H16
from .dsa_bwd_sm100_h32 import FlashAttentionDSABackwardSm100H32


def _layout(max_topk: int, heads: int) -> tuple[int, int, int, int]:
    """Return tile, shared capacity, packed stride, and threshold divisor."""
    if heads not in (16, 32):
        raise ValueError(f"adaptive q-cluster requires H16 or H32, got H{heads}")
    if not 0 < max_topk <= 2048:
        raise ValueError(f"adaptive q-cluster requires 1 <= topk <= 2048, got {max_topk}")
    tile = 128 if heads == 16 else 64
    if max_topk <= 2 * tile:
        return tile, 0, max_topk, 0
    shared_capacity = ((max_topk - 1) // tile) * tile
    packed_stride = -(-(max_topk + shared_capacity) // 32) * 32
    # H16 merges into the tuned H32 kernel; H32 merges into generic H64 and
    # needs a slightly larger segment to amortize that path.
    threshold_divisor = 8 if heads == 16 else 6
    return tile, shared_capacity, packed_stride, threshold_divisor


def workspace_size(total_q: int, total_kv: int, dim: int, heads: int, max_topk: int) -> int:
    """Return base scratch plus graph-stable q-cluster metadata storage."""
    lse_shape, dkv_shape = _workspace_shapes_sm100(total_q, total_kv, dim, heads, False)
    size = _align_workspace_bytes(math.prod(lse_shape)) + _align_workspace_bytes(math.prod(dkv_shape))
    _, shared_capacity, packed_stride, _ = _layout(max_topk, heads)
    if shared_capacity:
        size += _align_workspace_bytes(total_q * packed_stride * 4)
        size += _align_workspace_bytes((total_q + (total_q + 1) // 2) * 4)
        size += _align_workspace_bytes(2 * heads * 4)
    return size


def _fake(dtype, shape, stride=None, alignment=16):
    """Construct a metadata-only tensor for plan-time compilation."""
    if stride is None:
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=alignment)


def _set_policy(
    kernel,
    *,
    initialize=False,
    finalize=False,
    sum_odo=False,
    bwd=False,
    dsink=False,
    reduce_dq=False,
    workspace_heads=0,
    paired_q_view=False,
):
    """Apply compile-time launch policy to one fresh kernel instance."""
    kernel.initialize_dkv = initialize
    kernel.finalize_dkv = finalize
    kernel.run_sum_odo = sum_odo
    kernel.run_bwd = bwd
    kernel.run_dsink = dsink
    kernel.reduce_dq = reduce_dq
    kernel.workspace_num_heads = workspace_heads
    kernel.paired_q_view = paired_q_view
    return kernel


def _compile_stage(kernel, problem_shape, tensors, stream):
    """Compile one launch stage from graph metadata."""
    return cute.compile(kernel, problem_shape, *tensors, cutlass.Float32(1.0), stream, options=compile_options())


class QClusterPlanSm100:
    """Compiled q-pair transform and four-stage backward launch sequence."""

    def __init__(self, total_q: int, total_kv: int, heads: int, dim: int, max_topk: int, has_topk_length: bool, plans):
        """Store the fixed graph contract and its compiled stages."""
        self.total_q = total_q
        self.total_kv = total_kv
        self.heads = heads
        self.dim = dim
        self.max_topk = max_topk
        self.has_topk_length = has_topk_length
        _, self.shared_capacity, self.packed_stride, _ = _layout(max_topk, heads)
        self.transform, self.prelude, self.unique, self.shared, self.epilogue = plans

    def _buffers(self, workspace: torch.Tensor):
        """Carve all kernel scratch from the caller-owned workspace."""
        lse_shape, dkv_shape = _workspace_shapes_sm100(self.total_q, self.total_kv, self.dim, self.heads, False)
        required = workspace_size(self.total_q, self.total_kv, self.dim, self.heads, self.max_topk)
        carver = WorkspaceCarver(workspace, required, "adaptive q-cluster")
        workspace_lse = carver.take(math.prod(lse_shape), torch.uint8).view(lse_shape)
        workspace_dkv = carver.take(math.prod(dkv_shape), torch.uint8).view(dkv_shape)
        packed_flat = carver.take(self.total_q * self.packed_stride, torch.int32)
        packed = torch.as_strided(
            packed_flat,
            (self.total_q, self.max_topk + self.shared_capacity),
            (self.packed_stride, 1),
        )
        pair_slots = (self.total_q + 1) // 2
        lengths = carver.take(self.total_q + pair_slots, torch.int32)
        sink_abi = carver.take(2 * self.heads, torch.float32)
        return workspace_lse, workspace_dkv, packed, lengths[: self.total_q], lengths[self.total_q :], sink_abi

    @staticmethod
    def _pair_3d(x: torch.Tensor, pairs: int, heads: int) -> torch.Tensor:
        """View adjacent Q rows as one doubled-head row."""
        return torch.as_strided(x, (pairs, 2 * heads, x.shape[-1]), (2 * heads * x.shape[-1], x.shape[-1], 1))

    @staticmethod
    def _pair_2d(x: torch.Tensor, pairs: int, heads: int) -> torch.Tensor:
        """View adjacent statistic rows as one doubled-head row."""
        return torch.as_strided(x, (pairs, 2 * heads), (2 * heads, 1))

    def execute(
        self,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        topk_length,
        dq,
        dkv,
        d_sink,
        workspace,
        softmax_scale,
        current_stream,
    ):
        """Launch only precompiled kernels using caller-owned storage."""
        _validate_execute(self, q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, dq, dkv, d_sink, workspace)
        stream = resolve_stream(current_stream)
        workspace_lse, workspace_dkv, packed, unique_lengths, shared_lengths, sink_abi = self._buffers(workspace)
        pairs = self.total_q // 2

        self.transform(topk_idxs, packed, unique_lengths, shared_lengths, d_sink, topk_length, self.total_kv, stream)
        problem = (self.total_q, self.total_kv, self.dim, (self.heads, 1))
        base_args = (q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, dq, dkv, d_sink, workspace_lse, workspace_dkv)
        self.prelude(problem, *base_args, softmax_scale, stream)

        unique_idxs = torch.as_strided(
            packed,
            (self.total_q, self.max_topk),
            (self.packed_stride, 1),
            packed.storage_offset() + self.shared_capacity,
        )
        unique_args = (q, kv, out, dout, lse, attn_sink, unique_idxs, unique_lengths, dq, dkv, d_sink, workspace_lse, workspace_dkv)
        self.unique(problem, *unique_args, softmax_scale, stream)

        if pairs:
            shared_idxs = torch.as_strided(packed, (pairs, self.shared_capacity), (2 * self.packed_stride, 1))
            shared_problem = (pairs, self.total_kv, self.dim, (2 * self.heads, 1))
            shared_args = (
                self._pair_3d(q, pairs, self.heads),
                kv,
                self._pair_3d(out, pairs, self.heads),
                self._pair_3d(dout, pairs, self.heads),
                self._pair_2d(lse, pairs, self.heads),
                sink_abi,
                shared_idxs,
                shared_lengths[:pairs],
                self._pair_3d(dq, pairs, self.heads),
                dkv,
                sink_abi,
                workspace_lse,
                workspace_dkv,
            )
            self.shared(shared_problem, *shared_args, softmax_scale, stream)

        self.epilogue(problem, *base_args, softmax_scale, stream)
        return dq, dkv, d_sink


def _validate_execute(plan, q, kv, out, dout, lse, sink, indices, lengths, dq, dkv, d_sink, workspace):
    """Validate the precompiled tensor and caller-storage contract."""
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q).__name__}")
    specs = (
        ("q", q, (plan.total_q, plan.heads, plan.dim), torch.bfloat16, 16),
        ("kv", kv, (plan.total_kv, plan.dim), torch.bfloat16, 16),
        ("out", out, (plan.total_q, plan.heads, 512), torch.bfloat16, 16),
        ("dout", dout, (plan.total_q, plan.heads, 512), torch.bfloat16, 16),
        ("lse", lse, (plan.total_q, plan.heads), torch.float32, 8),
        ("attn_sink", sink, (plan.heads,), torch.float32, 16),
        ("topk_idxs", indices, (plan.total_q, plan.max_topk), torch.int32, 16),
        ("dq", dq, (plan.total_q, plan.heads, plan.dim), torch.bfloat16, 16),
        ("dkv", dkv, (plan.total_kv, plan.dim), torch.bfloat16, 16),
        ("d_sink", d_sink, (plan.heads,), torch.float32, 16),
    )
    if plan.has_topk_length:
        specs += (("topk_length", lengths, (plan.total_q,), torch.int32, 4),)
    elif lengths is not None:
        raise ValueError("topk_length presence must match the compiled q-cluster plan")
    for name, tensor, shape, dtype, alignment in specs:
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != q.device:
            raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {q.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")
    required = workspace_size(plan.total_q, plan.total_kv, plan.dim, plan.heads, plan.max_topk)
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8 or workspace.device != q.device or not workspace.is_contiguous():
        raise ValueError("workspace must be a contiguous uint8 tensor on Q's device")
    if workspace.numel() < required or workspace.data_ptr() % 16:
        raise ValueError(f"workspace must contain at least {required} bytes and be 16-byte aligned")

    def span(tensor, size=None):
        """Return the half-open byte range touched by one contiguous tensor."""
        start = tensor.data_ptr()
        return start, start + (tensor.numel() * tensor.element_size() if size is None else size)

    reads = (("q", q), ("kv", kv), ("out", out), ("dout", dout), ("lse", lse), ("attn_sink", sink), ("topk_idxs", indices))
    if lengths is not None:
        reads += (("topk_length", lengths),)
    read_spans = tuple((name, span(tensor)) for name, tensor in reads)
    writes = (("dq", span(dq)), ("dkv", span(dkv)), ("d_sink", span(d_sink)), ("workspace", span(workspace, required)))
    for index, (name, byte_span) in enumerate(writes):
        for other_name, other_span in writes[index + 1 :] + read_spans:
            if max(byte_span[0], other_span[0]) < min(byte_span[1], other_span[1]):
                raise ValueError(f"{name} must not overlap {other_name}")


@lru_cache(maxsize=None)
def _compile_artifacts(device_capability, heads: int, dim: int, max_topk: int, has_topk_length: bool):
    """Compile shape-dynamic stages from the immutable tensor contract."""
    if device_capability not in ((10, 0), (10, 3)) or heads not in (16, 32) or dim != 576:
        raise ValueError("adaptive q-cluster requires SM100/SM103 BF16 H16/H32 Dqk576")
    tile, shared_capacity, packed_stride, threshold_divisor = _layout(max_topk, heads)

    dtype = cutlass.BFloat16
    sq, skv, pair_count = cute.sym_int(), cute.sym_int(), cute.sym_int()
    q = _fake(dtype, (sq, heads, dim))
    kv = _fake(dtype, (skv, dim))
    out = _fake(dtype, (sq, heads, 512))
    lse = _fake(cutlass.Float32, (sq, heads), alignment=8)
    sink = _fake(cutlass.Float32, (heads,))
    indices = _fake(cutlass.Int32, (sq, max_topk))
    input_lengths = _fake(cutlass.Int32, (sq,), alignment=4) if has_topk_length else None
    workspace_lse = _fake(cutlass.Uint8, (1, heads, cute.sym_int(), 8))
    workspace_dkv = _fake(cutlass.Uint8, (1, 1, cute.sym_int(), dim * 4))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)

    packed = _fake(cutlass.Int32, (sq, max_topk + shared_capacity), (packed_stride, 1))
    unique_lengths = _fake(cutlass.Int32, (sq,), alignment=4)
    shared_lengths = _fake(cutlass.Int32, (pair_count,), alignment=4)
    transform = cute.compile(
        QPairTopKTransformSm100(max_topk, shared_capacity, tile, threshold_divisor),
        indices,
        packed,
        unique_lengths,
        shared_lengths,
        sink,
        input_lengths,
        cutlass.Int32(1),
        stream,
        options=compile_options(),
    )

    base_cls = FlashAttentionDSABackwardSm100H16 if heads == 16 else FlashAttentionDSABackwardSm100H32
    base_block = 128 if heads == 16 else 64
    problem = (cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(dim), (cutlass.Int32(heads), cutlass.Int32(1)))
    base_tensors = (q, kv, out, out, lse, sink, indices, input_lengths, q, kv, sink, workspace_lse, workspace_dkv)
    prelude = _compile_stage(
        _set_policy(base_cls(dtype, dim, 512, base_block, max_topk), initialize=True, sum_odo=True, dsink=True), problem, base_tensors, stream
    )
    unique_indices = _fake(cutlass.Int32, (sq, max_topk), (packed_stride, 1))
    unique_tensors = (q, kv, out, out, lse, sink, unique_indices, unique_lengths, q, kv, sink, workspace_lse, workspace_dkv)
    unique = _compile_stage(_set_policy(base_cls(dtype, dim, 512, base_block, max_topk), bwd=True), problem, unique_tensors, stream)

    merged_heads = 2 * heads
    pair_q = _fake(dtype, (pair_count, merged_heads, dim))
    pair_out = _fake(dtype, (pair_count, merged_heads, 512))
    pair_lse = _fake(cutlass.Float32, (pair_count, merged_heads), alignment=8)
    pair_sink = _fake(cutlass.Float32, (merged_heads,))
    shared_indices = _fake(cutlass.Int32, (pair_count, shared_capacity), (2 * packed_stride, 1))
    shared_cls = FlashAttentionDSABackwardSm100H32 if heads == 16 else FlashAttentionDSABackwardSm100
    shared_tensors = (
        pair_q,
        kv,
        pair_out,
        pair_out,
        pair_lse,
        pair_sink,
        shared_indices,
        _fake(cutlass.Int32, (pair_count,), alignment=4),
        pair_q,
        kv,
        pair_sink,
        workspace_lse,
        workspace_dkv,
    )
    shared = _compile_stage(
        _set_policy(
            shared_cls(dtype, dim, 512, 64, shared_capacity),
            bwd=True,
            reduce_dq=True,
            workspace_heads=heads,
            paired_q_view=True,
        ),
        (cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(dim), (cutlass.Int32(merged_heads), cutlass.Int32(1))),
        shared_tensors,
        stream,
    )
    epilogue = _compile_stage(_set_policy(base_cls(dtype, dim, 512, base_block, max_topk), finalize=True), problem, base_tensors, stream)
    return transform, prelude, unique, shared, epilogue


def compile_plan(device_capability, total_q: int, total_kv: int, heads: int, dim: int, max_topk: int, has_topk_length: bool):
    """Bind dynamic compiled stages to one caller-owned workspace geometry."""
    if device_capability not in ((10, 0), (10, 3)) or heads not in (16, 32) or dim != 576:
        raise ValueError("adaptive q-cluster requires SM100/SM103 BF16 H16/H32 Dqk576")
    _, shared_capacity, _, _ = _layout(max_topk, heads)
    if total_q < 2 or not shared_capacity:
        return None
    plans = _compile_artifacts(device_capability, heads, dim, max_topk, has_topk_length)
    return QClusterPlanSm100(total_q, total_kv, heads, dim, max_topk, has_topk_length, plans)
