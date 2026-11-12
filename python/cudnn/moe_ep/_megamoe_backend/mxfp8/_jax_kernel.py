# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe launch helpers for native JAX expert-parallel MoE."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

from ._training_stage_kernel import Mxfp8TrainingStageKernel


@cute.jit
def typed(tensor, dtype: cutlass.Constexpr, shape, stride=None):
    return cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=dtype), cute.make_layout(shape, stride=tensor.stride if cutlass.const_expr(stride is None) else stride)
    )


@cute.kernel
def reset_routing(indices: cute.Tensor, original: cute.Tensor, status: cute.Tensor, experts: cutlass.Constexpr):
    index = cute.arch.block_idx()[0] * Int32(256) + cute.arch.thread_idx()[0]
    capacity = cute.size(indices)
    original_size = cute.size(original)
    flat = cute.make_tensor(indices.iterator, cute.make_layout(capacity))
    source = cute.make_tensor(original.iterator, cute.make_layout(original_size))
    if index < capacity:
        flat[index] = Int32(-1)
    if index < original_size:
        expert = source[index]
        if (expert < Int32(0)) | (expert >= Int32(experts)):
            cute.arch.atomic_or((status.iterator + 1).llvm_ptr, Int32(1), sem="relaxed", scope="gpu")


@cute.jit
def stage_input(source, indices, scores, activation, activation_sf, routing, routing_scores, status, experts: cutlass.Constexpr, stream):
    reset_routing(indices=routing, original=indices, status=status, experts=experts).launch(
        grid=[cute.ceil_div(cute.size(routing), 256), 1, 1], block=[256, 1, 1], stream=stream
    )
    if source.shape[0] > 0:
        quantizer = Mxfp8TrainingStageKernel(source.shape[1], indices.shape[1])
        quantizer(
            source,
            indices,
            scores,
            typed(activation, cutlass.Float8E4M3FN, source.shape, (source.shape[1], 1)),
            typed(activation_sf, cutlass.Float8E8M0FNU, (source.shape[0], source.shape[1] // 32), (activation_sf.shape[1], 1)),
            cute.make_tensor(routing.iterator, cute.make_layout(indices.shape, stride=(indices.shape[1], 1))),
            cute.make_tensor(routing_scores.iterator, cute.make_layout(scores.shape, stride=(scores.shape[1], 1))),
            stream,
        )


@cute.kernel
def finish_output(source: cute.Tensor, output: cute.Tensor, overflow: cute.Tensor, status: cute.Tensor):
    index = cute.arch.block_idx()[0] * Int32(256) + cute.arch.thread_idx()[0]
    count = cute.size(output)
    source_flat = cute.make_tensor(source.iterator, cute.make_layout(cute.size(source)))
    output_flat = cute.make_tensor(output.iterator, cute.make_layout(count))
    if index < count:
        output_flat[index] = source_flat[index]
    if index == Int32(0):
        status[0] = overflow[0]


@cute.jit
def copy_output(source, output, overflow, status, stream):
    blocks = cute.ceil_div(cute.size(output), 256)
    if blocks < 1:
        blocks = Int32(1)
    finish_output(source, output, overflow, status).launch(grid=[blocks, 1, 1], block=[256, 1, 1], stream=stream)


@cute.kernel
def copy_dprob(source: cute.Tensor, output: cute.Tensor):
    index = cute.arch.block_idx()[0] * Int32(256) + cute.arch.thread_idx()[0]
    count = cute.size(output)
    source_flat = cute.make_tensor(source.iterator, cute.make_layout(cute.size(source)))
    output_flat = cute.make_tensor(output.iterator, cute.make_layout(count))
    if index < count:
        output_flat[index] = source_flat[index]


@cute.kernel
def expert_metadata(workspace: cute.Tensor, counts: cute.Tensor, offsets: cute.Tensor, byte_offset: cutlass.Constexpr, padding: cutlass.Constexpr):
    snapshot = cute.make_tensor(cute.recast_ptr(workspace.iterator + byte_offset, dtype=cutlass.Int32), cute.make_layout(counts.shape))
    running = Int32(0)
    for expert in cutlass.range_constexpr(counts.shape[0]):
        count = snapshot[expert]
        counts[expert] = count
        running += ((count + Int32(padding - 1)) // Int32(padding)) * Int32(padding)
        offsets[expert] = running


@cute.kernel
def fill_beta(beta: cute.Tensor):
    index = cute.arch.block_idx()[0] * Int32(256) + cute.arch.thread_idx()[0]
    if index < cute.size(beta):
        beta[index] = cutlass.Float32(1.0)
