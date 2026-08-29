# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal CuTe DSL implementation of normalized H128 plus MXFP4 QDQ.

This kernel was independently generated for Kernel Factory campaign
``yy6zxcr9y97kd7bysw2pfhbxaw``.  The initial integrated schedule comes from
candidate ``5aa0b53a9683291ab31b0fbff978246c2eb0bf5b4ec090ae031a23e7fe5671b6``;
external implementations are semantic and performance references only and no
external kernel source is included here.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

F32 = cutlass.Float32
U32 = cutlass.Uint32
I32 = cutlass.Int32

THREADS = 256
LANES_PER_ROW = 4
ELEMENTS_PER_LANE = 128 // LANES_PER_ROW
WORDS_PER_LANE = ELEMENTS_PER_LANE // 2
ROWS_PER_CTA = THREADS // LANES_PER_ROW

NORM_BITS = 0x3DB504F3  # float32(128 ** -0.5)
INV6_BITS = 0x3E2AAAAB  # float32(1 / 6)
FLOOR_BITS = 0x01C00000  # float32(6 * 2**-126)
CBASE_BITS = NORM_BITS + 0x3F800000

_WORD_INDICES = list(range(WORDS_PER_LANE))
_ELEMENT_INDICES = list(range(ELEMENTS_PER_LANE))

_LOCAL_STAGES = []
_half_width = 2
while _half_width < ELEMENTS_PER_LANE:
    _LOCAL_STAGES.append([(index, index + _half_width) for index in range(ELEMENTS_PER_LANE) if (index & _half_width) == 0])
    _half_width *= 2

_CROSS_LANE_STAGES = []
_lane_offset = 1
while _lane_offset < LANES_PER_ROW:
    _CROSS_LANE_STAGES.append((_lane_offset, 31 - _lane_offset.bit_length() + 1))
    _lane_offset *= 2


def _tree_max(values):
    while len(values) > 1:
        values = [cute.max(values[2 * index], values[2 * index + 1]) for index in range(len(values) // 2)]
    return values[0]


@dsl_user_op
def _cvt_bf16x2(high: F32, low: F32, *, loc=None, ip=None) -> U32:
    """Round two FP32 values to BF16 RNE and pack high/low halves."""

    return U32(
        llvm.inline_asm(
            T.i32(),
            [
                F32(high).ir_value(loc=loc, ip=ip),
                F32(low).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _qdq2(values: U32, packed_scale: U32, *, loc=None, ip=None) -> U32:
    """Quantize BF16x2 to finite E2M1, dequantize, and apply UE8M0 scale."""

    return U32(
        llvm.inline_asm(
            T.i32(),
            [
                U32(values).ir_value(loc=loc, ip=ip),
                U32(packed_scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .b8 q;\n\t"
            ".reg .b32 d;\n\t"
            "cvt.rn.satfinite.e2m1x2.bf16x2 q, $1;\n\t"
            "cvt.rn.bf16x2.e2m1x2 d, q;\n\t"
            "mul.rn.bf16x2 $0, d, $2;\n\t"
            "}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def _fwht_mxfp4_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor, rows: I32):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()

    lane_in_row = thread_index & (LANES_PER_ROW - 1)
    row = block_index * ROWS_PER_CTA + (thread_index >> (LANES_PER_ROW.bit_length() - 1))
    safe_row = cute.min(row, rows - 1)

    stage_signs = [(U32(0x3F800000) | ((U32(lane_in_row) & U32(lane_offset)) << sign_shift)).bitcast(F32) for lane_offset, sign_shift in _CROSS_LANE_STAGES]

    source = input_tensor[(None, lane_in_row, safe_row)]
    input_fragment = cute.make_fragment_like(source)
    cute.autovec_copy(source, input_fragment)

    values = [None] * ELEMENTS_PER_LANE
    for word_index in _WORD_INDICES:
        packed = input_fragment[word_index]
        even = (packed << 16).bitcast(F32)
        odd = (packed & U32(0xFFFF0000)).bitcast(F32)
        values[2 * word_index] = even + odd
        values[2 * word_index + 1] = even - odd

    for pairs in _LOCAL_STAGES:
        for low_index, high_index in pairs:
            low = values[low_index]
            high = values[high_index]
            values[low_index] = low + high
            values[high_index] = low - high

    for stage_index, (lane_offset, _) in enumerate(_CROSS_LANE_STAGES):
        sign = stage_signs[stage_index]
        partners = [cute.arch.shuffle_sync_bfly(values[index], lane_offset) for index in _ELEMENT_INDICES]
        for index in _ELEMENT_INDICES:
            values[index] = sign * values[index] + partners[index]

    norm = U32(NORM_BITS).bitcast(F32)
    inverse_six = U32(INV6_BITS).bitcast(F32)
    scale_floor = U32(FLOOR_BITS).bitcast(F32)

    group_amax = _tree_max([cute.abs(values[index]) for index in _ELEMENT_INDICES])
    rounded_amax = cute.max((group_amax * norm).to(cutlass.BFloat16).to(F32), scale_floor)
    biased_scale_exponent = ((rounded_amax * inverse_six).bitcast(U32) + U32(0x7FFFFF)) >> 23
    scale_bits = biased_scale_exponent << 23
    packed_scale = (biased_scale_exponent << 7) | scale_bits
    combined_normalization = (U32(CBASE_BITS) - scale_bits).bitcast(F32)

    output_fragment = cute.make_fragment_like(source)
    for word_index in _WORD_INDICES:
        output_fragment[word_index] = _qdq2(
            _cvt_bf16x2(
                values[2 * word_index + 1] * combined_normalization,
                values[2 * word_index] * combined_normalization,
            ),
            packed_scale,
        )

    if row < rows:
        cute.autovec_copy(output_fragment, output_tensor[(None, lane_in_row, safe_row)])


@cute.jit
def _fwht_mxfp4(input_tensor: cute.Tensor, output_tensor: cute.Tensor, stream):
    rows = I32(input_tensor.shape[0])
    input_pointer = cute.recast_ptr(input_tensor.iterator, dtype=U32)
    output_pointer = cute.recast_ptr(output_tensor.iterator, dtype=U32)
    layout = cute.make_layout(
        (WORDS_PER_LANE, LANES_PER_ROW, rows),
        stride=(1, WORDS_PER_LANE, 64),
    )
    global_input = cute.make_tensor(input_pointer, layout)
    global_output = cute.make_tensor(output_pointer, layout)
    grid = cute.ceil_div(rows, ROWS_PER_CTA)
    _fwht_mxfp4_kernel(global_input, global_output, rows).launch(grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream)


_COMPILED_KERNELS = {}


def run_fwht_mxfp4_qdq(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    """Launch the internal kernel on checked contiguous ``[M, 128]`` tensors."""

    from cuda.bindings import driver as cuda_driver

    device = input_tensor.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(device)
    cache_key = (device_index, compute_capability)
    with torch.cuda.device(device):
        stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

        compiled = _COMPILED_KERNELS.get(cache_key)
        if compiled is None:
            input_cute = from_dlpack(
                input_tensor,
                enable_tvm_ffi=True,
                assumed_align=32,
            ).mark_layout_dynamic()
            output_cute = from_dlpack(
                output_tensor,
                enable_tvm_ffi=True,
                assumed_align=32,
            ).mark_layout_dynamic()
            compiled = cute.compile(
                _fwht_mxfp4,
                input_cute,
                output_cute,
                stream,
                options="--enable-tvm-ffi",
            )
            _COMPILED_KERNELS[cache_key] = compiled

        compiled(input_tensor, output_tensor, stream)


__all__ = ["run_fwht_mxfp4_qdq"]
