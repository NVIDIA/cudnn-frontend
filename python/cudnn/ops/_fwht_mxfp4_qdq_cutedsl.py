# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal CuTe DSL implementation of normalized H128 plus MXFP4 QDQ.

The self-contained device kernel uses a 64-thread, four-lanes-per-row packed
schedule.  The FE integration owns the semantic public API, checked alignment
and empty-input contract, current-device stream launch, and per-device/compute-
capability compiled-kernel cache.  It does not invoke an external optimized
kernel at runtime.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

U16 = cutlass.Uint16
U32 = cutlass.Uint32
U64 = getattr(cutlass, "Uint64", None) or cutlass.Int64
I32 = cutlass.Int32

THREADS = 64
LANES_PER_ROW = 4
ELEMENTS_PER_LANE = 128 // LANES_PER_ROW
WORDS_PER_LANE = ELEMENTS_PER_LANE // 2
ROWS_PER_CTA = THREADS // LANES_PER_ROW
WORDS_PER_VECTOR = 8
VECTORS_PER_LANE = WORDS_PER_LANE // WORDS_PER_VECTOR
GROUPS_PER_LANE = VECTORS_PER_LANE
WORDS_PER_GROUP_PER_LANE = WORDS_PER_LANE // GROUPS_PER_LANE

NORM_BITS = 0x3DB504F3  # float32(128 ** -0.5)

_WORD_INDICES = list(range(WORDS_PER_LANE))
_VECTOR_INDICES = list(range(VECTORS_PER_LANE))
_GROUP_INDICES = list(range(GROUPS_PER_LANE))
_GROUP_WORD_INDICES = list(range(WORDS_PER_GROUP_PER_LANE))

# Pair register p holds elements 64*(p >> 3) + 16*lane + 2*(p & 7) and +1.
# Element bits 0..3 and 6 are register-local, while bits 4 and 5 select the
# lane.  Stages must run in the reference order h = 1, 2, 4, 8, 16, 32, 64.
_LOCAL_STAGE_MASKS_A = [1, 2, 4]
_LOCAL_STAGE_MASKS_B = [8]
_CROSS_LANE_STAGES = [(1, 31), (2, 30)]


def _pairs(mask):
    return [(index, index ^ mask) for index in _WORD_INDICES if (index & mask) == 0]


_LOCAL_STAGES_A = [_pairs(mask) for mask in _LOCAL_STAGE_MASKS_A]
_LOCAL_STAGES_B = [_pairs(mask) for mask in _LOCAL_STAGE_MASKS_B]


def _asm64(text, operands, constraints):
    return U64(
        llvm.inline_asm(
            T.i64(),
            operands,
            text,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _asm32(text, operands, constraints):
    return U32(
        llvm.inline_asm(
            T.i32(),
            operands,
            text,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _unpack_h1(word: U32, *, loc=None, ip=None) -> U64:
    """Unpack BF16x2 ``[b:a]`` to the FP32 pair ``(a+b, a-b)``."""

    return _asm64(
        "{\n\t"
        ".reg .b16 low, high;\n\t"
        ".reg .b32 tmp;\n\t"
        ".reg .f32 high_f32, sum, difference;\n\t"
        "and.b32 tmp, $1, -65536;\n\t"
        "mov.b32 high_f32, tmp;\n\t"
        "mov.b32 {low, high}, $1;\n\t"
        "add.rn.f32.bf16 sum, low, high_f32;\n\t"
        "sub.rn.f32.bf16 difference, low, high_f32;\n\t"
        "mov.b64 $0, {sum, difference};\n\t"
        "}",
        [U32(word).ir_value(loc=loc, ip=ip)],
        "=l,r",
    )


@dsl_user_op
def _add2(left: U64, right: U64, *, loc=None, ip=None) -> U64:
    return _asm64(
        "add.rn.f32x2 $0, $1, $2;",
        [U64(left).ir_value(loc=loc, ip=ip), U64(right).ir_value(loc=loc, ip=ip)],
        "=l,l,l",
    )


@dsl_user_op
def _sub2(left: U64, right: U64, *, loc=None, ip=None) -> U64:
    return _asm64(
        "sub.rn.f32x2 $0, $1, $2;",
        [U64(left).ir_value(loc=loc, ip=ip), U64(right).ir_value(loc=loc, ip=ip)],
        "=l,l,l",
    )


@dsl_user_op
def _mul2(left: U64, right: U64, *, loc=None, ip=None) -> U64:
    return _asm64(
        "mul.rn.f32x2 $0, $1, $2;",
        [U64(left).ir_value(loc=loc, ip=ip), U64(right).ir_value(loc=loc, ip=ip)],
        "=l,l,l",
    )


@dsl_user_op
def _butterfly_fma2(values: U64, sign: U64, lane_offset: int, *, loc=None, ip=None) -> U64:
    """Apply one cross-lane butterfly stage to one FP32 pair register."""

    return _asm64(
        "{\n\t"
        ".reg .b32 low, high, shuffled_low, shuffled_high;\n\t"
        ".reg .b64 shuffled;\n\t"
        "mov.b64 {low, high}, $1;\n\t"
        "shfl.sync.bfly.b32 shuffled_low, low, %d, 31, -1;\n\t"
        "shfl.sync.bfly.b32 shuffled_high, high, %d, 31, -1;\n\t"
        "mov.b64 shuffled, {shuffled_low, shuffled_high};\n\t"
        "fma.rn.f32x2 $0, $2, $1, shuffled;\n\t"
        "}" % (lane_offset, lane_offset),
        [U64(values).ir_value(loc=loc, ip=ip), U64(sign).ir_value(loc=loc, ip=ip)],
        "=l,l,l",
    )


@dsl_user_op
def _splat2(word: U32, *, loc=None, ip=None) -> U64:
    """Replicate one 32-bit word into both halves of a pair register."""

    return _asm64(
        "mov.b64 $0, {$1, $1};",
        [U32(word).ir_value(loc=loc, ip=ip)],
        "=l,r",
    )


@dsl_user_op
def _shuffle_group_max(values: U32, *, loc=None, ip=None) -> U32:
    return _asm32(
        "shfl.sync.bfly.b32 $0, $1, 1, 31, -1;",
        [U32(values).ir_value(loc=loc, ip=ip)],
        "=r,r",
    )


@dsl_user_op
def _byte_permute(left: U32, right: U32, selector: int, *, loc=None, ip=None) -> U32:
    """Permute bytes over the ``{right:left}`` 64-bit source."""

    return _asm32(
        "prmt.b32 $0, $1, $2, %d;" % selector,
        [U32(left).ir_value(loc=loc, ip=ip), U32(right).ir_value(loc=loc, ip=ip)],
        "=r,r,r",
    )


@dsl_user_op
def _cvt_bf16x2(values: U64, *, loc=None, ip=None) -> U32:
    """Round FP32 pair ``(low, high)`` to packed BF16x2 ``[high:low]``."""

    return _asm32(
        "{\n\t" ".reg .f32 low, high;\n\t" "mov.b64 {low, high}, $1;\n\t" "cvt.rn.bf16x2.f32 $0, high, low;\n\t" "}",
        [U64(values).ir_value(loc=loc, ip=ip)],
        "=r,l",
    )


@dsl_user_op
def _max_abs_bf16x2(left: U32, right: U32, *, loc=None, ip=None) -> U32:
    """Take the lane-wise BF16x2 maximum magnitude."""

    return _asm32(
        "max.xorsign.abs.bf16x2 $0, $1, $2;",
        [U32(left).ir_value(loc=loc, ip=ip), U32(right).ir_value(loc=loc, ip=ip)],
        "=r,r,r",
    )


@dsl_user_op
def _max_bf16x2(left: U32, right: U32, *, loc=None, ip=None) -> U32:
    """Take the lane-wise BF16x2 maximum."""

    return _asm32(
        "max.bf16x2 $0, $1, $2;",
        [U32(left).ir_value(loc=loc, ip=ip), U32(right).ir_value(loc=loc, ip=ip)],
        "=r,r,r",
    )


@dsl_user_op
def _to_u16(values: U32, *, loc=None, ip=None) -> U16:
    return U16(
        llvm.inline_asm(
            T.i16(),
            [U32(values).ir_value(loc=loc, ip=ip)],
            "cvt.u16.u32 $0, $1;",
            "=h,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _qdq4(values: U32, packed_inverse_scale: U32, packed_scale_bytes: U16, *, loc=None, ip=None) -> U32:
    """Apply packed BF16 scaling, E2M1 RNE, and UE8M0-scaled decode."""

    return _asm32(
        "{\n\t"
        ".reg .b8 quantized;\n\t"
        ".reg .b32 normalized;\n\t"
        "mul.rn.bf16x2 normalized, $1, $2;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 quantized, normalized;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 $0, quantized, $3;\n\t"
        "}",
        [
            U32(values).ir_value(loc=loc, ip=ip),
            U32(packed_inverse_scale).ir_value(loc=loc, ip=ip),
            U16(packed_scale_bytes).ir_value(loc=loc, ip=ip),
        ],
        "=r,r,r,h",
    )


def _tree_max_bf16x2(values):
    while len(values) > 1:
        values = [_max_abs_bf16x2(values[2 * index], values[2 * index + 1]) for index in range(len(values) // 2)]
    return values[0]


@cute.kernel
def _fwht_mxfp4_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor, rows: I32):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()

    lane_in_row = thread_index & (LANES_PER_ROW - 1)
    row = block_index * ROWS_PER_CTA + (thread_index >> (LANES_PER_ROW.bit_length() - 1))
    safe_row = cute.min(row, rows - 1)

    # Each lane owns two aligned 32-byte runs, 128 bytes apart.  The four lanes
    # of one row cover exactly one 128-byte cache line per vector access.
    source = [input_tensor[(None, vector, lane_in_row, safe_row)] for vector in _VECTOR_INDICES]
    input_fragments = [cute.make_fragment_like(tensor) for tensor in source]
    for vector in _VECTOR_INDICES:
        cute.autovec_copy(source[vector], input_fragments[vector])

    # Unpack BF16 pairs into FP32 pair registers and fold in butterfly h=1.
    values = [_unpack_h1(input_fragments[index >> 3][index & 7]) for index in _WORD_INDICES]

    # Register-local butterfly stages h=2,4,8.
    for pairs in _LOCAL_STAGES_A:
        for low_index, high_index in pairs:
            low = values[low_index]
            high = values[high_index]
            values[low_index] = _add2(low, high)
            values[high_index] = _sub2(low, high)

    # Cross-lane butterfly stages h=16,32.
    for lane_offset, sign_shift in _CROSS_LANE_STAGES:
        sign = _splat2(U32(0x3F800000) | ((U32(lane_in_row) & U32(lane_offset)) << sign_shift))
        for index in _WORD_INDICES:
            values[index] = _butterfly_fma2(values[index], sign, lane_offset)

    # Register-local butterfly stage h=64.
    for pairs in _LOCAL_STAGES_B:
        for low_index, high_index in pairs:
            low = values[low_index]
            high = values[high_index]
            values[low_index] = _add2(low, high)
            values[high_index] = _sub2(low, high)

    # Normalize and materialize the official BF16 boundary before group amax.
    norm = _splat2(U32(NORM_BITS))
    rounded_values = [_cvt_bf16x2(_mul2(values[index], norm)) for index in _WORD_INDICES]

    # Pair registers 0..7 and 8..15 belong to two separate group-32 values,
    # each shared by lane_in_row and lane_in_row^1.  Interleave the two partial
    # maxima so one packed maximum and one lane shuffle finish both groups.
    partial_group_0 = _tree_max_bf16x2(rounded_values[:WORDS_PER_GROUP_PER_LANE])
    partial_group_1 = _tree_max_bf16x2(rounded_values[WORDS_PER_GROUP_PER_LANE:])
    packed_amax = _max_abs_bf16x2(
        _byte_permute(partial_group_0, partial_group_1, 0x5410),
        _byte_permute(partial_group_0, partial_group_1, 0x7632),
    )
    packed_amax = packed_amax & U32(0x7FFF7FFF)
    packed_amax = _max_bf16x2(packed_amax, _shuffle_group_max(packed_amax))
    packed_amax = _max_bf16x2(packed_amax, U32(0x01C001C0))

    # Both halfwords are at least 0x01c0, so packed subtraction cannot borrow
    # across the 16-bit boundary.  The result directly encodes
    # scale_byte=ceil_exp_bias-2 for both groups.
    biased_amax = packed_amax - U32(0x00C100C1)

    output_fragments = [cute.make_fragment_like(tensor) for tensor in source]
    for group in _GROUP_INDICES:
        scale_byte = ((biased_amax & U32(0xFFFF)) >> 7) if group == 0 else (biased_amax >> 23)
        packed_inverse_scale = (U32(254) - scale_byte) * U32(0x00800080)
        packed_scale_bytes = _to_u16(scale_byte * U32(0x0101))
        base = group * WORDS_PER_GROUP_PER_LANE
        for index in _GROUP_WORD_INDICES:
            output_fragments[group][index] = _qdq4(
                rounded_values[base + index],
                packed_inverse_scale,
                packed_scale_bytes,
            )

    if row < rows:
        for vector in _VECTOR_INDICES:
            cute.autovec_copy(output_fragments[vector], output_tensor[(None, vector, lane_in_row, safe_row)])


@cute.jit
def _fwht_mxfp4(input_tensor: cute.Tensor, output_tensor: cute.Tensor, stream):
    rows = I32(input_tensor.shape[0])
    input_pointer = cute.recast_ptr(input_tensor.iterator, dtype=U32)
    output_pointer = cute.recast_ptr(output_tensor.iterator, dtype=U32)
    layout = cute.make_layout(
        (WORDS_PER_VECTOR, VECTORS_PER_LANE, LANES_PER_ROW, rows),
        stride=(1, WORDS_PER_VECTOR * LANES_PER_ROW, WORDS_PER_VECTOR, 64),
    )
    global_input = cute.make_tensor(input_pointer, layout)
    global_output = cute.make_tensor(output_pointer, layout)
    grid = cute.ceil_div(rows, ROWS_PER_CTA)
    _fwht_mxfp4_kernel(global_input, global_output, rows).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


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
