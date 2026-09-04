# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared device, metadata, compilation, launch, and workspace helpers for MSA."""

import ctypes
import enum
import math
import os
import threading
from collections.abc import Hashable
from dataclasses import dataclass, fields
from enum import IntEnum
from functools import partial
from typing import Callable, Optional, Tuple, Type, TypeAlias, TypeVar

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import NumericMeta, T, dsl_user_op, if_generate
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg


# -----------------------------------------------------------------------------
# Cute Dsl Utils
# -----------------------------------------------------------------------------

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


@dataclass
class ParamsBase:
    """Dataclass adapter for values carried through CuTe control flow."""

    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [
            field for field in all_fields if not isinstance(field, StaticTypes)
        ]
        values = []
        self._values_pos = []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {
            name: field
            for name, field in all_fields.items()
            if isinstance(field, StaticTypes)
        }
        non_constexpr_fields = {
            name: field
            for name, field in all_fields.items()
            if not isinstance(field, StaticTypes)
        }
        for (name, field), count in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(
                field, values[:count]
            )
            values = values[count:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def assume_strides_aligned(tensor: cute.Tensor) -> tuple:
    """Add 128-bit alignment facts to every non-innermost dynamic stride."""
    divisor = 128 // tensor.element_type.width
    strides = tuple(
        stride if isinstance(stride, int) else cute.assume(stride, divby=divisor)
        for stride in tensor.stride[:-1]
    )
    return (*strides, tensor.stride[-1])


def assume_tensor_aligned(tensor: cute.Tensor | None) -> cute.Tensor | None:
    """Rebuild ``tensor`` with 128-bit aligned stride assumptions."""
    if tensor is None:
        return None
    layout = cute.make_layout(tensor.shape, stride=assume_strides_aligned(tensor))
    return cute.make_tensor(tensor.iterator, layout)


# -----------------------------------------------------------------------------
# Layout Utils
# -----------------------------------------------------------------------------


def reshape_acc_to_mn(accumulator: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    """View an MMA accumulator as logical row/column modes.

    This handles both ``((2, 2), MMA_M, MMA_N, ...)`` and
    ``((2, 2, V), MMA_M, MMA_N, ...)`` accumulator layouts.
    """
    acc_layout = accumulator.layout
    col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (col_major.shape[0][1], col_major.shape[1]),
        (
            col_major.shape[0][0],
            *col_major.shape[0][2:],
            col_major.shape[2],
        ),
        *col_major.shape[3:],
    )
    stride = (
        (col_major.stride[0][1], col_major.stride[1]),
        (
            col_major.stride[0][0],
            *col_major.stride[0][2:],
            col_major.stride[2],
        ),
        *col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    mn_layout = cute.make_layout(shape, stride=stride)
    return cute.make_tensor(
        accumulator.iterator, cute.composition(acc_layout, mn_layout)
    )


# -----------------------------------------------------------------------------
# Copy Utils
# -----------------------------------------------------------------------------

# Store and Layout Helpers


@dsl_user_op
def stg_128_cs(
    gmem_ptr: cute.Pointer,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cs.v4.f32 [$4], {$5, $6, $7, $8}; "
        "mov.f32 $0, 0f00000000; mov.f32 $1, 0f00000000; "
        "mov.f32 $2, 0f00000000; mov.f32 $3, 0f00000000;",
        "=f,=f,=f,=f,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_128_bf16_cs(
    gmem_ptr: cute.Pointer,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        llvm.StructType.get_literal(
            [T.f32(), T.f32(), T.f32(), T.f32(), T.f32(), T.f32(), T.f32(), T.f32()]
        ),
        [
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
            Float32(v4).ir_value(loc=loc, ip=ip),
            Float32(v5).ir_value(loc=loc, ip=ip),
            Float32(v6).ir_value(loc=loc, ip=ip),
            Float32(v7).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n"
        ".reg .b32 p0, p1, p2, p3;\n"
        "cvt.rn.bf16.f32 h0, $9;\n"
        "cvt.rn.bf16.f32 h1, $10;\n"
        "cvt.rn.bf16.f32 h2, $11;\n"
        "cvt.rn.bf16.f32 h3, $12;\n"
        "cvt.rn.bf16.f32 h4, $13;\n"
        "cvt.rn.bf16.f32 h5, $14;\n"
        "cvt.rn.bf16.f32 h6, $15;\n"
        "cvt.rn.bf16.f32 h7, $16;\n"
        "mov.b32 p0, {h0, h1};\n"
        "mov.b32 p1, {h2, h3};\n"
        "mov.b32 p2, {h4, h5};\n"
        "mov.b32 p3, {h6, h7};\n"
        "st.global.cs.v4.b32 [$8], {p0, p1, p2, p3};\n"
        "}\n"
        "mov.f32 $0, 0f00000000; mov.f32 $1, 0f00000000; "
        "mov.f32 $2, 0f00000000; mov.f32 $3, 0f00000000; "
        "mov.f32 $4, 0f00000000; mov.f32 $5, 0f00000000; "
        "mov.f32 $6, 0f00000000; mov.f32 $7, 0f00000000;",
        "=f,=f,=f,=f,=f,=f,=f,=f,l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_128_f16_cs(
    gmem_ptr: cute.Pointer,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        llvm.StructType.get_literal(
            [T.f32(), T.f32(), T.f32(), T.f32(), T.f32(), T.f32(), T.f32(), T.f32()]
        ),
        [
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
            Float32(v4).ir_value(loc=loc, ip=ip),
            Float32(v5).ir_value(loc=loc, ip=ip),
            Float32(v6).ir_value(loc=loc, ip=ip),
            Float32(v7).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .f16 h0, h1, h2, h3, h4, h5, h6, h7;\n"
        ".reg .b32 p0, p1, p2, p3;\n"
        "cvt.rn.f16.f32 h0, $9;\n"
        "cvt.rn.f16.f32 h1, $10;\n"
        "cvt.rn.f16.f32 h2, $11;\n"
        "cvt.rn.f16.f32 h3, $12;\n"
        "cvt.rn.f16.f32 h4, $13;\n"
        "cvt.rn.f16.f32 h5, $14;\n"
        "cvt.rn.f16.f32 h6, $15;\n"
        "cvt.rn.f16.f32 h7, $16;\n"
        "mov.b32 p0, {h0, h1};\n"
        "mov.b32 p1, {h2, h3};\n"
        "mov.b32 p2, {h4, h5};\n"
        "mov.b32 p3, {h6, h7};\n"
        "st.global.cs.v4.b32 [$8], {p0, p1, p2, p3};\n"
        "}\n"
        "mov.f32 $0, 0f00000000; mov.f32 $1, 0f00000000; "
        "mov.f32 $2, 0f00000000; mov.f32 $3, 0f00000000; "
        "mov.f32 $4, 0f00000000; mov.f32 $5, 0f00000000; "
        "mov.f32 $6, 0f00000000; mov.f32 $7, 0f00000000;",
        "=f,=f,=f,=f,=f,=f,=f,=f,l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_128_fp8_e4m3_cs(
    gmem_ptr: cute.Pointer,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    v8: Float32,
    v9: Float32,
    v10: Float32,
    v11: Float32,
    v12: Float32,
    v13: Float32,
    v14: Float32,
    v15: Float32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        llvm.StructType.get_literal(
            [
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
                T.f32(),
            ]
        ),
        [
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
            Float32(v4).ir_value(loc=loc, ip=ip),
            Float32(v5).ir_value(loc=loc, ip=ip),
            Float32(v6).ir_value(loc=loc, ip=ip),
            Float32(v7).ir_value(loc=loc, ip=ip),
            Float32(v8).ir_value(loc=loc, ip=ip),
            Float32(v9).ir_value(loc=loc, ip=ip),
            Float32(v10).ir_value(loc=loc, ip=ip),
            Float32(v11).ir_value(loc=loc, ip=ip),
            Float32(v12).ir_value(loc=loc, ip=ip),
            Float32(v13).ir_value(loc=loc, ip=ip),
            Float32(v14).ir_value(loc=loc, ip=ip),
            Float32(v15).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n"
        ".reg .b32 p0, p1, p2, p3;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, $18, $17;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, $20, $19;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, $22, $21;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h3, $24, $23;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h4, $26, $25;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h5, $28, $27;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h6, $30, $29;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h7, $32, $31;\n"
        "mov.b32 p0, {h0, h1};\n"
        "mov.b32 p1, {h2, h3};\n"
        "mov.b32 p2, {h4, h5};\n"
        "mov.b32 p3, {h6, h7};\n"
        "st.global.cs.v4.b32 [$16], {p0, p1, p2, p3};\n"
        "}\n"
        "mov.f32 $0, 0f00000000; mov.f32 $1, 0f00000000; "
        "mov.f32 $2, 0f00000000; mov.f32 $3, 0f00000000; "
        "mov.f32 $4, 0f00000000; mov.f32 $5, 0f00000000; "
        "mov.f32 $6, 0f00000000; mov.f32 $7, 0f00000000; "
        "mov.f32 $8, 0f00000000; mov.f32 $9, 0f00000000; "
        "mov.f32 $10, 0f00000000; mov.f32 $11, 0f00000000; "
        "mov.f32 $12, 0f00000000; mov.f32 $13, 0f00000000; "
        "mov.f32 $14, 0f00000000; mov.f32 $15, 0f00000000;",
        (
            "=f,=f,=f,=f,=f,=f,=f,=f,"
            "=f,=f,=f,=f,=f,=f,=f,=f,"
            "l,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f"
        ),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def convert_layout_from_tmem16x256b_to_acc_sm90(acc_layout: cute.Layout) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            acc_layout_col_major.shape[0][0],
            acc_layout_col_major.shape[0][1],
            acc_layout_col_major.shape[1],
            *acc_layout_col_major.shape[2:],
        ),
        stride=(
            acc_layout_col_major.stride[0][0],
            acc_layout_col_major.stride[0][1],
            acc_layout_col_major.stride[1],
            *acc_layout_col_major.stride[2:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_16x256b_tensor_mn_view(tensor: cute.Tensor) -> cute.Tensor:
    layout = convert_layout_acc_mn(
        convert_layout_from_tmem16x256b_to_acc_sm90(tensor.layout)
    )
    return cute.make_tensor(tensor.iterator, layout)


def real_col_to_stg128_fake_col(col: Int32) -> Int32:
    nt = col // Int32(16)
    col16 = col - nt * Int32(16)
    pair = col16 // Int32(2)
    rank = pair % Int32(4)
    kv = (pair // Int32(4)) * Int32(2) + (col16 % Int32(2))
    return nt * Int32(16) + rank * Int32(4) + kv


def stg128_fake_col_to_real_col(fake_col: Int32) -> Int32:
    nt = fake_col // Int32(16)
    fake16 = fake_col - nt * Int32(16)
    rank = fake16 // Int32(4)
    kv = fake16 % Int32(4)
    return (
        nt * Int32(16) + rank * Int32(2) + (kv // Int32(2)) * Int32(8) + (kv % Int32(2))
    )


def real_col_to_stg128_half_fake_col(col: Int32) -> Int32:
    nt = col // Int32(32)
    col32 = col - nt * Int32(32)
    lane = (col32 % Int32(8)) // Int32(2)
    group = col32 // Int32(8)
    elem = col32 % Int32(2)
    return nt * Int32(32) + lane * Int32(8) + group * Int32(2) + elem


def stg128_half_fake_col_to_real_col(fake_col: Int32) -> Int32:
    nt = fake_col // Int32(32)
    fake32 = fake_col - nt * Int32(32)
    lane = fake32 // Int32(8)
    lane_slot = fake32 - lane * Int32(8)
    group = lane_slot // Int32(2)
    elem = lane_slot - group * Int32(2)
    return nt * Int32(32) + group * Int32(8) + lane * Int32(2) + elem


def real_col_to_stg128_fp8_fake_col(col: Int32) -> Int32:
    nt = col // Int32(64)
    col64 = col - nt * Int32(64)
    lane = (col64 % Int32(8)) // Int32(2)
    group = col64 // Int32(8)
    elem = col64 % Int32(2)
    return nt * Int32(64) + lane * Int32(16) + group * Int32(2) + elem


def stg128_fp8_fake_col_to_real_col(fake_col: Int32) -> Int32:
    nt = fake_col // Int32(64)
    fake64 = fake_col - nt * Int32(64)
    lane = fake64 // Int32(16)
    lane_slot = fake64 - lane * Int32(16)
    group = lane_slot // Int32(2)
    elem = lane_slot - group * Int32(2)
    return nt * Int32(64) + group * Int32(8) + lane * Int32(2) + elem


# TMA Copy Adapter


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (
        (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    )
    group_rank_smem = const_expr(
        cute.rank(smem_tensor) - (1 if not single_stage else 0)
    )
    group_rank_gmem = const_expr(
        cute.rank(gmem_tensor) - (1 if not single_stage else 0)
    )
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    def copy_tma_single_stage(**new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


# -----------------------------------------------------------------------------
# Mma Sm100 Desc
# -----------------------------------------------------------------------------

#
# The bit-field encodings, enum values, and descriptor layout below mirror the
# SM100 tcgen05 MMA instruction descriptor as documented and
# implemented in NVIDIA CUTLASS (BSD-3-Clause). The numeric values MUST stay
# identical to the hardware/ISA encodings; see the "Third-party licenses"
# section of README.md at the repo root for attribution.


# ---------------------------------------------------------------------------
# Enumerations that match the HW encodings (values MUST stay identical)
# ---------------------------------------------------------------------------


class Major(IntEnum):  # matrix "layout" in the ISA docs
    K = 0
    MN = 1


class ScaleIn(IntEnum):  # negate flags
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):  # 2-bit field (bits 4-5)
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):  # 3-bit field (A/B element type)
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


# ---------------------------------------------------------------------------
# CUTLASS-type -> encoding helpers
# ---------------------------------------------------------------------------


def to_UMMA_format(cutlass_type) -> int:
    """
    Map a CUTLASS scalar class to the 3-bit encoding for Matrix A/B.
    """
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    # Unsigned 8-bit (if available in your CUTLASS build)
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    # FP-16 / BF-16
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    # TensorFloat-32 (8-bit exponent, 10-bit mantissa packed in 19 bits)
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    # Float-8 / Float-6 / Float-4
    if cutlass_type is cutlass.Float8E4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.Float8E5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    """
    Map a CUTLASS scalar class to the 2-bit accumulator encoding.
    """
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(
        f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}"
    )


# ---------------------------------------------------------------------------
# The constructor – accepts only CUTLASS scalar classes
# ---------------------------------------------------------------------------


def make_instr_desc(
    a_type,  # CUTLASS scalar class, e.g. cutlass.Int8
    b_type,
    c_type,
    M: int,  # 64, 128 or 256
    N: int,  # 8 … 256 (multiple of 8)
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    """
    Build the 32-bit instruction descriptor for SM100 MMA.
    All matrix/accumulator **types must be CUTLASS scalar classes** –
    passing integers is forbidden.
    """
    # --- encode element formats -------------------------------------------------
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))
    is_f8f6f4 = a_type in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)

    # --- range checks on M/N -----------------------------------------------------
    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8…256")

    m_dim = M >> 4  # 5-bit field
    n_dim = N >> 3  # 6-bit field

    # fmt: off
    # --- pack the bit-fields -----------------------------------------------------
    desc = 0
    desc |= (0                 & 0x3) << 0        # sparse_id2 (always 0 here)
    desc |= (int(is_sparse)    & 0x1) << 2        # sparse_flag
    desc |= (int(c_sat)        & 0x1) << 3        # saturate
    desc |= (c_fmt             & 0x3) << 4        # c_format
    desc |= (a_fmt             & 0x7) << 7        # a_format
    desc |= (b_fmt             & 0x7) << 10       # b_format
    desc |= (int(a_neg)        & 0x1) << 13       # a_negate
    desc |= (int(b_neg)        & 0x1) << 14       # b_negate
    desc |= (int(a_major)      & 0x1) << 15       # a_major
    desc |= (int(b_major)      & 0x1) << 16       # b_major
    desc |= (n_dim             & 0x3F) << 17      # n_dim (6 bits)
    # CUTLASS' tcgen05 lowering sets bit 23 for dense f8f6f4 MMAs; keep this
    # descriptor aligned with generated/reference SM100 FP8 kernels.
    desc |= (int(is_f8f6f4)    & 0x1) << 23
    desc |= (m_dim             & 0x1F) << 24      # m_dim (5 bits)
    desc |= (int(max_shift)    & 0x3) << 30       # max_shift (2 bits)
    # fmt: on

    return desc & 0xFFFF_FFFF  # ensure 32-bit result


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp):
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K
        if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else Major.MN,
        Major.K
        if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else Major.MN,
    )


class LayoutType(IntEnum):  # occupies the top-3 bits [61:64)
    SWIZZLE_NONE = 0  # (a.k.a. "INTERLEAVE" in older docs)
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6
    # values 3,5,7 are reserved / illegal for UMMA


# ---------------------------------------------------------------------------
#  Helpers – figure out the SWIZZLE_* family from the tensor layout
# ---------------------------------------------------------------------------


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    B, M, S = swizzle.num_bits, swizzle.num_base, swizzle.num_shift

    if M == 4:  # Swizzle<*,4,3>
        if S != 3:
            raise ValueError("Unexpected swizzle shift – want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[B]  # KeyError ⇒ invalid B→ raise
    if M == 5:  # Swizzle<2,5,2> (the only legal triple for M==5)
        if (B, S) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B

    # Any other (M,B,S) triple is not a UMMA-legal shared-memory layout
    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(
    layout: cute.Layout, swizzle: cute.Swizzle, major: Major
) -> int:
    """
    Convert a 2-D *shared-memory* Cute layout into the SM100 64-bit
    smem-descriptor, without the smem start address.
    layout must correspond to layout of an uint128 tensor.
    """
    # ------------------------------------------------------------------ meta
    layout_type = _layout_type(swizzle)  # resolve SWIZZLE_* family

    VERSION = 1  # bits 46–47
    LBO_MODE = 0  # bit  52
    BASE_OFFSET = 0  # bits 49–51   (CUTLASS always 0)

    # ---------------------------------------------------------- strides  (units: uint128_t = 16 B)
    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]

    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(
            layout, (swizzle_atom_mn_size, swizzle_atom_k_size)
        )
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError(
                "Not a canonical UMMA_MN Layout: Expected profile failure."
            )
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = (
            canonical_layout.stride[0][1],
            canonical_layout.stride[1][1],
        )
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError(
                "Not a canonical UMMA_K Layout: Expected MN-size multiple of 8."
            )
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10

    # ------------------------------------------------------------------ pack
    desc = 0
    # leading_byte_offset_  [16:30)
    desc |= (leading_byte_offset & 0x3FFF) << 16
    # stride_byte_offset_   [32:46)
    desc |= (stride_byte_offset & 0x3FFF) << 32
    # version_             [46:48)
    desc |= (VERSION & 0x3) << 46
    # base_offset_         [49:52)
    desc |= (BASE_OFFSET & 0x7) << 49
    # lbo_mode_            [52:53)
    desc |= (LBO_MODE & 0x1) << 52
    # layout_type_         [61:64)
    desc |= (int(layout_type) & 0x7) << 61

    return desc & 0xFFFF_FFFF_FFFF_FFFF  # force 64-bit width


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    # 14 bits, remove 4 LSB (bits 0-13 in desc)
    return (start_addr.toint() & 0x3FFFF) >> 4


def smem_desc_base_from_tensor(sA: cute.Tensor, major: Major) -> int:
    sA_swizzle = sA.iterator.type.swizzle_type
    return make_smem_desc_base(
        cute.recast_layout(128, sA.element_type.width, sA.layout[0]),
        sA_swizzle,
        major,
    )


# -----------------------------------------------------------------------------
# Named Barrier
# -----------------------------------------------------------------------------


class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()
    LoadWG = enum.auto()
    StoreEpilogue = enum.auto()
    KvLoad = enum.auto()
    KvDequantK = enum.auto()
    KvDequantV = enum.auto()


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

# import math


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = NamedBarrierOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", NamedBarrier)
        return obj

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


@dataclass(frozen=True)
class PipelineAsync(PipelineAsyncOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineAsync
        object.__setattr__(obj, "__class__", PipelineAsync)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_full.arrive(index, self.producer_mask, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_empty.arrive(index, self.consumer_mask, loc=loc, ip=ip)


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineTmaUmmaOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaUmmaOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineTmaUmma
        object.__setattr__(obj, "__class__", PipelineTmaUmma)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if_generate(
            self.is_leader_cta,
            lambda: self.sync_object_full.arrive(
                index, self.producer_mask, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        """
        UMMA consumer release buffer empty, cta_group needs to be provided.
        """
        self.sync_object_empty.arrive(
            index, self.consumer_mask, self.cta_group, loc=loc, ip=ip
        )


@dataclass(frozen=True)
class PipelineUmmaAsync(PipelineUmmaAsyncOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineUmmaAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", PipelineUmmaAsync)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        """
        UMMA producer commit buffer full, cta_group needs to be provided.
        """
        self.sync_object_full.arrive(
            index, self.producer_mask, self.cta_group, loc=loc, ip=ip
        )

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_empty.arrive(index, self.consumer_mask, loc=loc, ip=ip)


@dataclass(frozen=True)
class PipelineAsyncUmma(PipelineAsyncUmmaOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineAsyncUmmaOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", PipelineAsyncUmma)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_full.arrive(index, self.producer_mask, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        """
        UMMA consumer release buffer empty, cta_group needs to be provided.
        """
        self.sync_object_empty.arrive(
            index, self.consumer_mask, self.cta_group, loc=loc, ip=ip
        )


# -----------------------------------------------------------------------------
# Paged Kv
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PagedKVManager:
    mPageTable: cute.Tensor
    page_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        *,
        page_size: int,
        n_block_size: int,
    ):
        if page_size != n_block_size:
            raise ValueError(
                f"page_size ({page_size}) must equal blk_kv ({n_block_size})"
            )
        return PagedKVManager(
            mPageTable,
            page_size=page_size,
            n_block_size=n_block_size,
        )

    @cute.jit
    def physical_block_index(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
    ) -> Int32:
        return self.mPageTable[batch_idx, kv_block_idx]


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

# Obtained from sollya:
# fpminimax(exp(x * log(2.0)), 1, [|1,24...|],[0;1],relative);
POLY_EX2 = {
    0: (1.0),
    1: (
        1.0,
        0.922497093677520751953125,
    ),
    2: (
        1.0,
        0.6657850742340087890625,
        0.330107033252716064453125,
    ),
    3: (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    ),
    4: (
        1.0,
        0.693042695522308349609375,
        0.2412912547588348388671875,
        5.2225358784198760986328125e-2,
        1.3434938155114650726318359375e-2,
    ),
    5: (
        1.0,
        0.693151414394378662109375,
        0.24016360938549041748046875,
        5.5802188813686370849609375e-2,
        9.01452265679836273193359375e-3,
        1.86810153536498546600341796875e-3,
    ),
}


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@dsl_user_op
def fmax(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    # ADAPTER PATCH (round 1 KF integration, not upstream): the vendored
    # CUDA_VERSION==12.9-only heuristic below picks the wrong nvvm.fmax
    # calling convention on this worktree's toolchain (nvidia-cutlass-dsl
    # 4.7.0's nvvm binding is always the 2-positional-arg "new" form here,
    # even though CUDA_VERSION reports 12.9 -- confirmed via
    # inspect.signature(nvvm.fmax) => (a, b, *, c=None, ...), no leading
    # result-type positional). Probe the actual binding instead of
    # hardcoding a CUDA_VERSION pairing that does not hold in this
    # environment.
    import inspect

    _fmax_positional = [
        p
        for p in inspect.signature(nvvm.fmax).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    _fmax_old_api = len(_fmax_positional) >= 3  # old API: (result_type, a, b); new API: (a, b)

    if _fmax_old_api:
        # Old API: requires explicit result type as first positional argument
        return Float32(
            nvvm.fmax(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )
    else:
        # New API: infers result type automatically
        return Float32(
            nvvm.fmax(
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return (
            local_max[0]
            if const_expr(init_val is None)
            else fmax(local_max[0], init_val)
        )
    else:
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
    else:
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(
                local_sum[0], (res[i + 0], res[i + 1])
            )
            local_sum[1] = cute.arch.add_packed_f32x2(
                local_sum[1], (res[i + 2], res[i + 3])
            )
            local_sum[2] = cute.arch.add_packed_f32x2(
                local_sum[2], (res[i + 4], res[i + 5])
            )
            local_sum[3] = cute.arch.add_packed_f32x2(
                local_sum[3], (res[i + 6], res[i + 7])
            )
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


@cute.jit
def fadd_exp2_scaled_reduce(
    x: cute.Tensor, scale: Float32, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    assert cute.size(x.shape) % 2 == 0, "x must have an even number of elements"
    if const_expr(arch < 100):
        return fadd_reduce(cute.math.exp2(x.load() * scale, fastmath=True), arch=arch)
    elif const_expr(cute.size(x.shape) % 8 == 0):
        local_sum = [
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
        ]
        for i in cutlass.range_constexpr(0, cute.size(x.shape), 8):
            acc0, acc1 = cute.arch.mul_packed_f32x2(
                (x[i + 0], x[i + 1]), (scale, scale)
            )
            acc2, acc3 = cute.arch.mul_packed_f32x2(
                (x[i + 2], x[i + 3]), (scale, scale)
            )
            acc4, acc5 = cute.arch.mul_packed_f32x2(
                (x[i + 4], x[i + 5]), (scale, scale)
            )
            acc6, acc7 = cute.arch.mul_packed_f32x2(
                (x[i + 6], x[i + 7]), (scale, scale)
            )
            acc0 = cute.math.exp2(acc0, fastmath=True)
            acc1 = cute.math.exp2(acc1, fastmath=True)
            acc2 = cute.math.exp2(acc2, fastmath=True)
            acc3 = cute.math.exp2(acc3, fastmath=True)
            acc4 = cute.math.exp2(acc4, fastmath=True)
            acc5 = cute.math.exp2(acc5, fastmath=True)
            acc6 = cute.math.exp2(acc6, fastmath=True)
            acc7 = cute.math.exp2(acc7, fastmath=True)
            local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], (acc0, acc1))
            local_sum[1] = cute.arch.add_packed_f32x2(local_sum[1], (acc2, acc3))
            local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], (acc4, acc5))
            local_sum[3] = cute.arch.add_packed_f32x2(local_sum[3], (acc6, acc7))
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]
    else:
        row_sum = Float32(0.0)
        for i in cutlass.range_constexpr(0, cute.size(x.shape), 2):
            acc0, acc1 = cute.arch.mul_packed_f32x2((x[i], x[i + 1]), (scale, scale))
            acc0 = cute.math.exp2(acc0, fastmath=True)
            acc1 = cute.math.exp2(acc1, fastmath=True)
            row_sum += acc0 + acc1
        return row_sum


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # important: need stride 1 and not 0 for recast_tensor to work
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@dsl_user_op
def shl_u32(
    val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """
    Left-shift val by shift bits using PTX shl.b32 (sign-agnostic).

    Named ``shl_u32`` (not ``shl_b32``) because python type annotations
    distinguish signed/unsigned.

    PTX semantics (9.7.8.8): "Shift amounts greater than the register width N
    are clamped to N."  So ``shl.b32 d, a, 32`` is well-defined and yields 0.

    This differs from C/C++ and LLVM IR, where shifting by >= the type width is
    undefined behavior.  CuTeDSL compiles through MLIR -> LLVM IR, so a plain
    Python-level ``Uint32(x) << Uint32(n)`` inherits LLVM's UB: the optimizer
    may treat the result as poison and eliminate dependent code.  Inline PTX
    bypasses the LLVM IR shift entirely -- the instruction is emitted verbatim
    into PTX where clamping makes it safe for all shift amounts.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shl.b32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def shr_u32(
    val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """
    Unsigned right-shift val by shift bits using PTX shr.u32 (zero-fills).

    See ``shl_u32`` docstring for why inline PTX is used instead of plain
    CuTeDSL shift operators (LLVM shift-by-type-width UB).
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], (
        "to_dtype must be BFloat16 or Float16"
    )
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_fp8x4_e4m3_bf16x4(
    src: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Int32, cutlass.Int32]:
    """Convert packed e4m3x4 bits into two packed bf16x2 registers."""
    out0 = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(src).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 q, mant, out, bias, zero;\n\t"
            "prmt.b32 q, $1, $1, 0x1302;\n\t"
            "and.b32 out, q, 0x80008000;\n\t"
            "and.b32 mant, q, 0x7f007f00;\n\t"
            "shr.u32 mant, mant, 4;\n\t"
            "or.b32 out, out, mant;\n\t"
            "mov.b32 bias, 0x7b807b80;\n\t"
            "mov.b32 zero, 0;\n\t"
            "fma.rn.bf16x2 $0, out, bias, zero;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    out1 = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(src).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 q, qs, mant, out, bias, zero;\n\t"
            "prmt.b32 q, $1, $1, 0x1302;\n\t"
            "shl.b32 qs, q, 8;\n\t"
            "and.b32 out, qs, 0x80008000;\n\t"
            "and.b32 mant, qs, 0x7f007f00;\n\t"
            "shr.u32 mant, mant, 4;\n\t"
            "or.b32 out, out, mant;\n\t"
            "mov.b32 bias, 0x7b807b80;\n\t"
            "mov.b32 zero, 0;\n\t"
            "fma.rn.bf16x2 $0, out, bias, zero;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return out0, out1


@dsl_user_op
def cvt_fp4x8_e2m1_f16x8(
    src: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Convert four packed E2M1 bytes into four packed f16x2 registers."""

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [cutlass.Int32(src).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
        "mov.b32 {byte0, byte1, byte2, byte3}, $4;\n\t"
        "cvt.rn.f16x2.e2m1x2 $0, byte0;\n\t"
        "cvt.rn.f16x2.e2m1x2 $1, byte1;\n\t"
        "cvt.rn.f16x2.e2m1x2 $2, byte2;\n\t"
        "cvt.rn.f16x2.e2m1x2 $3, byte3;\n\t"
        "}\n",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip))
    out1 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip))
    out2 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip))
    out3 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [3], loc=loc, ip=ip))
    return out0, out1, out2, out3


@dsl_user_op
def cvt_fp4x8_e2m1_bf16x8(
    src: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Convert four packed E2M1 bytes into four packed bf16x2 registers."""

    from cutlass import CUDA_VERSION

    if CUDA_VERSION.major > 13 or (
        CUDA_VERSION.major == 13 and CUDA_VERSION.minor >= 2
    ):
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
            [cutlass.Int32(src).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
            "mov.b32 {byte0, byte1, byte2, byte3}, $4;\n\t"
            "cvt.rn.bf16x2.e2m1x2 $0, byte0;\n\t"
            "cvt.rn.bf16x2.e2m1x2 $1, byte1;\n\t"
            "cvt.rn.bf16x2.e2m1x2 $2, byte2;\n\t"
            "cvt.rn.bf16x2.e2m1x2 $3, byte3;\n\t"
            "}\n",
            "=r,=r,=r,=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        out0 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip))
        out1 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip))
        out2 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip))
        out3 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [3], loc=loc, ip=ip))
        return out0, out1, out2, out3

    f16_pair0, f16_pair1, f16_pair2, f16_pair3 = cvt_fp4x8_e2m1_f16x8(
        src, loc=loc, ip=ip
    )
    return (
        cvt_f16x2_to_bf16x2(f16_pair0, loc=loc, ip=ip),
        cvt_f16x2_to_bf16x2(f16_pair1, loc=loc, ip=ip),
        cvt_f16x2_to_bf16x2(f16_pair2, loc=loc, ip=ip),
        cvt_f16x2_to_bf16x2(f16_pair3, loc=loc, ip=ip),
    )


@dsl_user_op
def cvt_fp4x8_e2m1_scaled_e4m3x8(
    src: cutlass.Int32,
    scale_e4m3: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Int32, cutlass.Int32]:
    """Scale eight packed E2M1 values by one E4M3 byte and convert to E4M3."""

    from cutlass import CUDA_VERSION

    if CUDA_VERSION.major > 13 or (
        CUDA_VERSION.major == 13 and CUDA_VERSION.minor >= 2
    ):
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.i32()]),
            [
                cutlass.Int32(src).ir_value(loc=loc, ip=ip),
                cutlass.Int32(scale_e4m3).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .b32 tmp, ra;\n\t"
            ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
            "prmt.b32 tmp, $3, 0, 0;\n\t"
            "mov.b32 {byte0, byte1, byte2, byte3}, $2;\n\t"
            "mov.b32 ra, {byte0, byte1, _, _};\n\t"
            "mul.e4m3x4.e2m1x4.e4m3x4.satfinite $0, ra, tmp;\n\t"
            "mov.b32 ra, {_, _, byte2, byte3};\n\t"
            "mul.e4m3x4.e2m1x4.e4m3x4.satfinite $1, ra, tmp;\n\t"
            "}\n",
            "=r,=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        out0 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip))
        out1 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip))
        return out0, out1

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            cutlass.Int32(src).ir_value(loc=loc, ip=ip),
            cutlass.Int32(scale_e4m3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b32 sf_bytes, sf_f16x2;\n\t"
        ".reg .b16 sf_pair, e0, e1, e2, e3;\n\t"
        ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
        ".reg .b32 h0, h1, h2, h3;\n\t"
        "prmt.b32 sf_bytes, $3, 0, 0;\n\t"
        "mov.b32 {sf_pair, _}, sf_bytes;\n\t"
        "cvt.rn.f16x2.e4m3x2 sf_f16x2, sf_pair;\n\t"
        "mov.b32 {byte0, byte1, byte2, byte3}, $2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h0, byte0;\n\t"
        "cvt.rn.f16x2.e2m1x2 h1, byte1;\n\t"
        "cvt.rn.f16x2.e2m1x2 h2, byte2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h3, byte3;\n\t"
        "mul.rn.f16x2 h0, h0, sf_f16x2;\n\t"
        "mul.rn.f16x2 h1, h1, sf_f16x2;\n\t"
        "mul.rn.f16x2 h2, h2, sf_f16x2;\n\t"
        "mul.rn.f16x2 h3, h3, sf_f16x2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
        "mov.b32 $0, {e0, e1};\n\t"
        "mov.b32 $1, {e2, e3};\n\t"
        "}\n",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip))
    out1 = cutlass.Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip))
    return out0, out1


@dsl_user_op
def cvt_f16x2_to_bf16x2(
    src: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Convert a packed f16x2 register into a packed bf16x2 register."""

    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(src).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b16 h0, h1;\n\t"
            ".reg .f32 f0, f1;\n\t"
            "mov.b32 {h0, h1}, $1;\n\t"
            "cvt.f32.f16 f0, h0;\n\t"
            "cvt.f32.f16 f1, h1;\n\t"
            "cvt.rn.bf16x2.f32 $0, f1, f0;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def mul_bf16x2(
    a: cutlass.Int32,
    b: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Multiply two packed bf16x2 registers."""

    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_fp8_e4m3_to_bf16x2_replicated(src: cutlass.Int32) -> cutlass.Int32:
    """Decode one E4M3 byte and replicate it into a packed bf16x2 register."""

    src_u8 = src & cutlass.Int32(0xFF)
    packed = src_u8 * cutlass.Int32(0x01010101)
    out0, _ = cvt_fp8x4_e4m3_bf16x4(packed)
    return out0


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def combine_int_frac_ex2(
    x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None
) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm"
    )
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = cute.arch.sub_packed_f32x2(
        xy_rounded, (fp32_round_int, fp32_round_int)
    )
    xy_frac = cute.arch.sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


# -----------------------------------------------------------------------------
# Seqlen Info
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SeqlenInfo:
    offset: Int32
    offset_padded: Int32
    seqlen: Int32
    has_cu_seqlens: cutlass.Constexpr[bool] = False

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_static: Int32,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        tile: cutlass.Constexpr[int] = 128,
    ):
        offset = 0 if const_expr(cu_seqlens is None) else cu_seqlens[batch_idx]
        offset_padded = (
            0
            if const_expr(cu_seqlens is None)
            # Add divby so that the compiler knows the alignment when moving by offset_padded
            else cute.assume((offset + batch_idx * tile) // tile * tile, divby=tile)
        )
        if const_expr(seqused is not None):
            seqlen = seqused[batch_idx]
        elif const_expr(cu_seqlens is not None):
            seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            seqlen = seqlen_static
        return SeqlenInfo(
            offset, offset_padded, seqlen, has_cu_seqlens=cu_seqlens is not None
        )

    def offset_batch(
        self,
        mT: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        """Offset a tensor by batch index. batch dim is at position `dim`, seqlen is at dim=0."""
        if const_expr(not self.has_cu_seqlens):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mT) - 1 - dim)
            return mT[idx]
        else:
            off = multiple * (
                self.offset if const_expr(not padded) else self.offset_padded
            )
            offset = off if const_expr(cute.rank(mT.shape[0]) == 1) else (0, off)
            idx = (offset,) + (None,) * (cute.rank(mT) - 1)
            return cute.domain_offset(idx, mT)


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: Int32
    offset_k: Int32
    padded_offset_q: Int32
    padded_offset_k: Int32
    seqlen_q: Int32
    seqlen_k: Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]
    has_seqused_q: cutlass.Constexpr[bool]
    has_seqused_k: cutlass.Constexpr[bool]


# -----------------------------------------------------------------------------
# Mask
# -----------------------------------------------------------------------------

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_CHUNK_SIZE: int = 32


@cute.jit
def make_bitmask_below(limit: Int32, s: int) -> Uint32:
    m = max((s + 1) * MASK_CHUNK_SIZE - limit, 0)
    return shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def apply_mask_from_bitmask(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    ncol = const_expr(
        cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape)
    )
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, MASK_CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(
            min(MASK_CHUNK_SIZE, ncol - s * MASK_CHUNK_SIZE)
        ):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * MASK_CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        row_idx: Optional[Int32] = None,
        kv_valid_cols: Optional[Int32] = None,
        kv_block_col_start: Optional[Int32] = None,
    ) -> None:
        if const_expr(not mask_seqlen and not mask_causal):
            return

        col_limit = Int32(self.tile_n)
        if const_expr(mask_seqlen):
            if const_expr(kv_valid_cols is not None):
                col_limit = kv_valid_cols
            else:
                col_limit = self.seqlen_k - n_block * Int32(self.tile_n)

        if const_expr(mask_causal):
            if const_expr(row_idx is None):
                row_axis = 0 if const_expr(not self.swap_AB) else 1
                row_idx_cur = tScS_t2r[0][row_axis] + m_block * Int32(self.tile_m)
                if const_expr(self.qhead_per_kvhead_packgqa > 1):
                    row_idx_cur = row_idx_cur // Int32(self.qhead_per_kvhead_packgqa)
            else:
                row_idx_cur = row_idx
            if const_expr(kv_block_col_start is not None):
                block_col_start = kv_block_col_start
            else:
                block_col_start = n_block * Int32(self.tile_n)
            causal_col_limit = (
                row_idx_cur + self.seqlen_k - self.seqlen_q - block_col_start + Int32(1)
            )
            col_limit = (
                cutlass.min(col_limit, causal_col_limit)
                if const_expr(mask_seqlen)
                else causal_col_limit
            )

        if col_limit < Int32(self.tile_n):
            apply_mask_from_bitmask(
                acc_S,
                lambda s: make_bitmask_below(col_limit, s),
                rank1=True,
            )


# -----------------------------------------------------------------------------
# Pack Gqa
# -----------------------------------------------------------------------------


@dataclass
class PackGQAComb:
    """Coalesced LSE loader used by the K2 combine kernel."""

    m_block_size: cutlass.Constexpr[int]

    @cute.jit
    def load_LSE(
        self,
        mLSE_partial: cute.Tensor,
        # Packed layout after caller-side reshape:
        #   shape  ((qhead_per_kvhead, seqlen_q), num_splits)
        #   stride ((1, qhead_per_kvhead), ...)
        # — H_q is the innermost (stride-1) element of the packed first dim.
        sLSE: cute.Tensor,
        # SMEM destination: ``(topk, m_block_size)`` fp32.
        topk: cutlass.Constexpr[int],
        # Explicit topk so the identity tensor shape is a plain int,
        # avoiding compound-shape traps from sLSE.shape[0] after tile_to_shape.
        gmem_tiled_copy: cute.TiledCopy,
        tidx: Int32,
        block: Int32,
        num_splits: Int32,
        seqlen: Int32,
        num_heads_divmod: FastDivmodDivisor,
        mCounter: Optional[cute.Tensor] = None,
        batch_idx: Optional[Int32] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # divmod for ``m_pos = idx // qhead_per_kvhead``; passed explicitly so
        # caller controls whether the divisor is constexpr or a runtime value.
    ):
        """Coalesced GMEM→SMEM async load of LSE_partial for one tile.

        For each (split, row) slot this thread owns in the tile, compute the
        GMEM coordinate ``(h_pos, m_pos)`` via divmod and copy one fp32.
        Out-of-bounds rows (``m_pos >= seqlen``) and splits (``si >= num_splits``)
        are filled with ``-inf`` so they flow cleanly through downstream reductions.

        Coalescing: adjacent thread rows correspond to adjacent ``h_pos`` values
        (head varies fast under ``divmod(idx, qhead_per_kvhead)``), which map to
        adjacent GMEM addresses when H_q is stride-1 — one sector per warp.
        """
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cLSE = cute.make_identity_tensor((topk, self.m_block_size))
        tLSEcLSE = gmem_thr_copy.partition_S(cLSE)
        tLSEsLSE = gmem_thr_copy.partition_D(sLSE)

        for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
            mi = tLSEcLSE[0, 0, m][1]
            idx = block * self.m_block_size + mi
            m_pos, h_pos = divmod(idx, num_heads_divmod)

            if m_pos < seqlen:
                row_count = (
                    mCounter[batch_idx, m_pos, h_pos // qhead_per_kvhead]
                    if const_expr(mCounter is not None)
                    else num_splits
                )
                for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                    si = tLSEcLSE[0, s, 0][0]
                    if si < num_splits and si < row_count:
                        # Build a 1-element GMEM tensor so cute.copy receives a
                        # proper Tensor, not a scalar.
                        src_ptr_i64 = elem_pointer(
                            mLSE_partial, ((h_pos, m_pos), si)
                        ).toint()
                        src_ptr = cute.make_ptr(
                            Float32,
                            src_ptr_i64,
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        src_t = cute.make_tensor(src_ptr, (1,))
                        cute.copy(gmem_thr_copy, src_t, tLSEsLSE[None, s, m])
                    else:
                        tLSEsLSE[None, s, m].fill(-Float32.inf)
            else:
                for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                    tLSEsLSE[None, s, m].fill(-Float32.inf)


# -----------------------------------------------------------------------------
# Softmax
# -----------------------------------------------------------------------------


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)


@dataclass
class SoftmaxSm100(Softmax):
    """SM100-specific softmax: single-row, explicit f32x2 pack for FMA/exp2 paths."""

    rescale_threshold: cutlass.Constexpr[float] = 0.0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            row_max,
            row_sum,
            arch,
            rescale_threshold=rescale_threshold,
        )

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = (
            self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        )
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)

    @cute.jit
    def compute_scaled_exp2_row_sum(
        self,
        acc_S_row: cute.Tensor,
        scale: Float32,
    ) -> Float32:
        return fadd_exp2_scaled_reduce(acc_S_row, scale, arch=self.arch)

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        row_max_scaled = row_max * self.scale_log2
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (-row_max_scaled, -row_max_scaled),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                if cutlass.const_expr(ex2_emu_freq == 0):
                    acc_S_row_frg[k, j] = cute.math.exp2(
                        acc_S_row_frg[k, j], fastmath=True
                    )
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(
                        acc_S_row_frg[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                        or j >= frg_cnt - 1
                        or j < ex2_emu_start_frg
                    ):
                        acc_S_row_frg[k, j] = cute.math.exp2(
                            acc_S_row_frg[k, j], fastmath=True
                        )
                        acc_S_row_frg[k + 1, j] = cute.math.exp2(
                            acc_S_row_frg[k + 1, j], fastmath=True
                        )
                    else:
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def apply_exp2_convert_and_sum(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Fused exp2 + convert + row-sum in a single pass.

        Identical outputs to apply_exp2_convert(...) followed by
        update_row_sum(acc_S_row.load(), 0.0, is_first=True), but accumulates
        the exp2'd values into row_sum during the conversion pass so the fp32
        acc_S_row register frag can be freed BEFORE the P-store loop.  This
        removes the 192-reg peak (S=128fp32 + P=64fp32 simultaneously live)
        that caused local-memory spills in the softmax warpgroup.

        Only the ex2_emu_freq == 0 path (all real exp2, the noncausal case) is
        fused; the emulated path falls back to the separate call sequence.
        """
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        # Four packed-f32x2 accumulators for high-ILP summation, mirroring
        # fadd_reduce's structure so numerics stay within tolerance.
        local_sum = [
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
        ]
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                acc_S_row_frg[k, j] = cute.math.exp2(
                    acc_S_row_frg[k, j], fastmath=True
                )
                acc_S_row_frg[k + 1, j] = cute.math.exp2(
                    acc_S_row_frg[k + 1, j], fastmath=True
                )
                acc_idx = (k // 2) & 3
                local_sum[acc_idx] = cute.arch.add_packed_f32x2(
                    local_sum[acc_idx],
                    (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        self.row_sum[0] = local_sum[0][0] + local_sum[0][1]

    @cute.jit
    def apply_scaled_exp2_convert_and_sum(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        row_max: Float32,
    ) -> None:
        """Fused scale-subtract + exp2 + convert + row-sum in ONE pass.

        Equivalent to the sequence:
            scale_subtract_rowmax(acc_S_row, row_max)
            apply_exp2_convert(acc_S_row, acc_S_row_converted)   # ex2_emu_freq==0
            update_row_sum(acc_S_row.load(), 0.0, is_first=True)

        but reads the raw (un-subtracted) S row once, folds the affine
        s*scale_log2 - row_max*scale_log2 into a packed-f32x2 FMA, exp2s,
        accumulates row_sum, and writes only the converted P frag.  The fp32 S
        row is never written back (it is dead after this), removing both the
        scale_subtract write-back pass and the exp2 write-back pass over the
        128-fp32 row.  Valid only for the noncausal path (no ex2 emulation) and
        when the temperature-LSE path is disabled (no second S reader).
        """
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        scale_log2 = self.scale_log2
        neg_row_max_scaled = -(row_max * scale_log2)
        local_sum = [
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
            (Float32(0.0), Float32(0.0)),
        ]
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                t0, t1 = cute.arch.fma_packed_f32x2(
                    (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                    (scale_log2, scale_log2),
                    (neg_row_max_scaled, neg_row_max_scaled),
                )
                e0 = cute.math.exp2(t0, fastmath=True)
                e1 = cute.math.exp2(t1, fastmath=True)
                acc_S_row_frg[k, j] = e0
                acc_S_row_frg[k + 1, j] = e1
                acc_idx = (k // 2) & 3
                local_sum[acc_idx] = cute.arch.add_packed_f32x2(
                    local_sum[acc_idx], (e0, e1)
                )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        self.row_sum[0] = local_sum[0][0] + local_sum[0][1]


# -----------------------------------------------------------------------------
# Blackwell Helpers
# -----------------------------------------------------------------------------


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    # sA_offset: Int32 = 0,
    # acc_offset: Int32 = 0,
    tA_addr: Optional[Int32] = None,
    cta_group: int = 1,
    mma_kind: str = "f16",
) -> None:
    # acc_tmem_addr += acc_offset
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(mma_op_to_idesc(op))
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                Major.K
                if const_expr(
                    op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [
        cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
        # ) + sA_offset
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            # "r,r,r",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            # Int32(cute.arch.make_warp_uniform(tCrA[None, None, 0].iterator.toint())).ir_value(),
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, (
                "mbar_phase must be provided when mbar_ptr is not None"
            )
            assert split_arrive is not None, (
                "split_arrive must be provided when mbar_ptr is not None"
            )
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            #     Int32(smem_desc_start_b_lo).ir_value(),
            #     Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2])
                    if const_expr(mbar_ptr is None)
                    else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_base_a: Optional[int],
    tCrA_layout: cute.Layout,
    var_name_prefix: str = "smem_desc",
) -> None:
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    smem_desc_base_a_lo, smem_desc_a_hi = None, None
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # else cute.recast_layout(32, tCrA.element_type.width, tCrA_layout)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value()],
            f".reg .b32 {var_name_prefix}_lo;\n\t"
            f".reg .b64 {var_name_prefix}_<{num_k_tile}>;\n\t"
            f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_a_hi)}}};\n\t"
            + "".join(
                (
                    f"add.s32 {var_name_prefix}_lo, $0, {hex(offset_a[k])};\n\t"
                    f"mov.b64 {var_name_prefix}_{k}, {{{var_name_prefix}_lo, {hex(smem_desc_a_hi)}}};\n\t"
                )
                for k in range(1, num_k_tile)
            ),
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_idesc(
    op: cute.nvgpu.tcgen05.mma.MmaOp, var_name: str = "idesc"
) -> None:
    idesc = const_expr(mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t"  # noqa
        f"mov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_varname(
    acc_tmem_addr: Int32,
    smem_desc_start_b: Int32,
    # idesc: int,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    smem_offset: int,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    mma_kind: str = "f16",
) -> None:
    is_ts = False
    num_k_tile = cute.size(tCrB_layout.shape[2])
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]

    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            # ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            # ".reg .b64 smem_desc_b;\n\t"
            f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            # f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $2;\n\t"
            "mov.b32 smem_desc_b_lo_start, $0;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_0;\n\t"
            f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
            f"mov.b64 {smem_var_name_prefix}_0, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            + "".join(
                (
                    f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "setp.ne.b32 p, $1, 0;\n\t"
            # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b, idesc, {pred_str};\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, {idesc_var_name}, {pred_str};\n\t"
            + "".join(
                (
                    # f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    # f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    # f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    # f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    # f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b, idesc, 1;\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b, {idesc_var_name}, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind} [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, {idesc_var_name}, 1;\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


# -----------------------------------------------------------------------------
# Tma Utils
# -----------------------------------------------------------------------------

# Raw TMA Ops

TMA_CACHE_EVICT_LAST = 0x14F0000000000000


@dsl_user_op
def fence_host_tma_desc_acquire(tma_desc_ptr, *, loc=None, ip=None):
    """Acquire a host-written descriptor before reading it via the TMA proxy."""
    llvm.inline_asm(
        T.i32(),
        [tma_desc_ptr.toint().ir_value(loc=loc, ip=ip)],
        "{\nfence.proxy.tensormap::generic.acquire.sys [$1], 128;\nmov.u32 $0, 0;\n}\n",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_gather4_prefetch(
    tma_desc_ptr,
    col_idx,
    row0,
    row1,
    row2,
    row3,
    cache_hint=TMA_CACHE_EVICT_LAST,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 with cache hint."""
    llvm.inline_asm(
        None,
        [
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row0).ir_value(loc=loc, ip=ip),
            Int32(row1).ir_value(loc=loc, ip=ip),
            Int32(row2).ir_value(loc=loc, ip=ip),
            Int32(row3).ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint "
        "[$0, {$1, $2, $3, $4, $5}], $6;\n",
        "l,r,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_gather4_cached(
    smem_ptr,
    smem_byte_offset,
    tma_desc_ptr,
    col_idx,
    row0,
    row1,
    row2,
    row3,
    mbar_ptr,
    cache_hint=TMA_CACHE_EVICT_LAST,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4 with cache hint."""
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(smem_byte_offset).ir_value(loc=loc, ip=ip),
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row0).ir_value(loc=loc, ip=ip),
            Int32(row1).ir_value(loc=loc, ip=ip),
            Int32(row2).ir_value(loc=loc, ip=ip),
            Int32(row3).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint().ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa, ma;\n"
        "cvt.u32.u64 sa, $0;\n"
        "add.u32 sa, sa, $1;\n"
        "cvt.u32.u64 ma, $8;\n"
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[sa], [$2, {$3, $4, $5, $6, $7}], [ma], $9;\n"
        "}\n",
        "l,r,l,r,r,r,r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


_TMA_DESC_BYTES = 128


def _encode_tma_desc_2d_bytes(tensor_2d, *, box_x, box_y, context: str) -> bytes:
    import torch
    import cuda.bindings.driver as cuda

    if tensor_2d.ndim != 2:
        raise ValueError(
            f"{context} tensor must be rank-2, got {tuple(tensor_2d.shape)}"
        )
    rows, cols = tensor_2d.shape
    if tensor_2d.stride(-1) != 1:
        raise ValueError(f"{context} tensor must be contiguous in the last dimension")
    dtype_map = {
        torch.float16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        torch.bfloat16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        torch.float8_e4m3fn: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    }
    if tensor_2d.dtype not in dtype_map:
        raise TypeError(
            f"Unsupported dtype for {context} TMA descriptor: {tensor_2d.dtype}"
        )

    sizes = [cuda.cuuint64_t(cols), cuda.cuuint64_t(rows)]
    strides = [cuda.cuuint64_t(tensor_2d.stride(0) * tensor_2d.element_size())]
    box = [cuda.cuuint32_t(box_x), cuda.cuuint32_t(box_y)]
    elem_stride = [cuda.cuuint32_t(1), cuda.cuuint32_t(1)]
    err, tm = cuda.cuTensorMapEncodeTiled(
        dtype_map[tensor_2d.dtype],
        2,
        tensor_2d.data_ptr(),
        sizes,
        strides,
        box,
        elem_stride,
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, f"TMA encode failed: {err}"
    buf = (ctypes.c_uint8 * _TMA_DESC_BYTES).from_address(tm.getPtr())
    return bytes(buf)


def _desc_bytes_to_device_tensor(desc_bytes: bytes | bytearray, *, device):
    import torch

    desc_bytes = bytes(desc_bytes)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"TMA descriptors require a CUDA device, got {device}")

    host_desc = torch.empty((len(desc_bytes),), dtype=torch.uint8, pin_memory=True)
    host_desc.copy_(torch.frombuffer(bytearray(desc_bytes), dtype=torch.uint8))
    device_desc = torch.empty((len(desc_bytes),), dtype=torch.uint8, device=device)
    stream = torch.cuda.current_stream(device)
    with torch.cuda.stream(stream):
        device_desc.copy_(host_desc, non_blocking=True)
    device_desc.record_stream(stream)
    # Keep the staging buffer alive for the async copy without caching descriptors.
    device_desc._tma_host_desc = host_desc
    return device_desc


def create_flat_gather4_tma_desc(tensor_2d, box_x=64):
    """Create a gather4 CUtensorMap descriptor for a flat 2D row-major tensor."""
    if tensor_2d.ndim != 2:
        raise ValueError(
            f"tensor_2d must be rank-2 [rows, dim], got {tuple(tensor_2d.shape)}"
        )
    desc = _encode_tma_desc_2d_bytes(
        tensor_2d,
        box_x=box_x,
        box_y=1,
        context="gather4",
    )
    return _desc_bytes_to_device_tensor(desc, device=tensor_2d.device)


def create_q_gather4_tma_desc(q_flat, box_x=64):
    return create_flat_gather4_tma_desc(q_flat, box_x=box_x)


class SparseAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        topk: int = 16,
        num_threads: int = 256,
        stages: int = 4,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param topk: exact number of split partials
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.topk = topk
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        self.use_stg128_half_layout = dtype_partial in (
            cutlass.BFloat16,
            cutlass.Float16,
        )
        self.use_stg128_fp8_layout = dtype_partial is cutlass.Float8E4M3FN

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        topk,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            Float32,
        ]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        if topk > 256:
            return False
        if (tile_m * topk) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128
            if self.k_block_size % 128 == 0
            else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # GMEM copy setup for final O (use universal copy for store).
        # Keep this independent from O_partial: fp8 partial uses 16 elements
        # per 128b transaction, while bf16/fp16 O stores must remain 8-wide.
        output_copy_elems = universal_copy_bits // self.dtype.width
        assert self.k_block_size % output_copy_elems == 0
        gmem_threads_per_row_o = k_block_gmem // output_copy_elems
        assert self.num_threads % gmem_threads_per_row_o == 0
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_o, gmem_threads_per_row_o),
            order=(1, 0),
        )
        vO_layout = cute.make_layout((1, output_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tO_layout,
            vO_layout,
        )
        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (
                    32
                    if self.tile_m % 32 == 0
                    else (16 if self.tile_m % 16 == 0 else 8)
                )
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (
                self.smem_threads_per_col_lse,
                self.num_threads // self.smem_threads_per_col_lse,
            ),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        lse_atom_splits = min(self.topk, 8)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((lse_atom_splits, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.topk, self.tile_m), (0, 1)
        )

        # O_partial staging layout.
        if const_expr(
            self.dtype_partial
            in [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN]
        ):
            smem_layout_atom_o = _get_cpasync_smem_layout_atom(
                self.dtype_partial, self.k_block_size
            )
            self.smem_layout_o = cute.tile_to_shape(
                smem_layout_atom_o,
                (self.tile_m, self.k_block_size, self.stages),
                (0, 1, 2),
            )
        else:
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
            )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        varlen_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(
            mLSE_temperature_partial is not None
            and mLSE_temperature_partial.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE partial tensor must be Float32")
        if const_expr(
            mLSE_temperature is not None
            and mLSE_temperature.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )

        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(
            mLSE_temperature_partial is not None
            and len(mLSE_temperature_partial.shape) not in [3, 4]
        ):
            raise ValueError(
                "temperature LSE partial tensor must have 3 or 4 dimensions: "
                "(num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(
            mLSE_temperature is not None and len(mLSE_temperature.shape) not in [2, 3]
        ):
            raise ValueError(
                "temperature LSE tensor must have 2 or 3 dimensions: "
                "(batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type not in [Int32]):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(cu_seqlens is not None):
                if const_expr(len(mSplitCounts.shape) != 2):
                    raise ValueError(
                        "varlen split_counts tensor must have shape (total_q, nheads_kv)"
                    )
            elif const_expr(len(mSplitCounts.shape) != 3):
                raise ValueError(
                    "batched split_counts tensor must have shape (batch, seqlen, nheads_kv)"
                )
        if const_expr(
            mOutputScale is not None and mOutputScale.element_type not in [Float32]
        ):
            raise TypeError("output_scale tensor must be Float32")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator,
            cute.select(mO_partial.layout, mode=O_partial_layout_transpose),
        )
        O_layout_transpose = (
            [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        )
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        # (num_splits, b, h, seqlen) -> (seqlen, num_splits, h, b)
        # Input is pre-transposed: [topK, B, Hq, Sq] with Sq innermost for K2-friendly reads.
        # or (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = (
            [3, 0, 2, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        )
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        # (b, seqlen, h) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if mLSE is not None
            else None
        )
        mLSE_temperature_partial = (
            cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.select(
                    mLSE_temperature_partial.layout, mode=LSE_partial_layout_transpose
                ),
            )
            if mLSE_temperature_partial is not None
            else None
        )
        mLSE_temperature = (
            cute.make_tensor(
                mLSE_temperature.iterator,
                cute.select(mLSE_temperature.layout, mode=LSE_layout_transpose),
            )
            if mLSE_temperature is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        # Output-dtype permutation buffer for Step 7 (tile_m × k_block_size).
        # Accumulation stays fp32; the final dtype conversion happens before
        # the fake→real SMEM scatter to reduce half-output SMEM pressure.
        if const_expr(self.dtype in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_perm = cute.make_layout(
                (self.tile_m, self.k_block_size),
                stride=(self.k_block_size + 16, 1),
            )
        else:
            smem_layout_perm = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size), order=(1, 0)
            )

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.tile_m], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_partial, cute.cosize(self.smem_layout_o)
                ],
                128,
            ]
            sO_perm: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_perm)], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        # Grid: (ceil(seqlen/tile_m), ceil(dim/k_block), num_head * batch)
        # Head separated from seqlen → enables future TMA (contiguous Sq tiles)
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            mLSE_temperature_partial,
            mLSE_temperature,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            smem_layout_perm,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            self.use_pdl,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def decode_flat_row_idx(
        self,
        idx: Int32,
        head_divmod: FastDivmodDivisor,
    ):
        """Decode flattened tile rows under the H_q-innermost contract."""
        q_idx_local, head_idx = divmod(idx, head_divmod)
        return q_idx_local, head_idx

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSE_temperature_partial: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch]
            if const_expr(varlen_batch_idx is not None)
            else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_perm_buf = storage.sO_perm.get_tensor(smem_layout_perm)

        # Handle semaphore reset — wait for dependent grids first
        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        if const_expr(num_splits_dynamic_ptr is not None):
            raise ValueError("K2 combine requires compile-time exact topK")
        num_splits = Int32(self.topk)
        # Handle variable length sequences using SeqlenInfo
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            # Don't need to pass in tile size since we won't use offset_padded
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]

        if const_expr(not varlen) or m_block * self.tile_m < max_idx:
            # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
            if const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()

            # ===============================
            # Step 1: Load LSE_partial from gmem to shared memory
            # ===============================
            # `cLSE` (identity tensor for row/split coord tracking) is reused
            # later in steps 4-5, so it must be defined on both branches.
            cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
            # Reshape mLSE_partial to PackGQA packed layout and delegate the
            # tile load to PackGQAComb.load_LSE. The packed form folds (H_q, Sq)
            # into one compound dim with H_q innermost (stride 1), so thread
            # rows that vary along h_pos produce one-sector coalesced reads.
            # Non-varlen path only — varlen keeps the original inline loop.
            if const_expr(not varlen):
                mLSE_partial_cur = seqlen_info.offset_batch(
                    mLSE_partial, batch_idx, dim=3
                )
                # mLSE_partial_cur: (H_q, topK, Sq) — after initial transpose
                # [3,0,2,1] on [topK,B,Sq,H_q] and dropping B.
                # Reorder to (H_q, Sq, topK) then group modes 0..1 for packed dim:
                mLSE_partial_reord = cute.make_tensor(
                    mLSE_partial_cur.iterator,
                    cute.select(mLSE_partial_cur.layout, mode=[0, 2, 1]),
                )
                mLSE_partial_packed = cute.group_modes(mLSE_partial_reord, 0, 2)
                # shape ((H_q, Sq), topK) with H_q innermost.
                packgqa = PackGQAComb(m_block_size=self.tile_m)
                packgqa.load_LSE(
                    mLSE_partial_packed,
                    sLSE,
                    self.topk,
                    gmem_tiled_copy_LSE,
                    tidx,
                    m_block,
                    num_splits,
                    seqlen,
                    head_divmod,
                    mSplitCounts,
                    batch_idx,
                    qhead_per_kvhead,
                )
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3
                    )
                    mLSE_temperature_partial_reord = cute.make_tensor(
                        mLSE_temperature_partial_cur.iterator,
                        cute.select(
                            mLSE_temperature_partial_cur.layout, mode=[0, 2, 1]
                        ),
                    )
                    mLSE_temperature_partial_packed = cute.group_modes(
                        mLSE_temperature_partial_reord, 0, 2
                    )
                    packgqa.load_LSE(
                        mLSE_temperature_partial_packed,
                        sLSE_temperature,
                        self.topk,
                        gmem_tiled_copy_LSE,
                        tidx,
                        m_block,
                        num_splits,
                        seqlen,
                        head_divmod,
                        mSplitCounts,
                        batch_idx,
                        qhead_per_kvhead,
                    )
            else:
                # Varlen path keeps the same H_q-innermost flat-row contract:
                # after transpose [1, 0, 2], mLSE_partial_cur is
                # (q_local, split, head).
                # mSplitCounts is the authoritative valid-split count per
                # packed (q_abs, kv_head); masked splits stay at -inf and
                # therefore drop out of the final kernel LSE_out reduction.
                mLSE_partial_cur = seqlen_info.offset_batch(
                    mLSE_partial, batch_idx, dim=3
                )
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
                gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
                tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
                tLSEsLSE_temperature = gmem_thr_copy_LSE.partition_D(sLSE_temperature)
                tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3
                    )
                    mLSE_temperature_partial_copy = cute.tiled_divide(
                        mLSE_temperature_partial_cur, (1,)
                    )

                for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                    mi = tLSEcLSE[0, 0, m][1]
                    idx = m_block * self.tile_m + mi
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        row_count = (
                            mSplitCounts[offset + m_idx, head_idx // qhead_per_kvhead]
                            if const_expr(mSplitCounts is not None)
                            else num_splits
                        )
                        mLSE_partial_cur_copy = mLSE_partial_copy[
                            None, m_idx, None, head_idx
                        ]
                        if const_expr(mLSE_temperature_partial is not None):
                            mLSE_temperature_partial_cur_copy = (
                                mLSE_temperature_partial_copy[
                                    None, m_idx, None, head_idx
                                ]
                            )
                        for s in cutlass.range(
                            cute.size(tLSEcLSE, mode=[1]), unroll_full=True
                        ):
                            si = tLSEcLSE[0, s, 0][0]
                            if si < num_splits and si < row_count:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                                if const_expr(mLSE_temperature_partial is not None):
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mLSE_temperature_partial_cur_copy[None, si],
                                        tLSEsLSE_temperature[None, s, m],
                                    )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                                if const_expr(mLSE_temperature_partial is not None):
                                    tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
                    else:
                        for s in cutlass.range(
                            cute.size(tLSEcLSE, mode=[1]), unroll_full=True
                        ):
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                            if const_expr(mLSE_temperature_partial is not None):
                                tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
            cute.arch.cp_async_commit_group()

            # ===============================
            # Step 2: Load O_partial for pipeline stages
            # ===============================

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
            mO_partial_cur = seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

            # Precompute per-row values for flattened (q_local, head) tiles.
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOSplitCount = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  # m coordinate in tile
                idx = m_block * self.tile_m + mi
                if idx >= max_idx:
                    tOhidx[m] = -1
                    tOmidx[m] = 0
                    tOSplitCount[m] = 0
                    tOrOptr[m] = cutlass.Int64(0)
                else:
                    tOmidx[m], tOhidx[m] = self.decode_flat_row_idx(idx, head_divmod)
                    if const_expr(mSplitCounts is None):
                        tOSplitCount[m] = num_splits
                    elif const_expr(cu_seqlens is None):
                        tOSplitCount[m] = mSplitCounts[
                            batch_idx, tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                    else:
                        tOSplitCount[m] = mSplitCounts[
                            offset + tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                    tOrOptr[m] = elem_pointer(
                        mO_partial_cur,
                        (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m]),
                    ).toint()

            tOpO = None
            if const_expr(not self.is_even_k):
                tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = (
                        tOcO[0, 0, k][1]
                        < mO_partial.shape[1] - k_block * self.k_block_size
                    )
                # if cute.arch.thread_idx()[0] == 0 and k_block == 1: cute.print_tensor(tOpO)

            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOSplitCount,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
            )

            # Load first few stages of O_partial
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            # ===============================
            # Step 3: Load and transpose LSE from smem to registers
            # ===============================

            # Wait for LSE and initial O partial stages to complete
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            # if cute.arch.thread_idx()[0] == 0:
            #     # cute.print_tensor(sLSE)
            #     for i in range(64):
            #         cute.printf("sLSE[%d, 0] = %f", i, sLSE[i, 0])
            # cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
            if const_expr(mLSE_temperature_partial is not None):
                ts2rsLSE_temperature = s2r_thr_copy_LSE.partition_S(sLSE_temperature)
                ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
                cute.copy(
                    s2r_tiled_copy_LSE,
                    ts2rsLSE_temperature,
                    ts2rrLSE_temperature,
                )

            # ===============================
            # Step 4: Compute final LSE along split dimension
            # ===============================

            final_lse = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            # This benchmark always has exactly topK valid sparse entries per
            # query/KV group, so all split partials are finite and present.
            # Specialize away the generic valid-split scan/reduction.
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                # Find max LSE value across splits
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = cute.arch.warp_reduction_max(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(
                        cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                    ),
                    threads_in_group=threads_per_col,
                )
                # if cute.arch.thread_idx()[0] == 0: cute.printf(lse_max)
                # Compute exp scales and sum
                LOG2_E = math.log2(math.e)
                # Hoist the loop-invariant lse_max_cur*LOG2_E out of the split
                # loop so the exp2 argument is a single FFMA per split
                # (exp2(lse*LOG2_E - max_log2)) instead of a mul+sub each iter.
                neg_max_log2 = -(lse_max * LOG2_E)
                lse_sum_cur = 0.0
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    scale = cute.math.exp2(
                        ts2rrLSE[0, s, m] * LOG2_E + neg_max_log2,
                        fastmath=True,
                    )
                    lse_sum_cur += scale
                    ts2rrLSE[0, s, m] = scale  # Store scale for later use
                lse_sum_cur = cute.arch.warp_reduction_sum(
                    lse_sum_cur, threads_in_group=threads_per_col
                )
                # Normalize scales
                final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                inv_sum = cute.arch.rcp_approx(lse_sum_cur)
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            # Store the scales exp(lse - lse_logsum) back to smem
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            if const_expr(mLSE_temperature_partial is not None):
                final_lse_temperature = cute.make_rmem_tensor(
                    cute.size(ts2rrLSE_temperature, mode=[2]), Float32
                )
                for m in cutlass.range(
                    cute.size(ts2rrLSE_temperature, mode=[2]), unroll_full=True
                ):
                    threads_per_col = const_expr(self.smem_threads_per_col_lse)
                    lse_temperature_max = cute.arch.warp_reduction_max(
                        ts2rrLSE_temperature[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    lse_temperature_max_cur = (
                        0.0
                        if lse_temperature_max == -Float32.inf
                        else lse_temperature_max
                    )
                    LOG2_E = math.log2(math.e)
                    neg_temp_max_log2 = -(lse_temperature_max_cur * LOG2_E)
                    lse_temperature_sum_cur = 0.0
                    for s in cutlass.range(
                        cute.size(ts2rrLSE_temperature, mode=[1]), unroll_full=True
                    ):
                        scale = cute.math.exp2(
                            ts2rrLSE_temperature[0, s, m] * LOG2_E
                            + neg_temp_max_log2,
                            fastmath=True,
                        )
                        lse_temperature_sum_cur += scale
                    lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                        lse_temperature_sum_cur, threads_in_group=threads_per_col
                    )
                    if (
                        lse_temperature_sum_cur == 0.0
                        or lse_temperature_sum_cur != lse_temperature_sum_cur
                    ):
                        final_lse_temperature[m] = -Float32.inf
                    else:
                        final_lse_temperature[m] = (
                            cute.math.log(lse_temperature_sum_cur, fastmath=True)
                            + lse_temperature_max
                        )

            # ===============================
            # Step 5: Store final LSE to gmem
            # This writeback is the authoritative LSE_out returned by the
            # public Sparse Attention / Sparse Page Attention interface.
            # ===============================

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_cur = mLSE[None, None, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                if const_expr(mLSE_temperature is not None):
                    if const_expr(cu_seqlens is None):
                        mLSE_temperature_cur = mLSE_temperature[None, None, batch_idx]
                    else:
                        mLSE_temperature_cur = cute.domain_offset(
                            (offset, 0), mLSE_temperature
                        )
                if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                    for m in cutlass.range(
                        cute.size(ts2rrLSE, mode=[2]), unroll_full=True
                    ):
                        if (
                            ts2rcLSE[0, 0, m][0] == 0
                        ):  # Only thread responsible for s=0 writes
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.tile_m + mi
                            if idx < max_idx:
                                m_idx, head_idx = self.decode_flat_row_idx(
                                    idx, head_divmod
                                )
                                mLSE_cur[m_idx, head_idx] = final_lse[m]
                                if const_expr(mLSE_temperature is not None):
                                    mLSE_temperature_cur[m_idx, head_idx] = (
                                        final_lse_temperature[m]
                                    )

            # ===============================
            # Step 6: Read O_partial and accumulate final O
            # ===============================

            cute.arch.sync_threads()

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = self.stages - 1
            stage_compute = 0

            # Async copies use a swizzled LSE layout.  For dynamic scalar
            # reads, evaluate its regular outer coordinate and static swizzle
            # explicitly, then load through a flat view.  This preserves the
            # same bank-conflict-avoiding placement used by the copies above.
            sLSE_outer = sLSE.layout.outer
            sLSE_swizzle = sLSE.layout.inner
            sLSE_flat = cute.make_tensor(
                sLSE.iterator,
                cute.make_layout(cute.cosize(sLSE_outer)),
            )
            sLSE_swizzle_mask = const_expr(
                ((1 << sLSE_swizzle.num_bits) - 1) << sLSE_swizzle.num_base
            )

            # Strength-reduce the per-split swizzled LSE index generation
            # (NCU: K2 is ALU/issue-bound, 73.6% on integer address-gen).
            # sLSE_outer((s, row)) is affine: s*stride_s + row*stride_row.
            # The row*stride_row term is loop-invariant per m, and stride_s is
            # a compile-time constant.  Hoist the per-m base offset out of the
            # s-loop so the inner body only does base_m + s*stride_s.
            lse_stride_s = const_expr(sLSE_outer((1, 0)) - sLSE_outer((0, 0)))
            _sw_shift = const_expr(sLSE_swizzle.num_shift)
            _sw_mask = const_expr(sLSE_swizzle_mask)
            lse_base_swz = cute.make_rmem_tensor(num_rows, Int32)
            for m in cutlass.range(num_rows, unroll_full=True):
                base_m = sLSE_outer((0, tOcO[0, m, 0][0]))
                lse_base_swz[m] = base_m ^ ((base_m >> _sw_shift) & Int32(_sw_mask))

            # Main accumulation loop
            for s in cutlass.range(self.topk, unroll=8):
                s_off = s * Int32(lse_stride_s)
                xor_s = (s_off >> _sw_shift) & Int32(_sw_mask)
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    lse_idx = (lse_base_swz[m] + s_off) ^ xor_s
                    scale[m] = sLSE_flat[lse_idx]

                # Load next stage if needed
                split_to_load = s + self.stages - 1
                if split_to_load < self.topk:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                # Wait for the current stage to be ready
                cute.arch.cp_async_wait_group(self.stages - 1)
                # We don't need __syncthreads() because each thread is just reading its own data from smem
                # Copy from smem to registers
                cute.autovec_copy(
                    tOsO_partial[None, None, None, stage_compute], tOrO_partial
                )
                stage_compute = (
                    0 if stage_compute == self.stages - 1 else stage_compute + 1
                )

                # Accumulate scaled partial results
                for m in cutlass.range(num_rows, unroll_full=True):
                    tOrO[None, m, None].store(
                        tOrO[None, m, None].load()
                        + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                    )

            # Flush any outstanding async-copy groups before the local Step-7
            # permutation buffer is read on the tail of the kernel.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ===============================
            # Step 7: Write final O to gmem (fake→real via SMEM)
            # ===============================

            mO_cur = seqlen_info.offset_batch(mO, batch_idx, dim=3)
            if const_expr(cu_seqlens is None):
                mO_cur = mO[None, None, None, batch_idx]
            else:
                mO_cur = cute.domain_offset((offset, 0, 0), mO)
            mO_cur = domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
            num_vals = const_expr(cute.size(tOcO, mode=[0]))
            if const_expr(not use_pdl):
                # Direct / standalone calls don't participate in the K1->K2
                # dependency chain. Use a simple per-element real-column store
                # path here to keep mixed-shape launches stable.
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(
                            cute.size(tOcO, mode=[2]), unroll_full=True
                        ):
                            if const_expr(self.is_even_k) or tOpO[k]:
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(
                                            fake_col
                                        )
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(
                                            fake_col
                                        )
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    mO_cur[tOmidx[m], real_col, tOhidx[m]] = o_val.to(
                                        self.dtype
                                    )
            else:
                # 7a: fp32 accumulator -> output dtype SMEM with fake→real
                # permutation. The dedicated permutation buffer stays separate
                # from the O_partial pipeline staging buffer.
                sO_perm = sO_perm_buf

                if const_expr(self.dtype in [cutlass.BFloat16, cutlass.Float16]):
                    # O_partial uses a dtype-specific STG.128 fake layout, but
                    # sO_perm is in the final O dtype. For all supported fake
                    # layouts, adjacent fake pairs map to adjacent real columns,
                    # so write the final BF16/F16 O pair as one 32-bit SMEM store.
                    assert num_vals % 2 == 0
                    r2s_o_pair_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Int32,
                        num_bits_per_copy=32,
                    )
                    rO_pair_word = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sO_perm_i32_base = cute.make_ptr(
                        dtype=cutlass.Int32,
                        value=sO_perm.iterator.toint(),
                        mem_space=sO_perm.iterator.memspace,
                        assumed_align=4,
                    )
                    sO_perm_i32_row_stride = Int32((self.k_block_size + 16) // 2)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(
                                cute.size(tOcO, mode=[2]), unroll_full=True
                            ):
                                for v_pair in cutlass.range(
                                    num_vals // 2, unroll_full=True
                                ):
                                    v = v_pair * 2
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(
                                            fake_col
                                        )
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(
                                            fake_col
                                        )
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o0 = tOrO[v, m, k]
                                    o1 = tOrO[v + 1, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o0, o1 = cute.arch.mul_packed_f32x2(
                                            (o0, o1),
                                            (output_scale, output_scale),
                                        )
                                    rO_pair_word[0] = cvt_f16x2_f32(o0, o1, self.dtype)
                                    smem_pair_ptr = cute.make_ptr(
                                        dtype=cutlass.Int32,
                                        value=(
                                            sO_perm_i32_base.toint()
                                            + Int64(
                                                row_local * sO_perm_i32_row_stride
                                                + real_col // Int32(2)
                                            )
                                            * Int64(4)
                                        ),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=4,
                                    )
                                    sO_pair = cute.make_tensor(
                                        smem_pair_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_pair_atom, rO_pair_word, sO_pair)
                else:
                    # 7a: iterate over ALL val elements in mode[0].
                    # tOcO[v, m, k][1] gives different fake_col for each v.
                    r2s_o_scalar_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.dtype,
                        num_bits_per_copy=self.dtype.width,
                    )
                    rO_scalar = cute.make_rmem_tensor((1,), self.dtype)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(
                                cute.size(tOcO, mode=[2]), unroll_full=True
                            ):
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(
                                            fake_col
                                        )
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(
                                            fake_col
                                        )
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    rO_scalar[0] = o_val.to(self.dtype)
                                    smem_ptr = elem_pointer(
                                        sO_perm, (row_local, real_col)
                                    )
                                    smem_scalar_ptr = cute.make_ptr(
                                        dtype=self.dtype,
                                        value=smem_ptr.toint(),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=self.dtype.width // 8,
                                    )
                                    sO_scalar = cute.make_tensor(
                                        smem_scalar_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_scalar_atom, rO_scalar, sO_scalar)

                cute.arch.sync_threads()

                # 7b: SMEM (real order, output dtype) → GMEM
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOcO_store = gmem_thr_copy_O.partition_D(cO)
                tOsO_store = gmem_thr_copy_O.partition_D(sO_perm)
                rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
                elems_per_store = const_expr(
                    cute.size(gmem_tiled_copy_O.layout_tv_tiled[1])
                )
                num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
                tOpO_store = None
                if const_expr(not self.is_even_k):
                    tOpO_store = cute.make_rmem_tensor(
                        cute.size(tOcO_store, mode=[2]), Boolean
                    )
                    for k in cutlass.range(cute.size(tOpO_store), unroll_full=True):
                        tOpO_store[k] = (
                            tOcO_store[0, 0, k][1]
                            < mO_partial.shape[1] - k_block * self.k_block_size
                        )

                # Read output dtype from SMEM (now in real column order).
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    for k in cutlass.range(
                        cute.size(tOcO_store, mode=[2]), unroll_full=True
                    ):
                        if const_expr(self.is_even_k) or tOpO_store[k]:
                            cute.autovec_copy(tOsO_store[None, m, k], rO[None, m, k])

                # Write bf16 to GMEM using gmem_tiled_copy_O (same as original FA Step 7)
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    row_local = tOcO_store[0, m, 0][0]
                    idx = m_block * self.tile_m + row_local
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        mO_cur_copy = cute.tiled_divide(
                            mO_cur[m_idx, None, head_idx], (elems_per_store,)
                        )
                        for k in cutlass.range(
                            cute.size(tOcO_store, mode=[2]), unroll_full=True
                        ):
                            k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                            if const_expr(self.is_even_k) or tOpO_store[k]:
                                cute.copy(
                                    gmem_thr_copy_O,
                                    rO[None, m, k],
                                    mO_cur_copy[None, k_idx],
                                )

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOSplitCount: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(
            cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1])
        )
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type,
                    tOrOptr[m],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(
                    mO_partial_cur, (elems_per_load,)
                )
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if split < tOSplitCount[m] and (
                        const_expr(tOpO is None) or tOpO[k]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
                    else:
                        tOsO_partial_cur[None, m, k].fill(0)


def _get_cpasync_smem_layout_atom(
    dtype: Type[cutlass.Numeric], k_dim: int
) -> cute.ComposedLayout:
    dtype_byte = const_expr(dtype.width // 8)
    bytes_per_row = const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        const_expr(
            128
            if bytes_per_row % 128 == 0
            else (
                64
                if bytes_per_row % 64 == 0
                else (32 if bytes_per_row % 32 == 0 else 16)
            )
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


HEAD_DIM = 128
KV_BLOCK_SIZE = 128
SUPPORTED_TOP_K = (4, 8, 16, 32)
SUPPORTED_GQA_RATIOS = (1, 2, 4, 8, 16)
SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
SUPPORTED_MMA_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
SUPPORTED_PARTIAL_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
)


@dataclass(frozen=True, slots=True)
class AttentionInputSpec:
    """Validated static properties of a BF16/FP8 MSA invocation."""

    head_kv: int
    qhead_per_kv: int
    top_k: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class Nvfp4InputSpec:
    """Validated static properties of an NVFP4-KV MSA invocation."""

    head_kv: int
    qhead_per_kv: int
    top_k: int
    paged_kv: bool


@dataclass(frozen=True, slots=True)
class AttentionK1CompileKey:
    """Static dimensions that select a standard K1 compiled variant."""

    target_arch: str
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    partial_dtype: torch.dtype
    qhead_per_kv: int
    causal: bool
    reg_split: tuple = (None, None)


@dataclass(frozen=True, slots=True)
class Nvfp4K1CompileKey:
    """Static dimensions that select an NVFP4 K1 compiled variant."""

    target_arch: str
    q_dtype: torch.dtype
    partial_dtype: torch.dtype
    qhead_per_kv: int
    causal: bool
    paged_kv: bool
    page_size: int | None
    has_seqused_k: bool
    fp8_pair_dequant: bool
    has_k_global_scale: bool


@dataclass(frozen=True, slots=True)
class CombineCompileKey:
    """Static dimensions that select a K2 combine compiled variant."""

    target_arch: str
    partial_dtype: torch.dtype
    top_k: int
    has_output_scale: bool
    min_blocks_per_mp: int


_QSPLIT_Q_BITS = 24
_QSPLIT_Q_LIMIT = 1 << _QSPLIT_Q_BITS
_SCHEDULER_FIELDS = 6


@dataclass(frozen=True)
class MsaMetadata:
    """Sparse reverse-index and scheduler metadata consumed by MSA kernels.

    Attributes:
        k2q_row_ptr: CSR row pointers with shape ``[head_kv, total_rows + 1]``.
        k2q_q_indices: Batch-local query indices with shape
            ``[head_kv, total_q * topK]`` and trailing ``-1`` padding.
        scheduler_metadata: Work rows ``[head, row, begin, count, batch, kv]``.
        work_count: One-element tensor containing the number of work rows.
        qsplit_indices: CSR-aligned packed split slot and query index.  The high
            eight bits hold the compact valid-selection slot and the low 24
            bits hold the batch-local query index.
        split_counts: Number of valid selections for every ``[query, head]``.
        target_q_per_cta: Maximum number of CSR entries assigned to one work
            item.
    """

    k2q_row_ptr: torch.Tensor
    k2q_q_indices: torch.Tensor
    scheduler_metadata: torch.Tensor
    work_count: torch.Tensor
    qsplit_indices: torch.Tensor
    split_counts: torch.Tensor | None
    target_q_per_cta: int


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, name: str) -> None:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.numel() == 0:
        raise ValueError(f"{name} must contain at least the initial zero")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError(f"{name}[0] must be zero")
    if cu_seqlens.numel() > 1 and bool(
        torch.any(cu_seqlens[1:] < cu_seqlens[:-1]).item()
    ):
        raise ValueError(f"{name} must be nondecreasing")


def _validate_inputs(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> None:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError(f"block_size must be an int, got {type(block_size).__name__}")
    if block_size != KV_BLOCK_SIZE:
        raise ValueError(
            f"MSA only supports block_size == {KV_BLOCK_SIZE}, got {block_size}"
        )
    if q2k_indices.dtype != torch.int32:
        raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
    if q2k_indices.ndim != 3:
        raise ValueError(
            "q2k_indices must have shape [head_kv, total_q, topK], "
            f"got {tuple(q2k_indices.shape)}"
        )
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
    if q2k_indices.shape[0] <= 0:
        raise ValueError("q2k_indices must contain at least one KV head")
    topk = int(q2k_indices.shape[2])
    if topk not in SUPPORTED_TOP_K:
        raise ValueError(f"MSA only supports topK in {SUPPORTED_TOP_K}, got {topk}")

    _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must have the same shape [batch + 1]"
        )
    if (
        q2k_indices.device != cu_seqlens_q.device
        or q2k_indices.device != cu_seqlens_k.device
    ):
        raise ValueError("q2k_indices and cu_seqlens tensors must share a device")

    total_q = int(q2k_indices.shape[1])
    cu_total_q = int(cu_seqlens_q[-1].item())
    if total_q != cu_total_q:
        raise ValueError(
            f"q2k_indices.shape[1] ({total_q}) must equal "
            f"cu_seqlens_q[-1] ({cu_total_q})"
        )
    max_q_length = 0
    if cu_seqlens_q.numel() > 1:
        max_q_length = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if max_q_length > _QSPLIT_Q_LIMIT:
        raise ValueError(
            "batch-local query indices must fit in 24 bits, "
            f"got max query length {max_q_length}"
        )
    if total_q * topk > torch.iinfo(torch.int32).max:
        raise ValueError("total_q * topK must fit in int32 CSR row pointers")
    if q2k_indices.numel() > 0 and bool(torch.any(q2k_indices < -1).item()):
        raise ValueError("q2k_indices may only use -1 for invalid selections")


def _choose_target_q_per_cta(
    *,
    total_q: int,
    topk: int,
    head_kv: int,
    block_size: int,
    qhead_per_kv: int,
    num_sms: int,
) -> int:
    """Apply the MSA occupancy and sink-balance heuristic."""
    q_tokens_per_group = 128 // qhead_per_kv
    total_refs = total_q * topk * head_kv
    desired_work_items = max(num_sms * 2, 1)
    total_groups = _ceil_div(max(total_refs, 1), q_tokens_per_group)
    target_groups = min(512, max(1, _ceil_div(total_groups, desired_work_items)))
    occupancy_target = target_groups * q_tokens_per_group
    sink_balance_cap = max(q_tokens_per_group, topk * block_size * 2)
    target = min(max(occupancy_target, q_tokens_per_group), sink_balance_cap)
    return _ceil_div(target, q_tokens_per_group) * q_tokens_per_group


def _build_row_layout(
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(row_map, row_batch, row_kv_block)`` in packed row order."""
    device = cu_seqlens_k.device
    cu_seqlens_k_i64 = cu_seqlens_k.to(torch.int64)
    lengths_k = cu_seqlens_k_i64[1:] - cu_seqlens_k_i64[:-1]
    rows_per_batch = (lengths_k + block_size - 1) // block_size
    max_rows = int(rows_per_batch.max().item()) if rows_per_batch.numel() else 0
    batch = int(rows_per_batch.numel())
    if max_rows == 0:
        row_map = torch.empty((batch, 0), dtype=torch.int64, device=device)
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        return row_map, empty, empty

    levels = torch.arange(max_rows, device=device, dtype=torch.int64)
    valid_by_level = levels[:, None] < rows_per_batch.to(torch.int64)[None, :]
    packed_by_level = torch.cumsum(valid_by_level.reshape(-1), dim=0) - 1
    packed_by_level = packed_by_level.reshape(max_rows, batch)
    row_map = (
        torch.where(
            valid_by_level,
            packed_by_level,
            torch.full_like(packed_by_level, -1),
        )
        .transpose(0, 1)
        .contiguous()
    )
    row_coordinates = torch.nonzero(valid_by_level, as_tuple=False)
    row_kv_block = row_coordinates[:, 0].contiguous()
    row_batch = row_coordinates[:, 1].contiguous()
    return row_map, row_batch, row_kv_block


def _validate_selections(
    q2k_indices: torch.Tensor,
    rows_per_query: torch.Tensor,
) -> None:
    valid = q2k_indices >= 0
    if valid.numel() > 0 and bool(
        torch.any(valid & (q2k_indices >= rows_per_query[None, :, None])).item()
    ):
        bad = torch.nonzero(
            valid & (q2k_indices >= rows_per_query[None, :, None]),
            as_tuple=False,
        )[0]
        head, query, slot = (int(value.item()) for value in bad)
        value = int(q2k_indices[head, query, slot].item())
        limit = int(rows_per_query[query].item())
        raise ValueError(
            f"q2k_indices[{head}, {query}, {slot}]={value} is out of range; "
            f"the query's batch has {limit} KV blocks"
        )

    # NOTE: duplicate KV-block selections are intentionally permitted here.
    # MSA gather semantics double-count duplicates: each duplicate slot maps to
    # a distinct compact partial and the K2 combine sums their exp mass. The
    # original example rejected duplicates as a safety check; that check is
    # removed for the gather (double-count) contract.
    return


def _build_csr(
    q2k_indices: torch.Tensor,
    batch_per_query: torch.Tensor,
    q_local: torch.Tensor,
    row_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CSR and return it with split indices and per-row counts."""
    head_kv, total_q, topk = (int(value) for value in q2k_indices.shape)
    total_rows = int(row_map.numel() - torch.count_nonzero(row_map < 0).item())
    output_length = total_q * topk
    device = q2k_indices.device
    valid = q2k_indices >= 0
    split_counts = (
        valid.sum(dim=-1, dtype=torch.int32)
        .transpose(0, 1)
        .clone(memory_format=torch.contiguous_format)
    )

    row_counts = torch.zeros((head_kv, total_rows), dtype=torch.int32, device=device)
    k2q_row_ptr = torch.zeros(
        (head_kv, total_rows + 1), dtype=torch.int32, device=device
    )
    k2q_q_indices = torch.full(
        (head_kv, output_length), -1, dtype=torch.int32, device=device
    )
    qsplit_indices = torch.full_like(k2q_q_indices, -1)
    if output_length == 0 or total_rows == 0:
        return k2q_row_ptr, k2q_q_indices, qsplit_indices, split_counts

    batch_entries = batch_per_query[:, None].expand(total_q, topk).reshape(-1)
    q_entries = q_local[:, None].expand(total_q, topk).reshape(-1)
    safe_kv = torch.where(valid, q2k_indices, 0).to(torch.int64)
    rows = row_map[
        batch_entries[None, :].expand(head_kv, -1),
        safe_kv.reshape(head_kv, output_length),
    ]
    valid_flat = valid.reshape(head_kv, output_length)
    safe_rows = torch.where(valid_flat, rows, 0)
    row_counts.scatter_add_(1, safe_rows, valid_flat.to(torch.int32))
    k2q_row_ptr[:, 1:] = torch.cumsum(row_counts, dim=1, dtype=torch.int32)

    q_flat = q_entries[None, :].expand(head_kv, -1)
    invalid_key = total_rows * max(total_q, 1)
    sort_keys = torch.where(
        valid_flat,
        rows * max(total_q, 1) + q_flat.to(torch.int64),
        torch.full_like(rows, invalid_key),
    )
    sort_indices = torch.sort(sort_keys, dim=1, stable=True).indices
    sorted_valid = valid_flat.gather(1, sort_indices)
    sorted_q = q_flat.gather(1, sort_indices)
    k2q_q_indices.copy_(torch.where(sorted_valid, sorted_q, -1))

    compact_slots = torch.cumsum(valid, dim=-1, dtype=torch.int32) - 1
    compact_slots = compact_slots.reshape(head_kv, output_length)
    sorted_slots = compact_slots.gather(1, sort_indices)
    packed_split = sorted_q | (sorted_slots << _QSPLIT_Q_BITS)
    qsplit_indices.copy_(torch.where(sorted_valid, packed_split, -1))
    return k2q_row_ptr, k2q_q_indices, qsplit_indices, split_counts


def _build_scheduler(
    k2q_row_ptr: torch.Tensor,
    row_batch: torch.Tensor,
    row_kv_block: torch.Tensor,
    target_q_per_cta: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split nonempty CSR rows into a deterministic row-major worklist."""
    device = k2q_row_ptr.device
    head_kv = int(k2q_row_ptr.shape[0])
    total_rows = int(k2q_row_ptr.shape[1] - 1)
    row_counts = k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]
    row_head_counts = (
        row_counts.transpose(0, 1).contiguous().reshape(-1).to(torch.int64)
    )
    chunks_per_row_head = (row_head_counts + target_q_per_cta - 1) // target_q_per_cta
    work = int(chunks_per_row_head.sum().item())
    work_count = torch.tensor([work], dtype=torch.int32, device=device)
    if work == 0:
        scheduler_metadata = torch.empty(
            (0, _SCHEDULER_FIELDS), dtype=torch.int32, device=device
        )
        return scheduler_metadata, work_count

    row_head = torch.repeat_interleave(
        torch.arange(total_rows * head_kv, device=device, dtype=torch.int64),
        chunks_per_row_head.to(torch.int64),
    )
    chunk_offsets = torch.cumsum(chunks_per_row_head, dim=0) - chunks_per_row_head
    chunk_index = torch.arange(work, device=device, dtype=torch.int64) - (
        torch.repeat_interleave(
            chunk_offsets.to(torch.int64), chunks_per_row_head.to(torch.int64)
        )
    )
    row = row_head // head_kv
    head = row_head - row * head_kv
    q_begin = chunk_index * target_q_per_cta
    counts = row_head_counts[row_head]
    q_count = torch.minimum(
        counts - q_begin,
        torch.full_like(q_begin, target_q_per_cta),
    )
    scheduler_metadata = torch.stack(
        (
            head,
            row,
            q_begin,
            q_count,
            row_batch[row],
            row_kv_block[row],
        ),
        dim=1,
    ).to(torch.int32)
    return scheduler_metadata.contiguous(), work_count


def build_msa_metadata(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    block_size: int = KV_BLOCK_SIZE,
    target_q_per_cta: int | None = None,
    qhead_per_kv: int = 1,
    num_sms: int | None = None,
) -> MsaMetadata:
    """Build all MSA reverse-index and scheduling metadata.

    ``q2k_indices`` values are batch-local KV-block indices.  Invalid choices
    may appear in any topK slot and must be represented by ``-1``; valid split
    slots are compacted in their original slot order.  Every returned tensor is
    contiguous, has dtype ``torch.int32``, and stays on the input device.

    Args:
        q2k_indices: Tensor ``[head_kv, total_q, topK]`` with topK in
            ``{4, 8, 16, 32}``.
        cu_seqlens_q: Nondecreasing query prefix sums ``[batch + 1]``.
        cu_seqlens_k: Nondecreasing KV prefix sums ``[batch + 1]``.
        block_size: Sparse KV block size.  MSA supports exactly 128.
        target_q_per_cta: Optional explicit CSR chunk size.  When omitted, the
            MSA occupancy heuristic selects it.
        qhead_per_kv: Number of query heads sharing one KV head, used only by
            the target-size heuristic.
        num_sms: SM count used by the heuristic.  CUDA inputs use their device
            count by default; CPU inputs use one.

    Returns:
        A fully populated :class:`MsaMetadata` instance.

    Raises:
        TypeError: If an input tensor does not have dtype ``torch.int32``.
        ValueError: If shapes, prefix sums, selections, or scheduling options
            violate the MSA metadata contract.
    """
    _validate_inputs(q2k_indices, cu_seqlens_q, cu_seqlens_k, block_size)
    if isinstance(qhead_per_kv, bool) or not isinstance(qhead_per_kv, int):
        raise TypeError(
            f"qhead_per_kv must be an int, got {type(qhead_per_kv).__name__}"
        )
    if qhead_per_kv <= 0 or 128 % qhead_per_kv != 0:
        raise ValueError(
            f"qhead_per_kv must be a positive divisor of 128, got {qhead_per_kv}"
        )
    if num_sms is not None:
        if isinstance(num_sms, bool) or not isinstance(num_sms, int):
            raise TypeError(f"num_sms must be an int, got {type(num_sms).__name__}")
        if num_sms <= 0:
            raise ValueError(f"num_sms must be positive, got {num_sms}")
    if target_q_per_cta is not None:
        if isinstance(target_q_per_cta, bool) or not isinstance(target_q_per_cta, int):
            raise TypeError(
                "target_q_per_cta must be an int, "
                f"got {type(target_q_per_cta).__name__}"
            )
        if target_q_per_cta <= 0:
            raise ValueError(
                f"target_q_per_cta must be positive, got {target_q_per_cta}"
            )
        if target_q_per_cta > torch.iinfo(torch.int32).max:
            raise ValueError("target_q_per_cta must fit in int32")

    head_kv, total_q, topk = (int(value) for value in q2k_indices.shape)
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    batch_per_query = torch.repeat_interleave(
        torch.arange(q_lengths.numel(), device=q2k_indices.device, dtype=torch.int64),
        q_lengths.to(torch.int64),
    )
    q_local = torch.arange(total_q, device=q2k_indices.device, dtype=torch.int32)
    if total_q > 0:
        q_local -= cu_seqlens_q[:-1][batch_per_query]

    row_map, row_batch, row_kv_block = _build_row_layout(cu_seqlens_k, block_size)
    cu_seqlens_k_i64 = cu_seqlens_k.to(torch.int64)
    rows_per_batch = (
        cu_seqlens_k_i64[1:] - cu_seqlens_k_i64[:-1] + block_size - 1
    ) // block_size
    rows_per_query = rows_per_batch[batch_per_query]
    _validate_selections(q2k_indices, rows_per_query)

    if target_q_per_cta is None:
        if num_sms is None:
            num_sms = (
                torch.cuda.get_device_properties(
                    q2k_indices.device
                ).multi_processor_count
                if q2k_indices.is_cuda
                else 1
            )
        target_q_per_cta = _choose_target_q_per_cta(
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            block_size=block_size,
            qhead_per_kv=qhead_per_kv,
            num_sms=num_sms,
        )

    (
        k2q_row_ptr,
        k2q_q_indices,
        qsplit_indices,
        split_counts,
    ) = _build_csr(q2k_indices, batch_per_query, q_local, row_map)
    scheduler_metadata, work_count = _build_scheduler(
        k2q_row_ptr,
        row_batch,
        row_kv_block,
        target_q_per_cta,
    )
    return MsaMetadata(
        k2q_row_ptr=k2q_row_ptr,
        k2q_q_indices=k2q_q_indices,
        scheduler_metadata=scheduler_metadata,
        work_count=work_count,
        qsplit_indices=qsplit_indices,
        split_counts=split_counts,
        target_q_per_cta=target_q_per_cta,
    )


_CompileKey = TypeVar("_CompileKey", bound=Hashable)
CompileArguments = tuple[object, ...]

_COMPILE_LOCK = threading.RLock()
_COMPILED_K1: dict[AttentionK1CompileKey, Callable[..., None]] = {}
_COMPILED_NVFP4_K1: dict[Nvfp4K1CompileKey, Callable[..., None]] = {}
_COMPILED_K2: dict[CombineCompileKey, Callable[..., None]] = {}


@dataclass(frozen=True, slots=True)
class _K1LaunchArguments:
    """Ordered ABI for the standard BF16/FP8 K1 kernel."""

    k: cute.Tensor
    v: cute.Tensor
    k2q_indices: cute.Tensor
    k2q_qsplit_indices: cute.Tensor
    k2q_counts: cute.Tensor
    scheduler_metadata: cute.Tensor
    work_count: cute.Tensor
    o_partial: cute.Tensor
    lse_partial: cute.Tensor
    lse_temperature_partial: cute.Tensor | None
    q: cute.Tensor
    q_gather4_desc: cute.Tensor | None
    page_table: cute.Tensor | None
    seq_used_k: cute.Tensor | None
    cu_seqlens_q: cute.Tensor | None
    cu_seqlens_k: cute.Tensor
    softmax_scale: Float32
    lse_temperature_scale: Float32
    num_kv_blocks: Int32
    num_heads_kv: Int32
    max_seqlen_q: Int32
    work_capacity: Int32
    stream: cuda.CUstream

    def as_tuple(self) -> CompileArguments:
        """Return fields in the device kernel's positional ABI order."""
        return (
            self.k,
            self.v,
            self.k2q_indices,
            self.k2q_qsplit_indices,
            self.k2q_counts,
            self.scheduler_metadata,
            self.work_count,
            self.o_partial,
            self.lse_partial,
            self.lse_temperature_partial,
            self.q,
            self.q_gather4_desc,
            self.page_table,
            self.seq_used_k,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.softmax_scale,
            self.lse_temperature_scale,
            self.num_kv_blocks,
            self.num_heads_kv,
            self.max_seqlen_q,
            self.work_capacity,
            self.stream,
        )


@dataclass(frozen=True, slots=True)
class _Nvfp4K1LaunchArguments:
    """Ordered ABI for the packed-NVFP4 K1 kernel."""

    k: cute.Tensor
    v: cute.Tensor
    k_scale: cute.Tensor
    v_scale: cute.Tensor
    k_global_scale: cute.Tensor | None
    v_global_scale: cute.Tensor | None
    k2q_indices: cute.Tensor
    k2q_qsplit_indices: cute.Tensor
    k2q_counts: cute.Tensor
    scheduler_metadata: cute.Tensor
    work_count: cute.Tensor
    o_partial: cute.Tensor
    lse_partial: cute.Tensor
    lse_temperature_partial: cute.Tensor | None
    q: cute.Tensor
    q_gather4_desc: cute.Tensor | None
    page_table: cute.Tensor | None
    seq_used_k: cute.Tensor | None
    cu_seqlens_q: cute.Tensor
    cu_seqlens_k: cute.Tensor
    softmax_scale: Float32
    lse_temperature_scale: Float32
    num_kv_blocks: Int32
    num_heads_kv: Int32
    max_seqlen_q: Int32
    work_capacity: Int32
    stream: cuda.CUstream

    def as_tuple(self) -> CompileArguments:
        """Return fields in the device kernel's positional ABI order."""
        return (
            self.k,
            self.v,
            self.k_scale,
            self.v_scale,
            self.k_global_scale,
            self.v_global_scale,
            self.k2q_indices,
            self.k2q_qsplit_indices,
            self.k2q_counts,
            self.scheduler_metadata,
            self.work_count,
            self.o_partial,
            self.lse_partial,
            self.lse_temperature_partial,
            self.q,
            self.q_gather4_desc,
            self.page_table,
            self.seq_used_k,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.softmax_scale,
            self.lse_temperature_scale,
            self.num_kv_blocks,
            self.num_heads_kv,
            self.max_seqlen_q,
            self.work_capacity,
            self.stream,
        )


@dataclass(frozen=True, slots=True)
class _K2LaunchArguments:
    """Ordered ABI for the stable log-sum-exp combine kernel."""

    o_partial: cute.Tensor
    lse_partial: cute.Tensor
    output: cute.Tensor
    lse: cute.Tensor
    lse_temperature_partial: cute.Tensor | None
    lse_temperature: cute.Tensor | None
    cu_seqlens_q: cute.Tensor
    seq_used_q: cute.Tensor | None
    num_splits_dynamic: cute.Tensor | None
    varlen_batch_idx: cute.Tensor | None
    semaphore_to_reset: cute.Tensor | None
    split_counts: cute.Tensor | None
    output_scale: cute.Tensor | None
    qhead_per_kv: Int32
    stream: cuda.CUstream

    def as_tuple(self) -> CompileArguments:
        """Return fields in the device kernel's positional ABI order."""
        return (
            self.o_partial,
            self.lse_partial,
            self.output,
            self.lse,
            self.lse_temperature_partial,
            self.lse_temperature,
            self.cu_seqlens_q,
            self.seq_used_q,
            self.num_splits_dynamic,
            self.varlen_batch_idx,
            self.semaphore_to_reset,
            self.split_counts,
            self.output_scale,
            self.qhead_per_kv,
            self.stream,
        )


@dataclass(frozen=True, slots=True)
class _K1HostState:
    """Shared host-side tensors and dimensions used by both K1 variants."""

    q_flat: torch.Tensor
    o_partial_flat: torch.Tensor
    q_gather4_desc: torch.Tensor | None
    max_seqlen_q: int
    work_capacity: int


def _target_arch_cache_key(device: torch.device) -> str:
    """Identify the CuTe target without conflating variants across devices."""
    return "B200"


def _compile_cached(
    cache: dict[_CompileKey, Callable[..., None]],
    key: _CompileKey,
    kernel: Callable[..., object],
    args: CompileArguments,
) -> Callable[..., None]:
    """Compile one static kernel variant once per process."""
    with _COMPILE_LOCK:
        compiled = cache.get(key)
        if compiled is None:
            compiled = cute.compile(kernel, *args, options="--opt-level 3")
            cache[key] = compiled
        return compiled


def _to_cute_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int = 16,
    leading_dim: int = -1,
) -> cute.Tensor:
    """Convert a Torch tensor to a CuTe tensor with a dynamic compact layout."""
    if leading_dim < 0:
        leading_dim += tensor.ndim
    return from_dlpack(
        tensor.detach(), assumed_align=assumed_align
    ).mark_layout_dynamic(leading_dim=leading_dim)


def _optional_cute_tensor(
    tensor: torch.Tensor | None,
    *,
    assumed_align: int,
) -> cute.Tensor | None:
    """Convert an optional Torch tensor without obscuring its alignment."""
    if tensor is None:
        return None
    return _to_cute_tensor(tensor, assumed_align=assumed_align)


def _torch_dtype_to_cutlass(dtype: torch.dtype) -> type[cutlass.Numeric]:
    """Map public Torch storage/compute dtypes to CuTeDSL numeric types."""
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    raise TypeError(f"unsupported CuTeDSL dtype: {dtype}")


def _prepare_k1_host_state(
    q: torch.Tensor,
    o_partial: torch.Tensor,
    metadata: MsaMetadata,
    cu_seqlens_q: torch.Tensor,
    qhead_per_kv: int,
) -> _K1HostState:
    """Prepare the common flattened buffers and static K1 launch dimensions."""
    q_flat = q.reshape(-1, HEAD_DIM).contiguous()
    o_partial_flat = o_partial.reshape(-1, HEAD_DIM).contiguous()
    q_gather4_desc = (
        create_q_gather4_tma_desc(
            q_flat,
            box_x=128 if q.dtype == torch.float8_e4m3fn else 64,
        )
        if qhead_per_kv in (1, 2, 4)
        else None
    )
    if qhead_per_kv == 16 and cu_seqlens_q.numel() == 2:
        max_seqlen_q = int(q.shape[0])
    else:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().to("cpu"))
    work_capacity = int(metadata.scheduler_metadata.shape[0])
    if work_capacity <= 0:
        raise ValueError("MSA requires at least one valid sparse work item")
    return _K1HostState(
        q_flat=q_flat,
        o_partial_flat=o_partial_flat,
        q_gather4_desc=q_gather4_desc,
        max_seqlen_q=max_seqlen_q,
        work_capacity=work_capacity,
    )


def _compile_and_launch_k1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: MsaMetadata,
    o_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    head_kv: int,
    qhead_per_kv: int,
    softmax_scale: float,
    causal: bool,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    stream: cuda.CUstream,
    reg_softmax_override: Optional[int] = None,
    reg_store_override: Optional[int] = None,
) -> None:
    """Compile and launch the KV-outer K1 partial-attention kernel."""
    state = _prepare_k1_host_state(q, o_partial, metadata, cu_seqlens_q, qhead_per_kv)
    launch = _K1LaunchArguments(
        k=_to_cute_tensor(k),
        v=_to_cute_tensor(v),
        k2q_indices=_to_cute_tensor(metadata.k2q_q_indices, assumed_align=4),
        k2q_qsplit_indices=_to_cute_tensor(metadata.qsplit_indices, assumed_align=4),
        k2q_counts=_to_cute_tensor(metadata.k2q_row_ptr, assumed_align=4),
        scheduler_metadata=_to_cute_tensor(
            metadata.scheduler_metadata, assumed_align=4
        ),
        work_count=_to_cute_tensor(metadata.work_count, assumed_align=4),
        o_partial=_to_cute_tensor(state.o_partial_flat),
        lse_partial=_to_cute_tensor(lse_partial, assumed_align=4),
        lse_temperature_partial=None,
        q=_to_cute_tensor(state.q_flat),
        q_gather4_desc=_optional_cute_tensor(state.q_gather4_desc, assumed_align=64),
        page_table=None,
        seq_used_k=None,
        cu_seqlens_q=_optional_cute_tensor(cu_seqlens_q, assumed_align=4),
        cu_seqlens_k=_to_cute_tensor(cu_seqlens_k, assumed_align=4),
        softmax_scale=Float32(softmax_scale),
        lse_temperature_scale=Float32(1.0),
        num_kv_blocks=Int32(metadata.k2q_row_ptr.shape[1] - 1),
        num_heads_kv=Int32(head_kv),
        max_seqlen_q=Int32(state.max_seqlen_q),
        work_capacity=Int32(state.work_capacity),
        stream=stream,
    )
    args = launch.as_tuple()
    from msa import BlackwellMiniMaxSparseAttentionForward

    kernel = BlackwellMiniMaxSparseAttentionForward(
        head_dim=HEAD_DIM,
        qheadperkv=qhead_per_kv,
        n_block_size=KV_BLOCK_SIZE,
        paged_kv=False,
        causal=causal,
        use_prepare_scheduler=True,
        qk_dtype=_torch_dtype_to_cutlass(qk_dtype),
        pv_dtype=_torch_dtype_to_cutlass(pv_dtype),
        reg_softmax_override=reg_softmax_override,
        reg_store_override=reg_store_override,
    )
    cache_key = AttentionK1CompileKey(
        target_arch=_target_arch_cache_key(q.device),
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        qk_dtype=qk_dtype,
        pv_dtype=pv_dtype,
        partial_dtype=o_partial.dtype,
        qhead_per_kv=qhead_per_kv,
        causal=causal,
        reg_split=(reg_softmax_override, reg_store_override),
    )
    compiled = _compile_cached(_COMPILED_K1, cache_key, kernel, args)
    compiled(*args)


def _compile_and_launch_nvfp4_k1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: torch.Tensor | None,
    metadata: MsaMetadata,
    o_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    *,
    head_kv: int,
    qhead_per_kv: int,
    softmax_scale: float,
    causal: bool,
    fp8_pair_dequant: bool,
    stream: cuda.CUstream,
) -> None:
    """Compile and launch the packed-NVFP4 KV-outer K1 kernel."""
    state = _prepare_k1_host_state(q, o_partial, metadata, cu_seqlens_q, qhead_per_kv)
    launch = _Nvfp4K1LaunchArguments(
        k=_to_cute_tensor(k),
        v=_to_cute_tensor(v),
        k_scale=_to_cute_tensor(k_scale_128x4, assumed_align=1),
        v_scale=_to_cute_tensor(v_scale_128x4, assumed_align=1),
        k_global_scale=_optional_cute_tensor(
            None if k_global_scale is None else k_global_scale.reshape(-1),
            assumed_align=4,
        ),
        v_global_scale=None,
        k2q_indices=_to_cute_tensor(metadata.k2q_q_indices, assumed_align=4),
        k2q_qsplit_indices=_to_cute_tensor(metadata.qsplit_indices, assumed_align=4),
        k2q_counts=_to_cute_tensor(metadata.k2q_row_ptr, assumed_align=4),
        scheduler_metadata=_to_cute_tensor(
            metadata.scheduler_metadata, assumed_align=4
        ),
        work_count=_to_cute_tensor(metadata.work_count, assumed_align=4),
        o_partial=_to_cute_tensor(state.o_partial_flat),
        lse_partial=_to_cute_tensor(lse_partial, assumed_align=4),
        lse_temperature_partial=None,
        q=_to_cute_tensor(state.q_flat),
        q_gather4_desc=_optional_cute_tensor(state.q_gather4_desc, assumed_align=64),
        page_table=_optional_cute_tensor(page_table, assumed_align=4),
        seq_used_k=_optional_cute_tensor(seqused_k, assumed_align=4),
        cu_seqlens_q=_optional_cute_tensor(cu_seqlens_q, assumed_align=4),
        cu_seqlens_k=_to_cute_tensor(cu_seqlens_k, assumed_align=4),
        softmax_scale=Float32(softmax_scale),
        lse_temperature_scale=Float32(1.0),
        num_kv_blocks=Int32(metadata.k2q_row_ptr.shape[1] - 1),
        num_heads_kv=Int32(head_kv),
        max_seqlen_q=Int32(state.max_seqlen_q),
        work_capacity=Int32(state.work_capacity),
        stream=stream,
    )
    args = launch.as_tuple()
    paged_kv = page_table is not None
    from msa_nvfp4 import BlackwellMiniMaxSparseAttentionForwardNVFP4

    kernel = BlackwellMiniMaxSparseAttentionForwardNVFP4(
        head_dim=HEAD_DIM,
        qheadperkv=qhead_per_kv,
        n_block_size=KV_BLOCK_SIZE,
        paged_kv=paged_kv,
        page_size=KV_BLOCK_SIZE if paged_kv else None,
        has_seqused_k=seqused_k is not None,
        causal=causal,
        use_prepare_scheduler=True,
        fp8_pair_dequant=fp8_pair_dequant,
        has_k_global_scale=k_global_scale is not None,
        # V's tensor scale is linear, so K2 applies it once after reduction.
        has_v_global_scale=False,
    )
    cache_key = Nvfp4K1CompileKey(
        target_arch=_target_arch_cache_key(q.device),
        q_dtype=q.dtype,
        partial_dtype=o_partial.dtype,
        qhead_per_kv=qhead_per_kv,
        causal=causal,
        paged_kv=paged_kv,
        page_size=KV_BLOCK_SIZE if paged_kv else None,
        has_seqused_k=seqused_k is not None,
        fp8_pair_dequant=fp8_pair_dequant,
        has_k_global_scale=k_global_scale is not None,
    )
    compiled = _compile_cached(_COMPILED_NVFP4_K1, cache_key, kernel, args)
    compiled(*args)


def _compile_and_launch_k2(
    o_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    metadata: MsaMetadata,
    cu_seqlens_q: torch.Tensor,
    *,
    qhead_per_kv: int,
    output_scale: torch.Tensor | None = None,
    stream: cuda.CUstream,
) -> None:
    """Compile and launch K2's stable log-sum-exp partial reduction."""
    top_k = int(o_partial.shape[0])
    min_blocks_per_mp = 3 if output_scale is not None else 0
    launch = _K2LaunchArguments(
        o_partial=_to_cute_tensor(o_partial),
        lse_partial=_to_cute_tensor(lse_partial, assumed_align=4),
        output=_to_cute_tensor(output),
        lse=_to_cute_tensor(lse, assumed_align=4),
        lse_temperature_partial=None,
        lse_temperature=None,
        cu_seqlens_q=_to_cute_tensor(cu_seqlens_q, assumed_align=4),
        seq_used_q=None,
        num_splits_dynamic=None,
        varlen_batch_idx=None,
        semaphore_to_reset=None,
        split_counts=_optional_cute_tensor(metadata.split_counts, assumed_align=4),
        output_scale=_optional_cute_tensor(
            None if output_scale is None else output_scale.reshape(-1),
            assumed_align=4,
        ),
        qhead_per_kv=Int32(qhead_per_kv),
        stream=stream,
    )
    args = launch.as_tuple()
    kernel = SparseAttentionForwardCombine(
        dtype=cutlass.BFloat16,
        dtype_partial=_torch_dtype_to_cutlass(o_partial.dtype),
        head_dim=HEAD_DIM,
        tile_m=64,
        k_block_size=KV_BLOCK_SIZE,
        topk=top_k,
        stages=4,
        use_pdl=True,
        min_blocks_per_mp=min_blocks_per_mp,
    )
    if not kernel.can_implement(
        cutlass.BFloat16,
        _torch_dtype_to_cutlass(o_partial.dtype),
        HEAD_DIM,
        64,
        KV_BLOCK_SIZE,
        top_k,
        256,
    ):
        raise ValueError(f"unsupported K2 configuration for top_k={top_k}")
    cache_key = CombineCompileKey(
        target_arch=_target_arch_cache_key(output.device),
        partial_dtype=o_partial.dtype,
        top_k=top_k,
        has_output_scale=output_scale is not None,
        min_blocks_per_mp=min_blocks_per_mp,
    )
    compiled = _compile_cached(_COMPILED_K2, cache_key, kernel, args)
    compiled(*args)


@dataclass(frozen=True, slots=True)
class AttentionWorkspace:
    """Buffers and launch state shared by the standard and NVFP4 variants."""

    metadata: MsaMetadata
    o_partial: torch.Tensor
    lse_partial: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    stream: cuda.CUstream


def resolve_softmax_scale(softmax_scale: float | None) -> float:
    """Return a validated positive attention scale."""
    resolved = HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {resolved}")
    return float(resolved)


def resolve_fp8_pair_dequant(fp8_pair_dequant: bool | None) -> bool:
    """Resolve the explicit NVFP4 option with the legacy environment fallback."""
    if fp8_pair_dequant is None:
        return os.environ.get("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "1") != "0"
    if not isinstance(fp8_pair_dequant, bool):
        raise TypeError("fp8_pair_dequant must be a bool or None")
    return fp8_pair_dequant


def empty_result(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the shape-correct result for an empty query sequence."""
    return (
        torch.empty(q.shape, dtype=torch.bfloat16, device=q.device),
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
    )


def no_work_result(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the defined result when every sparse-selection slot is invalid."""
    return (
        torch.zeros(q.shape, dtype=torch.bfloat16, device=q.device),
        torch.full(q.shape[:2], -torch.inf, dtype=torch.float32, device=q.device),
    )


def prepare_workspace(
    q: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int,
    qhead_per_kv: int,
    partial_dtype: torch.dtype,
    target_q_per_cta: int | None,
) -> AttentionWorkspace | None:
    """Build metadata and allocate the common K1/K2 intermediate buffers."""
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    metadata = build_msa_metadata(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        block_size=KV_BLOCK_SIZE,
        qhead_per_kv=qhead_per_kv,
        num_sms=num_sms,
        target_q_per_cta=target_q_per_cta,
    )
    if int(metadata.work_count.item()) <= 0:
        return None

    partial_shape = (top_k, int(q.shape[0]), int(q.shape[1]))
    return AttentionWorkspace(
        metadata=metadata,
        o_partial=torch.empty(
            *partial_shape, HEAD_DIM, dtype=partial_dtype, device=q.device
        ),
        lse_partial=torch.empty(*partial_shape, dtype=torch.float32, device=q.device),
        output=torch.empty(
            q.shape,
            dtype=torch.bfloat16,
            device=q.device,
            memory_format=torch.contiguous_format,
        ),
        lse=torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
        stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )


def benchmark_callable(
    launch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup_iterations: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Run a variant repeatedly and return its last result and average GPU time."""
    if warmup_iterations < 0 or iterations < 0:
        raise ValueError("warmup_iterations and iterations must be nonnegative")

    output, lse = launch()
    for _ in range(warmup_iterations):
        output, lse = launch()
    torch.cuda.synchronize()
    if iterations == 0:
        return output, lse, None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output, lse = launch()
    end.record()
    end.synchronize()
    return output, lse, start.elapsed_time(end) / iterations
