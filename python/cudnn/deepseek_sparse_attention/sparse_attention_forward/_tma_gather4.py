# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level CuTe DSL bridge for the SM100 2-D TMA gather4 instruction.

CUTLASS DSL 4.5+ can build a normal tiled TMA descriptor, but its Python
facade does not expose gather4 yet.  This bridge extracts the compiler-owned
descriptor address from an executable TMA atom and emits the single missing
instruction through MLIR's LLVM inline-assembly operation.  It can be removed
once the public gather4 atom constructor is available.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

TMA_GATHER4_AVAILABLE = hasattr(_cute_nvgpu_ir, "get_tma_desc_addr") and hasattr(llvm, "inline_asm")

_TMA_GATHER4_CTA1 = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
    "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
    "[$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;"
)

_TMA_GATHER4_CTA2 = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
    "mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
    "[$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;"
)

_TMA_GATHER4_CONSTRAINTS = "r,l,r,r,r,r,r,r,l"

# In a paired-CTA SM100 kernel this bit selects the peer CTA's shared-memory
# address.  Clearing it on both CTAs makes every gather transaction complete
# CTA0's full barrier, while the destination address remains CTA-local.
_SM100_MMA_PEER_BIT_MASK = -0x1000001  # int32 spelling of 0xFEFFFFFF

# TMA::CacheHintSm90::EVICT_LAST encoding.
_EVICT_LAST = 0x14F0000000000000


@dsl_user_op
def _get_tma_descriptor_ptr(
    tma_atom: cute.CopyAtom,
    mbar_ptr: cute.Pointer,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Pointer:
    """Return the 64-byte-aligned descriptor pointer owned by ``tma_atom``."""

    executable_atom = tma_atom._unpack(tma_bar_ptr=mbar_ptr, loc=loc, ip=ip)
    descriptor_ptr_type = ir.Type.parse("!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>")
    return _cute_nvgpu_ir.get_tma_desc_addr(descriptor_ptr_type, executable_atom, loc=loc, ip=ip)


@dsl_user_op
def tma_gather4_cta1(
    tma_atom: cute.CopyAtom,
    dst_smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    col: cutlass.Int32,
    row0: cutlass.Int32,
    row1: cutlass.Int32,
    row2: cutlass.Int32,
    row3: cutlass.Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one CTA-local gather4 transaction (four 16-bit rows x 64)."""

    descriptor_ptr = _get_tma_descriptor_ptr(tma_atom, mbar_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            dst_smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            descriptor_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(col).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row3).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int64(_EVICT_LAST).ir_value(loc=loc, ip=ip),
        ],
        _TMA_GATHER4_CTA1,
        _TMA_GATHER4_CONSTRAINTS,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_gather4_cta2_cta0_mbar(
    tma_atom: cute.CopyAtom,
    dst_smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    col: cutlass.Int32,
    row0: cutlass.Int32,
    row1: cutlass.Int32,
    row2: cutlass.Int32,
    row3: cutlass.Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one 2-CTA gather4 transaction and credit CTA0's barrier."""

    descriptor_ptr = _get_tma_descriptor_ptr(tma_atom, mbar_ptr, loc=loc, ip=ip)
    cta0_mbar_addr = mbar_ptr.toint(loc=loc, ip=ip) & cutlass.Int32(_SM100_MMA_PEER_BIT_MASK)
    llvm.inline_asm(
        None,
        [
            dst_smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            descriptor_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(col).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row3).ir_value(loc=loc, ip=ip),
            cta0_mbar_addr.ir_value(loc=loc, ip=ip),
            cutlass.Int64(_EVICT_LAST).ir_value(loc=loc, ip=ip),
        ],
        _TMA_GATHER4_CTA2,
        _TMA_GATHER4_CONSTRAINTS,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
