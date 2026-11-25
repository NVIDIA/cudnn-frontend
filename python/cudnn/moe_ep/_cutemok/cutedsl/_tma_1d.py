# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Descriptor-free 1D TMA helpers for runtime-selected peer addresses.

The symmetric-memory address of every EP rank's buffer is a runtime value.  A descriptor
cannot be selected with a runtime peer index, so the communication kernels use the
raw-address bulk-copy PTX form for one contiguous chunk.

The copies carry no L2 cache hint.  ``tma_store_1d_raw`` does not emit a generic->async
proxy fence: a caller that wrote the source chunk with ordinary stores must fence
(``cute.arch.fence_proxy("async.shared", space="cta")``) before the store.
"""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op


@dsl_user_op
def tma_load_1d_raw(
    dst_smem,
    src_gmem_addr: Int64,
    mbar_smem,
    num_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one runtime-address GMEM-to-SMEM bulk copy of ``num_bytes`` whose completion arrives on the
    mbarrier ``mbar_smem`` (the caller arms its transaction bytes)."""

    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem_addr.ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
        ],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes " "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_1d_raw(
    dst_gmem_addr: Int64,
    src_smem,
    num_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one runtime-address SMEM-to-GMEM bulk copy of ``num_bytes`` into the caller's bulk group
    (the caller commits and waits for the group)."""

    llvm.inline_asm(
        None,
        [
            dst_gmem_addr.ir_value(),
            src_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group " "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def zero_shared_bulk(
    dst_smem,
    num_bytes: int,
    *,
    loc=None,
    ip=None,
) -> None:
    """Zero ``num_bytes`` (a static multiple of 8) of shared memory with one ``st.bulk`` (PTX 8.6, sm_100+).
    The stores are weak generic-proxy stores: a TMA store that reads the chunk needs
    ``cute.arch.fence_proxy("async.shared", space="cta")`` first, as after ordinary stores."""

    assert num_bytes % 8 == 0
    llvm.inline_asm(
        None,
        [dst_smem.toint(loc=loc, ip=ip).ir_value()],
        f"st.bulk.weak.shared::cta [$0], {num_bytes}, 0;",
        "r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
