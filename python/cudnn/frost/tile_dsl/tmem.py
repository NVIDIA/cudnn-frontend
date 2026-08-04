# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute


@cute.jit
def tmem_alloc(tmem_ptr_i32, num_cols: int, cta_group_kind):
    # No is_exclusive: it only applies to >512-column allocations on sm_107/sm_109
    # (cute.arch.tmem.is_tmem_allocation_exclusive), so it is a no-op on the
    # <=512-column sm100/sm103 allocations here, and it is fenced out of the public
    # cutlass-dsl tcgen05_alloc wrapper.
    nvvm.tcgen05_alloc(
        tmem_ptr_i32,
        cutlass.Int32(num_cols),
        group=cta_group_kind,
    )
    nvvm.tcgen05_relinquish_alloc_permit(group=cta_group_kind)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def tmem_dealloc(tmem_ptr_i32, num_cols: int, cta_group_kind):
    tmem_ptr_for_dealloc = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    nvvm.tcgen05_dealloc(
        tmem_ptr_for_dealloc,
        cutlass.Int32(num_cols),
        group=cta_group_kind,
    )
