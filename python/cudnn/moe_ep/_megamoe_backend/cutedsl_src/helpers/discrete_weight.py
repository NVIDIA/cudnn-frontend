# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pointer-array discrete expert weights: descriptor workspace and slot addressing.

In ``contiguous`` mode an expert bank is one packed tensor and TMA addresses an expert
by stride. In ``discrete`` mode the host passes a GMEM array of per-expert base
pointers instead, and each CTA builds its own expert's weight TensorMaps at launch,
parking them in a workspace of four 128-byte descriptor slots per expert and cluster
variant. A uniform launch has one variant; a mixed-CGA launch has two (preferred,
fallback), laid out per expert as ``[preferred roles | fallback roles]``.

Arch-neutral on purpose: a Blackwell kernel must not import from
``sources.kernel_src.rubin.*`` (that package's ``__init__`` eagerly pulls in the
Rubin wgrad kernel and its ``rubin_helpers`` dependency), so this is the single copy
every architecture uses. The four slot names are labels, not semantics -- the
backward kernels reuse them for their dFC2/dFC1 operands.
"""

from typing import Literal

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir.dialects import cute as cute_ir, cute_nvgpu as cute_nvgpu_ir, llvm
from cutlass.cute.typing import AddressSpace, Pointer
from cutlass.cutlass_dsl import Int32, dsl_user_op


WeightStorageMode = Literal["contiguous", "discrete"]
DiscreteWeightRole = Literal["fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"]

_descriptor_byte_count = 128
_discrete_weight_descriptor_roles = (
    "fc1_weight",
    "fc1_weight_sf",
    "fc2_weight",
    "fc2_weight_sf",
)
_discrete_weight_descriptor_slot = {
    tensor_role: slot_index
    for slot_index, tensor_role in enumerate(_discrete_weight_descriptor_roles)
}


def discrete_weight_descriptor_workspace_size(
    expert_count: int, is_mixed_cga: bool = False
) -> int:
    """Return storage bytes for four 128-byte TensorMaps per expert and cluster variant."""
    if (
        not isinstance(expert_count, int)
        or isinstance(expert_count, bool)
        or expert_count <= 0
    ):
        raise ValueError(f"expert_count must be a positive integer, got {expert_count!r}.")
    if not isinstance(is_mixed_cga, bool):
        raise ValueError(f"is_mixed_cga must be a bool, got {is_mixed_cga!r}.")
    return (
        expert_count
        * (1 + int(is_mixed_cga))
        * len(_discrete_weight_descriptor_roles)
        * _descriptor_byte_count
    )


def discrete_weight_descriptor_workspace_alignment() -> int:
    return _descriptor_byte_count


@cute.jit
def raw_discrete_weight_descriptor_pointer(
    descriptor_workspace: Pointer,
    tensor_role: DiscreteWeightRole,
    expert_idx: Int32,
    cluster_variant: Int32 = Int32(0),
    is_mixed_cga: cutlass.Constexpr = False,
) -> Pointer:
    """Return the 128-byte-aligned descriptor slot for one expert / cluster-variant operand."""
    slot_index = _discrete_weight_descriptor_slot[tensor_role]
    byte_offset = (
        (expert_idx * Int32(1 + int(is_mixed_cga)) + cluster_variant)
        * Int32(len(_discrete_weight_descriptor_roles))
        + Int32(slot_index)
    ) * Int32(_descriptor_byte_count)
    return (descriptor_workspace + byte_offset).align(_descriptor_byte_count)


@dsl_user_op
def copy_discrete_weight_descriptor_pointer(
    descriptor_workspace: Pointer,
    tensor_role: DiscreteWeightRole,
    expert_idx: Int32,
    cluster_variant: Int32 = Int32(0),
    is_mixed_cga: cutlass.Constexpr = False,
    *,
    loc=None,
    ip=None,
) -> Pointer:
    """Convert a raw GMEM descriptor slot to the type accepted by TMA copy."""
    raw_pointer = raw_discrete_weight_descriptor_pointer(
        descriptor_workspace, tensor_role, expert_idx, cluster_variant, is_mixed_cga
    )
    generic_llvm_pointer = llvm.addrspacecast(
        llvm.PointerType.get(AddressSpace.generic),
        raw_pointer.llvm_ptr,
        loc=loc,
        ip=ip,
    )
    generic_pointer = cute.make_ptr(
        raw_pointer.dtype,
        generic_llvm_pointer,
        AddressSpace.generic,
        assumed_align=raw_pointer.alignment,
        loc=loc,
        ip=ip,
    )
    descriptor_pointer_type = cute_ir.PtrType.get(
        cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        generic_pointer.memspace,
        generic_pointer.alignment,
    )
    return cute_ir.recast_iter(descriptor_pointer_type, generic_pointer.value)


def int64_pointer_array_tensor(pointer_array: Pointer, element_count: int):
    """View the host-staged array of per-expert base pointers as Int64 GMEM."""
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Int64,
            pointer_array.toint(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        ),
        cute.make_layout((element_count,)),
    )


@cute.jit
def _write_expert_weight_descriptors(
    fc1_weight_ptrs: cute.Pointer,
    fc1_weight_sf_ptrs: cute.Pointer,
    fc2_weight_ptrs: cute.Pointer,
    fc2_weight_sf_ptrs: cute.Pointer,
    fc1_n: cutlass.Int32,
    fc1_k: cutlass.Int32,
    fc1_stride_n: cutlass.Int64,
    fc2_n: cutlass.Int32,
    fc2_k: cutlass.Int32,
    fc2_stride_n: cutlass.Int64,
    descriptor_workspace: cute.Pointer,
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    b_smem_layout,
    sfb_smem_layout,
    cluster_layout_vmnk_shapes: cutlass.Constexpr,
    cluster_layout_sfb_vmnk_shapes: cutlass.Constexpr,
    ab_dtype: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr,
    sf_vec_size: cutlass.Constexpr,
    cluster_shape_mn: cutlass.Constexpr,
    mma_tiler: cutlass.Constexpr,
    mma_tiler_sfb: cutlass.Constexpr,
    static_expert_shape: cutlass.Constexpr,
) -> None:
    """Build this CTA's expert weight TensorMaps, one set of four per cluster variant."""
    expert_idx = cute.arch.block_idx()[0]
    expert_count = static_expert_shape[0]
    is_mixed_cga = len(cluster_layout_vmnk_shapes) == 2

    b_operation = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape_mn, tiled_mma.thr_id)

    # (role, pointer array, dtype, n, k, stride_n); the expert broadcast axis stays static:
    # a dynamic unit mode / zero stride reaches an invalid TMA conversion in CuTe DSL 4.6.
    operands = (
        ("fc1_weight", fc1_weight_ptrs, ab_dtype, fc1_n, fc1_k, fc1_stride_n),
        ("fc1_weight_sf", fc1_weight_sf_ptrs, sf_dtype, fc1_n, fc1_k, None),
        ("fc2_weight", fc2_weight_ptrs, ab_dtype, fc2_n, fc2_k, fc2_stride_n),
        ("fc2_weight_sf", fc2_weight_sf_ptrs, sf_dtype, fc2_n, fc2_k, None),
    )
    for role, pointer_array, dtype, n, k, stride_n in operands:
        pointer = cute.make_ptr(
            dtype,
            int64_pointer_array_tensor(pointer_array, expert_count)[expert_idx],
            cute.AddressSpace.gmem,
        )
        if cutlass.const_expr(stride_n is None):
            tensor = cute.make_tensor(
                pointer, blockscaled_utils.tile_atom_to_shape_SF((n, k, 1), sf_vec_size)
            )
        else:
            tensor = cute.make_tensor(pointer, cute.make_layout((n, k, 1), stride=(stride_n, 1, 0)))
        for cluster_variant in cutlass.range_constexpr(len(cluster_layout_vmnk_shapes)):
            if cutlass.const_expr(stride_n is None):
                atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                    sfb_operation,
                    tensor,
                    sfb_smem_layout,
                    mma_tiler_sfb,
                    tiled_mma_sfb,
                    cluster_layout_sfb_vmnk_shapes[cluster_variant],
                    internal_type=cutlass.Uint64,
                )
            else:
                atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                    b_operation,
                    tensor,
                    b_smem_layout,
                    mma_tiler,
                    tiled_mma,
                    cluster_layout_vmnk_shapes[cluster_variant],
                )
            cpasync.copy_tensormap(
                atom,
                raw_discrete_weight_descriptor_pointer(
                    descriptor_workspace, role, expert_idx, Int32(cluster_variant), is_mixed_cga
                ),
            )


@cute.kernel
def initialize_discrete_weight_descriptors(
    fc1_weight_ptrs: cute.Pointer,
    fc1_weight_sf_ptrs: cute.Pointer,
    fc2_weight_ptrs: cute.Pointer,
    fc2_weight_sf_ptrs: cute.Pointer,
    fc1_n: cutlass.Int32,
    fc1_k: cutlass.Int32,
    fc1_stride_n: cutlass.Int64,
    fc2_n: cutlass.Int32,
    fc2_k: cutlass.Int32,
    fc2_stride_n: cutlass.Int64,
    descriptor_workspace: cute.Pointer,
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    b_smem_layout,
    sfb_smem_layout,
    cluster_layout_vmnk_shape: cutlass.Constexpr,
    cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
    ab_dtype: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr,
    sf_vec_size: cutlass.Constexpr,
    cluster_shape_mn: cutlass.Constexpr,
    mma_tiler: cutlass.Constexpr,
    mma_tiler_sfb: cutlass.Constexpr,
    static_expert_shape: cutlass.Constexpr,
) -> None:
    """Uniform-CGA launch: build this CTA's four expert weight TensorMaps."""
    _write_expert_weight_descriptors(
        fc1_weight_ptrs, fc1_weight_sf_ptrs, fc2_weight_ptrs, fc2_weight_sf_ptrs,
        fc1_n, fc1_k, fc1_stride_n, fc2_n, fc2_k, fc2_stride_n,
        descriptor_workspace, tiled_mma, tiled_mma_sfb, b_smem_layout, sfb_smem_layout,
        (cluster_layout_vmnk_shape,), (cluster_layout_sfb_vmnk_shape,),
        ab_dtype, sf_dtype, sf_vec_size, cluster_shape_mn, mma_tiler, mma_tiler_sfb,
        static_expert_shape,
    )


@cute.kernel
def initialize_mixed_cga_discrete_weight_descriptors(
    fc1_weight_ptrs: cute.Pointer,
    fc1_weight_sf_ptrs: cute.Pointer,
    fc2_weight_ptrs: cute.Pointer,
    fc2_weight_sf_ptrs: cute.Pointer,
    fc1_n: cutlass.Int32,
    fc1_k: cutlass.Int32,
    fc1_stride_n: cutlass.Int64,
    fc2_n: cutlass.Int32,
    fc2_k: cutlass.Int32,
    fc2_stride_n: cutlass.Int64,
    descriptor_workspace: cute.Pointer,
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    b_smem_layout,
    sfb_smem_layout,
    cluster_layout_vmnk_shape: cutlass.Constexpr,
    cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
    fallback_cluster_layout_vmnk_shape: cutlass.Constexpr,
    fallback_cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
    ab_dtype: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr,
    sf_vec_size: cutlass.Constexpr,
    cluster_shape_mn: cutlass.Constexpr,
    mma_tiler: cutlass.Constexpr,
    mma_tiler_sfb: cutlass.Constexpr,
    static_expert_shape: cutlass.Constexpr,
) -> None:
    """Mixed-CGA launch: build the preferred and the fallback-cluster TensorMap sets."""
    _write_expert_weight_descriptors(
        fc1_weight_ptrs, fc1_weight_sf_ptrs, fc2_weight_ptrs, fc2_weight_sf_ptrs,
        fc1_n, fc1_k, fc1_stride_n, fc2_n, fc2_k, fc2_stride_n,
        descriptor_workspace, tiled_mma, tiled_mma_sfb, b_smem_layout, sfb_smem_layout,
        (cluster_layout_vmnk_shape, fallback_cluster_layout_vmnk_shape),
        (cluster_layout_sfb_vmnk_shape, fallback_cluster_layout_sfb_vmnk_shape),
        ab_dtype, sf_dtype, sf_vec_size, cluster_shape_mn, mma_tiler, mma_tiler_sfb,
        static_expert_shape,
    )


__all__ = [
    "DiscreteWeightRole",
    "WeightStorageMode",
    "copy_discrete_weight_descriptor_pointer",
    "discrete_weight_descriptor_workspace_alignment",
    "discrete_weight_descriptor_workspace_size",
    "initialize_discrete_weight_descriptors",
    "initialize_mixed_cga_discrete_weight_descriptors",
    "int64_pointer_array_tensor",
    "raw_discrete_weight_descriptor_pointer",
]
