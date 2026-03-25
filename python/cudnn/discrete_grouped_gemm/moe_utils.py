# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
Online TMA Descriptor Construction Utilities.

Provides utilities for dynamically creating TMA descriptors at kernel runtime
based on runtime-provided information (problem sizes, pointers, etc.).

Key components:
- OnlineTensormapDescCreator: Simplified ABC for TMA descriptor builders (2 abstract methods)
- TensormapWorkspace: Helper for linear workspace layout of TMA descriptors
- DiscreteWeightTensormapConstructor: Per-expert B/SFB descriptors from pointer arrays
- Pointer utility functions (ptr_offset_bytes, gmem_ptr_to_generic, etc.)
- tensormap_ptr_for_copy: Convert raw desc ptr to cute.copy-compatible type
- compute_expert_token_range: Compute per-expert token offset and count from offs
- rewrite_tensor_shape: Debug-friendly tensor shape rewrite utility
"""

from abc import ABC, abstractmethod
from typing import Literal, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace, Pointer
from cutlass.cutlass_dsl import (
    dsl_user_op,
    Int32,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, builtin
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from dataclasses import dataclass

from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

TensormapDescBytes = 128
_NUM_DESC_I64 = TensormapDescBytes // 8


@dsl_user_op
def store_tma_desc(
    tma_atom: cute.CopyAtom,
    dest_ptr: Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Store the TMA descriptor from a non-exec TMA atom directly to a gmem pointer
    via atom_get_value + unrealized_conversion_cast + llvm.store, bypassing the
    copy_tma_desc op.

    :param tma_atom: The non-exec TMA CopyAtom containing the descriptor
    :param dest_ptr: Destination gmem pointer (must have >= TensormapDescBytes space)
    """
    i64_ty = ir.IntegerType.get_signless(64)
    llvm_struct_full_ty = ir.Type.parse("!llvm.struct<(array<16 x i64>)>")
    llvm_array_full_ty = ir.Type.parse("!llvm.array<16 x i64>")
    mlir_desc_ty = _cute_nvgpu_ir.TmaDescriptorTiledType.get()

    atom_value = tma_atom._trait.value

    field_attr_name = _get_tma_field_attr_name(atom_value.type)
    field_attr = ir.Attribute.parse(field_attr_name)
    tma_desc_val = _cute_nvgpu_ir.atom_get_value(mlir_desc_ty, atom_value, field_attr, loc=loc, ip=ip)

    desc_full = builtin.unrealized_conversion_cast([llvm_struct_full_ty], [tma_desc_val], loc=loc, ip=ip)

    arr_full = llvm.extractvalue(llvm_array_full_ty, desc_full, [0], loc=loc, ip=ip)

    if _NUM_DESC_I64 == 16:
        store_struct = desc_full
    else:
        llvm_array_store_ty = ir.Type.parse(f"!llvm.array<{_NUM_DESC_I64} x i64>")
        llvm_struct_store_ty = ir.Type.parse(f"!llvm.struct<(array<{_NUM_DESC_I64} x i64>)>")
        arr_store = llvm.mlir_undef(llvm_array_store_ty, loc=loc, ip=ip)
        for i in range(_NUM_DESC_I64):
            elem = llvm.extractvalue(i64_ty, arr_full, [i], loc=loc, ip=ip)
            arr_store = llvm.insertvalue(arr_store, elem, [i], loc=loc, ip=ip)
        store_struct = llvm.mlir_undef(llvm_struct_store_ty, loc=loc, ip=ip)
        store_struct = llvm.insertvalue(store_struct, arr_store, [0], loc=loc, ip=ip)

    # dst_llvm_ptr = dest_ptr.to_llvm_ptr(loc=loc, ip=ip)
    dst_llvm_ptr = dest_ptr.llvm_ptr
    llvm.store(store_struct, dst_llvm_ptr, alignment=TensormapDescBytes, loc=loc, ip=ip)


def _get_tma_field_attr_name(atom_type: ir.Type) -> str:
    """Return the field attribute string for extracting tma_desc from a non-exec TMA atom."""
    type_str = str(atom_type)
    if "non_exec_tiled_tma_load" in type_str:
        return "#cute_nvgpu.atom_copy_field_non_exec_tma_load<tma_desc>"
    elif "non_exec_tiled_tma_store" in type_str:
        return "#cute_nvgpu.atom_copy_field_non_exec_tma_store<tma_desc>"
    elif "non_exec_tiled_tma_reduce" in type_str:
        return "#cute_nvgpu.atom_copy_field_non_exec_tma_reduce<tma_desc>"
    elif "non_exec_im2col_tma_load" in type_str:
        return "#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_load<tma_desc>"
    elif "non_exec_im2col_tma_store" in type_str:
        return "#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_store<tma_desc>"
    raise ValueError(f"Unsupported TMA atom type: {type_str}")


# =============================================================================
# Pointer Utilities
# =============================================================================


@dsl_user_op
def gmem_ptr_to_generic(gmem_ptr: Pointer, *, loc=None, ip=None) -> Pointer:
    if gmem_ptr.memspace != AddressSpace.gmem:
        raise ValueError(f"gmem_ptr_to_generic requires pointer in gmem address space, " f"got {gmem_ptr.memspace}")
    # Get LLVM pointer and cast to generic address space
    # llvm_ptr = gmem_ptr.to_llvm_ptr(loc=loc, ip=ip)
    llvm_ptr = gmem_ptr.llvm_ptr
    generic_llvm_ptr = llvm.addrspacecast(llvm.PointerType.get(AddressSpace.generic), llvm_ptr, loc=loc, ip=ip)
    # Create a new cute.Pointer with generic address space, preserving alignment
    return cute.make_ptr(
        gmem_ptr.dtype,
        generic_llvm_ptr,
        AddressSpace.generic,
        assumed_align=gmem_ptr.alignment,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def generic_ptr_to_gmem(generic_ptr: Pointer, *, loc=None, ip=None) -> Pointer:
    if generic_ptr.memspace != AddressSpace.generic:
        raise ValueError(f"generic_ptr_to_gmem requires pointer in generic address space, " f"got {generic_ptr.memspace}")
    # Get LLVM pointer and cast to gmem address space
    # llvm_ptr = generic_ptr.to_llvm_ptr(loc=loc, ip=ip)
    llvm_ptr = generic_ptr.llvm_ptr
    gmem_llvm_ptr = llvm.addrspacecast(llvm.PointerType.get(AddressSpace.gmem), llvm_ptr, loc=loc, ip=ip)
    # Create a new cute.Pointer with gmem address space, preserving alignment
    return cute.make_ptr(
        generic_ptr.dtype,
        gmem_llvm_ptr,
        AddressSpace.gmem,
        assumed_align=generic_ptr.alignment,
        loc=loc,
        ip=ip,
    )


def ptr_offset_bytes(ptr: Pointer, byte_offset: int) -> Pointer:
    """Offset a pointer by a given number of bytes."""
    element_offset = byte_offset * 8 // ptr.dtype.width
    return ptr + element_offset


@dsl_user_op
def tensormap_ptr_for_copy(raw_ptr: Pointer, *, loc=None, ip=None) -> Pointer:
    """
    Convert a raw TMA descriptor gmem pointer to the type expected by cute.copy.

    cute.copy requires the tma_desc_ptr to be in generic address space and
    recast to TmaDescriptorTiledType. This utility performs both conversions.

    :param raw_ptr: Raw pointer to TMA descriptor in gmem
    :type raw_ptr: Pointer
    :return: Pointer compatible with cute.copy's tma_desc_ptr parameter
    :rtype: Pointer
    """
    generic_ptr = gmem_ptr_to_generic(raw_ptr, loc=loc, ip=ip)
    tma_desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        generic_ptr.memspace,
        generic_ptr.alignment,
    )
    return _cute_ir.recast_iter(tma_desc_ptr_ty, generic_ptr.value)


# =============================================================================
# MoE Utilities
# =============================================================================


@dsl_user_op
@cute.jit
def compute_expert_token_range(
    offs: cute.Tensor,
    expert_idx: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Int32, Int32]:
    """
    Compute token offset and count for a given expert from the cumsum offs tensor.

    :param offs: Cumulative sum tensor of token counts per expert, shape (experts,)
    :param expert_idx: Index of the expert
    :return: (token_offset, tokens_i) where token_offset is the start position
             and tokens_i is the number of tokens for this expert
    """
    token_offset = Int32(0)
    if expert_idx > Int32(0):
        token_offset = offs[expert_idx - 1]
    tokens_i = offs[expert_idx] - token_offset
    return token_offset, tokens_i


@dsl_user_op
def rewrite_tensor_shape(
    tensor: cute.Tensor,
    new_shape: Tuple,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    """
    Rewrite tensor shape while keeping the same stride and iterator.

    This is primarily for debug friendliness - shows the actual expert's shape
    instead of the fake global shape. No runtime overhead as it becomes
    dead code in non-debug builds.

    :param tensor: Source tensor whose stride and iterator to preserve
    :param new_shape: New shape to apply
    :return: New tensor with the given shape but original stride and iterator
    """
    new_layout = cute.make_layout(new_shape, stride=tensor.stride, loc=loc, ip=ip)
    return cute.make_tensor(tensor.iterator, new_layout, loc=loc, ip=ip)


# =============================================================================
# TMA Descriptor Workspace Helper
# =============================================================================


class TensormapWorkspace:
    """
    Helper for linear workspace layout of TMA descriptors.

    Manages address calculation for a workspace buffer containing TMA descriptors
    organized as: for each executor (e.g., expert or group), a fixed set of
    named descriptor slots.

    Layout: [slot_0_exec_0, slot_1_exec_0, ..., slot_0_exec_1, slot_1_exec_1, ...]

    Example:
        # 2Dx3D MoE: only C is expert-wise
        workspace = TensormapWorkspace(workspace_ptr, ["c"])

        # 2Dx2D MoE: A and B are expert-wise
        workspace = TensormapWorkspace(workspace_ptr, ["a", "b"])

        # General grouped GEMM: all three tensors
        workspace = TensormapWorkspace(workspace_ptr, ["a", "b", "c"])
    """

    def __init__(self, workspace_ptr: Pointer, slot_names: list):
        """
        :param workspace_ptr: Pointer to the beginning of the workspace buffer
        :param slot_names: Ordered list of tensor names, defining the slot layout
                           per executor. e.g., ["a", "b", "c"]
        """
        self.workspace_ptr = workspace_ptr
        self._slot_names = slot_names
        self._name_to_slot = {name: i for i, name in enumerate(slot_names)}
        self._slots_per_executor = len(slot_names)

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.workspace_ptr)

    def __new_from_mlir_values__(self, values):
        new_ptr = new_from_mlir_values(self.workspace_ptr, values)
        return TensormapWorkspace(new_ptr, self._slot_names)

    @cute.jit
    def get_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        """
        Get the workspace pointer for a specific TMA descriptor.

        :param tensor_name: Name of the tensor (must be one of the slot_names)
        :param executor_idx: Index of the executor (e.g., group_idx or expert_idx)
        :return: Aligned pointer to the TMA descriptor in workspace
        """
        slot = self._name_to_slot[tensor_name]  # codegen-time constant lookup
        byte_offset = (executor_idx * self._slots_per_executor + slot) * TensormapDescBytes
        return ptr_offset_bytes(self.workspace_ptr, byte_offset).align(TensormapDescBytes)

    @cute.jit
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        """Alias for get_ptr, matching the OnlineTensormapDescCreator interface."""
        return self.get_ptr(tensor_name, executor_idx)

    @staticmethod
    def size_bytes(num_slots: int, num_executors: int) -> int:
        """
        Calculate workspace size in bytes.

        :param num_slots: Number of descriptor slots per executor
        :param num_executors: Total number of executors (e.g., expert_cnt or group_cnt)
        :return: Total workspace size in bytes
        """
        return num_slots * num_executors * TensormapDescBytes


# =============================================================================
# Online TMA Descriptor Creator (Abstract Base Class)
# =============================================================================


@dataclass(frozen=True)
class OnlineTensormapDescCreator(ABC):
    """
    Abstract base class for building TMA descriptors online (at kernel runtime).

    Subclasses store all needed parameters (both codegen-time configs and runtime
    values) as explicit instance attributes in __init__. No dict-based APIs.

    Subclasses must implement exactly 2 abstract methods:
    - construct_and_write: Build TMA descriptor(s) for one executor and write to workspace
    - get_desc_ptr: Return raw gmem pointer to a specific descriptor in workspace

    To convert the raw pointer for use with cute.copy, callers should use the
    standalone tensormap_ptr_for_copy() utility.
    """

    @abstractmethod
    def construct_and_write(self, executor_idx: Int32, dependency=None) -> None:
        """
        Build TMA descriptor(s) for one executor and write to workspace.

        :param executor_idx: Index of the executor (e.g., group_idx or expert_idx).
            Semantics may vary by subclass when ``dependency`` is provided.
        :param dependency: Optional pipeline consumer for inter-warp-group
            synchronization. When provided, the subclass decides when to wait
            (via ``dependency.wait_and_advance()``) and release. The subclass
            also decides how to interpret ``executor_idx`` in this mode.
        """
        ...

    @abstractmethod
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        """
        Get the raw gmem pointer to a specific TMA descriptor in workspace.

        :param tensor_name: Name identifying which tensor's descriptor
        :param executor_idx: Index of the executor (e.g., group_idx or expert_idx)
        :return: Raw pointer (gmem) to the TMA descriptor
        """
        ...


# =============================================================================
# Discrete Weight TMA Descriptor Constructor
# =============================================================================


class DiscreteWeightTensormapConstructor(OnlineTensormapDescCreator):
    """
    TMA descriptor constructor for discrete (non-contiguous) weight tensors.

    Each expert has an independent B tensor (N, K) and SFB tensor with separate
    base pointers. Pointers are provided as i64 arrays, and N/K are uniform
    across experts (passed as Int32).

    Workspace layout per expert: [B, SFB]  (2 slots)

    :param b_dtype: Data type for tensor B
    :param sf_dtype: Data type for scale factors (SFB)
    :param b_smem_layout: SMEM layout for B TMA
    :param sfb_smem_layout: SMEM layout for SFB TMA
    :param b_tma_op: TMA operation for B (G2S or G2S multicast)
    :param sfb_tma_op: TMA operation for SFB
    :param tiled_mma: TiledMma for B TMA atom construction
    :param tiled_mma_sfb: TiledMma for SFB (separate due to 2CTA replication)
    :param mma_tiler: MMA tiler shape (M, N, K)
    :param mma_tiler_sfb: MMA tiler shape for SFB
    :param cluster_layout_vmnk_shape: Cluster layout shape for B multicast
    :param cluster_layout_sfb_vmnk_shape: Cluster layout shape for SFB multicast
    :param sf_vec_size: Scale factor vector size (32 for MXFP8/MXFP4, 16 for NVFP4)
    :param b_ptrs: (expert_cnt,) i64 tensor of B base pointers
    :param sfb_ptrs: (expert_cnt,) i64 tensor of SFB base pointers
    :param n: N dimension (uniform across experts)
    :param k: K dimension (uniform across experts)
    :param b_stride_n: Stride of B along N dimension (in elements)
    :param b_stride_k: Stride of B along K dimension (in elements)
    :param workspace_ptr: Pointer to workspace for TMA descriptors
    """

    def __init__(
        self,
        # Codegen-time configs
        b_dtype,
        sf_dtype,
        b_smem_layout,
        sfb_smem_layout,
        b_tma_op,
        sfb_tma_op,
        tiled_mma,
        tiled_mma_sfb,
        mma_tiler,
        mma_tiler_sfb,
        cluster_layout_vmnk_shape,
        cluster_layout_sfb_vmnk_shape,
        sf_vec_size: int,
        # Runtime params
        b_ptrs: cute.Tensor,
        sfb_ptrs: cute.Tensor,
        n: Int32,
        k: Int32,
        b_stride_n: Int32,
        b_stride_k: Int32,
        workspace_ptr: Pointer,
    ) -> None:
        super().__init__()
        # Codegen-time configs
        self.b_dtype = b_dtype
        self.sf_dtype = sf_dtype
        self.b_smem_layout = b_smem_layout
        self.sfb_smem_layout = sfb_smem_layout
        self.b_tma_op = b_tma_op
        self.sfb_tma_op = sfb_tma_op
        self.tiled_mma = tiled_mma
        self.tiled_mma_sfb = tiled_mma_sfb
        self.mma_tiler = mma_tiler
        self.mma_tiler_sfb = mma_tiler_sfb
        self.cluster_layout_vmnk_shape = cluster_layout_vmnk_shape
        self.cluster_layout_sfb_vmnk_shape = cluster_layout_sfb_vmnk_shape
        self.sf_vec_size = sf_vec_size
        # Runtime params
        self.b_ptrs = b_ptrs
        self.sfb_ptrs = sfb_ptrs
        self.n = n
        self.k = k
        self.b_stride_n = b_stride_n
        self.b_stride_k = b_stride_k
        self.workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])

    @staticmethod
    def get_workspace_size(expert_cnt: int) -> int:
        """Calculate workspace size in bytes."""
        return TensormapWorkspace.size_bytes(2, expert_cnt)

    @cute.jit
    def get_desc_ptr(self, tensor_name: str, executor_idx: Int32) -> Pointer:
        return self.workspace.get_ptr(tensor_name, executor_idx)

    @cute.jit
    def construct_and_write(self, expert_idx: Int32, dependency=None) -> None:
        """
        Build B and SFB TMA descriptors for one expert from pointer arrays.
        """
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # --- B descriptor ---
        b_ptr_val = self.b_ptrs[expert_idx]
        b_ptr = cute.make_ptr(self.b_dtype, b_ptr_val, AddressSpace.gmem)
        b_layout = cute.make_layout(
            (self.n, self.k, c1),
            stride=(self.b_stride_n, self.b_stride_k, c0),
        )
        b_tensor = cute.make_tensor(b_ptr, b_layout)

        tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
            self.b_tma_op,
            b_tensor,
            self.b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk_shape,
        )
        store_tma_desc(tma_atom_b, self.get_desc_ptr("b", expert_idx))

        # --- SFB descriptor ---
        sfb_ptr_val = self.sfb_ptrs[expert_idx]
        sfb_ptr = cute.make_ptr(self.sf_dtype, sfb_ptr_val, AddressSpace.gmem)
        k_sf = (self.k + self.sf_vec_size - 1) // self.sf_vec_size
        sfb_shape = (self.n, k_sf, c1)
        sfb_layout = tile_atom_to_shape_SF(sfb_shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
            self.sfb_tma_op,
            sfb_tensor,
            self.sfb_smem_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk_shape,
            internal_type=cutlass.Uint64,
        )
        store_tma_desc(tma_atom_sfb, self.get_desc_ptr("sfb", expert_idx))
