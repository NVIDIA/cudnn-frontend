# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Sched extension for the fused fc1+fc2 dGLU-backward MXFP8 kernel."""

import dataclasses
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32, Int64, extract_mlir_values, new_from_mlir_values

from ..fwd_glu.glu_mxfp8_fc12_extension import GluMxFp8Fc12SchedExtension
from .....schedulers.fc12_mapping import NonSwapAbFc12WorkTileInfo


@dataclasses.dataclass(frozen=True)
class DgluMxFp8Fc12SchedExtension(GluMxFp8Fc12SchedExtension):
    """dGLU adapter for token-major auxiliary data and per-expert blocked SF."""

    expert_token_sizes: Optional[cute.Tensor] = None
    token_padding_block: int = 128
    sf_padding_block: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.expert_token_sizes is None:
            raise ValueError("dGLU auxiliaries require expert_token_sizes.")
        if self.token_padding_block != self.sf_padding_block or self.token_padding_block % 128 != 0:
            raise ValueError("dGLU auxiliaries require equal token/SF padding divisible by 128.")

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self.expert_token_sizes))
        return values

    def __new_from_mlir_values__(self, values: list) -> "DgluMxFp8Fc12SchedExtension":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            value_index += field_value_count
            return result

        fc1_done_counter_pointer = rebuild(self.fc1_done_counter_pointer)
        fc2_spin_threshold = rebuild(self.fc2_spin_threshold)
        fc1_ready_counter_pointer = (
            rebuild(self.fc1_ready_counter_pointer) if self.fc1_ready_counter_pointer is not None else None
        )
        expert_token_sizes = rebuild(self.expert_token_sizes)
        if value_index != len(values):
            raise ValueError(
                f"DgluMxFp8Fc12SchedExtension MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter_pointer,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=fc1_ready_counter_pointer,
            cluster_m=self.cluster_m,
            expert_token_sizes=expert_token_sizes,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
        )

    @cute.jit
    def _physical_token_count(self, work_tile_info: NonSwapAbFc12WorkTileInfo, padding_block: int):
        expert_idx = work_tile_info.expert_idx
        valid_tokens = Int32(self.expert_token_sizes[expert_idx])
        return ((valid_tokens + Int32(padding_block - 1)) // Int32(padding_block)) * Int32(padding_block)

    @cute.jit
    def _aux_data_tensor(
        self,
        tensor: cute.Tensor,
        work_tile_info: NonSwapAbFc12WorkTileInfo,
        feature_extent,
    ) -> cute.Tensor:
        physical_tokens = self._physical_token_count(work_tile_info, self.token_padding_block)
        real = cute.domain_offset(
            (work_tile_info.cumulative_data_physical_row, 0, 0),
            tensor,
        )
        return cute.make_tensor(
            real.iterator,
            cute.make_layout(
                (physical_tokens, feature_extent, Int32(1)),
                stride=real.stride,
            ),
        )

    @cute.jit
    def _aux_sf_tensor(
        self,
        tensor: cute.Tensor,
        work_tile_info: NonSwapAbFc12WorkTileInfo,
        feature_padded,
    ) -> cute.Tensor:
        physical_tokens = self._physical_token_count(work_tile_info, self.sf_padding_block)
        feature_atoms = Int32(feature_padded) // Int32(128)
        token_atoms = physical_tokens // Int32(128)
        element_offset = Int64(feature_padded) * (
            Int64(work_tile_info.cumulative_sf_physical_row) // Int64(self.sf_vec_size)
        )
        return cute.make_tensor(
            tensor.iterator + element_offset,
            cute.make_layout(
                (feature_atoms, token_atoms, Int32(512)),
                stride=(token_atoms * Int32(512), Int32(512), Int32(1)),
            ),
        )

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info: NonSwapAbFc12WorkTileInfo,
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """dGLU-backward operand views; every other name delegates to the base."""
        shape = gmem_tensor_in_moe_view.shape

        if cutlass.const_expr(tensor_name == "recompute"):
            return (self._aux_data_tensor(gmem_tensor_in_moe_view, work_tile_info, shape[1]), None)

        elif cutlass.const_expr(tensor_name == "sfrecompute"):
            return (self._aux_sf_tensor(gmem_tensor_in_moe_view, work_tile_info, shape[0]), None)

        elif cutlass.const_expr(tensor_name == "col_output"):
            return (self._aux_data_tensor(gmem_tensor_in_moe_view, work_tile_info, shape[1]), None)

        elif cutlass.const_expr(tensor_name == "sfcol_output"):
            return (self._aux_sf_tensor(gmem_tensor_in_moe_view, work_tile_info, shape[0]), None)

        return GluMxFp8Fc12SchedExtension.get_gmem_tensor(
            self, tensor_name, gmem_tensor_in_moe_view, work_tile_info
        )
