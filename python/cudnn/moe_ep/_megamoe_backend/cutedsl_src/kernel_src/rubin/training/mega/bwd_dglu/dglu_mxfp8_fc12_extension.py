# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Sched extension for the fused fc1+fc2 dGLU-backward MXFP8 kernel."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer

from ..fwd_glu.glu_mxfp8_fc12_extension import GluMxFp8Fc12SchedExtension
from .....schedulers.fc12_mapping import NonSwapAbFc12WorkTileInfo


def _rewrite_tensor_shape(tensor: cute.Tensor, new_shape: Tuple) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.make_layout(new_shape, stride=tensor.stride))


class DgluMxFp8Fc12SchedExtension(GluMxFp8Fc12SchedExtension):
    """
    Sched extension for the fused fc1+fc2 dGLU-backward MXFP8 kernel.
    """

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info: NonSwapAbFc12WorkTileInfo,
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """dGLU-backward operand views; every other name delegates to the base."""
        data_token_offset = work_tile_info.cumulative_data_physical_row
        sf_token_offset = work_tile_info.cumulative_sf_physical_row

        shape = gmem_tensor_in_moe_view.shape
        c1 = cutlass.Int32(1)
        sf_vec_size = self.sf_vec_size

        if cutlass.const_expr(tensor_name == "recompute"):
            # Forward-swiglu recompute data tensor.
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = _rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfrecompute"):
            # Per-expert base for atom-packed col-SF of the forward-swiglu recompute.
            real = cute.domain_offset(
                (sf_token_offset // sf_vec_size, 0, 0), gmem_tensor_in_moe_view
            )
            real = _rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "col_output"):
            # Col-quant grad_y1 data tensor (alongside row-quant "d").
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = _rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfcol_output"):
            # Per-expert base for atom-packed col-SF of the grad_y1 col output.
            real = cute.domain_offset(
                (sf_token_offset // sf_vec_size, 0, 0), gmem_tensor_in_moe_view
            )
            real = _rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        return GluMxFp8Fc12SchedExtension.get_gmem_tensor(
            self, tensor_name, gmem_tensor_in_moe_view, work_tile_info
        )
