# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed-address WGrad operand materialization for the training graph path."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

from ..._types import MoeEpTrainingWgradOperands
from ._launch import _to_cute

if TYPE_CHECKING:
    from ._training_resources import Mxfp8TrainingSlotViews


class Mxfp8TrainingWgradExporter:
    """Own scale-expansion compiles; every export writes existing buffers."""

    def __init__(
        self,
        *,
        experts: int,
        hidden: int,
        intermediate: int,
        sf_padding: int = 128,
    ) -> None:
        self.experts = int(experts)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.sf_padding = int(sf_padding)
        self._compiled: dict[tuple[int, int, int | None], object] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _copy_gate_up_data(
        target: torch.Tensor,
        source: torch.Tensor,
        intermediate: int,
    ) -> None:
        pool_rows = source.shape[0]
        pairs = intermediate // 32
        source_view = source.view(pool_rows, pairs, 2, 32).permute(
            0,
            2,
            1,
            3,
        )
        target_view = target.as_strided(
            (pool_rows, 2, pairs, 32),
            (
                target.stride(0),
                intermediate * target.stride(1),
                32 * target.stride(1),
                target.stride(1),
            ),
        )
        target_view.copy_(source_view)

    def _expand_scales(
        self,
        source: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        output: torch.Tensor,
        *,
        non_k_size: int,
        deinterleave_gate_up: int | None = None,
    ) -> None:
        if source.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError("WGrad source scales must use Uint8 or E8M0")
        if output.dtype is not torch.float8_e8m0fnu:
            raise TypeError("WGrad output scales must use E8M0")
        source_bytes = source.view(torch.uint8).reshape(-1)
        output_bytes = output.view(torch.uint8).reshape(-1)
        key = (int(non_k_size), self.sf_padding, deinterleave_gate_up)
        import cuda.bindings.driver as cuda

        stream = torch.cuda.current_stream(output.device)
        args = (
            _to_cute(source_bytes, dynamic_layout=False),
            _to_cute(counts, assumed_align=4, dynamic_layout=False),
            _to_cute(offsets, assumed_align=4, dynamic_layout=False),
            _to_cute(output_bytes, dynamic_layout=False),
            cuda.CUstream(stream.cuda_stream),
        )
        with self._lock:
            compiled = self._compiled.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "WGrad scale expansion must be compiled before "
                        "CUDA graph capture"
                    )
                import cutlass.cute as cute

                from ._training_wgrad_kernel import (
                    Mxfp8TrainingScaleExpandKernel,
                )

                kernel = Mxfp8TrainingScaleExpandKernel(
                    non_k_size=non_k_size,
                    expert_count=self.experts,
                    source_sf_padding=self.sf_padding,
                    deinterleave_gate_up=deinterleave_gate_up,
                )
                compiled = cute.compile(kernel, *args)
                self._compiled[key] = compiled
        compiled(*args)

    def export(
        self,
        slot: "Mxfp8TrainingSlotViews",
    ) -> MoeEpTrainingWgradOperands:
        """Write and return the fixed-capacity grouped-WGrad operands."""

        if slot.col_quant_data is None or slot.col_quant_sf is None:
            raise RuntimeError("training WGrad export requires forward col-quant")
        pool_rows = slot.fc1_recompute.shape[0]
        if slot.col_quant_data.shape[0] != pool_rows:
            raise RuntimeError("forward/backward WGrad pool capacities differ")

        self._copy_gate_up_data(
            slot.wgrad_fc1_b,
            slot.fc1_col_output,
            self.intermediate,
        )
        slot.wgrad_fc2_a.copy_(slot.fc1_recompute.transpose(0, 1))
        self._expand_scales(
            slot.col_quant_sf,
            slot.valid_route_counts,
            slot.expert_offsets,
            slot.wgrad_fc1_sfa,
            non_k_size=self.hidden,
        )
        self._expand_scales(
            slot.fc1_col_output_sf,
            slot.valid_route_counts,
            slot.expert_offsets,
            slot.wgrad_fc1_sfb,
            non_k_size=2 * self.intermediate,
            deinterleave_gate_up=self.intermediate,
        )
        self._expand_scales(
            slot.fc1_recompute_sf,
            slot.valid_route_counts,
            slot.expert_offsets,
            slot.wgrad_fc2_sfa,
            non_k_size=self.intermediate,
        )
        self._expand_scales(
            slot.grad_y2_sf,
            slot.valid_route_counts,
            slot.expert_offsets,
            slot.wgrad_fc2_sfb,
            non_k_size=self.hidden,
        )

        return MoeEpTrainingWgradOperands(
            fc1_a=slot.col_quant_data.transpose(0, 1),
            fc1_sfa=slot.wgrad_fc1_sfa,
            fc1_b=slot.wgrad_fc1_b,
            fc1_sfb=slot.wgrad_fc1_sfb,
            fc2_a=slot.wgrad_fc2_a,
            fc2_sfa=slot.wgrad_fc2_sfa,
            fc2_b=slot.grad_y2,
            fc2_sfb=slot.wgrad_fc2_sfb,
            expert_offsets=slot.expert_offsets,
            valid_route_counts=slot.valid_route_counts,
        )


__all__ = ["Mxfp8TrainingWgradExporter"]
