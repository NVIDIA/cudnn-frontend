# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape dense and packed training prototype for causal conv.

This module intentionally stays separate from the inference API while the
backward schedules are being compared.  Construction compiles the exact-shape
backward once; calls then expose an ordinary PyTorch autograd edge. Packed
calls consume device ``cu_seqlens`` and optional BF16 channel bias. Bias-free
calls may retain their width-four filter in FP32. Optional
full-width initial/final state uses the same prefill-to-decode ABI as forward;
width variants remain outside this experiment.
"""

from __future__ import annotations

import torch

from cudnn.api_base import TupleDict

from .api import causal_conv1d_bulk_fwd_wrapper_sm100
from .backward import (
    CausalConv1dBulkBwdPrototype,
    compile_causal_conv1d_bulk_bwd_prototype,
)


class _CausalConv1dBulkAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        backend,
        bias: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ):
        result = causal_conv1d_bulk_fwd_wrapper_sm100(
            x,
            weight,
            cu_seqlens_tensor=cu_seqlens,
            bias_tensor=bias,
            initial_state_tensor=initial_state,
            output_final_state=output_final_state,
        )
        saved = [x, weight]
        if bias is not None:
            saved.append(bias)
        if cu_seqlens is not None:
            saved.append(cu_seqlens)
        if initial_state is not None:
            saved.append(initial_state)
        ctx.save_for_backward(*saved)
        ctx.has_bias = bias is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.has_initial_state = initial_state is not None
        ctx.output_final_state = output_final_state
        ctx.backend = backend
        if output_final_state:
            return result["output_tensor"], result["final_state_tensor"]
        return result["output_tensor"]

    @staticmethod
    def backward(ctx, dy: torch.Tensor, d_final_state: torch.Tensor | None = None):
        x, weight = ctx.saved_tensors[:2]
        saved_index = 2
        bias = None
        if ctx.has_bias:
            bias = ctx.saved_tensors[saved_index]
            saved_index += 1
        cu_seqlens = None
        if ctx.has_cu_seqlens:
            cu_seqlens = ctx.saved_tensors[saved_index]
            saved_index += 1
        initial_state = ctx.saved_tensors[saved_index] if ctx.has_initial_state else None
        backend: CausalConv1dBulkBwdPrototype = ctx.backend
        dy = dy.contiguous()
        if ctx.output_final_state:
            if d_final_state is None:
                raise RuntimeError("autograd did not materialize the required d_final_state")
            d_final_state = d_final_state.contiguous()
        elif d_final_state is not None:
            raise RuntimeError("received d_final_state for a forward without final-state output")
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        dw_accum = torch.empty(
            weight.shape,
            dtype=torch.float32,
            device=weight.device,
        )
        db_accum = None
        if bias is not None:
            db_accum = torch.empty(
                bias.shape,
                dtype=torch.float32,
                device=bias.device,
            )
        d_initial_state = None
        if initial_state is not None:
            d_initial_state = torch.empty_like(initial_state, memory_format=torch.contiguous_format)
        workspace = None
        if backend.dweight_workspace_numel:
            workspace = torch.empty(
                backend.dweight_workspace_numel,
                dtype=torch.float32,
                device=weight.device,
            )
        packed_tile_map = None
        if backend.packed_tile_map_numel:
            packed_tile_map = torch.empty(
                backend.packed_tile_map_numel,
                dtype=torch.int32,
                device=weight.device,
            )
        backend.execute(
            x,
            weight,
            dy,
            dx,
            dw_accum,
            cu_seqlens=cu_seqlens,
            packed_tile_map=packed_tile_map,
            dweight_workspace=workspace,
            bias=bias,
            db_accum=db_accum,
            initial_state=initial_state,
            d_final_state=d_final_state,
            d_initial_state=d_initial_state,
        )
        db = db_accum.to(bias.dtype) if db_accum is not None else None
        return dx, dw_accum.to(weight.dtype), None, None, db, d_initial_state, None


class CausalConv1dBulkAutogradPrototype:
    """One exact-shape dense or packed BF16 training callable."""

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_cu_seqlens: torch.Tensor | None = None,
        *,
        schedule: str = "auto",
        sample_bias: torch.Tensor | None = None,
        sample_initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> None:
        if not isinstance(output_final_state, bool):
            raise TypeError(f"output_final_state must be bool, got {type(output_final_state).__name__}")
        sample_dy = torch.empty_like(sample_x, memory_format=torch.contiguous_format)
        sample_d_final_state = None
        if output_final_state:
            num_sequences = sample_x.shape[0] if sample_cu_seqlens is None else sample_cu_seqlens.shape[0] - 1
            sample_d_final_state = torch.empty(
                (num_sequences, sample_x.shape[2], 4),
                dtype=sample_x.dtype,
                device=sample_x.device,
            )
        self.backward_backend = compile_causal_conv1d_bulk_bwd_prototype(
            sample_x,
            sample_weight,
            sample_dy,
            sample_cu_seqlens,
            schedule=schedule,
            bias=sample_bias,
            initial_state=sample_initial_state,
            d_final_state=sample_d_final_state,
        )
        self.output_final_state = output_final_state

    @property
    def dweight_workspace_bytes(self) -> int:
        return self.backward_backend.dweight_workspace_bytes

    @property
    def total_workspace_bytes(self) -> int:
        return self.backward_backend.total_workspace_bytes

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        *,
        bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool | None = None,
    ) -> torch.Tensor | TupleDict:
        self.backward_backend._validate_runtime_tensor(x, self.backward_backend.x_desc, "X")
        self.backward_backend._validate_runtime_tensor(weight, self.backward_backend.weight_desc, "Weight")
        if (cu_seqlens is None) != (self.backward_backend.cu_seqlens_desc is None):
            raise ValueError("cu_seqlens presence must match the compiled signature")
        if cu_seqlens is not None:
            assert self.backward_backend.cu_seqlens_desc is not None
            self.backward_backend._validate_runtime_tensor(cu_seqlens, self.backward_backend.cu_seqlens_desc, "cu_seqlens")
        if (bias is None) != (self.backward_backend.bias_desc is None):
            raise ValueError("Bias presence must match the compiled signature")
        if bias is not None:
            assert self.backward_backend.bias_desc is not None
            self.backward_backend._validate_runtime_tensor(bias, self.backward_backend.bias_desc, "Bias")
        if (initial_state is None) != (self.backward_backend.initial_state_desc is None):
            raise ValueError("Initial state presence must match the compiled signature")
        if initial_state is not None:
            assert self.backward_backend.initial_state_desc is not None
            self.backward_backend._validate_runtime_tensor(initial_state, self.backward_backend.initial_state_desc, "Initial state")
        if output_final_state is None:
            output_final_state = self.output_final_state
        elif not isinstance(output_final_state, bool):
            raise TypeError(f"output_final_state must be bool or None, got {type(output_final_state).__name__}")
        if output_final_state != self.output_final_state:
            raise ValueError("output_final_state must match the compiled signature")
        result = _CausalConv1dBulkAutogradFunction.apply(
            x,
            weight,
            cu_seqlens,
            self.backward_backend,
            bias,
            initial_state,
            output_final_state,
        )
        if output_final_state:
            output, final_state = result
            return TupleDict(output_tensor=output, final_state_tensor=final_state)
        return result


__all__ = ["CausalConv1dBulkAutogradPrototype"]
