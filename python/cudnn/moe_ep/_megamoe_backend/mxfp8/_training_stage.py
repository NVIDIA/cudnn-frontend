# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Cached BF16/FP32 staging into private symmetric training scratch."""

from __future__ import annotations

import threading

import torch

from ._launch import _to_cute


class Mxfp8TrainingStager:
    """Own one compile cache; steady-state staging is allocation-free."""

    def __init__(self, hidden: int, top_k: int) -> None:
        self.hidden = int(hidden)
        self.top_k = int(top_k)
        self._compiled: dict[tuple, object] = {}
        self._lock = threading.RLock()

    def _validate(
        self,
        source: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
        output_sf: torch.Tensor,
        output_topk_idx: torch.Tensor,
        output_topk_weights: torch.Tensor,
    ) -> int:
        if source.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("training staging source must be BF16 or FP32, " f"got {source.dtype}")
        if source.ndim != 2 or source.shape[1] != self.hidden:
            raise ValueError(f"training staging source must have shape (T, {self.hidden})")
        if not source.is_contiguous():
            raise ValueError("training staging source must be contiguous")
        token_count = int(source.shape[0])
        if topk_idx.shape != (token_count, self.top_k):
            raise ValueError("training staging topk_idx shape mismatch")
        if topk_idx.dtype is not torch.int32 or not topk_idx.is_contiguous():
            raise TypeError("training staging topk_idx must be contiguous Int32")
        if topk_weights.shape != topk_idx.shape:
            raise ValueError("training staging topk_weights shape mismatch")
        if topk_weights.dtype is not torch.float32 or not topk_weights.is_contiguous():
            raise TypeError("training staging topk_weights must be contiguous FP32")
        if output.dtype is not torch.float8_e4m3fn or output.ndim != 2 or output.shape[1] != self.hidden or not output.is_contiguous():
            raise ValueError("training staging output must be contiguous E4M3 " f"(capacity, {self.hidden})")
        if token_count > output.shape[0]:
            raise ValueError(f"token count {token_count} exceeds capacity {output.shape[0]}")
        logical_sf_columns = self.hidden // 32
        if (
            output_sf.dtype is not torch.float8_e8m0fnu
            or output_sf.ndim != 2
            or output_sf.shape[0] != output.shape[0]
            or output_sf.shape[1] < logical_sf_columns
            or not output_sf.is_contiguous()
        ):
            raise ValueError("training staging output_sf has an invalid ABI")
        for name, tensor, dtype in (
            ("output_topk_idx", output_topk_idx, torch.int32),
            ("output_topk_weights", output_topk_weights, torch.float32),
        ):
            if tensor.shape != (output.shape[0], self.top_k) or tensor.dtype is not dtype or not tensor.is_contiguous():
                raise ValueError(f"training staging {name} has an invalid ABI")
        devices = {
            source.device,
            topk_idx.device,
            topk_weights.device,
            output.device,
            output_sf.device,
            output_topk_idx.device,
            output_topk_weights.device,
        }
        if len(devices) != 1:
            raise ValueError("all training staging tensors must share one device")
        return token_count

    def stage(
        self,
        source: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
        output_sf: torch.Tensor,
        output_topk_idx: torch.Tensor,
        output_topk_weights: torch.Tensor,
    ) -> None:
        """Enqueue tail reset plus one fused quant-and-routing launch."""

        token_count = self._validate(
            source,
            topk_idx,
            topk_weights,
            output,
            output_sf,
            output_topk_idx,
            output_topk_weights,
        )
        output_sf.zero_()
        routing_in_place = topk_idx.data_ptr() == output_topk_idx.data_ptr() and topk_weights.data_ptr() == output_topk_weights.data_ptr()
        routing_partially_aliased = (topk_idx.data_ptr() == output_topk_idx.data_ptr()) != (topk_weights.data_ptr() == output_topk_weights.data_ptr())
        if routing_partially_aliased:
            raise ValueError("training staging routing inputs must either both alias " "their outputs or neither alias")
        if not routing_in_place:
            output_topk_idx.fill_(-1)
            output_topk_weights.zero_()
        if token_count == 0:
            return

        logical_sf_columns = self.hidden // 32
        import cuda.bindings.driver as cuda

        stream = torch.cuda.current_stream(source.device)
        args = (
            _to_cute(source, dynamic_layout=False),
            _to_cute(topk_idx, assumed_align=4, dynamic_layout=False),
            _to_cute(topk_weights, assumed_align=4, dynamic_layout=False),
            _to_cute(output[:token_count], dynamic_layout=False),
            _to_cute(
                output_sf[:token_count, :logical_sf_columns],
                assumed_align=4,
                dynamic_layout=False,
            ),
            _to_cute(
                output_topk_idx[:token_count],
                assumed_align=4,
                dynamic_layout=False,
            ),
            _to_cute(
                output_topk_weights[:token_count],
                assumed_align=4,
                dynamic_layout=False,
            ),
            cuda.CUstream(stream.cuda_stream),
        )
        key = (
            source.device.index,
            source.dtype,
            token_count,
            self.hidden,
            self.top_k,
            tuple(output_sf.stride()),
        )
        with self._lock:
            compiled = self._compiled.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("MXFP8 training stager must be compiled before " "CUDA graph capture")
                import cutlass.cute as cute

                from ._training_stage_kernel import Mxfp8TrainingStageKernel

                kernel = Mxfp8TrainingStageKernel(self.hidden, self.top_k)
                compiled = cute.compile(kernel, *args)
                self._compiled[key] = compiled
        compiled(*args)


__all__ = ["Mxfp8TrainingStager"]
