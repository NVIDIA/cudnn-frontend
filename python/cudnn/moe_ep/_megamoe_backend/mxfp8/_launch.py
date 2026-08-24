# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CuTe tensor conversion and current-stream MXFP8 launch."""

from __future__ import annotations

from typing import Any

import torch

from .._plan import PreparedResources
from ._adapter import Mxfp8LaunchInputs


def _to_cute(
    tensor: torch.Tensor,
    assumed_align: int = 16,
    *,
    dynamic_layout: bool = True,
):
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(
        tensor,
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    if not dynamic_layout:
        return cute_tensor
    return cute_tensor.mark_layout_dynamic(
        leading_dim=cutlass_torch.get_leading_dim(tensor)
    )


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 128):
    """Build the opaque byte-pointer ABI used by Rubin workspaces."""

    import cutlass
    from cutlass.cute.runtime import make_ptr
    from cutlass.cute.typing import AddressSpace

    address = int(tensor.data_ptr())
    if address % assumed_align:
        raise ValueError(
            f"Rubin workspace address {address:#x} is not "
            f"{assumed_align}-byte aligned"
        )
    return make_ptr(
        cutlass.Uint8,
        address,
        AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def build_runtime_kwargs(
    inputs: Mxfp8LaunchInputs,
    resources: PreparedResources,
) -> dict[str, Any]:
    import cuda.bindings.driver as cuda

    stream = resources.runtime.current_stream()
    weights = inputs.weights
    kwargs = {
        "activation": _to_cute(inputs.activation),
        "activation_sf": _to_cute(inputs.activation_sf),
        "topk_indices": _to_cute(inputs.topk_indices),
        "topk_scores": _to_cute(inputs.topk_scores, assumed_align=4),
        "fc1_weight": _to_cute(weights.fc1_weight),
        "fc1_weight_sf": _to_cute(weights.fc1_weight_sf),
        "fc2_weight": _to_cute(weights.fc2_weight),
        "fc2_weight_sf": _to_cute(weights.fc2_weight_sf),
        "fc1_c": (
            None
            if inputs.fc1_c is None
            else _to_cute(inputs.fc1_c, dynamic_layout=False)
        ),
        "output_activation": _to_cute(inputs.output_data),
        "col_quant_data": (
            None
            if inputs.col_quant_data is None
            else _to_cute(
                inputs.col_quant_data,
                assumed_align=128,
                dynamic_layout=False,
            )
        ),
        "col_quant_sf": (
            None
            if inputs.col_quant_sf is None
            else _to_cute(
                inputs.col_quant_sf,
                dynamic_layout=False,
            )
        ),
        "overflow_flag": _to_cute(
            inputs.overflow_flag,
            assumed_align=4,
            dynamic_layout=False,
        ),
        "local_workspace": _to_cute_ptr(inputs.local_workspace),
        "shared_workspace": _to_cute_ptr(inputs.shared_workspace),
        "peer_rank_ptr_mapper_host": (
            resources.workspace.peer_mapping.to_sym_buffer_host()
        ),
        "stream": cuda.CUstream(stream.cuda_stream),
    }
    return kwargs


def layout_signature(inputs: Mxfp8LaunchInputs) -> tuple:
    tensors = (
        inputs.activation,
        inputs.activation_sf,
        inputs.topk_indices,
        inputs.topk_scores,
        inputs.weights.fc1_weight,
        inputs.weights.fc1_weight_sf,
        inputs.weights.fc2_weight,
        inputs.weights.fc2_weight_sf,
        inputs.fc1_c,
        inputs.col_quant_data,
        inputs.col_quant_sf,
        inputs.output_data,
        inputs.overflow_flag,
        inputs.local_workspace,
        inputs.shared_workspace,
    )
    return tuple(
        None
        if tensor is None
        else (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        for tensor in tensors
    )


def _check_overflow(overflow_flag: torch.Tensor) -> None:
    message = (
        "Rubin MegaMoE receive route-pool overflow; the output is invalid for "
        "this routing distribution"
    )
    assert_async = getattr(torch, "_assert_async", None)
    if assert_async is not None:
        assert_async(overflow_flag == 0, message)
        return
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "CUDA graph capture requires torch._assert_async to surface "
            "Rubin MegaMoE overflow"
        )
    # Compatibility fallback for PyTorch builds without a device-side assert.
    value = int(overflow_flag.item())
    if value != 0:
        raise RuntimeError(f"{message} (overflow_flag={value})")


def launch_forward(
    compiled,
    inputs: Mxfp8LaunchInputs,
    resources: PreparedResources,
) -> torch.Tensor:
    runtime_kwargs = build_runtime_kwargs(inputs, resources)
    compiled.callable(**runtime_kwargs)
    _check_overflow(inputs.overflow_flag)

    output_data = torch.empty(
        (inputs.token_count, inputs.output_data.shape[1]),
        dtype=inputs.output_data.dtype,
        device=inputs.output_data.device,
    )
    output_data.copy_(inputs.output_data[: inputs.token_count])
    return output_data


__all__ = [
    "build_runtime_kwargs",
    "launch_forward",
    "layout_signature",
]
