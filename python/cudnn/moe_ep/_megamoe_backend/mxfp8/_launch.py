# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe tensor conversion and current-stream MXFP8 launch."""

from __future__ import annotations

from typing import Any

import torch

from .._plan import PreparedResources
from ._adapter import Mxfp8LaunchInputs
from ._overflow import apply_overflow_policy


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
    return cute_tensor.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 128):
    """Build the opaque byte-pointer ABI used by Rubin workspaces."""

    import cutlass
    from cutlass.cute.runtime import make_ptr
    from cutlass.cute.typing import AddressSpace

    address = int(tensor.data_ptr())
    if address % assumed_align:
        raise ValueError(f"Rubin workspace address {address:#x} is not " f"{assumed_align}-byte aligned")
    return make_ptr(
        cutlass.Uint8,
        address,
        AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def _to_discrete_ptr_table(tensor: torch.Tensor):
    """Build the device Int64 pointer-table ABI used by discrete weights."""

    import cutlass
    from cutlass.cute.runtime import make_ptr
    from cutlass.cute.typing import AddressSpace

    address = int(tensor.data_ptr())
    if address % 8:
        raise ValueError(f"Rubin discrete pointer-table address {address:#x} is not 8-byte aligned")
    return make_ptr(
        cutlass.Int64,
        address,
        AddressSpace.gmem,
        assumed_align=8,
    )


def build_runtime_kwargs(
    inputs: Mxfp8LaunchInputs,
    resources: PreparedResources,
    *,
    weight_storage_mode: str = "contiguous",
) -> dict[str, Any]:
    import cuda.bindings.driver as cuda

    stream = resources.runtime.current_stream()
    weights = inputs.weights
    if weight_storage_mode == "contiguous":
        convert_weight = _to_cute
    elif weight_storage_mode == "discrete":
        convert_weight = _to_discrete_ptr_table
    else:
        raise ValueError("weight_storage_mode must be 'contiguous' or 'discrete', " f"got {weight_storage_mode!r}")
    kwargs = {
        "activation": _to_cute(inputs.activation),
        "activation_sf": _to_cute(inputs.activation_sf),
        "topk_indices": _to_cute(inputs.topk_indices),
        "topk_scores": _to_cute(inputs.topk_scores, assumed_align=4),
        "fc1_weight": convert_weight(weights.fc1_weight),
        "fc1_weight_sf": convert_weight(weights.fc1_weight_sf),
        "fc2_weight": convert_weight(weights.fc2_weight),
        "fc2_weight_sf": convert_weight(weights.fc2_weight_sf),
        "fc1_c": (None if inputs.fc1_c is None else _to_cute(inputs.fc1_c, dynamic_layout=False)),
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
        "peer_rank_ptr_mapper_host": (resources.workspace.peer_mapping.to_sym_buffer_host()),
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
    return tuple(None if tensor is None else (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype) for tensor in tensors)


def launch_forward(
    compiled,
    inputs: Mxfp8LaunchInputs,
    resources: PreparedResources,
    *,
    drop_on_overflow: bool,
) -> torch.Tensor:
    runtime_kwargs = build_runtime_kwargs(inputs, resources)
    compiled.callable(**runtime_kwargs)
    if inputs.overflow_ok is None:
        raise RuntimeError("inference launch requires a prepared overflow_ok buffer")
    apply_overflow_policy(
        inputs.overflow_flag,
        drop_on_overflow=drop_on_overflow,
        overflow_ok=inputs.overflow_ok,
        message=("Rubin MegaMoE receive route-pool overflow; the output is invalid " "for this routing distribution"),
    )

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
