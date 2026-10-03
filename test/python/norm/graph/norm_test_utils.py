# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared graph execution and quantization helpers for LayerNorm and RMSNorm tests."""

from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple

import cudnn
import torch

_HEURISTICS = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
_BLOCK_SCALE_TORCH_TYPES = {
    "mxfp8": ("float8_e5m2", "float8_e8m0fnu"),
    "nvfp4": ("float8_e4m3fn", "float4_e2m1fn_x2"),
}


class NormPlanDiscoveryUnsupportedError(RuntimeError):
    pass


class NormSupportUnsupportedError(RuntimeError):
    pass


@dataclass(frozen=True)
class NormForwardOutputs:
    y: torch.Tensor
    mean: Optional[torch.Tensor]
    inv_variance: Optional[torch.Tensor]


@dataclass(frozen=True)
class NormBackwardOutputs:
    dx: torch.Tensor
    dscale: torch.Tensor
    dbias: Optional[torch.Tensor]


@dataclass(frozen=True)
class BlockScaleQuantizeSpec:
    block_size: int
    axis: int
    output_data_type: object
    output_torch_dtype: torch.dtype
    scale_data_type: object
    scale_torch_dtype: torch.dtype
    transpose: bool = False


@dataclass(frozen=True)
class NormBlockScaleOutputs:
    quantized: Tuple[torch.Tensor, ...]
    scales: Tuple[torch.Tensor, ...]
    mean: Optional[torch.Tensor]
    inv_variance: Optional[torch.Tensor]


@dataclass(frozen=True)
class _NormForwardGraph:
    graph: object
    x: object
    scale: object
    bias: Optional[object]
    epsilon: object
    y: object
    mean: Optional[object]
    inv_variance: Optional[object]


def block_scale_quantize_skip_reason(quantization: str) -> Optional[str]:
    if quantization not in _BLOCK_SCALE_TORCH_TYPES:
        raise ValueError(f"Unsupported block-scale quantization format: {quantization}")
    if cudnn.backend_version() < 90700:
        return "Block-scale quantized norm output requires cuDNN 9.7.0 or newer"
    if torch.cuda.get_device_capability()[0] < 10:
        return "Block-scale quantized norm output requires Blackwell or newer"
    missing_types = [name for name in _BLOCK_SCALE_TORCH_TYPES[quantization] if not hasattr(torch, name)]
    if missing_types:
        return f"PyTorch does not provide the required data types: {', '.join(missing_types)}"
    return None


def make_seeded_randn(shape: Tuple[int, ...], dtype: torch.dtype, seed: int, scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=dtype, generator=generator).mul_(scale).add_(shift)


def make_bsh_block_scale_quantize_specs(quantization: str, *, include_column_output: bool) -> Tuple[BlockScaleQuantizeSpec, ...]:
    if quantization == "mxfp8":
        specs = (
            BlockScaleQuantizeSpec(
                block_size=32,
                axis=2,
                output_data_type=cudnn.data_type.FP8_E5M2,
                output_torch_dtype=torch.float8_e5m2,
                scale_data_type=cudnn.data_type.FP8_E8M0,
                scale_torch_dtype=torch.float8_e8m0fnu,
            ),
        )
        if include_column_output:
            specs += (
                BlockScaleQuantizeSpec(
                    block_size=32,
                    axis=1,
                    output_data_type=cudnn.data_type.FP8_E5M2,
                    output_torch_dtype=torch.float8_e5m2,
                    scale_data_type=cudnn.data_type.FP8_E8M0,
                    scale_torch_dtype=torch.float8_e8m0fnu,
                    transpose=True,
                ),
            )
        return specs
    if quantization == "nvfp4" and not include_column_output:
        return (
            BlockScaleQuantizeSpec(
                block_size=16,
                axis=2,
                output_data_type=cudnn.data_type.FP4_E2M1,
                output_torch_dtype=torch.uint8,
                scale_data_type=cudnn.data_type.FP8_E4M3,
                scale_torch_dtype=torch.float8_e4m3fn,
            ),
        )
    raise ValueError(f"Unsupported block-scale quantization configuration: {quantization}, include_column_output={include_column_output}")


def _packed_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    stride = 1
    result = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _new_graph(cudnn_handle):
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    return cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )


def _build_norm_forward_graph(
    *,
    layernorm: bool,
    phase,
    x: torch.Tensor,
    scale_gpu: torch.Tensor,
    bias_gpu: Optional[torch.Tensor],
    epsilon_cpu: torch.Tensor,
    cudnn_handle,
) -> _NormForwardGraph:
    graph = _new_graph(cudnn_handle)
    x_tensor = graph.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    scale = graph.tensor(name="scale", dim=scale_gpu.size(), stride=scale_gpu.stride(), data_type=scale_gpu.dtype)
    bias = None
    if bias_gpu is not None:
        bias = graph.tensor(name="bias", dim=bias_gpu.size(), stride=bias_gpu.stride(), data_type=bias_gpu.dtype)
    epsilon = graph.tensor_like(epsilon_cpu, name="epsilon")

    if layernorm:
        y, mean, inv_variance = graph.layernorm(
            name="LayerNorm",
            norm_forward_phase=phase,
            input=x_tensor,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
        )
    else:
        y, inv_variance = graph.rmsnorm(
            name="RMSNorm",
            norm_forward_phase=phase,
            input=x_tensor,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
        )
        mean = None

    return _NormForwardGraph(graph, x_tensor, scale, bias, epsilon, y, mean, inv_variance)


def _mark_output(tensor, shape: Tuple[int, ...], stride: Tuple[int, ...], dtype: torch.dtype):
    tensor.set_output(True).set_data_type(dtype)
    tensor.set_dim(shape)
    tensor.set_stride(stride)
    return tensor


def _finalize_graph(graph) -> None:
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans(_HEURISTICS)
    except cudnn.cudnnGraphNotSupportedError as error:
        raise NormPlanDiscoveryUnsupportedError(str(error)) from error
    try:
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as error:
        raise NormSupportUnsupportedError(str(error)) from error
    graph.build_plans()


def _nan_output(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.full(shape, float("nan"), device=device, dtype=dtype)


def _allocate_quantized_output(tensor, spec: BlockScaleQuantizeSpec, device: torch.device) -> torch.Tensor:
    if spec.output_data_type == cudnn.data_type.FP4_E2M1:
        return torch.zeros((prod(tensor.get_dim()) + 1) // 2, device=device, dtype=torch.uint8)
    return torch.zeros(tuple(tensor.get_dim()), device=device, dtype=spec.output_torch_dtype).as_strided(tuple(tensor.get_dim()), tuple(tensor.get_stride()))


def block_scale_quantize_reference(
    values: torch.Tensor,
    spec: BlockScaleQuantizeSpec,
) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = spec.axis % values.ndim
    axis_last = values.float().movedim(axis, -1).contiguous()
    if axis_last.shape[-1] % spec.block_size:
        raise ValueError("The quantization axis must be divisible by the block size")
    blocks = axis_last.view(*axis_last.shape[:-1], axis_last.shape[-1] // spec.block_size, spec.block_size)

    output_max = 6.0 if spec.output_data_type == cudnn.data_type.FP4_E2M1 else torch.finfo(spec.output_torch_dtype).max
    scale_values = blocks.abs().amax(dim=-1) / output_max
    if spec.scale_data_type == cudnn.data_type.FP8_E8M0:
        nonzero_scale = torch.where(scale_values > 0, scale_values, 1.0)
        scale_values = torch.where(scale_values > 0, torch.pow(2.0, torch.ceil(torch.log2(nonzero_scale))), 0.0)
    block_scales = scale_values.to(spec.scale_torch_dtype)
    # NVFP4 quantizes with its FP32 scale before storing that scale as E4M3;
    # MXFP8 quantizes with the emitted E8M0 scale.
    quantization_scale = scale_values if spec.output_data_type == cudnn.data_type.FP4_E2M1 else block_scales.float()
    inverse_scale = torch.where(quantization_scale > 0, quantization_scale.reciprocal(), 0.0)
    normalized = (blocks * inverse_scale.unsqueeze(-1)).clamp(-output_max, output_max)

    if spec.output_data_type == cudnn.data_type.FP4_E2M1:
        magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=values.device)
        boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device)
        absolute_values = normalized.abs()
        indices = torch.bucketize(absolute_values, boundaries, right=True)
        # Resolve exact midpoints with round-to-nearest-even.
        is_midpoint = (indices > 0) & (absolute_values == boundaries[(indices - 1).clamp(min=0)])
        indices = torch.where(is_midpoint & (indices % 2 == 1), indices - 1, indices)
        quantized = torch.where(normalized < 0, -magnitudes[indices], magnitudes[indices])
    else:
        quantized = normalized.to(spec.output_torch_dtype).float()

    quantized = quantized.view(axis_last.shape).movedim(-1, axis)
    block_scales = block_scales.movedim(-1, axis)
    return quantized, block_scales


def unpack_last_dim_fp4(packed: torch.Tensor, logical_shape: Tuple[int, ...]) -> torch.Tensor:
    packed = packed.view(*logical_shape[:-1], logical_shape[-1] // 2)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
    )
    low = values[(packed & 0xF).long()]
    high = values[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).view(logical_shape)


def dequantize_block_scaled(quantized: torch.Tensor, block_scales: torch.Tensor, spec: BlockScaleQuantizeSpec) -> torch.Tensor:
    return quantized.float() * block_scales.float().repeat_interleave(spec.block_size, dim=spec.axis)


def _execute_block_scaled_forward(
    *,
    layernorm: bool,
    phase,
    x: torch.Tensor,
    scale_gpu: torch.Tensor,
    bias_gpu: Optional[torch.Tensor],
    epsilon_cpu: torch.Tensor,
    quantize_specs: Tuple[BlockScaleQuantizeSpec, ...],
    cudnn_handle,
) -> NormBlockScaleOutputs:
    if x.ndim != 3 or scale_gpu.ndim != 3 or (bias_gpu is not None and bias_gpu.ndim != 3):
        raise ValueError("Block-scaled norm test inputs must use BSH descriptors")

    # cuDNN's fused norm block-scale path uses rank-4 descriptors. The trailing
    # singleton preserves the test's BSH semantics.
    x_graph = x.unsqueeze(-1)
    scale_graph = scale_gpu.unsqueeze(-1)
    bias_graph = bias_gpu.unsqueeze(-1) if bias_gpu is not None else None
    epsilon_graph = epsilon_cpu.reshape(1, 1, 1, 1)

    forward = _build_norm_forward_graph(
        layernorm=layernorm,
        phase=phase,
        x=x_graph,
        scale_gpu=scale_graph,
        bias_gpu=bias_graph,
        epsilon_cpu=epsilon_graph,
        cudnn_handle=cudnn_handle,
    )

    quantized_tensors = []
    scale_tensors = []
    for index, spec in enumerate(quantize_specs):
        quantized, block_scale = forward.graph.block_scale_quantize(
            input=forward.y,
            block_size=spec.block_size,
            axis=spec.axis,
            transpose=spec.transpose,
            name=f"quantize_{index}",
        )
        quantized.set_output(True).set_data_type(spec.output_data_type)
        block_scale.set_output(True).set_data_type(spec.scale_data_type)
        quantized_tensors.append(quantized)
        scale_tensors.append(block_scale)

    is_training = phase == cudnn.norm_forward_phase.TRAINING
    stats_shape = tuple(input_extent if parameter_extent == 1 else 1 for input_extent, parameter_extent in zip(x_graph.size(), scale_graph.size()))
    stats_stride = _packed_strides(stats_shape)
    if is_training:
        if forward.mean is not None:
            _mark_output(forward.mean, stats_shape, stats_stride, torch.float32)
        _mark_output(forward.inv_variance, stats_shape, stats_stride, torch.float32)

    _finalize_graph(forward.graph)

    quantized_outputs = tuple(_allocate_quantized_output(tensor, spec, x.device) for tensor, spec in zip(quantized_tensors, quantize_specs))
    scale_outputs = tuple(
        torch.zeros(tuple(tensor.get_dim()), device=x.device, dtype=spec.scale_torch_dtype).as_strided(tuple(tensor.get_dim()), tuple(tensor.get_stride()))
        for tensor, spec in zip(scale_tensors, quantize_specs)
    )
    variant_pack = {
        forward.x: x_graph.detach(),
        forward.scale: scale_graph.detach(),
        forward.epsilon: epsilon_graph,
    }
    if forward.bias is not None:
        variant_pack[forward.bias] = bias_graph.detach()
    variant_pack.update(zip(quantized_tensors, quantized_outputs))
    variant_pack.update(zip(scale_tensors, scale_outputs))

    mean_output = None
    inv_variance_output = None
    if is_training:
        if forward.mean is not None:
            mean_output = _nan_output(stats_shape, torch.float32, x.device)
            variant_pack[forward.mean] = mean_output
        inv_variance_output = _nan_output(stats_shape, torch.float32, x.device)
        variant_pack[forward.inv_variance] = inv_variance_output

    workspace = torch.empty(forward.graph.get_workspace_size(), device=x.device, dtype=torch.uint8)
    forward.graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    logical_quantized_outputs = tuple(
        output if spec.output_data_type == cudnn.data_type.FP4_E2M1 else output.squeeze(-1) for output, spec in zip(quantized_outputs, quantize_specs)
    )
    logical_scale_outputs = tuple(output.squeeze(-1) for output in scale_outputs)
    return NormBlockScaleOutputs(
        quantized=logical_quantized_outputs,
        scales=logical_scale_outputs,
        mean=mean_output.squeeze(-1) if mean_output is not None else None,
        inv_variance=inv_variance_output.squeeze(-1) if inv_variance_output is not None else None,
    )


def execute_layernorm_block_scaled_forward(
    *,
    phase,
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    epsilon: torch.Tensor,
    quantize_specs: Tuple[BlockScaleQuantizeSpec, ...],
    cudnn_handle,
) -> NormBlockScaleOutputs:
    return _execute_block_scaled_forward(
        layernorm=True,
        phase=phase,
        x=x,
        scale_gpu=scale,
        bias_gpu=bias,
        epsilon_cpu=epsilon,
        quantize_specs=quantize_specs,
        cudnn_handle=cudnn_handle,
    )


def execute_rmsnorm_block_scaled_forward(
    *,
    phase,
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    epsilon: torch.Tensor,
    quantize_specs: Tuple[BlockScaleQuantizeSpec, ...],
    cudnn_handle,
) -> NormBlockScaleOutputs:
    return _execute_block_scaled_forward(
        layernorm=False,
        phase=phase,
        x=x,
        scale_gpu=scale,
        bias_gpu=bias,
        epsilon_cpu=epsilon,
        quantize_specs=quantize_specs,
        cudnn_handle=cudnn_handle,
    )


def _execute_forward(
    *,
    layernorm: bool,
    phase,
    x: torch.Tensor,
    scale_gpu: torch.Tensor,
    bias_gpu: Optional[torch.Tensor],
    epsilon_cpu: torch.Tensor,
    cudnn_handle,
) -> NormForwardOutputs:
    forward = _build_norm_forward_graph(
        layernorm=layernorm,
        phase=phase,
        x=x,
        scale_gpu=scale_gpu,
        bias_gpu=bias_gpu,
        epsilon_cpu=epsilon_cpu,
        cudnn_handle=cudnn_handle,
    )

    x_shape = tuple(x.size())
    _mark_output(forward.y, x_shape, tuple(x.stride()), x.dtype)
    is_training = phase == cudnn.norm_forward_phase.TRAINING
    stats_shape = (*x_shape[:-1], 1)
    stats_stride = _packed_strides(stats_shape)
    if is_training:
        if forward.mean is not None:
            _mark_output(forward.mean, stats_shape, stats_stride, torch.float32)
        _mark_output(forward.inv_variance, stats_shape, stats_stride, torch.float32)

    _finalize_graph(forward.graph)

    y_actual = _nan_output(x_shape, x.dtype, x.device)
    variant_pack = {
        forward.x: x.detach(),
        forward.scale: scale_gpu.detach(),
        forward.epsilon: epsilon_cpu,
        forward.y: y_actual,
    }
    if forward.bias is not None:
        variant_pack[forward.bias] = bias_gpu.detach()

    mean_actual = None
    inv_variance_actual = None
    if is_training:
        if forward.mean is not None:
            mean_actual = _nan_output(stats_shape, torch.float32, x.device)
            variant_pack[forward.mean] = mean_actual
        inv_variance_actual = _nan_output(stats_shape, torch.float32, x.device)
        variant_pack[forward.inv_variance] = inv_variance_actual

    workspace = torch.empty(forward.graph.get_workspace_size(), device=x.device, dtype=torch.uint8)
    forward.graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    return NormForwardOutputs(y=y_actual, mean=mean_actual, inv_variance=inv_variance_actual)


def execute_layernorm_forward(
    *,
    phase,
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    epsilon: torch.Tensor,
    cudnn_handle,
) -> NormForwardOutputs:
    return _execute_forward(
        layernorm=True,
        phase=phase,
        x=x,
        scale_gpu=scale,
        bias_gpu=bias,
        epsilon_cpu=epsilon,
        cudnn_handle=cudnn_handle,
    )


def execute_rmsnorm_forward(
    *,
    phase,
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    epsilon: torch.Tensor,
    cudnn_handle,
) -> NormForwardOutputs:
    return _execute_forward(
        layernorm=False,
        phase=phase,
        x=x,
        scale_gpu=scale,
        bias_gpu=bias,
        epsilon_cpu=epsilon,
        cudnn_handle=cudnn_handle,
    )


def _execute_backward(
    *,
    layernorm: bool,
    x: torch.Tensor,
    scale_gpu: torch.Tensor,
    grad_gpu: torch.Tensor,
    mean_gpu: Optional[torch.Tensor],
    inv_variance_gpu: torch.Tensor,
    has_dbias: bool,
    cudnn_handle,
) -> NormBackwardOutputs:
    graph = _new_graph(cudnn_handle)
    x_tensor = graph.tensor(name="X", dim=x.size(), stride=x.stride(), data_type=x.dtype)
    scale = graph.tensor(name="scale", dim=scale_gpu.size(), stride=scale_gpu.stride(), data_type=scale_gpu.dtype)
    grad = graph.tensor(name="grad", dim=grad_gpu.size(), stride=grad_gpu.stride(), data_type=grad_gpu.dtype)
    inv_variance = graph.tensor(
        name="inv_variance",
        dim=inv_variance_gpu.size(),
        stride=inv_variance_gpu.stride(),
        data_type=inv_variance_gpu.dtype,
    )

    mean = None
    if layernorm:
        if mean_gpu is None:
            raise ValueError("LayerNorm backward requires saved mean values")
        mean = graph.tensor(name="mean", dim=mean_gpu.size(), stride=mean_gpu.stride(), data_type=mean_gpu.dtype)
        dx, dscale, dbias = graph.layernorm_backward(
            name="LayerNormBackward",
            grad=grad,
            input=x_tensor,
            scale=scale,
            mean=mean,
            inv_variance=inv_variance,
        )
    else:
        dx, dscale, dbias = graph.rmsnorm_backward(
            name="RMSNormBackward",
            grad=grad,
            input=x_tensor,
            scale=scale,
            inv_variance=inv_variance,
            has_dbias=has_dbias,
        )

    _mark_output(dx, tuple(x.size()), tuple(x.stride()), x.dtype)
    _mark_output(dscale, tuple(scale_gpu.size()), tuple(scale_gpu.stride()), scale_gpu.dtype)
    if dbias is not None:
        _mark_output(dbias, tuple(scale_gpu.size()), tuple(scale_gpu.stride()), scale_gpu.dtype)
    _finalize_graph(graph)

    dx_actual = _nan_output(tuple(x.size()), x.dtype, x.device)
    dscale_actual = _nan_output(tuple(scale_gpu.size()), scale_gpu.dtype, scale_gpu.device)
    dbias_actual = _nan_output(tuple(scale_gpu.size()), scale_gpu.dtype, scale_gpu.device) if dbias is not None else None
    variant_pack = {
        x_tensor: x.detach(),
        scale: scale_gpu.detach(),
        grad: grad_gpu.detach(),
        inv_variance: inv_variance_gpu.detach(),
        dx: dx_actual,
        dscale: dscale_actual,
    }
    if mean is not None:
        variant_pack[mean] = mean_gpu.detach()
    if dbias is not None:
        variant_pack[dbias] = dbias_actual

    workspace = torch.empty(graph.get_workspace_size(), device=x.device, dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    return NormBackwardOutputs(dx=dx_actual, dscale=dscale_actual, dbias=dbias_actual)


def execute_layernorm_backward(
    *,
    x: torch.Tensor,
    scale: torch.Tensor,
    grad: torch.Tensor,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
    cudnn_handle,
) -> NormBackwardOutputs:
    return _execute_backward(
        layernorm=True,
        x=x,
        scale_gpu=scale,
        grad_gpu=grad,
        mean_gpu=mean,
        inv_variance_gpu=inv_variance,
        has_dbias=True,
        cudnn_handle=cudnn_handle,
    )


def execute_rmsnorm_backward(
    *,
    x: torch.Tensor,
    scale: torch.Tensor,
    grad: torch.Tensor,
    inv_variance: torch.Tensor,
    has_dbias: bool,
    cudnn_handle,
) -> NormBackwardOutputs:
    return _execute_backward(
        layernorm=False,
        x=x,
        scale_gpu=scale,
        grad_gpu=grad,
        mean_gpu=None,
        inv_variance_gpu=inv_variance,
        has_dbias=has_dbias,
        cudnn_handle=cudnn_handle,
    )
