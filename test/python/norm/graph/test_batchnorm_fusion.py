# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple

import cudnn
import pytest
import torch
import torch.nn.functional as functional

from batchnorm_fusion_cases import BATCHNORM_CASES, EPSILON, FP8_E4M3, FP8_E5M2, MOMENTUM, BatchNormCase, BatchNormPattern
from batchnorm_test_utils import execute_graph, finalize_graph, new_batchnorm_graph, preserve_handle_stream, torch_to_cudnn_data_type

FP8_DTYPES = tuple(dtype for dtype in (FP8_E4M3, FP8_E5M2) if dtype is not None)
FP32_OUTPUT_RTOL = 1e-4
FP32_OUTPUT_ATOL = 1e-5
FP32_REDUCTION_RTOL = 2e-3
FP32_REDUCTION_ATOL = 2e-4
FP8_OUTLIER_RTOL = 0.5
FP8_OUTLIER_ATOL = 0.5
RELU_BOUNDARY_ATOL = 1e-4
MASK_OUTPUT_INITIAL_VALUE = 0xA5
# The largest FP8 references use several GiB, so keep them in the shared exclusive xdist group.
pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@dataclass(frozen=True)
class BatchNormForwardTensors:
    x: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor
    running_mean: torch.Tensor
    running_variance: torch.Tensor
    z: Optional[torch.Tensor]
    epsilon: torch.Tensor
    momentum: torch.Tensor
    zero: torch.Tensor


@dataclass(frozen=True)
class BatchNormBackwardTensors:
    x: torch.Tensor
    dy: torch.Tensor
    scale: torch.Tensor
    mask: Optional[torch.Tensor]


@dataclass(frozen=True)
class FP8ForwardTensors:
    x: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor
    running_mean: torch.Tensor
    running_variance: torch.Tensor
    z: Optional[torch.Tensor]
    x_descale: torch.Tensor
    z_descale: torch.Tensor
    y_scale: torch.Tensor
    epsilon: torch.Tensor
    momentum: torch.Tensor
    zero: torch.Tensor


@dataclass(frozen=True)
class FP8BackwardTensors:
    x: torch.Tensor
    dy: torch.Tensor
    scale: torch.Tensor
    mask: Optional[torch.Tensor]
    x_descale: torch.Tensor
    dy_descale: torch.Tensor
    dx_scale: torch.Tensor
    add_scale: torch.Tensor


@dataclass(frozen=True)
class BatchNormMoments:
    mean: torch.Tensor
    variance: torch.Tensor
    inv_variance: torch.Tensor


@dataclass(frozen=True)
class BatchNormTrainingStatistics:
    saved_mean: torch.Tensor
    saved_inv_variance: torch.Tensor
    running_mean: torch.Tensor
    running_variance: torch.Tensor


def _packed_nhwc_strides(shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    _n, channels, height, width = shape
    return (height * width * channels, 1, width * channels, channels)


def _randn_strided(
    shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    seed: int,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    random_dtype = torch.float16 if dtype in FP8_DTYPES else dtype
    values = torch.randn(shape, device="cuda", dtype=random_dtype, generator=generator).mul_(scale).add_(shift)
    result = torch.empty_strided(shape, _packed_nhwc_strides(shape), device="cuda", dtype=dtype)
    result.copy_(values)
    return result


def _to_packed_nhwc(tensor: torch.Tensor) -> torch.Tensor:
    shape = tuple(tensor.shape)
    result = torch.empty_strided(shape, _packed_nhwc_strides(shape), device=tensor.device, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def _batchnorm_moments(activation: torch.Tensor) -> BatchNormMoments:
    reduction_dims = (0, 2, 3)
    mean = _to_packed_nhwc(activation.mean(dim=reduction_dims, keepdim=True))
    variance = _to_packed_nhwc(activation.var(dim=reduction_dims, correction=0, keepdim=True))
    inv_variance = _to_packed_nhwc(torch.rsqrt(variance + EPSILON))
    return BatchNormMoments(mean, variance, inv_variance)


def _compute_batchnorm_training_statistics(
    activation: torch.Tensor,
    running_mean: torch.Tensor,
    running_variance: torch.Tensor,
) -> BatchNormTrainingStatistics:
    moments = _batchnorm_moments(activation)
    sample_count = activation.shape[0] * activation.shape[2] * activation.shape[3]
    unbiased_variance = moments.variance * (sample_count / (sample_count - 1))
    return BatchNormTrainingStatistics(
        saved_mean=moments.mean,
        saved_inv_variance=moments.inv_variance,
        running_mean=(1.0 - MOMENTUM) * running_mean + MOMENTUM * moments.mean,
        running_variance=(1.0 - MOMENTUM) * running_variance + MOMENTUM * unbiased_variance,
    )


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_mismatch_rate: float = 0.0,
    allowed_mismatch_mask: Optional[torch.Tensor] = None,
    outlier_rtol: Optional[float] = None,
    outlier_atol: Optional[float] = None,
) -> None:
    """Checks finite values and bounds mismatch location, count, and magnitude."""

    actual_float = actual.float()
    expected_float = expected.float()
    if not torch.isfinite(actual_float).all():
        raise AssertionError(f"{name} contains non-finite actual values")
    if not torch.isfinite(expected_float).all():
        raise AssertionError(f"{name} contains non-finite reference values")

    mismatches = ~torch.isclose(actual_float, expected_float, rtol=rtol, atol=atol)
    if allowed_mismatch_mask is not None:
        mismatches_outside_allowed_region = mismatches & ~allowed_mismatch_mask
        if mismatches_outside_allowed_region.any():
            count = mismatches_outside_allowed_region.count_nonzero().item()
            raise AssertionError(f"{name} has {count} out-of-tolerance elements outside the allowed boundary region")
    if outlier_rtol is not None and outlier_atol is not None:
        unbounded_outliers = mismatches & ~torch.isclose(actual_float, expected_float, rtol=outlier_rtol, atol=outlier_atol)
        if unbounded_outliers.any():
            count = unbounded_outliers.count_nonzero().item()
            raise AssertionError(f"{name} has {count} elements outside the bounded outlier tolerance")

    mismatch_count = mismatches.count_nonzero().item()
    allowed_mismatches = int(max_mismatch_rate * mismatches.numel())
    if mismatch_count > allowed_mismatches:
        raise AssertionError(
            f"{name} mismatch: {mismatch_count}/{mismatches.numel()} elements differ; " f"allowed {allowed_mismatches} at rtol={rtol}, atol={atol}"
        )


def _assert_batchnorm_training_statistics(
    *,
    saved_mean: torch.Tensor,
    saved_inv_variance: torch.Tensor,
    running_mean: torch.Tensor,
    running_variance: torch.Tensor,
    expected: BatchNormTrainingStatistics,
) -> None:
    for name, actual, reference in (
        ("saved mean", saved_mean, expected.saved_mean),
        ("saved inverse variance", saved_inv_variance, expected.saved_inv_variance),
        ("running mean", running_mean, expected.running_mean),
        ("running variance", running_variance, expected.running_variance),
    ):
        _assert_close(name, actual, reference, rtol=FP32_OUTPUT_RTOL, atol=FP32_OUTPUT_ATOL)


def _require_fp8_support(bn_case: BatchNormCase) -> None:
    if FP8_E4M3 is None or FP8_E5M2 is None:
        pytest.skip("PyTorch FP8 dtypes are unavailable")
    if cudnn.backend_version() < 90000:
        pytest.skip("FP8 BatchNorm fusion coverage requires cuDNN 9.0 or newer")
    if bn_case.min_compute_capability is None:
        raise ValueError(f"missing minimum compute capability for FP8 case {bn_case.case_id}")
    if torch.cuda.get_device_capability() < bn_case.min_compute_capability:
        required_major, required_minor = bn_case.min_compute_capability
        pytest.skip(f"FP8 BatchNorm fusion coverage requires compute capability {required_major}.{required_minor} or newer")


def _make_batchnorm_forward_tensors(bn_case: BatchNormCase) -> BatchNormForwardTensors:
    x = _randn_strided(bn_case.shape, bn_case.input_dtype, bn_case.seed, scale=0.5, shift=-0.125)
    scale = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 1, scale=0.5, shift=1.0)
    bias = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 2, scale=0.25)
    running_mean = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 3, scale=0.25)
    running_variance = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 4, scale=0.1, shift=1.0)
    running_variance.abs_()
    z = _randn_strided(bn_case.shape, bn_case.input_dtype, bn_case.seed + 5, scale=0.25) if bn_case.pattern.has_add else None
    epsilon = torch.full((1, 1, 1, 1), EPSILON, device="cpu", dtype=torch.float64)
    momentum = torch.full((1, 1, 1, 1), MOMENTUM, device="cpu", dtype=torch.float64)
    zero = torch.zeros((1, 1, 1, 1), device="cpu", dtype=bn_case.output_dtype)
    return BatchNormForwardTensors(x, scale, bias, running_mean, running_variance, z, epsilon, momentum, zero)


def _batchnorm_forward_reference(bn_case: BatchNormCase, tensors: BatchNormForwardTensors):
    x_high_precision = tensors.x.float()
    statistics = _compute_batchnorm_training_statistics(x_high_precision, tensors.running_mean, tensors.running_variance)
    with torch.backends.cudnn.flags(enabled=False):
        activation = functional.batch_norm(
            x_high_precision,
            None,
            None,
            weight=tensors.scale.reshape(-1),
            bias=tensors.bias.reshape(-1),
            training=True,
            momentum=MOMENTUM,
            eps=EPSILON,
        ).to(bn_case.output_dtype)

    if bn_case.pattern.has_add:
        if tensors.z is None:
            raise ValueError(f"missing Z tensor for add case {bn_case.case_id}")
        activation = (activation.float() + tensors.z.float()).to(bn_case.output_dtype)

    mask = None
    relu_boundary = None
    relu_boundary_packed = None
    if bn_case.pattern.has_relu:
        activation_high_precision = activation.float()
        mask = _pack_boolean_mask(activation_high_precision > 0)
        relu_boundary = activation_high_precision.abs() <= RELU_BOUNDARY_ATOL
        relu_boundary_packed = _pack_boolean_mask(relu_boundary)
        activation = torch.relu(activation)

    return {
        "y": activation,
        "statistics": statistics,
        "relu_boundary": relu_boundary,
        "relu_boundary_packed": relu_boundary_packed,
        "mask": mask,
    }


def _run_batchnorm_forward_case(bn_case: BatchNormCase, cudnn_handle) -> None:
    tensors = _make_batchnorm_forward_tensors(bn_case)
    reference = _batchnorm_forward_reference(bn_case, tensors)

    graph = new_batchnorm_graph(cudnn_handle, bn_case.input_dtype)
    x = graph.tensor(name="X", dim=tensors.x.size(), stride=tensors.x.stride(), data_type=tensors.x.dtype)
    scale = graph.tensor_like(tensors.scale)
    bias = graph.tensor_like(tensors.bias)
    in_running_mean = graph.tensor_like(tensors.running_mean)
    in_running_variance = graph.tensor_like(tensors.running_variance)
    epsilon = graph.tensor_like(tensors.epsilon, name="epsilon")
    momentum = graph.tensor_like(tensors.momentum, name="momentum")

    activation, saved_mean, saved_inv_variance, out_running_mean, out_running_variance = graph.batchnorm(
        name="BatchNorm",
        input=x,
        scale=scale,
        bias=bias,
        in_running_mean=in_running_mean,
        in_running_var=in_running_variance,
        epsilon=epsilon,
        momentum=momentum,
    )
    activation.set_data_type(torch_to_cudnn_data_type(bn_case.output_dtype))

    variant_pack = {
        x: tensors.x,
        scale: tensors.scale,
        bias: tensors.bias,
        in_running_mean: tensors.running_mean,
        in_running_variance: tensors.running_variance,
        epsilon: tensors.epsilon,
        momentum: tensors.momentum,
    }
    if bn_case.pattern.has_add:
        if tensors.z is None:
            raise ValueError(f"missing Z tensor for add case {bn_case.case_id}")
        z = graph.tensor(name="Z", dim=tensors.z.size(), stride=tensors.z.stride(), data_type=tensors.z.dtype)
        activation = graph.add(a=activation, b=z, name="add")
        activation.set_data_type(torch_to_cudnn_data_type(bn_case.output_dtype))
        variant_pack[z] = tensors.z

    mask = None
    if bn_case.pattern.has_relu:
        activation = graph.relu(name="relu", input=activation)
        activation.set_data_type(torch_to_cudnn_data_type(bn_case.output_dtype))
        zero = graph.tensor_like(tensors.zero, name="zero")
        mask = graph.cmp_gt(name="mask", input=activation, comparison=zero)
        mask.set_output(True).set_data_type(cudnn.data_type.BOOLEAN)
        variant_pack[zero] = tensors.zero

    activation.set_output(True)
    for output in (saved_mean, saved_inv_variance, out_running_mean, out_running_variance):
        output.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    finalize_graph(graph)

    y_actual = torch.empty_strided(bn_case.shape, _packed_nhwc_strides(bn_case.shape), device="cuda", dtype=bn_case.output_dtype)
    saved_mean_actual = torch.empty_like(tensors.scale)
    saved_inv_variance_actual = torch.empty_like(tensors.scale)
    running_mean_actual = torch.empty_like(tensors.running_mean)
    running_variance_actual = torch.empty_like(tensors.running_variance)

    variant_pack.update(
        {
            activation: y_actual,
            saved_mean: saved_mean_actual,
            saved_inv_variance: saved_inv_variance_actual,
            out_running_mean: running_mean_actual,
            out_running_variance: running_variance_actual,
        }
    )
    mask_actual = None
    if mask is not None:
        packed_mask_shape = (bn_case.shape[0], bn_case.shape[1] // 8, bn_case.shape[2], bn_case.shape[3])
        mask_actual = torch.empty_strided(packed_mask_shape, _packed_nhwc_strides(packed_mask_shape), device="cuda", dtype=torch.uint8)
        mask_actual.fill_(MASK_OUTPUT_INITIAL_VALUE)
        variant_pack[mask] = mask_actual

    execute_graph(graph, variant_pack, cudnn_handle)

    _assert_close(
        "Y",
        y_actual,
        reference["y"],
        rtol=bn_case.rtol,
        atol=bn_case.atol,
        max_mismatch_rate=bn_case.max_mismatch_rate,
        allowed_mismatch_mask=reference["relu_boundary"],
        outlier_rtol=bn_case.rtol,
        # Allow one unit-scale rounding step near ReLU cancellation, not arbitrary outliers.
        outlier_atol=bn_case.atol + torch.finfo(bn_case.output_dtype).eps,
    )
    _assert_batchnorm_training_statistics(
        saved_mean=saved_mean_actual,
        saved_inv_variance=saved_inv_variance_actual,
        running_mean=running_mean_actual,
        running_variance=running_variance_actual,
        expected=reference["statistics"],
    )
    if mask_actual is not None:
        mask_mismatch_bits = torch.bitwise_xor(mask_actual, reference["mask"])
        mask_mismatch_bits.bitwise_and_(torch.bitwise_not(reference["relu_boundary_packed"]))
        if mask_mismatch_bits.any():
            raise AssertionError(f"mask has mismatches in {mask_mismatch_bits.count_nonzero().item()} packed bytes away from the ReLU boundary")


def _make_batchnorm_backward_tensors(bn_case: BatchNormCase) -> BatchNormBackwardTensors:
    if bn_case.grad_dtype is None:
        raise ValueError(f"missing gradient dtype for backward case {bn_case.case_id}")
    x = _randn_strided(bn_case.shape, bn_case.input_dtype, bn_case.seed, scale=0.5)
    dy = _randn_strided(bn_case.shape, bn_case.grad_dtype, bn_case.seed + 1, scale=0.5)
    scale = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 2, scale=0.25, shift=1.0)
    mask = None
    if bn_case.pattern.has_backward_mask:
        mask_values = _randn_strided(bn_case.shape, torch.float16, bn_case.seed + 3)
        mask = mask_values > 0
    return BatchNormBackwardTensors(x, dy, scale, mask)


def _batchnorm_backward_reference(
    bn_case: BatchNormCase,
    tensors: BatchNormBackwardTensors,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
):
    if bn_case.grad_dtype is None:
        raise ValueError(f"missing gradient dtype for backward case {bn_case.case_id}")

    grad = tensors.dy
    if bn_case.pattern.has_backward_mask:
        if tensors.mask is None:
            raise ValueError(f"missing mask tensor for dReLU case {bn_case.case_id}")
        grad = (grad.float() * tensors.mask).to(bn_case.grad_dtype)

    x_high_precision = tensors.x.float()
    grad_high_precision = grad.float()
    reduction_dims = (0, 2, 3)
    normalized = (x_high_precision - mean) * inv_variance
    dscale = (grad_high_precision * normalized).sum(dim=reduction_dims, keepdim=True)
    dbias = grad_high_precision.sum(dim=reduction_dims, keepdim=True)
    sample_count = tensors.x.shape[0] * tensors.x.shape[2] * tensors.x.shape[3]
    dx = (tensors.scale * inv_variance / sample_count) * (sample_count * grad_high_precision - dbias - normalized * dscale)

    return {
        "dx": dx.to(bn_case.output_dtype),
        "dscale": dscale,
        "dbias": dbias,
        "dadd": grad if bn_case.pattern.has_add else None,
    }


def _run_batchnorm_backward_case(bn_case: BatchNormCase, cudnn_handle) -> None:
    if bn_case.grad_dtype is None:
        raise ValueError(f"missing gradient dtype for backward case {bn_case.case_id}")
    tensors = _make_batchnorm_backward_tensors(bn_case)
    moments = _batchnorm_moments(tensors.x.float())
    reference = _batchnorm_backward_reference(bn_case, tensors, moments.mean, moments.inv_variance)

    graph = new_batchnorm_graph(cudnn_handle, bn_case.grad_dtype)
    x = graph.tensor(name="X", dim=tensors.x.size(), stride=tensors.x.stride(), data_type=tensors.x.dtype)
    dy = graph.tensor(name="DY", dim=tensors.dy.size(), stride=tensors.dy.stride(), data_type=tensors.dy.dtype)
    variant_pack = {x: tensors.x, dy: tensors.dy}

    grad = dy
    dadd = None
    if bn_case.pattern.has_backward_mask:
        if tensors.mask is None:
            raise ValueError(f"missing mask tensor for dReLU case {bn_case.case_id}")
        mask = graph.tensor(
            name="mask",
            dim=bn_case.shape,
            stride=_packed_nhwc_strides(bn_case.shape),
            data_type=cudnn.data_type.BOOLEAN,
        )
        grad = graph.mul(a=dy, b=mask, name="mask_mul")
        grad.set_data_type(torch_to_cudnn_data_type(bn_case.grad_dtype))
        variant_pack[mask] = _pack_boolean_mask(tensors.mask)
        if bn_case.pattern.has_add:
            dadd = grad.set_output(True)

    scale = graph.tensor_like(tensors.scale)
    mean = graph.tensor_like(moments.mean)
    inv_variance = graph.tensor_like(moments.inv_variance)
    dx, dscale, dbias = graph.batchnorm_backward(
        name="BatchNormBackward",
        grad=grad,
        input=x,
        scale=scale,
        mean=mean,
        inv_variance=inv_variance,
    )
    dx.set_output(True).set_data_type(torch_to_cudnn_data_type(bn_case.output_dtype))
    dscale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    dbias.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    variant_pack.update({scale: tensors.scale, mean: moments.mean, inv_variance: moments.inv_variance})

    finalize_graph(graph)

    dx_actual = torch.empty_strided(bn_case.shape, _packed_nhwc_strides(bn_case.shape), device="cuda", dtype=bn_case.output_dtype)
    dscale_actual = torch.empty_like(tensors.scale)
    dbias_actual = torch.empty_like(tensors.scale)
    variant_pack.update({dx: dx_actual, dscale: dscale_actual, dbias: dbias_actual})

    dadd_actual = None
    if dadd is not None:
        dadd_actual = torch.empty_strided(bn_case.shape, _packed_nhwc_strides(bn_case.shape), device="cuda", dtype=bn_case.grad_dtype)
        variant_pack[dadd] = dadd_actual

    execute_graph(graph, variant_pack, cudnn_handle)

    _assert_close("DX", dx_actual, reference["dx"], rtol=bn_case.rtol, atol=bn_case.atol)
    _assert_close("DScale", dscale_actual, reference["dscale"], rtol=bn_case.rtol, atol=bn_case.atol)
    _assert_close("DBias", dbias_actual, reference["dbias"], rtol=bn_case.rtol, atol=bn_case.atol)
    if dadd_actual is not None:
        _assert_close("DAdd", dadd_actual, reference["dadd"], rtol=bn_case.rtol, atol=bn_case.atol)


def _make_fp8_forward_tensors(bn_case: BatchNormCase) -> FP8ForwardTensors:
    x = _randn_strided(bn_case.shape, bn_case.input_dtype, bn_case.seed, scale=0.5)
    scale = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 1, scale=0.25, shift=1.0)
    bias = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 2, scale=0.25)
    running_mean = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 3, scale=0.25)
    running_variance = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 4, scale=0.1, shift=1.0)
    running_variance.abs_()
    z = _randn_strided(bn_case.shape, bn_case.input_dtype, bn_case.seed + 5, scale=0.25) if bn_case.pattern.has_add else None
    scalar = (1, 1, 1, 1)
    x_descale = torch.full(scalar, 0.5, device="cuda", dtype=torch.float32)
    z_descale = torch.full(scalar, 0.75, device="cuda", dtype=torch.float32)
    y_scale = torch.full(scalar, 1.25, device="cuda", dtype=torch.float32)
    epsilon = torch.full(scalar, EPSILON, device="cpu", dtype=torch.float64)
    momentum = torch.full(scalar, MOMENTUM, device="cpu", dtype=torch.float64)
    zero = torch.zeros(scalar, device="cpu", dtype=torch.float16)
    return FP8ForwardTensors(x, scale, bias, running_mean, running_variance, z, x_descale, z_descale, y_scale, epsilon, momentum, zero)


def _fp8_forward_reference(bn_case: BatchNormCase, tensors: FP8ForwardTensors):
    activation = tensors.x.float()
    activation.mul_(tensors.x_descale)
    statistics = _compute_batchnorm_training_statistics(activation, tensors.running_mean, tensors.running_variance)
    activation.sub_(statistics.saved_mean).mul_(statistics.saved_inv_variance).mul_(tensors.scale).add_(tensors.bias)

    if bn_case.pattern.has_add:
        if tensors.z is None:
            raise ValueError(f"missing Z tensor for add case {bn_case.case_id}")
        z_high_precision = tensors.z.float()
        z_high_precision.mul_(tensors.z_descale)
        activation.add_(z_high_precision)
        del z_high_precision

    mask = None
    relu_boundary = None
    relu_boundary_packed = None
    if bn_case.pattern.has_relu:
        mask = _pack_boolean_mask(activation > 0)
        relu_boundary = activation >= -RELU_BOUNDARY_ATOL
        relu_boundary.logical_and_(activation <= RELU_BOUNDARY_ATOL)
        relu_boundary_packed = _pack_boolean_mask(relu_boundary)
        activation.relu_()

    amax = activation.abs().amax().reshape(1, 1, 1, 1)
    activation.mul_(tensors.y_scale)

    return {
        "y": activation.to(bn_case.output_dtype),
        "statistics": statistics,
        "amax": amax,
        "relu_boundary": relu_boundary,
        "relu_boundary_packed": relu_boundary_packed,
        "mask": mask,
    }


def _pack_boolean_mask(mask: torch.Tensor) -> torch.Tensor:
    # cuDNN stores logical channel c in bit c % 8 of byte c // 8 while retaining the full logical tensor dimensions.
    batch, channels, height, width = mask.shape
    if channels % 8 != 0:
        raise ValueError("packed BatchNorm masks require a channel count divisible by 8")
    channel_groups = mask.permute(0, 2, 3, 1).reshape(batch, height, width, channels // 8, 8)
    shifts = torch.arange(8, device=mask.device, dtype=torch.uint8)
    packed = (channel_groups.to(torch.uint8) << shifts).sum(dim=-1).to(torch.uint8)
    return packed.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


def _run_fp8_forward_case(bn_case: BatchNormCase, cudnn_handle) -> None:
    _require_fp8_support(bn_case)
    tensors = _make_fp8_forward_tensors(bn_case)
    reference = _fp8_forward_reference(bn_case, tensors)

    graph = new_batchnorm_graph(cudnn_handle, bn_case.input_dtype)
    x = graph.tensor(name="X", dim=tensors.x.size(), stride=tensors.x.stride(), data_type=bn_case.input_dtype)
    x_descale = graph.tensor_like(tensors.x_descale)
    x_high_precision = graph.mul(a=x, b=x_descale, name="x_descale")
    x_high_precision.set_data_type(cudnn.data_type.FLOAT)
    scale = graph.tensor_like(tensors.scale)
    bias = graph.tensor_like(tensors.bias)
    in_running_mean = graph.tensor_like(tensors.running_mean)
    in_running_variance = graph.tensor_like(tensors.running_variance)
    epsilon = graph.tensor_like(tensors.epsilon, name="epsilon")
    momentum = graph.tensor_like(tensors.momentum, name="momentum")

    activation, saved_mean, saved_inv_variance, out_running_mean, out_running_variance = graph.batchnorm(
        name="BatchNorm",
        input=x_high_precision,
        scale=scale,
        bias=bias,
        in_running_mean=in_running_mean,
        in_running_var=in_running_variance,
        epsilon=epsilon,
        momentum=momentum,
    )
    activation.set_data_type(cudnn.data_type.FLOAT)

    variant_pack = {
        x: tensors.x,
        x_descale: tensors.x_descale,
        scale: tensors.scale,
        bias: tensors.bias,
        in_running_mean: tensors.running_mean,
        in_running_variance: tensors.running_variance,
        epsilon: tensors.epsilon,
        momentum: tensors.momentum,
    }
    if bn_case.pattern.has_add:
        if tensors.z is None:
            raise ValueError(f"missing Z tensor for add case {bn_case.case_id}")
        z = graph.tensor(name="Z", dim=tensors.z.size(), stride=tensors.z.stride(), data_type=bn_case.input_dtype)
        z_descale = graph.tensor_like(tensors.z_descale)
        z_high_precision = graph.mul(a=z, b=z_descale, name="z_descale")
        z_high_precision.set_data_type(cudnn.data_type.FLOAT)
        activation = graph.add(a=activation, b=z_high_precision, name="add")
        activation.set_data_type(cudnn.data_type.FLOAT)
        variant_pack.update({z: tensors.z, z_descale: tensors.z_descale})

    mask = None
    if bn_case.pattern.has_relu:
        activation = graph.relu(name="relu", input=activation)
        activation.set_data_type(cudnn.data_type.FLOAT)
        zero = graph.tensor_like(tensors.zero, name="zero")
        mask = graph.cmp_gt(name="mask", input=activation, comparison=zero)
        mask.set_output(True).set_data_type(cudnn.data_type.BOOLEAN)
        variant_pack[zero] = tensors.zero

    y_scale = graph.tensor_like(tensors.y_scale)
    y = graph.mul(a=activation, b=y_scale, name="y_scale")
    y.set_output(True).set_data_type(torch_to_cudnn_data_type(bn_case.output_dtype))
    amax = graph.reduction(input=activation, mode=cudnn.reduction_mode.AMAX, name="amax")
    _mark_scalar_float_output(amax)
    for output in (saved_mean, saved_inv_variance, out_running_mean, out_running_variance):
        output.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    variant_pack[y_scale] = tensors.y_scale

    finalize_graph(graph)

    y_actual = torch.empty_strided(bn_case.shape, _packed_nhwc_strides(bn_case.shape), device="cuda", dtype=bn_case.output_dtype)
    saved_mean_actual = torch.empty_like(tensors.scale)
    saved_inv_variance_actual = torch.empty_like(tensors.scale)
    running_mean_actual = torch.empty_like(tensors.running_mean)
    running_variance_actual = torch.empty_like(tensors.running_variance)
    amax_actual = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
    variant_pack.update(
        {
            y: y_actual,
            saved_mean: saved_mean_actual,
            saved_inv_variance: saved_inv_variance_actual,
            out_running_mean: running_mean_actual,
            out_running_variance: running_variance_actual,
            amax: amax_actual,
        }
    )
    mask_actual = None
    if mask is not None:
        packed_mask_shape = (bn_case.shape[0], bn_case.shape[1] // 8, bn_case.shape[2], bn_case.shape[3])
        mask_actual = torch.empty_strided(packed_mask_shape, _packed_nhwc_strides(packed_mask_shape), device="cuda", dtype=torch.uint8)
        mask_actual.fill_(MASK_OUTPUT_INITIAL_VALUE)
        variant_pack[mask] = mask_actual

    execute_graph(graph, variant_pack, cudnn_handle)

    _assert_close(
        "Y",
        y_actual,
        reference["y"],
        rtol=bn_case.rtol,
        atol=bn_case.atol,
        max_mismatch_rate=bn_case.max_mismatch_rate,
        allowed_mismatch_mask=reference["relu_boundary"],
        outlier_rtol=FP8_OUTLIER_RTOL if reference["relu_boundary"] is not None else None,
        outlier_atol=FP8_OUTLIER_ATOL if reference["relu_boundary"] is not None else None,
    )
    _assert_batchnorm_training_statistics(
        saved_mean=saved_mean_actual,
        saved_inv_variance=saved_inv_variance_actual,
        running_mean=running_mean_actual,
        running_variance=running_variance_actual,
        expected=reference["statistics"],
    )
    _assert_close("amax", amax_actual, reference["amax"], rtol=FP32_OUTPUT_RTOL, atol=FP32_OUTPUT_ATOL)
    if mask_actual is not None:
        mask_mismatch_bits = torch.bitwise_xor(mask_actual, reference["mask"])
        mask_mismatch_bits.bitwise_and_(torch.bitwise_not(reference["relu_boundary_packed"]))
        if mask_mismatch_bits.any():
            raise AssertionError(f"mask has mismatches in {mask_mismatch_bits.count_nonzero().item()} packed bytes away from the ReLU boundary")


def _make_fp8_backward_tensors(bn_case: BatchNormCase) -> FP8BackwardTensors:
    x = _randn_strided(bn_case.shape, bn_case.input_dtype, bn_case.seed, scale=0.5)
    dy = _randn_strided(bn_case.shape, bn_case.grad_dtype, bn_case.seed + 1, scale=0.5)
    scale = _randn_strided(bn_case.parameter_shape, torch.float32, bn_case.seed + 2, scale=0.25, shift=1.0)
    mask = None
    if bn_case.pattern.has_backward_mask:
        mask_values = _randn_strided(bn_case.shape, torch.float16, bn_case.seed + 3)
        mask = mask_values > 0
    scalar = (1, 1, 1, 1)
    x_descale = torch.full(scalar, 0.5, device="cuda", dtype=torch.float32)
    dy_descale = torch.full(scalar, 0.5, device="cuda", dtype=torch.float32)
    dx_scale = torch.full(scalar, 1.25, device="cuda", dtype=torch.float32)
    add_scale = torch.full(scalar, 0.75, device="cuda", dtype=torch.float32)
    return FP8BackwardTensors(x, dy, scale, mask, x_descale, dy_descale, dx_scale, add_scale)


def _fp8_backward_reference(
    bn_case: BatchNormCase,
    tensors: FP8BackwardTensors,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
):
    x_high_precision = tensors.x.float()
    x_high_precision.mul_(tensors.x_descale)
    dy_high_precision = tensors.dy.float()
    dy_high_precision.mul_(tensors.dy_descale)
    if bn_case.pattern.has_backward_mask:
        if tensors.mask is None:
            raise ValueError(f"missing mask tensor for dReLU case {bn_case.case_id}")
        dy_high_precision.mul_(tensors.mask)

    reduction_dims = (0, 2, 3)
    normalized = x_high_precision.sub_(mean).mul_(inv_variance)
    dscale = (dy_high_precision * normalized).sum(dim=reduction_dims, keepdim=True)
    dbias = dy_high_precision.sum(dim=reduction_dims, keepdim=True)
    sample_count = x_high_precision.shape[0] * x_high_precision.shape[2] * x_high_precision.shape[3]
    dx_high_precision = (tensors.scale * inv_variance / sample_count) * (sample_count * dy_high_precision - dbias - normalized * dscale)

    return {
        "dx": (dx_high_precision * tensors.dx_scale).to(bn_case.output_dtype),
        "dscale": dscale,
        "dbias": dbias,
        "dx_amax": dx_high_precision.abs().amax().reshape(1, 1, 1, 1),
        "dadd": (dy_high_precision * tensors.add_scale).to(bn_case.grad_dtype) if bn_case.pattern.has_add else None,
        "dadd_amax": dy_high_precision.abs().amax().reshape(1, 1, 1, 1) if bn_case.pattern.has_add else None,
    }


def _mark_scalar_float_output(tensor):
    return tensor.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1))


def _run_fp8_backward_case(bn_case: BatchNormCase, cudnn_handle) -> None:
    _require_fp8_support(bn_case)
    tensors = _make_fp8_backward_tensors(bn_case)
    x_high_precision = tensors.x.float()
    x_high_precision.mul_(tensors.x_descale)
    moments = _batchnorm_moments(x_high_precision)
    del x_high_precision
    mean_gpu = moments.mean
    inv_variance_gpu = moments.inv_variance
    reference = _fp8_backward_reference(bn_case, tensors, mean_gpu, inv_variance_gpu)

    graph = new_batchnorm_graph(cudnn_handle, bn_case.grad_dtype)
    x = graph.tensor(name="X", dim=tensors.x.size(), stride=tensors.x.stride(), data_type=bn_case.input_dtype)
    dy = graph.tensor(name="DY", dim=tensors.dy.size(), stride=tensors.dy.stride(), data_type=bn_case.grad_dtype)
    x_descale = graph.tensor_like(tensors.x_descale)
    dy_descale = graph.tensor_like(tensors.dy_descale)
    x_high_precision = graph.mul(a=x, b=x_descale, name="x_descale")
    dy_high_precision = graph.mul(a=dy, b=dy_descale, name="dy_descale")
    x_high_precision.set_data_type(cudnn.data_type.FLOAT)
    dy_high_precision.set_data_type(cudnn.data_type.FLOAT)

    variant_pack = {x: tensors.x, dy: tensors.dy, x_descale: tensors.x_descale, dy_descale: tensors.dy_descale}
    dadd = None
    dadd_amax = None
    if bn_case.pattern.has_backward_mask:
        if tensors.mask is None:
            raise ValueError(f"missing mask tensor for dReLU case {bn_case.case_id}")
        mask = graph.tensor(
            name="mask",
            dim=bn_case.shape,
            stride=_packed_nhwc_strides(bn_case.shape),
            data_type=cudnn.data_type.BOOLEAN,
        )
        dy_high_precision = graph.mul(a=dy_high_precision, b=mask, name="mask_mul")
        dy_high_precision.set_data_type(cudnn.data_type.FLOAT)
        variant_pack[mask] = _pack_boolean_mask(tensors.mask)

        if bn_case.pattern.has_add:
            add_scale = graph.tensor_like(tensors.add_scale)
            dadd = graph.mul(a=dy_high_precision, b=add_scale, name="add_scale")
            dadd.set_output(True).set_data_type(torch_to_cudnn_data_type(bn_case.grad_dtype))
            dadd_amax = graph.reduction(input=dy_high_precision, mode=cudnn.reduction_mode.AMAX, name="dadd_amax")
            _mark_scalar_float_output(dadd_amax)
            variant_pack[add_scale] = tensors.add_scale

    scale = graph.tensor_like(tensors.scale)
    mean = graph.tensor_like(mean_gpu)
    inv_variance = graph.tensor_like(inv_variance_gpu)
    dx_high_precision, dscale, dbias = graph.batchnorm_backward(
        name="BatchNormBackward",
        grad=dy_high_precision,
        input=x_high_precision,
        scale=scale,
        mean=mean,
        inv_variance=inv_variance,
    )
    dx_high_precision.set_data_type(cudnn.data_type.FLOAT)
    dscale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    dbias.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    dx_scale = graph.tensor_like(tensors.dx_scale)
    dx = graph.mul(a=dx_high_precision, b=dx_scale, name="dx_scale")
    dx.set_output(True).set_data_type(torch_to_cudnn_data_type(bn_case.output_dtype))
    dx_amax = graph.reduction(input=dx_high_precision, mode=cudnn.reduction_mode.AMAX, name="dx_amax")
    _mark_scalar_float_output(dx_amax)
    variant_pack.update(
        {
            scale: tensors.scale,
            mean: mean_gpu,
            inv_variance: inv_variance_gpu,
            dx_scale: tensors.dx_scale,
        }
    )

    finalize_graph(graph)

    dx_actual = torch.empty_strided(bn_case.shape, _packed_nhwc_strides(bn_case.shape), device="cuda", dtype=bn_case.output_dtype)
    dscale_actual = torch.empty_like(tensors.scale)
    dbias_actual = torch.empty_like(tensors.scale)
    dx_amax_actual = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
    variant_pack.update({dx: dx_actual, dscale: dscale_actual, dbias: dbias_actual, dx_amax: dx_amax_actual})

    dadd_actual = None
    dadd_amax_actual = None
    if dadd is not None:
        dadd_actual = torch.empty_strided(bn_case.shape, _packed_nhwc_strides(bn_case.shape), device="cuda", dtype=bn_case.grad_dtype)
        dadd_amax_actual = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
        variant_pack.update({dadd: dadd_actual, dadd_amax: dadd_amax_actual})

    execute_graph(graph, variant_pack, cudnn_handle)

    _assert_close("DX", dx_actual, reference["dx"], rtol=bn_case.rtol, atol=bn_case.atol)
    _assert_close("DScale", dscale_actual, reference["dscale"], rtol=FP32_REDUCTION_RTOL, atol=FP32_REDUCTION_ATOL)
    _assert_close("DBias", dbias_actual, reference["dbias"], rtol=FP32_REDUCTION_RTOL, atol=FP32_REDUCTION_ATOL)
    _assert_close("DX AMAX", dx_amax_actual, reference["dx_amax"], rtol=FP32_OUTPUT_RTOL, atol=FP32_OUTPUT_ATOL)
    if dadd_actual is not None:
        _assert_close("DAdd", dadd_actual, reference["dadd"], rtol=bn_case.rtol, atol=bn_case.atol)
        _assert_close("DAdd AMAX", dadd_amax_actual, reference["dadd_amax"], rtol=FP32_OUTPUT_RTOL, atol=FP32_OUTPUT_ATOL)


def _run_case(bn_case: BatchNormCase, cudnn_handle) -> None:
    if bn_case.pattern.is_forward and not bn_case.uses_fp8:
        _run_batchnorm_forward_case(bn_case, cudnn_handle)
    elif bn_case.pattern.is_forward:
        _run_fp8_forward_case(bn_case, cudnn_handle)
    elif bn_case.pattern.is_backward and not bn_case.uses_fp8:
        _run_batchnorm_backward_case(bn_case, cudnn_handle)
    elif bn_case.pattern.is_backward:
        _run_fp8_backward_case(bn_case, cudnn_handle)
    else:
        raise ValueError(f"unsupported BatchNorm pattern: {bn_case.pattern}")


def _case_parameter(bn_case: BatchNormCase):
    try:
        level_marker = {"L0": pytest.mark.L0, "L1": pytest.mark.L1}[bn_case.level]
    except KeyError as exc:
        raise ValueError(f"unsupported pytest level for {bn_case.case_id}: {bn_case.level}") from exc
    return pytest.param(bn_case, marks=level_marker, id=bn_case.case_id)


@pytest.mark.parametrize("bn_case", tuple(_case_parameter(bn_case) for bn_case in BATCHNORM_CASES))
@preserve_handle_stream
def test_batchnorm_fusion(bn_case: BatchNormCase, cudnn_handle):
    _run_case(bn_case, cudnn_handle)
